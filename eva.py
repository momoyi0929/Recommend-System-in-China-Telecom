import torch
import joblib
import pandas as pd
from model import TwoTowerModel
import matplotlib
import torch
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy.stats import norm


# 设备选择
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

# 读取数据
user_df = pd.read_csv("/Users/momoyi0929/Desktop/测试数据3/user.csv")
item_df = pd.read_csv("/Users/momoyi0929/Desktop/测试数据3/item.csv")
interaction_df = pd.read_csv("/Users/momoyi0929/Desktop/测试数据3/label.csv")

item_df = item_df.drop_duplicates(subset=["产品ID"]).reset_index(drop=True)
user_df = user_df.drop_duplicates(subset=["phone"]).reset_index(drop=True)

# 从文件中加载编码器和scaler
encoders = joblib.load("/Users/momoyi0929/Desktop/模型数据3/encoders.pkl")
user_encoder = joblib.load("/Users/momoyi0929/Desktop/模型数据3/user_id_encoder.pkl")
item_encoder = joblib.load("/Users/momoyi0929/Desktop/模型数据3/item_id_encoder.pkl")
user_scaler = joblib.load("/Users/momoyi0929/Desktop/模型数据3/user_scaler.pkl")
item_scaler = joblib.load("/Users/momoyi0929/Desktop/模型数据3/item_scaler.pkl")

# 读取测试数据
test_data = torch.load('/Users/momoyi0929/Desktop/模型数据3/train_data.pt')
print("数据读取完毕！")

# 预处理类别特征
user_categ_features = ["pd_inst_id"]
item_categ_features = ["业务", "分类1", "支付方式", "产品有效期", "是否有附属权益", "是否有协议期", "收费策略",
                       "退订规则"]

for col in user_categ_features:
    user_df[col] = encoders[col].transform(user_df[col])
for col in item_categ_features:
    item_df[col] = encoders[col].transform(item_df[col])

# 预处理数值特征
user_num_features = [
    "po_type", "city_nm", "gender", "cust_tp", "age_f",
    "main_limit_mon_chrg_lvl", "ob_f", "data_flow_f", "date_flow_use_pr", "qy_by", "qy_dc",
    "short_video_f", "game_f", "webcast_f", "music_f", "entertainment_f", "local_life_f",
    "webclass_f", "parenting_education_f", "employee_f", "trip_f", "catering_f",
    "read_and_listen_f", "ebusiness_f", "payment_f", "txvideo_f", "youku_f", "bilibili_f",
    "mango_f", "txsyjsq_f", "iqiyi_f", "wangzhe_f", "bdwp_f", "keep_f", "meituan_f",
    "zyb_f", "tyyp_f", "xunlei_f", "zhihu_f", "xmly_kid_f", "yax_f", "didi_f", "gddt_f",
    "mtwm_f", "mcdonold_f", "kfc_f", "luckin_f", "naixue_f", "xicha_f", "chayan_f",
    "mixue_f", "qqyuedu_f", "zhangyue_f", "xmly_f", "ksjgs_f", "qmxs_f", "jd_f", "yzf_f",
    "didacx_f"
]
item_num_features = ["定价策略（元）"]

user_df[user_num_features] = user_scaler.transform(user_df[user_num_features])
item_df[item_num_features] = item_scaler.transform(item_df[item_num_features])

# 处理ID
user_df["phone"] = user_encoder.transform(user_df["phone"])
item_df["产品ID"] = item_encoder.transform(item_df["产品ID"])
interaction_df["phone"] = user_encoder.transform(interaction_df["phone"])
interaction_df["goods_id"] = item_encoder.transform(interaction_df["goods_id"])

print("数据处理完毕！")

# 建立模型
num_users = len(user_df)
num_items = len(item_df)
user_categ_sizes = [user_df[col].nunique() for col in user_categ_features]
item_categ_sizes = [item_df[col].nunique() for col in item_categ_features]
embedding_dim = 512

model = TwoTowerModel(len(user_num_features), len(item_num_features),
                      user_categ_sizes, item_categ_sizes, embedding_dim).to(device)

# 加载模型
checkpoint = torch.load("/Users/momoyi0929/Desktop/模型数据3/best_model.pth", map_location=device)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()
print("模型加载完毕！")


def calculate_scores(user_phone, model):
    # 1. 获取用户ID
    user_id = torch.tensor([user_encoder.transform([user_phone])[0]], dtype=torch.long).to(device)

    # 2. 获取用户特征
    user_row = user_df[user_df["phone"] == user_id.item()].iloc[0]
    user_features = torch.tensor(user_row[user_num_features].values, dtype=torch.float32).unsqueeze(0).to(device)
    user_categ = torch.tensor(user_row[user_categ_features].values, dtype=torch.long).unsqueeze(0).to(device)

    # 3. 获取所有商品特征
    item_ids = torch.tensor(item_df["产品ID"].values, dtype=torch.long).to(device)
    item_features = torch.tensor(item_df[item_num_features].values, dtype=torch.float32).to(device)
    item_categ = torch.tensor(item_df[item_categ_features].values, dtype=torch.long).to(device)

    # 扩展用户特征以匹配商品数量
    user_id = user_id.expand(len(item_ids))
    user_features = user_features.expand(len(item_ids), -1)
    user_categ = user_categ.expand(len(item_ids), -1)

    # 4. 计算分数
    model.eval()
    with torch.no_grad():
        scores = model(user_id, item_ids, user_features, user_categ, item_features, item_categ)

    # 转换为numpy数组并获取原始商品ID
    scores = scores.cpu().numpy().flatten()
    original_item_ids = item_encoder.inverse_transform(item_ids.cpu().numpy())

    return scores, original_item_ids

def plot_similarity_distribution(phone, model):
    scores, _ = calculate_scores(phone, model)

    plt.figure(figsize=(10, 6))
    sns.histplot(scores, bins=50, kde=True, stat="density", label="Predicted Scores")

    mu, sigma = np.mean(scores), np.std(scores)
    x = np.linspace(mu - 3 * sigma, mu + 3 * sigma, 100)
    plt.plot(x, norm.pdf(x, mu, sigma), 'r-', label=f'Normal Dist. ($\mu$={mu:.2f}, $\sigma$={sigma:.2f})')

    plt.xlabel("Predicted Similarity Score")
    plt.ylabel("Density")
    plt.title(f"Similarity Score Distribution for User {phone}")
    plt.legend()
    plt.grid()
    plt.show()

def recall_at_k(user_phone, model, k=3, report=False):
    # 1. 获取用户历史购买商品
    user_id = user_encoder.transform([user_phone])[0]
    purchased_items = interaction_df[(interaction_df["phone"] == user_id) &
                                     (interaction_df["label"] == 1)]["goods_id"].values
    purchased_items = item_encoder.inverse_transform(purchased_items)

    if len(purchased_items) == 0:
        if report:
            print(f"用户 {user_phone} 没有历史购买记录，无法计算 Recall@{k}")
        return None

    # 2. 获取Top-K推荐
    scores, original_item_ids = calculate_scores(user_phone, model)
    top_k_indices = np.argsort(scores)[-k:][::-1]  # 降序排列
    top_k_items = original_item_ids[top_k_indices]

    # 3. 计算Recall@K
    hits = sum(1 for item in purchased_items if item in top_k_items)
    recall = hits / len(purchased_items)

    if report:
        print(f"用户 {user_phone} 的 Top-{k} 推荐商品: {top_k_items}")
        print(f"用户实际购买商品: {purchased_items}")
        print(f"Recall@{k}: {recall:.4f}")
        print()

    return recall

def precision_at_k(user_phone, model, k=3, report=False):
    # 1. 获取用户历史购买商品
    user_id = user_encoder.transform([user_phone])[0]
    purchased_items = interaction_df[(interaction_df["phone"] == user_id) &
                                     (interaction_df["label"] == 1)]["goods_id"].values
    purchased_items = item_encoder.inverse_transform(purchased_items)

    if len(purchased_items) == 0:
        if report:
            print(f"用户 {user_phone} 没有历史购买记录，无法计算 Recall@{k}")
        return None

    # 2. 获取Top-K推荐
    scores, original_item_ids = calculate_scores(user_phone, model)
    top_k_indices = np.argsort(scores)[-k:][::-1]  # 降序排列
    top_k_items = original_item_ids[top_k_indices]

    # 3. 计算Recall@K
    hits = sum(1 for item in purchased_items if item in top_k_items)
    precision = hits / k

    if report:
        print(f"用户 {user_phone} 的 Top-{k} 推荐商品: {top_k_items}")
        print(f"用户实际购买商品: {purchased_items}")
        print(f"Precision@{k}: {precision:.4f}")
        print()

    return precision

def F_value_at_k(user_phone, model, k=3, report=False):
    r = recall_at_k(user_phone, model, k, report)
    p = precision_at_k(user_phone, model, k, report)

    if None in [r,p]:
        return None

    smooth = 1e-6
    f = r * p * 2 / (r + p + smooth)

    if report:
        print(f"F-value@{k}: {f:.4f}")
        print()
    return f

def evaluate_model(model, test_loader, k=3, device='cpu'):
    """评估模型在整个测试集上的表现"""
    model.eval()
    recalls = []
    precisions = []
    f1_scores = []
    count = 0
    length = len(test_loader)
    with torch.no_grad():
        for user_id, item_id, user_features, user_categ, item_features, item_categ, label in test_loader:
            # 移动到设备
            user_id = user_id.to(device)
            ids = user_encoder.inverse_transform(user_id.cpu().numpy())
            for id in ids:
                r = recall_at_k(id, model, k)
                p = precision_at_k(id, model, k)
                f = F_value_at_k(id, model, k)

                if None in [r,p,f]:
                    continue

                recalls.append(r)
                precisions.append(p)
                f1_scores.append(f)

            count += 1

            print(f"\n===== {count} over {length} =====")
            print(f"Users evaluated: {len(recalls)}")
            print(f"Avg Recall@{k}: {np.mean(recalls):.4f}")
            print(f"Avg Precision@{k}: {np.mean(precisions):.4f}")
            print(f"Avg F1@{k}: {np.mean(f1_scores):.4f}")

    # 计算平均指标
    avg_recall = np.mean(recalls) if recalls else 0
    avg_precision = np.mean(precisions) if precisions else 0
    avg_f1 = np.mean(f1_scores) if f1_scores else 0

    print(f"\n===== Evaluation Results @{k} =====")
    print(f"Users evaluated: {len(recalls)}")
    print(f"Avg Recall@{k}: {avg_recall:.4f}")
    print(f"Avg Precision@{k}: {avg_precision:.4f}")
    print(f"Avg F1@{k}: {avg_f1:.4f}")

    return {
        'recall': avg_recall,
        'precision': avg_precision,
        'f1': avg_f1
    }

def recall_at_k_output(user_phone, model, k=3, report=False):
    # 1. 获取用户历史购买商品
    user_id = user_encoder.transform([user_phone])[0]
    purchased_items = interaction_df[(interaction_df["phone"] == user_id) &
                                     (interaction_df["label"] == 1)]["goods_id"].values
    purchased_items = item_encoder.inverse_transform(purchased_items)

    if len(purchased_items) == 0:
        if report:
            print(f"用户 {user_phone} 没有历史购买记录，无法计算 Recall@{k}")
        return None

    # 2. 获取Top-K推荐
    scores, original_item_ids = calculate_scores(user_phone, model)
    top_k_indices = np.argsort(scores)[-k:][::-1]  # 降序排列
    top_k_items = original_item_ids[top_k_indices]

    # 3. 计算Recall@K
    hits = sum(1 for item in purchased_items if item in top_k_items)
    recall = hits / len(purchased_items)

    if report:
        print(f"用户 {user_phone} 的 Top-{k} 推荐商品: {top_k_items}")
        print(f"用户实际购买商品: {purchased_items}")
        print(f"Recall@{k}: {recall:.4f}")
        print()

    return user_phone, top_k_items, purchased_items

def get_top_k_in_target(sorted_list, target_items_list, k=3):
    """
    从已排序的列表中提取前K个存在于目标列表中的元素

    参数:
        sorted_list: 已排序的列表（降序排列，分数从高到低）
        target_items_list: 目标产品ID列表
        k: 要提取的数量

    返回:
        list: 前K个存在于目标列表中的产品ID
    """
    result = []
    for item in sorted_list:
        if item in target_items_list:
            result.append(item)
            if len(result) >= k:
                break
    return result



target_items_list = [
    "Bus0019_1", "Bus0019_3", "Bus0019_4", "Bus0019_5",
    "Bus0019_6", "Bus0019_7", "Bus0019_22", "Bus0019_27", "Bus0019_10"
]

# 示例使用
# user_phone = 17706109463
# user_phone = 19996621765
user_phone = 15358999386
# user_phone = 13306174995
# user_phone = 17321582807

recall_k = 5
get_item_k = 5

# 评估单个用户
# F_value_at_k(user_phone, model, recall_k, True)
# plot_similarity_distribution(user_phone, model)

results = evaluate_model(model, test_data, recall_k, device)

# 保存模型推荐结果
# results_list = []
# all_user_ids = pd.read_csv("/Users/momoyi0929/Desktop/测试数据3/user.csv")
# count = 0
# all_count = len(all_user_ids)
# for id in all_user_ids["phone"]:
#     if None == recall_at_k_output(id, model, recall_k):
#         continue
#     _, topKitem, purchasedItem = recall_at_k_output(id, model, recall_k)
#
#     # r = get_top_k_in_target(topKitem, target_items_list, get_item_k)
#
#     # print("id: ",id)
#     # print("result: ", r)
#     # print("purchased: ",purchasedItem)
#     # print("--------------")
#     results_list.append({
#         "phone": id,
#         "recommended_items": ",".join(topKitem),
#         "purchased_items": ",".join(purchasedItem)
#     })
#     count+=1
#     print(count,"/", all_count)
#
# # ✅ 存入 CSV 文件
# results_df = pd.DataFrame(results_list)
# results_df.to_csv("/Users/momoyi0929/Desktop/recommend_results_all.csv", index=False)
# print("推荐结果已保存到 recommend_results.csv")


