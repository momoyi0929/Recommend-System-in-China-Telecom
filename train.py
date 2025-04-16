import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import joblib
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset, ConcatDataset
from model import TwoTowerModel

# **设备选择**
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

# **1️⃣ 读取数据**
user_df = pd.read_csv("/Users/momoyi0929/Desktop/测试数据3/user.csv")
item_df = pd.read_csv("/Users/momoyi0929/Desktop/测试数据3/item.csv")
interaction_df = pd.read_csv("/Users/momoyi0929/Desktop/测试数据3/label.csv")

user_df = user_df.drop_duplicates(subset=["phone"]).reset_index(drop=True)
item_df = item_df.drop_duplicates(subset=["产品ID"]).reset_index(drop=True)
print("✅ 数据读取完毕！")

# **2️⃣ 预处理类别特征**
user_categ_features = ["pd_inst_id"]
item_categ_features = ["业务", "分类1", "支付方式", "产品有效期", "是否有附属权益", "是否有协议期", "收费策略", "退订规则"]

encoders = {}
for col in user_categ_features:
    encoder = LabelEncoder()
    user_df[col] = encoder.fit_transform(user_df[col])
    encoders[col] = encoder

for col in item_categ_features:
    encoder = LabelEncoder()
    item_df[col] = encoder.fit_transform(item_df[col])
    encoders[col] = encoder

# **3️⃣ 预处理数值特征**
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

user_scaler = MinMaxScaler()
item_scaler = MinMaxScaler()

user_df[user_num_features] = user_scaler.fit_transform(user_df[user_num_features])
item_df[item_num_features] = item_scaler.fit_transform(item_df[item_num_features])


# **4️⃣ 处理 ID**
user_encoder = LabelEncoder()
item_encoder = LabelEncoder()

user_df["phone"] = user_encoder.fit_transform(user_df["phone"])
item_df["产品ID"] = item_encoder.fit_transform(item_df["产品ID"])

interaction_df["phone"] = user_encoder.transform(interaction_df["phone"])
interaction_df["goods_id"] = item_encoder.transform(interaction_df["goods_id"])

print("✅ 数据处理完毕！")

# 保存编码器
joblib.dump(encoders, "/Users/momoyi0929/Desktop/模型数据3/encoders.pkl")

joblib.dump(user_scaler, "/Users/momoyi0929/Desktop/模型数据3/user_scaler.pkl")
joblib.dump(item_scaler, "/Users/momoyi0929/Desktop/模型数据3/item_scaler.pkl")

joblib.dump(user_encoder, "/Users/momoyi0929/Desktop/模型数据3/user_id_encoder.pkl")
joblib.dump(item_encoder, "/Users/momoyi0929/Desktop/模型数据3/item_id_encoder.pkl")

print("✅ 编码器以保存！")

# **5️⃣ 合并数据**
merged_df = interaction_df.merge(user_df, on="phone").merge(item_df, left_on="goods_id", right_on="产品ID")
final_features = ["phone", "goods_id"] + user_num_features + item_num_features + user_categ_features + item_categ_features
train_df, test_df = train_test_split(merged_df[final_features + ["label"]], test_size=0.2, random_state=42)

# **6️⃣ 获取类别特征的唯一值数量**
user_categ_sizes = [user_df[col].nunique() for col in user_categ_features]
item_categ_sizes = [item_df[col].nunique() for col in item_categ_features]

# **7️⃣ 转换为 Tensor**
def to_tensor(df):
    return {
        "user_id": torch.tensor(df["phone"].values, dtype=torch.long),
        "item_id": torch.tensor(df["goods_id"].values, dtype=torch.long),
        "user_features": torch.tensor(df[user_num_features].values, dtype=torch.float32),
        "item_features": torch.tensor(df[item_num_features].values, dtype=torch.float32),
        "user_categ": torch.tensor(df[user_categ_features].values, dtype=torch.long),
        "item_categ": torch.tensor(df[item_categ_features].values, dtype=torch.long),
        "label": torch.tensor(df["label"].values, dtype=torch.float32)
    }

train_data = to_tensor(train_df)
test_data = to_tensor(test_df)


# **8️⃣ 创建 DataLoader**
batch_size = 4096
train_dataset = TensorDataset(train_data["user_id"], train_data["item_id"],
                              train_data["user_features"], train_data["user_categ"],
                              train_data["item_features"], train_data["item_categ"], train_data["label"])

test_dataset = TensorDataset(test_data["user_id"], test_data["item_id"],
                             test_data["user_features"], test_data["user_categ"],
                             test_data["item_features"], test_data["item_categ"], test_data["label"])

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
test_loader = DataLoader(test_dataset, batch_size=batch_size, num_workers=0)

torch.save(test_loader, "/Users/momoyi0929/Desktop/模型数据3/test_data.pt")
torch.save(train_loader, "/Users/momoyi0929/Desktop/模型数据3/train_data.pt")
print("✅ 创建保存 DataLoader！")

# **9️⃣ 初始化模型**
embedding_dim = 512
model = TwoTowerModel(len(user_num_features), len(item_num_features),
                      user_categ_sizes, item_categ_sizes, embedding_dim).to(device)

# criterion = nn.MSELoss()
criterion = nn.BCELoss()
# criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
print("✅ 模型初始化！")

# **🔟 训练模型**
num_epochs = 40
print("🚀 开始训练！")
best_auc = 0

for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    i = 0
    j = 0
    for user_id, item_id, user_features, user_categ, item_features, item_categ, label in test_loader:
        # **数据移动到 MPS/CPU**
        user_id, item_id = user_id.to(device), item_id.to(device)
        user_features, user_categ = user_features.to(device), user_categ.to(device)
        item_features, item_categ, label = item_features.to(device), item_categ.to(device), label.to(device)

        optimizer.zero_grad()
        outputs = model(user_id, item_id, user_features, user_categ, item_features, item_categ)
        label = label.view(-1, 1)  # 调整标签形状
        loss = criterion(outputs, label)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        i += 1

    # **11️⃣ 验证模型**
    model.eval()
    total_test_loss = 0
    all_labels, all_preds = [], []

    with torch.no_grad():
        for user_id, item_id, user_features, user_categ, item_features, item_categ, label in test_loader:
            user_id, item_id = user_id.to(device), item_id.to(device)
            user_features, user_categ = user_features.to(device), user_categ.to(device)
            item_features, item_categ, label = item_features.to(device), item_categ.to(device), label.to(device)

            test_outputs = model(user_id, item_id, user_features, user_categ, item_features, item_categ)
            label = label.view(-1, 1)  # 调整标签形状
            test_loss = criterion(test_outputs, label)
            total_test_loss += test_loss.item()
            j += 1

            all_labels.extend(label.cpu().numpy())
            all_preds.extend(test_outputs.cpu().numpy())


    # **计算 AUC**
    auc = roc_auc_score(all_labels, all_preds)

    # **12️⃣ 保存最佳模型**
    if auc > best_auc:
        best_auc = auc
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': total_loss,
        }, '/Users/momoyi0929/Desktop/模型数据3/best_model.pth')
        print(f"✅ 第 {epoch+1} 轮：模型保存，AUC = {auc:.4f}")

    print(f"✅ Epoch {epoch+1} | Train Loss: {total_loss/i:.4f} | Test Loss: {total_test_loss/j:.4f} | Test AUC: {auc:.4f}")

print("🎉 训练完成！")
