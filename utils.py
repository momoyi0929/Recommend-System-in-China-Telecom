# import pandas as pd
# interaction_df = pd.read_csv("/Users/momoyi0929/Desktop/测试数据3/label.csv")
#
# negative_samples = interaction_df[interaction_df["label"] == 0]
# positive_samples = interaction_df[interaction_df["label"] == 1]
#
# print("negative samples:", len(negative_samples))
# print("positive samples:",len(positive_samples))
# print("all samples",len(interaction_df))
#
# print(positive_samples["goods_id"].value_counts().head(66))  # 查看出现次数最多的商品
#
#
# user_df = pd.read_csv("/Users/momoyi0929/Desktop/测试数据3/user.csv")
# item_df = pd.read_csv("/Users/momoyi0929/Desktop/测试数据3/item.csv")
# interaction_df = pd.read_csv("/Users/momoyi0929/Desktop/测试数据3/label.csv")
#
# user_df = user_df.drop_duplicates(subset=["phone"]).reset_index(drop=True)
# item_df = item_df.drop_duplicates(subset=["产品ID"]).reset_index(drop=True)
#
# print("item nums:",len(item_df))

import statistics

input_file = "/Users/momoyi0929/Desktop/测试数据3/item_user.txt"
output_file = "/Users/momoyi0929/Desktop/测试数据3/item_user2.txt"

ages = []

# ✅ 第一步：收集所有合法的年龄
with open(input_file, "r", encoding="utf-8") as fin:
    for line in fin:
        parts = line.strip().split("|")
        if len(parts) >= 6:
            age_str = parts[5].strip()
            if age_str.isdigit():
                ages.append(int(age_str))

# ✅ 计算中位数
median_age = statistics.median(ages) if ages else -1
print(f"中位年龄是：{median_age}")

# ✅ 第二步：处理并写入文件
with open(input_file, "r", encoding="utf-8") as fin, open(output_file, "w", encoding="utf-8") as fout:
    for line in fin:
        parts = line.strip().split("|")

        # 补全性别
        if len(parts) < 5 or parts[4].strip() == "":
            while len(parts) < 5:
                parts.append("")
            parts[4] = "未知"

        # 补全年龄
        if len(parts) < 6 or not parts[5].strip().isdigit():
            while len(parts) < 6:
                parts.append("")
            parts[5] = str(int(median_age))  # 中位数转成字符串

        # 补足字段到12列
        while len(parts) < 12:
            parts.append("")

        fout.write("|".join(parts) + "\n")

print(f"✅ 已补全性别和年龄，结果写入 {output_file}")
