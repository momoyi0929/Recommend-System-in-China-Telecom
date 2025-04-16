import pandas as pd

# **1. 读取数据**
item_df = pd.read_csv("/Users/momoyi0929/Desktop/测试数据3/item.txt", sep="|", dtype=str, encoding="utf-8")
user_df = pd.read_csv("/Users/momoyi0929/Desktop/测试数据3/user.txt", sep=",", dtype=str, encoding="utf-8")
item_user_df = pd.read_csv("/Users/momoyi0929/Desktop/测试数据3/item_user.txt", sep="|", dtype=str, encoding="utf-8")

print("数据读取完毕")

# **2. 提取有效的产品ID 和 用户手机号**
valid_goods_ids = set(item_df["产品ID"].astype(str).str.strip())
valid_phones = set(user_df["phone"].astype(str).str.strip())

print("ID提取完毕")
print(valid_phones)

# **3. 过滤 `item_user_df`，确保 `item_user` 里的数据都是 `item` 和 `user` 里有的**
filtered_item_user_df = item_user_df[
    item_user_df["goods_id"].astype(str).str.strip().isin(valid_goods_ids) &
    item_user_df["phone"].astype(str).str.strip().isin(valid_phones)
]

print(f"交互数据筛选完成，剩余记录数: {len(filtered_item_user_df)}")

# **4. 反向过滤 `item_df` 和 `user_df`，去掉 `item_user` 里未出现的产品和用户**
used_goods_ids = set(filtered_item_user_df["goods_id"].astype(str).str.strip())
used_phones = set(filtered_item_user_df["phone"].astype(str).str.strip())

filtered_item_df = item_df[item_df["产品ID"].astype(str).str.strip().isin(used_goods_ids)]
filtered_user_df = user_df[user_df["phone"].astype(str).str.strip().isin(used_phones)]

print(f"商品数据筛选完成，剩余商品数: {len(filtered_item_df)}")
print(f"用户数据筛选完成，剩余用户数: {len(filtered_user_df)}")

# **5. 保存最终的匹配数据**
filtered_item_user_df.to_csv("/Users/momoyi0929/Desktop/测试数据3/item_user.csv", index=False, encoding="utf-8")
filtered_item_df.to_csv("/Users/momoyi0929/Desktop/测试数据3/item.csv", index=False, encoding="utf-8")
filtered_user_df.to_csv("/Users/momoyi0929/Desktop/测试数据3/user.csv", index=False, encoding="utf-8")

print("数据保存完毕，所有数据已互相匹配")


#
# # 处理新的用户文本
# import csv
# input_file = "/Users/momoyi0929/Desktop/ds/hb_qw_qy_label_20250309.txt"
# output_file = "/Users/momoyi0929/Desktop//user2.txt"
# data = []
# with open(input_file, 'r', encoding='utf-8') as file:
#     for line in file:
#         fields = line.strip().split('|')
#         if "" in fields:
#             continue
#         data.append(fields)
#
# with open(output_file, 'w', encoding='utf-8', newline='') as csvfile:
#     writer = csv.writer(csvfile)
#     writer.writerows(data)
# print(f"数据已保存到 {output_file}（已去除空行）")
#
#
# def add_header_to_txt(input_file, output_file, header_line):
#     """
#     在TXT文件的第一行插入指定的列名（管道符分隔）
#
#     :param input_file: 输入TXT文件路径
#     :param output_file: 输出TXT文件路径
#     :param header_line: 列名列表（会自动用 | 拼接）
#     """
#     # 拼接列名（用 | 分隔）
#     header = ",".join(header_line) + "\n"
#
#     # 读取原始文件内容
#     with open(input_file, 'r', encoding='utf-8') as f:
#         original_content = f.read()
#
#     # 写入新文件（列名 + 原始内容）
#     with open(output_file, 'w', encoding='utf-8') as f:
#         f.write(header + original_content)
#
#     print(f"列名已添加到 {output_file}")
#
#
# # 使用示例
# if __name__ == "__main__":
#     input_file = "/Users/momoyi0929/Desktop/测试数据3/user.txt"  # 原始文件路径
#     output_file = "/Users/momoyi0929/Desktop/测试数据3/user.txt"  # 输出文件路径
#
#     # 列名列表（按你的需求顺序）
#     header_line = [
#         "po_type", "pd_inst_id", "main_accs_nmbr", "city_nm", "gender", "cust_tp", "age_f",
#         "main_limit_mon_chrg_lvl", "ob_f", "data_flow_f", "date_flow_use_pr", "qy_by", "qy_dc",
#         "short_video_f", "game_f", "webcast_f", "music_f", "entertainment_f", "local_life_f",
#         "webclass_f", "parenting_education_f", "employee_f", "trip_f", "catering_f",
#         "read_and_listen_f", "ebusiness_f", "payment_f", "txvideo_f", "youku_f", "bilibili_f",
#         "mango_f", "txsyjsq_f", "iqiyi_f", "wangzhe_f", "bdwp_f", "keep_f", "meituan_f",
#         "zyb_f", "tyyp_f", "xunlei_f", "zhihu_f", "xmly_kid_f", "yax_f", "didi_f", "gddt_f",
#         "mtwm_f", "mcdonold_f", "kfc_f", "luckin_f", "naixue_f", "xicha_f", "chayan_f",
#         "mixue_f", "qqyuedu_f", "zhangyue_f", "xmly_f", "ksjgs_f", "qmxs_f", "jd_f", "yzf_f",
#         "didacx_f", "qqmusic_f"
#     ]
#
#     # 执行添加列名
#     add_header_to_txt(input_file, output_file, header_line)
