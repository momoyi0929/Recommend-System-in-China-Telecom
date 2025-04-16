import csv
import random
from collections import defaultdict


def balance_labels(item_user_path, output_path, ratio=3):
    """在原始数据基础上增加负样本，保持正样本不变"""
    # 读取原始数据
    with open(item_user_path, 'r', encoding='utf-8') as f_in:
        reader = csv.DictReader(f_in)
        original_data = list(reader)

    # 分离正负样本（原始数据保持不变）
    positive_samples = [row for row in original_data if int(row['is_order']) > 0]
    negative_samples = [row for row in original_data if int(row['is_order']) == 0]

    num_pos = len(positive_samples)
    num_neg = len(negative_samples)
    print(f"原始正样本数: {num_pos}, 负样本数: {num_neg}")

    # 计算需要增加的负样本数
    target_neg = num_pos * ratio
    additional_needed = max(0, target_neg - num_neg)

    if additional_needed > 0:
        # 上采样：随机复制现有负样本（不改变原始数据）
        additional_samples = random.choices(negative_samples, k=additional_needed)
        negative_samples += additional_samples
        print(f"新增负样本数: {additional_needed}")
    else:
        print("负样本已足够，无需新增")

    print(f"处理后正样本数: {num_pos}, 负样本数: {len(negative_samples)}")

    # 写入新数据（包含原始数据+新增负样本）
    with open(output_path, 'w', newline='', encoding='utf-8') as f_out:
        writer = csv.writer(f_out)
        writer.writerow(['phone', 'goods_id', 'label'])  # 表头

        # 写入所有正样本（保持不变）
        for row in positive_samples:
            writer.writerow([row['phone'], row['goods_id'], 1])

        # 写入原始+新增的负样本
        for row in negative_samples:
            writer.writerow([row['phone'], row['goods_id'], 0])

    print(f"处理完成，已生成: {output_path}")


if __name__ == "__main__":
    item_user_path = "/Users/momoyi0929/Desktop/测试数据3/item_user.csv"
    output_path = "/Users/momoyi0929/Desktop/测试数据3/label_upsampled.csv"
    balance_labels(item_user_path, output_path, ratio=3)
#
# import csv
#
#
# def generate_labels(item_user_path, output_path):
#     """直接根据 is_order 生成标签"""
#     with open(item_user_path, 'r', encoding='utf-8') as f_in, \
#             open(output_path, 'w', newline='', encoding='utf-8') as f_out:
#         reader = csv.DictReader(f_in)
#         writer = csv.writer(f_out)
#
#         # 写入表头
#         writer.writerow(['phone', 'goods_id', 'label'])
#
#         # 直接读取 is_order 作为 label
#         for row in reader:
#             phone = row['phone']
#             goods_id = row['goods_id']
#             label = 1 if int(row['is_order']) > 0 else 0
#             writer.writerow([phone, goods_id, label])
#
#     print(f"处理完成，已生成: {output_path}")
#
#
# if __name__ == "__main__":
#     item_user_path = "/Users/momoyi0929/Desktop/测试数据3/item_user.csv"
#     output_path = "/Users/momoyi0929/Desktop/测试数据3/label.csv"
#
#     generate_labels(item_user_path, output_path)