import csv
from collections import defaultdict

def validate_data_integrity():
    # 加载有效数据
    valid_items = set()
    with open('/Users/momoyi0929/Desktop/测试数据/item.csv', 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        valid_items = {row['产品ID'] for row in reader}

    valid_phones = set()
    with open('/Users/momoyi0929/Desktop/测试数据/user.csv', 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        valid_phones = {row['phone'] for row in reader}

    # 验证item_user数据
    invalid_records = []
    with open('/Users/momoyi0929/Desktop/测试数据/item_user.csv', 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader, 2):  # 从第2行开始计数
            error = []
            if row['goods_id'] not in valid_items:
                error.append(f"无效商品ID: {row['goods_id']}")
            if row['phone'] not in valid_phones:
                error.append(f"无效手机号: {row['phone']}")
            if error:
                invalid_records.append((i, '|'.join(error)))

    # 输出结果
    if invalid_records:
        print("发现无效记录：")
        for line_num, error in invalid_records:
            print(f"第 {line_num} 行: {error}")
    else:
        print("所有记录的商品ID和手机号均有效")


# ----------------------------------------------------------
# ----------------------------------------------------------

def validate_sampling():
    # 加载原始购买记录
    original_purchases = defaultdict(set)
    with open('/Users/momoyi0929/Desktop/测试数据/item_user.csv', 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if int(row['quanity']) > 0:
                original_purchases[row['phone']].add(row['goods_id'])

    # 验证采样文件
    sampling_errors = []
    with open('/Users/momoyi0929/Desktop/测试数据/label.csv', 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            phone = row['phone']
            item = row['产品ID']
            label = row['label']

            # 验证标签是否正确
            actual_label = 1 if item in original_purchases.get(phone, set()) else 0
            if int(label) != actual_label:
                sampling_errors.append(
                    f"手机号 {phone} 商品 {item} 标签错误：标记为 {label}，实际应为 {actual_label}"
                )

    # 输出结果
    if sampling_errors:
        print("发现采样错误：")
        for error in sampling_errors[:5]:  # 最多显示前5个错误
            print(error)
        print(f"共发现 {len(sampling_errors)} 处错误")
    else:
        print("所有采样标签均正确")

    # 额外统计信息
    positive_count = 0
    negative_count = 0
    with open('/Users/momoyi0929/Desktop/测试数据/label.csv', 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row['label'] == '1':
                positive_count += 1
            else:
                negative_count += 1
    print(f"\n样本分布统计：")
    print(f"正样本数：{positive_count}")
    print(f"负样本数：{negative_count}")
    print(f"正负比例：{positive_count / negative_count:.2f}:1" if negative_count else "无负样本")


if __name__ == "__main__":
    validate_data_integrity()

    validate_sampling()