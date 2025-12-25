import pandas as pd
import numpy as np

# 读取CSV文件
df_a = pd.read_csv('A.csv')  # 包含test_id, method_id, label
df_b = pd.read_csv('B.csv')  # 包含project, code_id, code_body

# 为A.csv添加两列递增序列id1和id2
df_a['id1'] = range(1, len(df_a) + 1)
df_a['id2'] = range(1, len(df_a) + 1)

# 步骤1: 匹配code_id和test_id
# 使用内连接匹配A.csv的test_id和B.csv的code_id
merged_df = pd.merge(df_a, df_b, left_on='test_id', right_on='code_id', how='inner')

# 选择需要的列：id1, id2, project, test_id, method_id, label
matched_data = merged_df[['id1', 'id2', 'project', 'test_id', 'method_id', 'label']]

print(f"成功匹配 {len(matched_data)} 条记录")

# 步骤2: 混合所有数据，按9:1比例划分训练测试数据集，测试集正负例比例接近7:3
# 计算总记录数和目标测试集大小（10%）
total_records = len(matched_data)
target_test_size = int(total_records * 0.1)

print(f"\n总记录数: {total_records}")
print(f"目标测试集大小: {target_test_size} (约10%)")

# 计算正负例比例
positive_count_total = (matched_data['label'] == 1).sum()
negative_count_total = (matched_data['label'] == 0).sum()
print(f"\n总体正负例统计:")
print(f"正例数 (1): {positive_count_total}, 比例: {positive_count_total/total_records:.2%}")
print(f"负例数 (0): {negative_count_total}, 比例: {negative_count_total/total_records:.2%}")

# 分离正例和负例数据
positive_data = matched_data[matched_data['label'] == 1]
negative_data = matched_data[matched_data['label'] == 0]

print(f"\n正例数据: {len(positive_data)} 条")
print(f"负例数据: {len(negative_data)} 条")

# 计算测试集中需要的正负例数量
target_positive_test = int(target_test_size * 0.05)  # 70%正例
target_negative_test = target_test_size - target_positive_test  # 30%负例

print(f"\n目标测试集组成:")
print(f"正例: {target_positive_test} 条")
print(f"负例: {target_negative_test} 条")

# 从正例和负例中随机抽取测试集数据
np.random.seed(114512)  # 设置随机种子以确保可重复性

# 检查是否有足够的正例和负例
if len(positive_data) < target_positive_test:
    print(f"警告: 正例数据不足，只有 {len(positive_data)} 条，但需要 {target_positive_test} 条")
    target_positive_test = len(positive_data)

if len(negative_data) < target_negative_test:
    print(f"警告: 负例数据不足，只有 {len(negative_data)} 条，但需要 {target_negative_test} 条")
    target_negative_test = len(negative_data)

# 随机选择测试集的正例和负例
positive_test_indices = np.random.choice(positive_data.index, size=target_positive_test, replace=False)
negative_test_indices = np.random.choice(negative_data.index, size=target_negative_test, replace=False)

# 创建测试集
test_indices = np.concatenate([positive_test_indices, negative_test_indices])
test_df = matched_data.loc[test_indices]

# 创建训练集（剩余的数据）
train_df = matched_data.drop(test_indices)

# 步骤3: 保存训练集和测试集，只保留id1, id2, test_id, method_id, label
# 删除project列
test_df_final = test_df.drop('project', axis=1)
train_df_final = train_df.drop('project', axis=1)

# 保存文件
test_df_final.to_csv('flaky_test.csv', index=False)
train_df_final.to_csv('flaky_train.csv', index=False)

# 打印结果
print(f"\n划分结果:")
print(f"测试集记录数: {len(test_df)} ({len(test_df)/total_records*100:.1f}%)")
print(f"训练集记录数: {len(train_df)} ({len(train_df)/total_records*100:.1f}%)")

# 统计训练集和测试集中正负例的比例
positive_count_test = (test_df['label'] == 1).sum()
negative_count_test = (test_df['label'] == 0).sum()
total_count_test = positive_count_test + negative_count_test

positive_count_train = (train_df['label'] == 1).sum()
negative_count_train = (train_df['label'] == 0).sum()
total_count_train = positive_count_train + negative_count_train

if total_count_test > 0:
    positive_ratio_test = positive_count_test / total_count_test
    negative_ratio_test = negative_count_test / total_count_test
    
    print(f"\n测试集正负例统计:")
    print(f"正例数 (1): {positive_count_test}, 比例: {positive_ratio_test:.2%}")
    print(f"负例数 (0): {negative_count_test}, 比例: {negative_ratio_test:.2%}")
    print(f"目标比例: 7:3 (正例70%)")
else:
    print("\n警告: 测试集没有记录")

if total_count_train > 0:
    positive_ratio_train = positive_count_train / total_count_train
    negative_ratio_train = negative_count_train / total_count_train
    
    print(f"\n训练集正负例统计:")
    print(f"正例数 (1): {positive_count_train}, 比例: {positive_ratio_train:.2%}")
    print(f"负例数 (0): {negative_count_train}, 比例: {negative_ratio_train:.2%}")
else:
    print("\n警告: 训练集没有记录")

# 步骤4: 保存B.csv删除第一列project后的文件
# 只保留code_id和code_body两列
flaky_db = df_b[['code_id', 'code_body']]
flaky_db.to_csv('flaky_db.csv', index=False)

print(f"\n已保存 flaky_db.csv，包含 {len(flaky_db)} 条记录，列: code_id, code_body")
print(f"已保存 flaky_test.csv，包含 {len(test_df_final)} 条记录，列: id1, id2, test_id, method_id, label")
print(f"已保存 flaky_train.csv，包含 {len(train_df_final)} 条记录，列: id1, id2, test_id, method_id, label")