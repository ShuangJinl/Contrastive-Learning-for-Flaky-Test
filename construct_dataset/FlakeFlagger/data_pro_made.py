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

# 步骤2: 按项目划分训练测试数据集，比例9:1，测试集正负例比例接近7:3
# 统计项目信息
unique_projects = matched_data['project'].unique()
project_counts = matched_data['project'].value_counts()

print(f"\n总共有 {len(unique_projects)} 个项目")
print("\n每个项目的记录数:")
print(project_counts.to_string())

# 计算总记录数和目标测试集大小（10%）
total_records = len(matched_data)
target_test_size = int(total_records * 0.1)
target_min = target_test_size - 10  # 允许一定误差范围
target_max = target_test_size + 10

print(f"\n总记录数: {total_records}")
print(f"目标测试集大小: {target_test_size} (约10%)")
print(f"允许范围: {target_min} - {target_max}")

# 计算正负例比例
positive_count_total = (matched_data['label'] == 1).sum()
negative_count_total = (matched_data['label'] == 0).sum()
print(f"\n总体正负例统计:")
print(f"正例数 (1): {positive_count_total}, 比例: {positive_count_total/total_records:.2%}")
print(f"负例数 (0): {negative_count_total}, 比例: {negative_count_total/total_records:.2%}")
division_ratio = negative_count_total/total_records

# 初始化变量
best_combination = None
best_size = 0
best_positive_ratio = 0
best_score = float('inf')

# 随机尝试多次，找到最接近目标大小且正负例比例接近7:3的组合
np.random.seed(114512)  # 设置随机种子以确保可重复性
for _ in range(10000):  # 尝试多次随机组合
    # 随机选择项目
    selected_projects = np.random.choice(unique_projects, size=np.random.randint(1, len(unique_projects)), replace=False)
    
    # 计算选中项目的总记录数
    test_df = matched_data[matched_data['project'].isin(selected_projects)]
    selected_size = len(test_df)
    
    # 计算测试集的正负例比例
    positive_count = (test_df['label'] == 1).sum()
    negative_count = (test_df['label'] == 0).sum()
    total_count = positive_count + negative_count
    
    if total_count == 0:
        continue
    
    positive_ratio = positive_count / total_count
    
    # 计算与目标大小和正负例比例的差异
    size_diff = abs(selected_size - target_test_size) / target_test_size
    ratio_diff = abs(positive_ratio - division_ratio)  # 目标正例比例为70%
    
    # 综合评分（权重可以调整）
    score = 0.6 * size_diff + 0.4 * ratio_diff
    
    # 检查是否更接近目标
    if (score < best_score) and (target_min <= selected_size <= target_max):
        best_score = score
        best_combination = selected_projects
        best_size = selected_size
        best_positive_ratio = positive_ratio

# 如果没有找到在范围内的组合，选择评分最好的组合
if best_combination is None:
    print("未找到在允许范围内的组合，选择评分最好的组合")
    for _ in range(10000):
        selected_projects = np.random.choice(unique_projects, size=np.random.randint(1, len(unique_projects)), replace=False)
        test_df = matched_data[matched_data['project'].isin(selected_projects)]
        selected_size = len(test_df)
        
        positive_count = (test_df['label'] == 1).sum()
        negative_count = (test_df['label'] == 0).sum()
        total_count = positive_count + negative_count
        
        if total_count == 0:
            continue
            
        positive_ratio = positive_count / total_count
        
        size_diff = abs(selected_size - target_test_size) / target_test_size
        ratio_diff = abs(positive_ratio - 0.03)  # 目标正例比例为70%
        score = 0.6 * size_diff + 0.4 * ratio_diff
        
        if score < best_score:
            best_score = score
            best_combination = selected_projects
            best_size = selected_size
            best_positive_ratio = positive_ratio

# 创建测试集和训练集
test_df = matched_data[matched_data['project'].isin(best_combination)]
train_df = matched_data[~matched_data['project'].isin(best_combination)]

# 步骤3: 保存训练集和测试集，只保留id1, id2, test_id, method_id, label
# 删除project列
test_df_final = test_df.drop('project', axis=1)
train_df_final = train_df.drop('project', axis=1)

# 保存文件
test_df_final.to_csv('flaky_test.csv', index=False)
train_df_final.to_csv('flaky_train.csv', index=False)

# 打印结果
print(f"\n划分结果:")
print(f"测试集包含 {len(best_combination)} 个项目，总记录数: {best_size} ({best_size/total_records*100:.1f}%)")
print(f"训练集包含 {train_df['project'].nunique()} 个项目，总记录数: {len(train_df)} ({len(train_df)/total_records*100:.1f}%)")
print(f"测试集项目: {', '.join(best_combination)}")

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