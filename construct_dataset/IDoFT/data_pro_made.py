import pandas as pd
import numpy as np

# 读取CSV文件
# 请确保目录下存在 A.csv 和 B.csv
df_a = pd.read_csv('A.csv')  # 包含test_id, method_id, label
df_b = pd.read_csv('B.csv')  # 包含project, code_id, code_body

# 为A.csv添加两列递增序列id1和id2
df_a['id1'] = range(1, len(df_a) + 1)
df_a['id2'] = range(1, len(df_a) + 1)

# 步骤1: 匹配code_id和test_id
# 使用内连接匹配A.csv的test_id和B.csv的code_id
merged_df = pd.merge(df_a, df_b, left_on='test_id', right_on='code_id', how='inner')

# 选择需要的列，保留 project
matched_data = merged_df[['id1', 'id2', 'project', 'test_id', 'method_id', 'label']]

print(f"成功匹配 {len(matched_data)} 条记录")

# 步骤2: 按项目划分训练测试数据集，比例9:1，测试集正负例比例接近7:3
# 统计项目信息
unique_projects = matched_data['project'].unique()

# 计算总记录数和目标测试集大小（10%）
total_records = len(matched_data)
target_test_size = int(total_records * 0.1)
target_min = target_test_size - 10  # 允许一定误差范围
target_max = target_test_size + 10

print(f"\n总记录数: {total_records}")
print(f"目标测试集大小: {target_test_size} (约10%)")

# 初始化变量
best_combination = None
best_size = 0
best_positive_ratio = 0
best_score = float('inf')

# 随机尝试多次，找到最接近目标大小且正负例比例接近7:3的组合
np.random.seed(114512)  # 设置随机种子以确保可重复性

print("正在寻找最佳数据集划分...")

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
    # 目标正例比例为70% (0.7)
    ratio_diff = abs(positive_ratio - 0.7) 
    
    # 综合评分
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
        total_count = len(test_df)
        
        if total_count == 0:
            continue
            
        positive_ratio = positive_count / total_count
        
        size_diff = abs(selected_size - target_test_size) / target_test_size
        ratio_diff = abs(positive_ratio - 0.7)
        score = 0.6 * size_diff + 0.4 * ratio_diff
        
        if score < best_score:
            best_score = score
            best_combination = selected_projects
            best_size = selected_size
            best_positive_ratio = positive_ratio

# 创建测试集和训练集
test_df = matched_data[matched_data['project'].isin(best_combination)]
train_df = matched_data[~matched_data['project'].isin(best_combination)]

# 步骤3: 保存训练集和测试集
# 修改：保留 project 列
# 映射关系: id1->id, id2->id.1, project->project, test_id->code_id_1, method_id->code_id_2, label->label
rename_map = {
    'id1': 'id',
    'id2': 'id.1',
    'project': 'project',
    'test_id': 'code_id_1',
    'method_id': 'code_id_2',
    'label': 'label'
}

# 指定列的顺序
output_columns = ['id1', 'id2', 'project', 'test_id', 'method_id', 'label']

# 处理测试集
test_df_final = test_df[output_columns].rename(columns=rename_map)
test_df_final.to_csv('flaky_test.csv', index=False)

# 处理训练集
train_df_final = train_df[output_columns].rename(columns=rename_map)
train_df_final.to_csv('flaky_train.csv', index=False)

# 打印结果
print(f"\n划分结果:")
print(f"测试集包含 {len(best_combination)} 个项目，总记录数: {best_size} ({best_size/total_records*100:.1f}%)")
print(f"训练集包含 {train_df['project'].nunique()} 个项目，总记录数: {len(train_df)} ({len(train_df)/total_records*100:.1f}%)")

# 统计训练集和测试集中正负例的比例
positive_count_test = (test_df['label'] == 1).sum()
total_count_test = len(test_df)

if total_count_test > 0:
    positive_ratio_test = positive_count_test / total_count_test
    print(f"\n测试集正负例统计:")
    print(f"正例数 (1): {positive_count_test}, 比例: {positive_ratio_test:.2%}")
    print(f"目标比例: 70%")
else:
    print("\n警告: 测试集没有记录")

# 步骤4: 保存B.csv删除第一列project后的文件
# 保持 flaky_db 为 id, code 格式
flaky_db = df_b[['code_id', 'code_body']].rename(columns={'code_id': 'id', 'code_body': 'code'})
flaky_db.to_csv('flaky_db.csv', index=False)

print(f"\n文件保存完毕:")
print(f"1. flaky_db.csv (列: id, code)")
print(f"2. flaky_test.csv (列: id, id.1, project, code_id_1, code_id_2, label)")
print(f"3. flaky_train.csv (列: id, id.1, project, code_id_1, code_id_2, label)")