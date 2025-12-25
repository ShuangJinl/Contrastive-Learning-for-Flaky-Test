import pandas as pd
from imblearn.over_sampling import RandomOverSampler
import numpy as np

# 读取数据
df = pd.read_csv('flaky_train.csv')

# 分离特征和标签（第五列是标签）
X = df.iloc[:, :4]  # 前四列作为特征
y = df.iloc[:, 4]   # 第五列作为标签（0或1）

print("原始数据分布：")
print(f"负例(0)数量: {sum(y == 0)}")
print(f"正例(1)数量: {sum(y == 1)}")
print(f"正负比例: {sum(y == 1)/sum(y == 0):.3f}")

# 创建RandomOverSampler对象，设置正负例比例为1:1
ros = RandomOverSampler(sampling_strategy=1, random_state=42)

# 进行上采样
X_resampled, y_resampled = ros.fit_resample(X, y)

print("\n上采样后数据分布：")
print(f"负例(0)数量: {sum(y_resampled == 0)}")
print(f"正例(1)数量: {sum(y_resampled == 1)}")
print(f"正负比例: {sum(y_resampled == 1)/sum(y_resampled == 0):.3f}")

# 将上采样后的数据合并为DataFrame
# 注意：需要确保列名正确
resampled_df = pd.DataFrame(X_resampled, columns=X.columns)
resampled_df['label'] = y_resampled  # 添加标签列

# 保存到新文件
resampled_df.to_csv('flaky_train_s.csv', index=False)

print(f"\n上采样完成！数据已保存到 flaky_train_s.csv")
print(f"新数据集形状: {resampled_df.shape}")