# 该代码用于检查是否存在数据泄露
import pandas as pd

def find_common_ids(file1, file2):
    """
    比较两个CSV文件的第一列ID，找出相同的ID
    
    参数:
    file1: 第一个CSV文件路径
    file2: 第二个CSV文件路径
    
    返回:
    common_ids: 相同的ID列表
    """
    try:
        # 读取CSV文件，只读取第一列
        df1 = pd.read_csv(file1, usecols=[0])
        df2 = pd.read_csv(file2, usecols=[0])
        
        # 获取第一列的数据（假设列名是第一行或者默认列名）
        # 如果CSV有表头，使用列名；如果没有表头，使用第一列默认列名
        if df1.columns[0] != '0':  # 有表头的情况
            ids1 = df1.iloc[:, 0].tolist()
            ids2 = df2.iloc[:, 0].tolist()
            print(f"文件1的列名: {df1.columns[0]}")
            print(f"文件2的列名: {df2.columns[0]}")
        else:  # 无表头的情况
            ids1 = df1.iloc[:, 0].tolist()
            ids2 = df2.iloc[:, 0].tolist()
        
        # 转换为集合进行快速查找
        set1 = set(ids1)
        set2 = set(ids2)
        
        # 找出相同的ID
        common_ids = list(set1.intersection(set2))
        
        # 打印结果
        print(f"文件1的ID数量: {len(ids1)}")
        print(f"文件2的ID数量: {len(ids2)}")
        print(f"相同的ID数量: {len(common_ids)}")
        
        if common_ids:
            print("\n相同的ID:")
            for i, common_id in enumerate(common_ids[:20]):  # 只显示前20个
                print(f"{i+1}. {common_id}")
            if len(common_ids) > 20:
                print(f"... 还有 {len(common_ids) - 20} 个相同的ID")
        else:
            print("没有找到相同的ID")
            
        return common_ids
        
    except FileNotFoundError as e:
        print(f"文件未找到: {e}")
        return []
    except Exception as e:
        print(f"处理文件时出错: {e}")
        return []

# 使用方法
if __name__ == "__main__":
    # 替换为你的文件路径
    file1_path = "flaky_train_s.csv"  # 第一个CSV文件路径
    file2_path = "flaky_test.csv"  # 第二个CSV文件路径
    
    common_ids = find_common_ids(file1_path, file2_path)
    
    # 如果需要将结果保存到文件
    if common_ids:
        result_df = pd.DataFrame(common_ids, columns=['common_ids'])
        result_df.to_csv('common_ids.csv', index=False)
        print("\n相同的ID已保存到 common_ids.csv")