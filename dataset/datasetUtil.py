import pandas as pd
from sklearn.preprocessing import StandardScaler

read_path = 'dataset_test.csv'
write_path = 'dataset_test_clean.csv'

# 1. 读取CSV文件
df = pd.read_csv(read_path, encoding='gbk')

# 2. # 查找包含异常值的股票编号
invalid_stock_ids = df[df['成交量'] < 0]['股票']

# 3. 从DataFrame中删除这些股票编号的所有行
# 使用 ~ (取反) 操作符来选择不是无效股票编号的所有行
df_cleaned = df[~df['股票'].isin(invalid_stock_ids)]

# 4. 按股票编号和日期排序
# 假设日期的列名为'datetime'
df_cleaned = df_cleaned.sort_values(by=['股票', '日期代码'])

# 4. 对相同的股票编号下的所有数据列（除了日期）进行标准化
# 假设除了 'code', 'datetime', 'vol' 外的其他列需要标准化
columns_to_standardize = [col for col in df_cleaned.columns if col not in ['股票', '日期代码']]
scaler = StandardScaler()

# 将股票编号和日期设置为索引，以便按股票编号分组
df_cleaned.set_index(['股票', '日期代码'], inplace=True)

# 按股票编号分组，并对每组进行标准化
df_standardized = pd.DataFrame()
for stock_code in df_cleaned.index.get_level_values('股票').unique():
    group = df_cleaned.loc[stock_code]
    # 确保数据是二维的
    if group[columns_to_standardize].shape[0] == 1:
        # 如果只有一个样本，将其转换为二维数组
        group_to_standardize = group[columns_to_standardize].values.reshape(1, -1)
    else:
        # 如果有多个样本，确保它是二维数组
        group_to_standardize = group[columns_to_standardize].values

    # 现在对二维数组进行标准化
    group_standardized = scaler.fit_transform(group_to_standardize)

    # 将标准化后的数据转换回DataFrame，并追加到df_standardized
    standardized_df = pd.DataFrame(group_standardized, columns=columns_to_standardize, index=group.index)
    df_standardized = pd.concat([df_standardized, standardized_df], axis=0)

# 将标准化后的数据重置索引
df_standardized.reset_index(inplace=True)

# 5. 将清理并标准化后的数据写回CSV文件
df_standardized.to_csv(write_path, index=False)
