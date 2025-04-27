import pandas as pd
import numpy as np

# 读取Excel文件
df = pd.read_excel('tdx_block_pre_data.xlsx', sheet_name='Sheet1')

# 计算每一列大于0的百分比
percent_positive = {}
for column in df.columns:
    if df[column].dtype in [np.float64, np.int64]:  # 只处理数值列
        total = len(df[column])
        positive = (df[column] > 0).sum()
        percent = (positive / total) * 100 if total > 0 else 0
        percent_positive[column] = percent

# 创建结果DataFrame
result_df = pd.DataFrame.from_dict(percent_positive, orient='index', columns=['大于0的百分比(%)'])
result_df = result_df.sort_values(by='大于0的百分比(%)', ascending=False)

# 打印结果
print("各列大于0的百分比:")
print(result_df)

# 保存结果到新Excel文件
result_df.to_excel('大于0百分比结果.xlsx')