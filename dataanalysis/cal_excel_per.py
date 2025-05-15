import pandas as pd
import numpy as np

def calculate_positive_percentage(input_filename):
    # 读取Excel文件
    df = pd.read_excel(input_filename, sheet_name='Sheet1')
    # 过滤列名：保留包含两个下划线 '_' 的列，并排除以 '0_' 开头的列
    filtered_columns = [col for col in df.columns if col.count('_') == 2 and not col.startswith('0_')]
    # 计算每一列大于0的百分比
    percent_positive = {}
    for column in filtered_columns:
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
    output_filename = input_filename.replace('.xlsx', '-result.xlsx')
    result_df.to_excel(output_filename)

# 主程序
if __name__ == "__main__":
    # 示例调用
    input_filename = 'tdx_block_pre_data_401-430.xlsx'
    calculate_positive_percentage(input_filename)