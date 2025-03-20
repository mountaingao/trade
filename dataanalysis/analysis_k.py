import pandas as pd
from stockrating.read_local_info_tdx import  get_stock_data_from_date
def read_excel_and_get_top_n(file_path, n):
    # 读取 Excel 文件
    df = pd.read_excel(file_path)
    # 获取前 N 条数据
    top_n_data = df.head(n)
    return top_n_data

def calculate_stock_statistics(stock_code, stock_name, start_date):
    # 检查 stock_code 是否为 NaN
    if pd.isna(stock_code):
        return None
    
    # 计算股票利润
    stock_code = str(int(stock_code))
    start_date = str(int(start_date))

    stock_data = get_stock_data_from_date(stock_code, start_date, days=5)
    print(stock_data)
    return_data = {}
    if stock_data is not None:
        return_data[stock_code] = calculate_price_comparison(stock_data)

    return return_data

def generate_analysis_excel(stock_data, output_file):
    # 创建一个空的 DataFrame，用于存储统计结果
    print(stock_data)
    result_df = pd.DataFrame()
    for _, row in stock_data.iterrows():
        print(row)

        stock_name = row['stock_name']  # 股票名称
        stock_code = row['stock_code']  # 股票代码
        start_date = row['alert_date']  #

        print(stock_code)
        if stock_name is not None:
            # 计算统计结果
            statistics = calculate_stock_statistics(stock_code, stock_name, start_date)
            # 将结果转换为 DataFrame
            if statistics is not None:
                result_df[stock_code] = statistics
        exit()
    # 生成 Excel 文件
    result_df.to_excel(output_file, index=False)

def calculate_price_comparison(stock_data):
    # 计算最高价、最低价和收盘价是否比前一个日期高
    stock_data['high_higher'] = (stock_data['high'] > stock_data['high'].shift(1)).astype(int)
    stock_data['low_higher'] = (stock_data['low'] > stock_data['low'].shift(1)).astype(int)
    stock_data['close_higher'] = (stock_data['close'] > stock_data['close'].shift(1)).astype(int)
    stock_data['open_higher'] = (stock_data['open'] > stock_data['open'].shift(1)).astype(int)

    print(stock_data['high_higher'])
    print(stock_data['low_higher'])
    print(stock_data['close_higher'])
    print(stock_data['open_higher'])
    # 计算比例
    high_ratio = stock_data['high_higher'].mean()
    low_ratio = stock_data['low_higher'].mean()
    close_ratio = stock_data['close_higher'].mean()

    print(high_ratio, low_ratio, close_ratio)
    return high_ratio, low_ratio, close_ratio

# 示例用法
if __name__ == "__main__":
    file_path = '202501.xlsx'
    top_n_data = read_excel_and_get_top_n(file_path, 34)
    output_file = 'stock_analysis.xlsx'
    generate_analysis_excel(top_n_data, output_file)
    
    # # 示例数据
    # data = {
    #     'date': ['2025-01-23', '2025-01-24', '2025-01-27', '2025-02-05', '2025-02-06', '2025-02-07'],
    #     'open': [18.18, 19.01, 24.10, 28.92, 34.70, 36.39],
    #     'high': [19.99, 20.50, 24.10, 28.92, 34.70, 41.60],
    #     'low': [18.18, 18.80, 24.10, 28.92, 34.11, 35.85],
    #     'close': [19.30, 20.08, 24.10, 28.92, 34.70, 39.94],
    #     'amount': [1.837448e+09, 1.713788e+09, 4.871454e+08, 1.145861e+08, 2.707779e+09, 7.230999e+09],
    #     'volume': [963461.00, 872764.00, 202135.02, 39621.75, 780545.60, 1863297.60]
    # }
    # df = pd.DataFrame(data)
    # df.set_index('date', inplace=True)
    #
    # high_ratio, low_ratio, close_ratio = calculate_price_comparison(df)
    # print(f"最高价比前一日高的比例: {high_ratio:.2%}")
    # print(f"最低价比前一日高的比例: {low_ratio:.2%}")
    # print(f"收盘价比前一日高的比例: {close_ratio:.2%}")