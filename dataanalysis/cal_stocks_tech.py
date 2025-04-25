import datetime
import time

from stockrating.tech import get_macd,sma_base,cal_ma_amount,cal_boll



def process_stock_data(input_file, output_file):
    """
    读取包含日期和股票代码的文件，逐行处理数据，并计算技术指标，最后将结果写入 Excel 文件。

    :param input_file: 输入文件路径，包含日期和股票代码
    :param output_file: 输出文件路径，用于保存计算结果
    """
    import pandas as pd
    from stockrating.read_local_info_tdx import get_stock_history_by_local

    # 读取输入文件
    df_input = pd.read_csv(input_file, sep='\t')

    # 创建一个空的 DataFrame 用于存储结果
    results = []

    # 逐行处理数据
    for index, row in df_input.iterrows():
        date = row['date']
        code = str(row['code'])  # 将code转换为字符串

        # 获取股票历史数据
        data = get_stock_history_by_local(code)

        # 计算技术指标
        sma_result = sma_base(data, 6.5, 1)
        macd_result = get_macd(data)
        boll_result = cal_boll(data, date)
        ma_amount_result = cal_ma_amount(data, date, 'amount')

        # 将结果添加到列表中
        results.append({
            'date': date,
            'code': code,
            'sma': sma_result['sma'].iloc[-1],
            'macd': macd_result['MACD'].iloc[-1],
            'boll': boll_result,
            'ma_amount_3_days_ratio': ma_amount_result['3_days_ratio'],
            'ma_amount_5_days_ratio': ma_amount_result['5_days_ratio'],
            'ma_amount_8_days_ratio': ma_amount_result['8_days_ratio'],
            'ma_amount_11_days_ratio': ma_amount_result['11_days_ratio']
        })

    # 将结果转换为 DataFrame
    df_results = pd.DataFrame(results)

    # 将结果写入 Excel 文件
    df_results.to_excel(output_file, index=False)

# 筛选符合条件的行（每隔1分钟）
if __name__ == '__main__':

    file_path = r"202504.txt"

    process_stock_data(file_path, 'output.xlsx')
