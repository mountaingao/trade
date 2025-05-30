import datetime
import time
import os
import json
from stockrating.tech import get_macd,sma_base,cal_ma_amount,cal_boll,cal_ema
import pandas as pd
from stockrating.read_local_info_tdx import get_stock_history_by_local
import logging


# 新增代码：读取配置文件
config_path = os.path.join(os.path.dirname(__file__), '..', 'config', 'config.json')
with open(config_path, 'r', encoding='utf-8') as config_file:  # 修改编码为utf-8
    configs = json.load(config_file)


# 配置日志
logging.basicConfig(level=logging.DEBUG if configs['debug'] else logging.INFO)
# 使用示例
# logging.debug("这是调试信息")  # 仅在 DEBUG 为 True 时输出
# logging.info("这是普通信息")   # 无论 DEBUG 为何值都会输出

def process_stock_data(input_file, output_file):
    """
    读取包含日期和股票代码的文件，逐行处理数据，并计算技术指标，最后将结果写入 Excel 文件。

    :param input_file: 输入文件路径，包含日期和股票代码
    :param output_file: 输出文件路径，用于保存计算结果
    """

    # 读取输入文件
    df_input = pd.read_csv(input_file, sep='\t')

    # 创建一个空的 DataFrame 用于存储结果
    results = []
    logging.debug(df_input)

    # 逐行处理数据
    for index, row in df_input.iterrows():
        # 处理每一行数据
        date = str(row['date'])
        code = str(row['code'])  # 将code转换为字符串
        logging.debug(row)
        result = process_single_stock(code,date)
        if result:
            results.append(result)

    # 将结果转换为 DataFrame
    df_results = pd.DataFrame(results)

    # 将结果写入 Excel 文件
    df_results.to_excel(output_file, index=False)

def process_single_stock(code,date):
    """
    处理单只股票的数据，计算技术指标并返回结果。

    :param row: 包含日期和股票代码的行数据
    :return: 包含技术指标的结果字典
    """

    # 获取股票历史数据
    data = get_stock_history_by_local(code)
    # date 转化为时间格式
    date = datetime.datetime.strptime(date, '%Y%m%d')
    # 计算技术指标

    data = sma_base(data, 6.5, 1) #上轨
    data = sma_base(data, 13.5, 1)  #下轨
    data = get_macd(data)
    # logging.debug(data)
    if date not in data.index:
        return None
    ma_result = cal_ma_amount(data, date, 'amount')
    # logging.debug(ma_result)
    boll_result = cal_boll(data, date)

    ema_result = cal_ema(data, date)

    date_index = data.index.get_loc(date)
    date_data = data.iloc[date_index]
    # logging.debug(date_data)
    # exit
    #比较date_data['close'] 和 date_data['sma'] ，大于0，则返回1，否则为0
    sma_result_up = 1 if date_data['close'] > date_data['sma-6.5'] else 0
    sma_result_down = 1 if date_data['close'] > date_data['sma-13.5'] else 0

    # 返回结果
    return {
        'date': date,
        'code': code,
        'close': date_data['close'],
        'amount': date_data['amount'],
        'zhang': ma_result['zhang'],
        'zhen': ma_result['zhen'],
        'sma_up': sma_result_up,
        'sma_down': sma_result_down,
        'macd': date_data['MACD'],
        'is_up': boll_result['is_up'],
        'consecutive_upper_days': boll_result['consecutive_upper_days'],
        'upper_days_counts': boll_result['upper_count_in_days'],
        'ma_amount_days_ratio_3': ma_result['3_days_ratio'],
        'ma_amount_days_ratio_5': ma_result['5_days_ratio'],
        'ma_amount_days_ratio_8': ma_result['8_days_ratio'],
        'ma_amount_days_ratio_11': ma_result['11_days_ratio']
    }

# 筛选符合条件的行（每隔1分钟）
if __name__ == '__main__':

    # file_path = r"20250501.txt"
    file_path = r"202504.txt"
    # output_file 为输入文件去除文件扩展名后增加后缀.xlsx
    base_name = os.path.splitext(file_path)[0]  # 去除文件扩展名
    output_file = base_name + '1.xlsx'  # 增加 .xlsx 后缀

    process_stock_data(file_path, output_file)