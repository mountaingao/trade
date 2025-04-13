import talib
import pandas as pd
import json
import os
from mootdx.reader import Reader
from datetime import datetime

# 新增代码：读取配置文件
config_path = os.path.join(os.path.dirname(__file__), '..', 'config', 'config.json')
with open(config_path, 'r', encoding='utf-8') as config_file:  # 修改编码为utf-8
    config = json.load(config_file)

db_config = config['db_config']

reader = Reader.factory(market='std', tdxdir=config['tdxdir'])
symbol = "300871"
stock_data = reader.daily(symbol=symbol)




print(stock_data)

# 复制最后 100 条数据
stock_data_last_100 = stock_data[-100:].copy()

# 假设 stock_data 是一个包含 '收盘' 列的 DataFrame
macd, signal, hist = talib.MACD(stock_data_last_100['close'], fastperiod=12, slowperiod=26, signalperiod=9)

# 将结果添加到 DataFrame
stock_data_last_100['MACD'] = macd
stock_data_last_100['Signal_Line'] = signal
stock_data_last_100['MACD_Histogram'] = hist

stock_data_last_100.index = stock_data_last_100.index.strftime('%Y-%m-%d')

print(stock_data_last_100)

# 打印结果
print(stock_data_last_100[[ 'close', 'MACD', 'Signal_Line', 'MACD_Histogram']].tail())



# import numpy as np
# from numpy import NaN as npNaN
#
# # 替换 pandas_ta 中的 NaN 引用
# import pandas_ta as ta
# ta.core.npNaN = npNaN
#
# # 示例数据
#
# # 计算MACD
# stock_data.ta.macd(append=True)
#
# # 打印结果
# print(stock_data[['date', 'close', 'MACD_12_26_9', 'MACDs_12_26_9', 'MACDh_12_26_9']].tail())


# 计算MACD
def calculate_macd(data, short_window=12, long_window=26, signal_window=9):
    # 计算短期和长期EMA
    data['EMA_short'] = data['close'].ewm(span=short_window, adjust=False).mean()
    data['EMA_long'] = data['close'].ewm(span=long_window, adjust=False).mean()

    # 计算MACD线
    data['MACD'] = data['EMA_short'] - data['EMA_long']

    # 计算信号线
    data['Signal_Line'] = data['MACD'].ewm(span=signal_window, adjust=False).mean()

    # 计算MACD柱状图
    data['MACD_Histogram'] = data['MACD'] - data['Signal_Line']

    return data

def get_macd(data):
    # 编写计算函数
    # 上市首日，DIFF、DEA、MACD = 0
    # 用来装变量的list
    EMA12 = []
    EMA26 = []
    DIFF = []
    DEA = []
    BAR = []
    # 如果是上市首日
    if len(data) == 1:
        # 则DIFF、DEA、MACD均为0
        DIFF = [0]
        DEA = [0]
        BAR = [0]

    # 如果不是首日
    else:
        # 第一日的EMA要用收盘价代替
        EMA12.append(data['close'].iloc[0])
        EMA26.append(data['close'].iloc[0])
        DIFF.append(0)
        DEA.append(0)
        BAR.append(0)

        # 计算接下来的EMA
        # 搜集收盘价
        close = list(data['close'].iloc[1:])    # 从第二天开始计算，去掉第一天
        for i in close:
            ema12 = EMA12[-1] * (11/13) + i * (2/13)
            ema26 = EMA26[-1] * (25/27) + i * (2/27)
            diff = ema12 - ema26
            dea = DEA[-1] * (8/10) + diff * (2/10)
            bar = 2 * (diff - dea)

            # 将计算结果写进list中
            EMA12.append(ema12)
            EMA26.append(ema26)
            DIFF.append(diff)
            DEA.append(dea)
            BAR.append(bar)

    # 将计算出的 MACD 值直接添加到 data 中
    data['DIFF'] = DIFF
    data['DEA'] = DEA
    data['MACD'] = BAR

    return data

def get_macd_value(stock_history, date):
    macd = get_macd(stock_history)
    # 将 alert_date 转换为 datetime 对象
    cur_date = datetime.strptime(date, "%Y%m%d").strftime("%Y-%m-%d")
    print(f"cur_date {cur_date}")
    
    # 从 DataFrame 中获取指定日期的 MACD 值
    macd_value = macd.loc[macd.index == cur_date, 'MACD'].values[0] if cur_date in macd.index else None
    print(f"macd_value {macd_value} ")

    # 获取第二日
    # 将日期转换为 datetime 对象以便比较
    cur_date_dt = datetime.strptime(cur_date, "%Y-%m-%d")
    # 获取 cur_date_dt 之前的两条数据
    prev_days = macd[macd.index < cur_date].tail(1)
    # 得到第一条数据
    prev_macd_value = prev_days['MACD'].values[0] if not prev_days.empty else None
    print(f"prev_macd_value {prev_macd_value}")
    
    if macd_value is not None and prev_macd_value is not None and macd_value > prev_macd_value:
        return 1
    else:
        return 0

# 计算MACD
stock_data_macd = calculate_macd(stock_data[-100:].copy())
print(stock_data_macd)

stock_data_macd = get_macd(stock_data[-100:].copy())
print(stock_data_macd)

print(get_macd_value(stock_data[-100:].copy(), "20250408"))

# print(stock_data[['date', 'close', 'MACD_12_26_9', 'MACDs_12_26_9', 'MACDh_12_26_9']].tail())