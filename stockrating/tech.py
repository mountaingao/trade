# 假设有一个包含每日成交量的列表 volume_data

import pandas as pd
import json
import os




# 新增代码：读取配置文件
config_path = os.path.join(os.path.dirname(__file__), '..', 'config', 'config.json')
with open(config_path, 'r', encoding='utf-8') as config_file:  # 修改编码为utf-8
    config = json.load(config_file)

db_config = config['db_config']


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

    # 返回全部的macd
    MACD = pd.DataFrame({'DIFF':DIFF,'DEA':DEA,'MACD':BAR})
    # 将计算出的 MACD 值直接添加到 data 中
    data['DIFF'] = DIFF
    data['DEA'] = DEA
    data['MACD'] = BAR

    return data

def ma(data,val,window=5):
    ma_5 = data[f"{val}"].rolling(window).mean()
    return ma_5

def cal_ma_amount(data, date, val='close'):
    # 查找和定位到该日期数据
    date_index = data.index.get_loc(date)
    
    # 计算3日、5日、8日、11日的成交金额数据
    ma_amount_3 = ma(data, val, 3)
    ma_amount_5 = ma(data, val, 5)
    ma_amount_8 = ma(data, val, 8)
    ma_amount_11 = ma(data, val, 11)
    
    # 获取当日数据
    current_amount = ma_amount_3.iloc[date_index]
    
    # 计算3日前的数据
    amount_3_days_ago = ma_amount_3.iloc[date_index - 3]
    print(ma_amount_3)
    # 计算比值
    ratio_3 = current_amount / amount_3_days_ago
    print(f"3日比值: {ratio_3} = {current_amount}/{amount_3_days_ago}")
    # 返回结果
    result = {
        'date': date,
        '3_days_ratio': ratio_3.round(3),
        '5_days_ratio': (ma_amount_5.iloc[date_index]/ma_amount_5.iloc[date_index - 5]).round(3),
        '8_days_ratio': (ma_amount_8.iloc[date_index]/ma_amount_8.iloc[date_index - 8]).round(3),
        '11_days_ratio': (ma_amount_11.iloc[date_index]/ma_amount_11.iloc[date_index - 11]).round(3),
    }
    
    return result

def boll(data,val,window=20):
    ma = data[f"{val}"].rolling(window).mean()
    upper = (ma + 2 * data[f"{val}"].rolling(window).std()).round(2)
    lower = (ma - 2 * data[f"{val}"].rolling(window).std()).round(2)
    data['ma'] = ma.round(2)
    data['upper'] = upper
    data['lower'] = lower
    return data


def sma(data, val, window, weight=1):
    """
    计算加权移动平均 (Weighted Simple Moving Average)

    :param data: DataFrame 包含时间序列数据
    :param val: 字符串，需要计算移动平均的列名
    :param window: 窗口大小
    :param weight: 最近M个数据点的权重
    :return: 计算后的加权移动平均值
    """
    # 确保窗口大小大于权重
    if window < weight:
        raise ValueError("窗口大小必须大于或等于权重")

    # 计算加权移动平均
    def calculate_weighted_sma(series):
        # 获取最近N个数据点
        recent_data = series[-window:]
        # 计算加权部分
        weighted_sum = sum(recent_data[-weight:] * weight)
        # 计算非加权部分
        non_weighted_sum = sum(recent_data[:-weight])
        # 计算加权移动平均
        return (weighted_sum + non_weighted_sum) / window

    # 应用到DataFrame的每一行
    data['sma'] = data[val].rolling(window=window).apply(calculate_weighted_sma, raw=False)

    return data

def sma_base(data, period, weight,val='close'):
    """
    计算通达信公式中的SMA（加权移动平均）。

    参数:
        data (list): 数据列表，例如收盘价列表。
        period (float): 计算周期，例如6.5。
        weight (float): 权重参数，例如1。

    返回:
        list: 计算得到的SMA值列表。
    """
    if len(data) < period:
        raise ValueError("数据长度必须大于或等于周期")
    print(data)
    sma_values = []
    # 初始值使用前几个数据点的平均值
    initial_value = sum(data[:int(period)][val]) / int(period)
    # sma_values.append(initial_value)
    for i in range(len(data)):
        if i < int(period):
            # 在周期内，使用简单的平均值
            sma_values.append(sum(data[:i+1][val]) / (i+1))
            # 初始值为第一个数据点
            # sma_values.append(data.iloc[i][val])
        else:
            # 使用递推公式计算SMA .round(3)
            sma_values.append(((weight * data.iloc[i][val] + (period - weight) * sma_values[i - 1]) / period).round(3))
    print(sma_values)
    data['sma'] = sma_values
    return data


# # 示例数据
# data = [10, 15, 20, 25, 30, 35, 40, 45, 50, 55]
#
# # 计算SMA(C, 6.5, 1)
# period = 6.5
# weight = 1
# sma_result = sma(data, period, weight)
#
# # 输出结果
# print(f"SMA(C, {period}, {weight}) 的结果是: {sma_result}")

def weighted_sma_manual(data, val, window, weight):
    """
    手动计算加权移动平均 (Weighted Simple Moving Average)

    :param data: DataFrame 包含时间序列数据
    :param val: 字符串，需要计算移动平均的列名
    :param window: 窗口大小（可以是非整数）
    :param weight: 最近M个数据点的权重
    :return: 计算后的加权移动平均值
    """
    # 确保窗口大小大于权重
    if window < weight:
        raise ValueError("窗口大小必须大于或等于权重")

    # 创建一个新的列来存储加权移动平均值
    data['weighted_sma'] = 0.0

    # 手动计算加权移动平均
    for i in range(window - 1, len(data)):
        # 获取最近N个数据点
        recent_data = data[val].iloc[i - window + 1:i + 1]
        # 计算加权部分
        weighted_sum = sum(recent_data[-weight:] * weight)
        # 计算非加权部分
        non_weighted_sum = sum(recent_data[:-weight])
        # 计算加权移动平均
        data.at[i, 'weighted_sma'] = (weighted_sum + non_weighted_sum) / window

    return data


def cal_boll(data,date):
    result = boll(data, 'close',60)
    # 查找和定位到该日期数据
    date_index = result.index.get_loc(date)
    # 返回当日数据,上轨以上，返回1
    date_data =  result.iloc[date_index]
    if date_data['close'] > date_data['upper']:
        return 1
    else:
        return 0
def MA(DF, N):
    return pd.Series.rolling(DF, N).mean().round(2)

def EMA(DF, N):
    return pd.Series(DF).ewm(alpha=2/(N+1), adjust=True).mean().round(2)

def SMA(DF, N, M):
    return pd.Series(DF).ewm(alpha=M / N, adjust=True).mean().round(2)

if __name__ == '__main__':
    from stockrating.read_local_info_tdx import get_stock_history_by_local
    data = get_stock_history_by_local('300100')
    # data = get_stock_history_by_local('300005')

    # 假设data是包含股票历史数据的DataFrame，'close'是收盘价列
    result = sma_base(data, 6.5, 1)
    print(result)
    data = SMA(data['close'], 6.5, 1)
    print(data)

    # data = weighted_sma_manual(data, 'close', 6.5, 1)
    # print(data.head())

    # result =  cal_ma_amount(data, '2025-04-21', 'amount')
    # print(result)
    exit()
    result = cal_boll(data, '2025-04-21')
    print(result)
    print(cal_boll(data, '2025-04-15'))



    exit()

    result = get_macd(data)
    print(result)


#
# # 计算最近3日、5日、8日、13日的成交量之和
# recent_3_days = sum(volume_data[-3:])
# recent_5_days = sum(volume_data[-5:])
# recent_8_days = sum(volume_data[-8:])
# recent_13_days = sum(volume_data[-13:])
#
# # 计算前3日、5日、8日、13日的成交量之和
# previous_3_days = sum(volume_data[-6:-3])
# previous_5_days = sum(volume_data[-10:-5])
# previous_8_days = sum(volume_data[-16:-8])
# previous_13_days = sum(volume_data[-26:-13])
#
# # 计算放量比例
# volume_ratio_3_days = recent_3_days / previous_3_days
# volume_ratio_5_days = recent_5_days / previous_5_days
# volume_ratio_8_days = recent_8_days / previous_8_days
# volume_ratio_13_days = recent_13_days / previous_13_days
#
# # 输出结果
# print(f"最近3日放量比例: {volume_ratio_3_days}")
# print(f"最近5日放量比例: {volume_ratio_5_days}")
# print(f"最近8日放量比例: {volume_ratio_8_days}")
# print(f"最近13日放量比例: {volume_ratio_13_days}")
#
#
# 一、常见技术指标的中英文对照与计算方法
# 1. 移动平均线 (Moving Average, MA)
# 简单移动平均线 (SMA):
#
# python
# df['SMA_10'] = df['Close'].rolling(window=10).mean()
# 英文: Simple Moving Average (SMA)
#
# 计算: 最近N个收盘价的算术平均值
#
# 指数移动平均线 (EMA):
#
# python
# df['EMA_12'] = df['Close'].ewm(span=12, adjust=False).mean()
# 英文: Exponential Moving Average (EMA)
#
# 计算: 给予近期价格更高权重的移动平均
#
# 2. 相对强弱指数 (Relative Strength Index, RSI)
# python
# delta = df['Close'].diff()
# gain = delta.where(delta > 0, 0)
# loss = -delta.where(delta < 0, 0)
# avg_gain = gain.rolling(window=14).mean()
# avg_loss = loss.rolling(window=14).mean()
# rs = avg_gain / avg_loss
# df['RSI'] = 100 - (100 / (1 + rs))
# 英文: Relative Strength Index (RSI)
#
# 计算: 100 - (100 / (1 + 平均涨幅/平均跌幅))
#
# 3. 移动平均收敛发散 (MACD)
# python
# df['EMA_12'] = df['Close'].ewm(span=12, adjust=False).mean()
# df['EMA_26'] = df['Close'].ewm(span=26, adjust=False).mean()
# df['DIF'] = df['EMA_12'] - df['EMA_26']
# df['DEA'] = df['DIF'].ewm(span=9, adjust=False).mean()
# df['MACD'] = (df['DIF'] - df['DEA']) * 2
# 英文: Moving Average Convergence Divergence (MACD)
#
# 组成:
#
# DIF (Difference Line): 12日EMA - 26日EMA
#
# DEA (Signal Line): DIF的9日EMA
#
# MACD柱状图: (DIF-DEA)×2
#
# 4. 布林带 (Bollinger Bands)
# python
# df['MA_20'] = df['Close'].rolling(window=20).mean()
# df['Upper'] = df['MA_20'] + 2 * df['Close'].rolling(window=20).std()
# df['Lower'] = df['MA_20'] - 2 * df['Close'].rolling(window=20).std()
# 英文: Bollinger Bands (BB)
#
# 组成:
#
# 中轨: N日移动平均线
#
# 上轨: 中轨 + K×标准差
#
# 下轨: 中轨 - K×标准差
#
# (通常N=20，K=2)
#
# 5. 随机指标 (Stochastic Oscillator)
# python
# low_14 = df['Low'].rolling(window=14).min()
# high_14 = df['High'].rolling(window=14).max()
# df['%K'] = 100 * ((df['Close'] - low_14) / (high_14 - low_14))
# df['%D'] = df['%K'].rolling(window=3).mean()
# 英文: Stochastic Oscillator
#
# 计算:
#
# %K = (当前收盘价 - N日内最低价)/(N日内最高价 - N日内最低价) × 100
#
# %D = %K的M日移动平均
#
# (通常N=14，M=3)
#
# 二、技术指标分类英文表达
# 中文名称	英文表达	主要用途
# 趋势指标	Trend Indicators	判断市场趋势方向
# 动量指标	Momentum Indicators	衡量价格变化速度
# 波动率指标	Volatility Indicators	测量价格波动幅度
# 成交量指标	Volume Indicators	分析交易量变化
# 超买超卖指标	Overbought/Oversold	判断市场极端状态
# 三、技术指标应用场景英文表达
# 趋势跟踪 (Trend Following):
#
# "We use moving averages to identify the prevailing market trend."
#
# 中文: 我们使用移动平均线来识别当前市场趋势。
#
# 超买超卖信号 (Overbought/Oversold Signals):
#
# "The RSI above 70 indicates overbought conditions, while below 30 suggests oversold."
#
# 中文: RSI高于70表示超买，低于30表示超卖。
#
# 背离分析 (Divergence Analysis):
#
# "Bearish divergence occurs when price makes higher highs while the indicator shows lower highs."
#
# 中文: 当价格创出更高高点而指标出现更低高点时，形成看跌背离。
#
# 交叉信号 (Crossover Signals):
#
# "A bullish signal is generated when the fast MA crosses above the slow MA."
#
# 中文: 当快速均线上穿慢速均线时产生看涨信号。