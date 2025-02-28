
# 此部分分为两种情况：
# 1、竞价结束，获取当时成交额，可根据昨日成交额，预测当日总成交额；
# 2、正常交易时间，获取当时成交额，预测当日总成交额。

import pandas as pd
from scipy.interpolate import interp1d

import akshare as ak

def get_minute_data(stock_code: str, date: str, period: str = "1") -> pd.DataFrame:
    """
    获取指定股票在指定日期的分钟级别行情数据。

    参数:
        stock_code (str): 股票代码，如 "300718"。
        date (str): 日期，格式为 "YYYYMMDD"，如 "20250226"。
        period (str): 时间周期，默认为 "1"，表示1分钟。

    返回:
        pd.DataFrame: 分钟级别行情数据。
    """
    try:
        # 获取分钟级别行情数据
        minute_data = ak.stock_zh_a_minute(symbol=stock_code, period=period)
        return minute_data
    except Exception as e:
        print(f"获取分钟级别行情数据时出错：{e}")
        return pd.DataFrame()
stock_code = "sz300718"  # 长盛轴承
date = "20250226"  # 日期
period = "1"  # 1分钟

minute_data = get_minute_data(stock_code, date, period)
print("分钟级别行情数据：")
print(minute_data)

def estimate_daily_volume(current_volume, current_time, total_trading_minutes=240, historical_factors=None):
    """
    估算当日成交额
    :param current_volume: 当前成交额（单位：亿）
    :param current_time: 当前时间（格式：HH:MM）
    :param total_trading_minutes: 总交易时间（分钟，默认240分钟）
    :param historical_factors: 历史活跃度调整因子（字典，键为累计分钟数，值为调整因子）
    :return: 估算的当日总成交额（单位：亿）
    """
    try:
        # 将时间转换为分钟
        current_hour, current_minute = map(int, current_time.split(':'))
        if current_hour < 9 or (current_hour == 9 and current_minute < 30):
            print("当前时间早于9:30，无法进行估算。")
            return None

        elapsed_minutes = (current_hour - 9) * 60 + current_minute - 30  # 从9:30开始计算

        # 避免除零错误
        if elapsed_minutes <= 0 or elapsed_minutes > total_trading_minutes:
            print("当前时间无效，无法进行估算。")
            return None

        # 计算时间比例
        time_ratio = elapsed_minutes / total_trading_minutes

        # 获取当前时间点的 historical_factor
        if historical_factors is not None:
            minutes_list = sorted(historical_factors.keys())
            factors_list = [historical_factors[minute] for minute in minutes_list]

            # 创建插值函数
            interpolation_func = interp1d(minutes_list, factors_list, kind='linear', fill_value="extrapolate")

            # 获取当前时间点的 historical_factor
            historical_factor = float(interpolation_func(elapsed_minutes))
        else:
            historical_factor = 1.0  # 默认值

        # 估算当日总成交额
        estimated_daily_volume = current_volume / (time_ratio * historical_factor)

        return estimated_daily_volume

    except ValueError:
        print("时间格式错误，请输入正确的HH:MM格式。")
        return None
    except Exception as e:
        print(f"发生未知错误: {e}")
        return None

# 示例
current_volume = 2  # 当前成交额（亿）
current_time = "10:00"  # 当前时间
historical_factors = {
    0: 0.8,
    30: 0.9,
    60: 1.0,
    120: 1.1,
    180: 1.2,
    240: 1.3
}  # 示例历史活跃度调整因子

estimated_volume = estimate_daily_volume(current_volume, current_time, historical_factors=historical_factors)
if estimated_volume is not None:
    print(f"估算的当日总成交额为：{estimated_volume:.2f} 亿")


def generate_historical_factors(csv_file):
    """
    从CSV文件中读取历史活跃度调整因子，并生成每分钟的历史活跃度调整因子。

    :param csv_file: 包含时间和历史活跃度调整因子的CSV文件路径
    :return: 每分钟的历史活跃度调整因子字典
    """
    # 读取CSV文件
    df = pd.read_csv(csv_file)

    # 将时间转换为累计分钟数
    def time_to_minutes(time_str):
        hour, minute = map(int, time_str.split(':'))
        return (hour - 9) * 60 + minute - 30

    df['elapsed_minutes'] = df['time'].apply(time_to_minutes)

    # 提取时间和历史活跃度调整因子
    minutes_list = df['elapsed_minutes'].values
    factors_list = df['historical_factor'].values

    # 创建插值函数
    interpolation_func = interp1d(minutes_list, factors_list, kind='linear', fill_value="extrapolate")

    # 生成每分钟的历史活跃度调整因子
    historical_factors = {}
    for minute in range(0, 240):  # 总交易时间为240分钟
        historical_factors[minute] = float(interpolation_func(minute))

    return historical_factors

# # 示例
# csv_file = '300718_2023-02-25.csv'
# historical_factors = generate_historical_factors(csv_file)
#
# # 打印部分结果以验证
# for minute in range(0, 240, 15):  # 每15分钟打印一次
#     print(f"Minute {minute}: {historical_factors[minute]:.2f}")