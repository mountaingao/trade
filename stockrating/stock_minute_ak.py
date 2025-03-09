import akshare as ak
import pandas as pd
from datetime import datetime, timedelta

def add_market_prefix(stock_code: str) -> str:
    """
    根据股票代码自动添加市场前缀（sz 或 sh）。
    """
    if stock_code.startswith("60"):
        return f"sh{stock_code}"
    elif stock_code.startswith(("00", "30")):
        return f"sz{stock_code}"
    else:
        raise ValueError("无法识别的股票代码。")

def get_minute_data(stock_code: str, start_date: str, end_date: str) -> pd.DataFrame:
    """
    获取指定股票在指定日期范围内的分钟级别行情数据。  没有成交额，比较麻烦，通过成交量估算成交额
         day   open   high    low  close  volume
0     2025-02-18 13:53:00  84.00  84.46  84.00  84.46   35800
    """
    try:
    # 添加市场前缀
        stock_code = add_market_prefix(stock_code)
        minute_data = ak.stock_zh_a_minute(symbol=stock_code, period="1",  adjust="qfq")
    # stock_zh_a_spot_em
        return minute_data
    except Exception as e:
        print(f"获取分钟级别行情数据时出错：{e}")
        return pd.DataFrame()

def get_daily_high(data: pd.DataFrame, date: str) -> float:
    """
    获取指定日期内的最高价。
    """
    # 确保 'day' 字段是 DatetimeIndex
    if not isinstance(data['day'], pd.DatetimeIndex):
        data['day'] = pd.to_datetime(data['day'])
    # 筛选出指定日期的数据
    daily_data = data[data['day'].dt.date == pd.to_datetime(date).date()]
    if daily_data.empty:
        raise ValueError(f"没有找到 {date} 的数据。")
    return daily_data['high'].max()

def calculate_volume_percentage(data: pd.DataFrame, timestamp: str) -> pd.Series:
    """
    计算指定时间点之前的所有每分钟成交量占全天成交量的百分比。
    """
    target_time = pd.to_datetime(timestamp)
    # 确保 'day' 字段是 DatetimeIndex
    if not isinstance(data['day'], pd.DatetimeIndex):
        data['day'] = pd.to_datetime(data['day'])
    # 筛选出指定日期的数据
    daily_data = data[data['day'].dt.date == target_time.date()]
    if daily_data.empty:
        raise ValueError(f"没有找到 {target_time.date()} 的数据。")

    print(f"{target_time.date()} 的数据：", daily_data)
    # 计算全天成交量
    total_volume = daily_data['volume'].sum()

    # 获取指定时间点之前的数据
    filtered_data = daily_data[daily_data['day'] <= target_time]

    # 计算每分钟成交量占比
    filtered_data.loc[:, 'volume_percentage'] = filtered_data['volume'].astype(float) / float(total_volume)

    return filtered_data['volume_percentage']


def calculate_percentage(data: pd.DataFrame) -> pd.DataFrame:
    """
    计算每分钟成交额占全天成交额的百分比。
    """
    # 将 'volume' 列转换为 float 类型
    data['volume'] = data['volume'].astype(float)
    total_volume = data['volume'].sum()
    data['percentage'] = data['volume'] / total_volume
    return data

def main(stock_code: str, days: int = 10):
    """
    主函数：获取过去N天的数据，计算每分钟成交额的百分比，并平均这些百分比。
    """
    end_date = datetime.now().strftime('%Y%m%d')
    start_date = (datetime.now() - timedelta(days=days)).strftime('%Y%m%d')

    # 获取过去N天的分钟级别数据
    combined_data = get_minute_data(stock_code, start_date, end_date)
    if combined_data.empty:
        print("没有获取到足够的数据。")
        return

    # 确保 combined_data 的索引是 DatetimeIndex
    if not isinstance(combined_data.index, pd.DatetimeIndex):
        combined_data.index = pd.to_datetime(combined_data.index)

    # 示例：获取2025-02-25的最高价
    try:
        high_price = get_daily_high(combined_data, "2025-03-06")
        print(f"2025-02-25 的最高价: {high_price}")
    except ValueError as e:
        print(e)

    # 示例：计算2025-02-25 13:53:00的每分钟成交量占比
    try:
        volume_percentage = calculate_volume_percentage(combined_data, "2025-03-06 13:53:00")
        print("2025-02-25 13:53:00 之前的每分钟成交量占比:")
        print(volume_percentage)
    except ValueError as e:
        print(e)

    # print(combined_data)
    # 按日期分组，计算每分钟的百分比
    grouped_data = combined_data.groupby(combined_data.index.date).apply(calculate_percentage)
    print(grouped_data)
    # 计算每分钟的平均百分比
    average_percentage = grouped_data.groupby(grouped_data.index.day)['percentage'].mean()

    # 输出结果
    print("每分钟成交额占全天成交额的平均百分比：")
    print(average_percentage)

    return average_percentage

# 示例：获取股票300718过去10天的数据
stock_code = "300718"
average_percentage = main(stock_code, days=10)