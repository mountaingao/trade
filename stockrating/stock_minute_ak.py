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

def calculate_percentage(data: pd.DataFrame) -> pd.DataFrame:
    """
    计算每分钟成交额占全天成交额的百分比。
    """
    total_volume = data['成交额'].sum()
    data['percentage'] = data['成交额'] / total_volume
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

    print(combined_data)
    # 按日期分组，计算每分钟的百分比
    grouped_data = combined_data.groupby(combined_data.index.date).apply(calculate_percentage)

    # 计算每分钟的平均百分比
    average_percentage = grouped_data.groupby(grouped_data.index.time)['percentage'].mean()

    # 输出结果
    print("每分钟成交额占全天成交额的平均百分比：")
    print(average_percentage)

    return average_percentage

# 示例：获取股票300718过去10天的数据
stock_code = "300718"
average_percentage = main(stock_code, days=10)