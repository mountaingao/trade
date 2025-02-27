import akshare as ak
import pandas as pd
from datetime import datetime, timedelta
import os

# 配置
data_dir = "stock_data"  # 本地数据存储目录
os.makedirs(data_dir, exist_ok=True)

def get_stock_data(stock_code, days=20):
    """
    获取指定股票过去20天内的平均成交额、最高成交价和最低成交价。
    如果本地存在数据文件，则优先读取本地数据。

    参数:
    - stock_code: 股票代码，如 "000001"（平安银行）。
    - days: 查询的天数，默认为20天。

    返回:
    - 一个字典，包含平均成交额、最高成交价和最低成交价。
    """
    # 计算日期范围
    end_date = datetime.now().strftime("%Y%m%d")
    start_date = (datetime.now() - timedelta(days=days)).strftime("%Y%m%d")

    # 本地数据文件路径
    file_name = f"{data_dir}/{stock_code}.csv"

    # 检查本地是否存在数据文件
    if os.path.exists(file_name):
        print(f"本地数据文件存在，直接读取：{file_name}")
        stock_data = pd.read_csv(file_name)
    else:
        print(f"本地数据文件不存在，通过 akshare 获取数据并保存到本地：{file_name}")
        try:
            # 获取股票日线行情数据
            stock_data = ak.stock_zh_a_hist(
                symbol=stock_code,
                period="daily",
                start_date=start_date,
                end_date=end_date,
                adjust=""
            )
            # 将数据保存为 CSV 文件
            stock_data.to_csv(file_name, index=False)
        except Exception as e:
            print(f"获取数据时出错：{e}")
            return None

    # 计算所需指标
    avg_turnover = stock_data["成交额"].mean()  # 平均成交额
    max_price = stock_data["最高"].max()  # 最高成交价
    min_price = stock_data["最低"].min()  # 最低成交价

    return {
        "average_turnover": avg_turnover,
        "max_price": max_price,
        "min_price": min_price
    }

# 示例：获取股票 "000001" 的数据
if __name__ == "__main__":
    stock_code = "000001"  # 平安银行
    result = get_stock_data(stock_code)

    if result:
        print(f"股票代码：{stock_code}")
        print(f"过去20天的平均成交额：{result['average_turnover']:.2f}")
        print(f"过去20天的最高成交价：{result['max_price']:.2f}")
        print(f"过去20天的最低成交价：{result['min_price']:.2f}")