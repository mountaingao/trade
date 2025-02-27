import akshare as ak
import os

os.makedirs("stock_data", exist_ok=True)
# 获取指定股票的历史成交数据
def get_stock_history_data(stock_code, start_date, end_date, period="daily", adjust=""):
    """
    获取指定股票的历史成交数据。

    参数:
    - stock_code: 股票代码，如 "000001"（平安银行）。
    - start_date: 查询开始日期，格式为 "YYYYMMDD"。
    - end_date: 查询结束日期，格式为 "YYYYMMDD"。
    - period: 数据周期，可选值为 "daily"（日线）、"weekly"（周线）、"monthly"（月线）。
    - adjust: 复权方式，可选值为 ""（不复权）、"qfq"（前复权）、"hfq"（后复权）。

    返回:
    - pandas.DataFrame，包含历史成交数据。
    """
    try:
        # 获取历史行情数据
        stock_data = ak.stock_zh_a_hist(
            symbol=stock_code,
            period=period,
            start_date=start_date,
            end_date=end_date,
            adjust=adjust
        )
        return stock_data
    except Exception as e:
        print(f"获取数据时出错：{e}")
        return None

# 示例：获取股票 "000001" 的历史数据
if __name__ == "__main__":
    stock_code = "000001"  # 平安银行
    start_date = "20240101"  # 查询开始日期
    end_date = "20250101"  # 查询结束日期
    period = "daily"  # 日线数据
    adjust = "qfq"  # 前复权

    # 获取历史数据
    stock_data = get_stock_history_data(stock_code, start_date, end_date, period, adjust)

    # 打印数据
    if stock_data is not None:
        print(stock_data)
        # 将数据保存为 CSV 文件
        file_name = f"stock_data/{stock_code}.csv"
        stock_data.to_csv(file_name, index=False)
        print(f"（{stock_code}）的数据已成功保存到 {file_name}")
    #     "日期",
    # "股票代码",
    # "开盘",
    # "收盘",
    # "最高",
    # "最低",
    # "成交量",
    # "成交额",
    # "振幅",
    # "涨跌幅",
    # "涨跌额",
    # "换手率",