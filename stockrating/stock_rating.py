

# 股票评级的几个方向，30个交易日内的
# 1、近期成交额 10亿以上100分，5-10亿 5分，5-10亿 4分，10-20亿 3分，20-50亿 2分，50亿以上 1分
# 2、近期涨幅
# 3、近期振幅
# 4、股本大小
# 5、是否热门板块和概念
# 6、当日预估成交额

import akshare as ak
import pandas as pd
import numpy as np
from ak_stock_block_ths import stock_profit_forecast_ths

# 获取股票数据
def get_stock_data(stock_code):
    """
    获取股票的近期成交额、涨幅、振幅、股本等信息。
    """
    # 获取股票历史行情数据
    stock_df = ak.stock_zh_a_hist(symbol=stock_code, period="daily", start_date="20230101", end_date="20231010", adjust="qfq")

    # 计算近期成交额、涨幅、振幅
    recent_df = stock_df.tail(30)  # 取最近30个交易日的数据
    recent_turnover = recent_df["成交额"].mean()  # 近期平均成交额
    recent_pct_change = (recent_df["收盘"].iloc[-1] - recent_df["收盘"].iloc[0]) / recent_df["收盘"].iloc[0]  # 近期涨幅
    recent_amplitude = (recent_df["最高"].max() - recent_df["最低"].min()) / recent_df["最低"].min()  # 近期振幅

    # 获取股本信息
    stock_info = ak.stock_individual_info_em(symbol=stock_code)
    print(stock_info)
    # total_shares = float(stock_info.loc[stock_info["item"] == "总股本", "value"].values[0].replace("亿", ""))  # 总股本（亿股）

    # 确保 "value" 列的数据是字符串类型
    total_shares = stock_info.loc[stock_info["item"] == "流通市值", "value"].values[0]
    print(total_shares)


    return {
        "stock_code": stock_code,
        "recent_turnover": recent_turnover,
        "recent_pct_change": recent_pct_change,
        "recent_amplitude": recent_amplitude,
        "total_shares": total_shares,
    }

# 获取热门板块和概念
def is_hot_stock(stock_code):
    """
    判断股票是否属于热门板块或概念。
    """
    # 这里可以根据问财或其他数据源判断股票是否属于热门板块或概念
    # 假设我们有一个热门板块列表
    hot_sectors = ["新能源", "半导体", "医药", "消费"]
    # stock_sectors = ak.stock_sector_detail(sector=stock_code)  # 获取股票所属板块
    stock_sectors = stock_profit_forecast_ths(symbol=stock_code)
    print(f"获取到股票 {stock_code} 的概念板块数据: {stock_sectors}")

    for sector in stock_sectors:
        if sector in hot_sectors:
            return True
    return False

# 评分函数
def score_stock(stock_data, is_hot):
    """
    根据股票数据进行评分。
    """
    # 1. 近期成交额评分
    turnover_score = np.where(
        stock_data["recent_turnover"] > np.percentile(stock_data["recent_turnover"], 80), 5,
        np.where(stock_data["recent_turnover"] > np.percentile(stock_data["recent_turnover"], 60), 4,
                 np.where(stock_data["recent_turnover"] > np.percentile(stock_data["recent_turnover"], 40), 3,
                          np.where(stock_data["recent_turnover"] > np.percentile(stock_data["recent_turnover"], 20), 2, 1)))
    )

    # 2. 近期涨幅评分
    pct_change_score = np.where(
        stock_data["recent_pct_change"] > np.percentile(stock_data["recent_pct_change"], 80), 5,
        np.where(stock_data["recent_pct_change"] > np.percentile(stock_data["recent_pct_change"], 60), 4,
                 np.where(stock_data["recent_pct_change"] > np.percentile(stock_data["recent_pct_change"], 40), 3,
                          np.where(stock_data["recent_pct_change"] > np.percentile(stock_data["recent_pct_change"], 20), 2, 1)))
    )

    # 3. 近期振幅评分
    amplitude_score = np.where(
        stock_data["recent_amplitude"] > np.percentile(stock_data["recent_amplitude"], 80), 1,
        np.where(stock_data["recent_amplitude"] > np.percentile(stock_data["recent_amplitude"], 60), 2,
                 np.where(stock_data["recent_amplitude"] > np.percentile(stock_data["recent_amplitude"], 40), 3,
                          np.where(stock_data["recent_amplitude"] > np.percentile(stock_data["recent_amplitude"], 20), 4, 5)))
    )

    # 4. 股本大小评分
    shares_score = np.where(
        stock_data["total_shares"] < np.percentile(stock_data["total_shares"], 20), 5,
        np.where(stock_data["total_shares"] < np.percentile(stock_data["total_shares"], 40), 4,
                 np.where(stock_data["total_shares"] < np.percentile(stock_data["total_shares"], 60), 3,
                          np.where(stock_data["total_shares"] < np.percentile(stock_data["total_shares"], 80), 2, 1)))
    )

    # 5. 是否热门板块和概念评分
    hot_score = 5 if is_hot else 1

    print(f"近期成交额评分: {turnover_score}")
    print(f"近期涨幅评分: {pct_change_score}")
    print(f"近期振幅评分: {amplitude_score}")
    print(f"股本大小评分: {shares_score}")
    print(f"热门板块: {hot_score}")

    # 综合评分
    total_score = turnover_score + pct_change_score + amplitude_score + shares_score + hot_score
    return total_score

# 示例：对某只股票进行评级
stock_code = "600519"  # 贵州茅台
stock_data = get_stock_data(stock_code)
is_hot = is_hot_stock(stock_code)
score = score_stock(stock_data, is_hot)
print(f"股票 {stock_code} 的综合评分为: {score}")