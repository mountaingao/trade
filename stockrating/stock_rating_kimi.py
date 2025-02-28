import akshare as ak
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# 设置日期范围
end_date = datetime.now().strftime('%Y%m%d')
start_date = (datetime.now() - timedelta(days=30)).strftime('%Y%m%d')  # 获取过去30天的数据

# 获取沪深300成分股作为评级对象
stock_code = "600519"  # 贵州茅台
# stock_data = get_stock_data(stock_code)
stocks = ["000001", "000002", "000037"]

# 初始化评级结果列表
ratings = []

# 评级函数
def rate_stock(stock_code):
    # 获取历史行情数据
    stock_daily_df = ak.stock_zh_a_hist(symbol=stock_code, period="daily",
                                        start_date=start_date, end_date=end_date,
                                        adjust="qfq")
    stock_daily_df['date'] = pd.to_datetime(stock_daily_df['日期'])
    stock_daily_df.set_index('date', inplace=True)
    stock_daily_df['close'] = stock_daily_df['收盘'].astype(float)
    stock_daily_df['pctChg'] = stock_daily_df['涨跌幅'].astype(float)
    stock_daily_df['volume'] = stock_daily_df['成交量'].astype(float)
    stock_daily_df['amount'] = stock_daily_df['成交额'].astype(float)

    # 板块地位评分
    sector_score = 10  # 假设默认为10分，根据实际数据调整

    # 涨停次数评分
    limit_up_count = (stock_daily_df['pctChg'] >= 9.9).sum()  # 涨停定义为涨幅>=9.9%
    if limit_up_count >= 3:
        limit_up_score = 20
    elif limit_up_count == 2:
        limit_up_score = 15
    elif limit_up_count == 1:
        limit_up_score = 10
    else:
        limit_up_score = 5

    # 资金流向评分
    if stock_daily_df['amount'].mean() > 1e8:  # 平均成交额大于1亿
        fund_flow_score = 20
    elif stock_daily_df['amount'].mean() > 5e7:  # 平均成交额大于5000万
        fund_flow_score = 15
    else:
        fund_flow_score = 5

    # 技术形态评分
    stock_daily_df['MA5'] = stock_daily_df['close'].rolling(window=5).mean()
    stock_daily_df['MA10'] = stock_daily_df['close'].rolling(window=10).mean()
    stock_daily_df['MA20'] = stock_daily_df['close'].rolling(window=20).mean()
    if stock_daily_df.iloc[-1]['MA5'] > stock_daily_df.iloc[-1]['MA10'] > stock_daily_df.iloc[-1]['MA20']:
        tech_score = 30
    elif stock_daily_df.iloc[-1]['MA5'] > stock_daily_df.iloc[-1]['MA10']:
        tech_score = 20
    else:
        tech_score = 5

    # 综合评分
    total_score = (sector_score * 0.3 + limit_up_score * 0.2 + fund_flow_score * 0.2 + tech_score * 0.3)
    return total_score

# 对每只股票进行评级
for stock in stocks:
    try:
        score = rate_stock(stock)
        rating = 1
        if score >= 80:
            rating = 5
        elif score >= 60:
            rating = 4
        elif score >= 40:
            rating = 3
        elif score >= 20:
            rating = 2
        ratings.append({'stock_code': stock, 'score': score, 'rating': rating})
    except Exception as e:
        print(f"Error processing {stock}: {e}")

# 将评级结果存储到CSV文件
ratings_df = pd.DataFrame(ratings)
ratings_df.to_csv("stock_ratings.csv", index=False)

print("评级完成，结果已保存到 stock_ratings.csv 文件中。")