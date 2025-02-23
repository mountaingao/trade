import akshare as ak
import pandas as pd

import akshare as ak

stock_profit_forecast_ths_df = ak.stock_profit_forecast_ths(symbol="600519", indicator="预测年报每股收益")
print(stock_profit_forecast_ths_df)
exit()
# 获取A股历史行情数据
df = ak.stock_zh_a_hist_min_em_df(symbol="000001",  # 股票代码（平安银行）
                                  period="daily",  # 数据周期（日线）
                                  start_date="20240101",  # 开始日期
                                  end_date="20240228",  # 结束日期
                                  adjust="qfq")  # 复权方式（前复权）

# 打印数据
print(df)

# 获取A股分钟级行情数据
df = ak.stock_zh_a_minute(symbol="000001",  # 股票代码（平安银行）
                          period="1",  # 分钟周期（1分钟）
                          adjust="qfq")  # 复权方式（前复权）

# 打印数据
print(df)

# 新浪接口
stock_zh_a_daily_qfq_df = ak.stock_zh_a_daily(symbol="sz000001", start_date="19910403", end_date="20231027", adjust="qfq")
# 腾讯接口
stock_zh_a_hist_tx_df = ak.stock_zh_a_hist_tx(symbol="sz000001", start_date="20200101", end_date="20231027", adjust="")
# 新浪分时数据
stock_zh_a_minute_df = ak.stock_zh_a_minute(symbol='sh600751', period='1', adjust="qfq")
# 东方财富分时数据
stock_zh_a_hist_min_em_df = ak.stock_zh_a_hist_min_em(symbol="000001", start_date="2024-03-20 09:30:00", end_date="2024-03-20 15:00:00", period="1", adjust="")
# 东方财富日内分时数据
stock_intraday_em_df = ak.stock_intraday_em(symbol="000001")
# 新浪日内分时数据
stock_intraday_sina_df = ak.stock_intraday_sina(symbol="sz000001", date="20240321")
# 盘前数据
stock_zh_a_hist_pre_min_em_df = ak.stock_zh_a_hist_pre_min_em(symbol="000001", start_time="09:00:00", end_time="15:40:00")
