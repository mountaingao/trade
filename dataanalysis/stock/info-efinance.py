"""
pip install efinance


"""


import efinance as ef
import pandas as pd
# 1. 批量拉取历史日K线
stocks = ['600519', '000001']
kline = ef.stock.get_quote_history(stock_codes=stocks, beg='20240101', end='20240501')
print(kline)
# print(kline.groupby('股票代码').head())
# 2. 实时行情
realtime = ef.stock.get_realtime_quotes()
print(realtime[['股票代码', '最新价', '涨跌幅']].head())
# 3. 基金净值
funds = ['161725', '005827']
fund_nav = ef.fund.get_quote_history(funds)
print(fund_nav.head())
# 4. 输出文件备份
kline.to_csv('stocks_kline.csv', index=False)
fund_nav.to_csv('funds_nav.csv', index=False)