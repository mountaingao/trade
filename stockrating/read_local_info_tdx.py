from mootdx.quotes import Quotes
from mootdx.reader import Reader
import datetime
# 本地日线数据
def get_stock_history_by_local():
    # 创建 Reader 对象
    # reader = Reader.factory(market='std', tdxdir='D:/new_haitong/')
    reader = Reader.factory(market='std', tdxdir='D:/zd_haitong/')

    # 获取历史日线数据
    daily_data = reader.daily(symbol='300264')
    print("日线数据：", daily_data)
    return daily_data


client = Quotes.factory(market='std')
# client = Quotes.factory(market='std', multithread=True, heartbeat=True, bestip=False, timeout=15)

# 远程日线数据
def get_stock_history_by_remote( symbol='', begin=None, end=None, **kwargs):

    """
    读取k线信息

    :param symbol:  股票代码
    :param begin:   开始日期
    :param end:     截止日期
    :return: pd.dataFrame or None
    """
    return client.k(symbol, begin, end)


# 实时行情，列表形式返回
def get_stock_quotes_by_remote(symbols):
    client_data = client.quotes(symbol=["000001", "600300"])
    print(client_data)
    return client_data



# from mootdx import consts
#
# # client = Quotes.factory(market='std')
# symbol = client.stocks(market=consts.MARKET_SH)

# 历史分时
from mootdx.quotes import Quotes

# client = Quotes.factory(market='std')
# 当日和历史分时
# info = client.F10C(symbol='301210')
# 股票F10数据
def get_stock_f10_by_remote(symbol=''):
    info = client.F10(symbol=symbol, name='最新提示')
    print(info)

def get_stock_minutes_by_remote(symbol='', date=''):
# minu = client.minutes(symbol='301210', date='20250310')
    """
   读取某日分钟数据
   :param symbol:  股票代码
   :param date:   开始日期
   """
    if date == '':
        date = datetime.now().strftime('%Y%m%d')
    minutes = client.minutes(symbol, date)
    print(minutes)
    return minutes