from mootdx.quotes import Quotes

client = Quotes.factory(market='std')
# client = Quotes.factory(market='std', multithread=True, heartbeat=True, bestip=False, timeout=15)

client_data = client.quotes(symbol=["000001", "600300"])

print(client_data)

from mootdx import consts

# client = Quotes.factory(market='std')
symbol = client.stocks(market=consts.MARKET_SH)

# 历史分时
from mootdx.quotes import Quotes

# client = Quotes.factory(market='std')
# 当日和历史分时
# info = client.F10C(symbol='301210')
info = client.F10(symbol='301210', name='最新提示')
print(info)


# minu = client.minutes(symbol='301210', date='20250310')
minu = client.minutes(symbol='301210', date='20250307')
print(minu)

# 当日分时

minu = client.minutes(symbol='301210', date='20171010')
print(minu)