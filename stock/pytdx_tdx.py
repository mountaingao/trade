from pytdx.hq import TdxHq_API

# 创建 API 对象
#todo 数据不出来，后续再研究
api = TdxHq_API()

def get_server_ip():
    from pytdx.util.best_ip import select_best_ip
    best_ip = select_best_ip()
    print(f"最优服务器：IP={best_ip['ip']}, 端口={best_ip['port']}")
    # 最优服务器：IP=180.153.18.170, 端口=7709

# 连接服务器
# if api.connect('123.125.108.90', 7709): # 海通
if api.connect('180.153.18.170', 7709):
    print("连接成功！")
    # 做一些操作
    api.disconnect() #关闭连接
else:
    print("连接失败！")

# 获取日 K 线数据
# K线种类
# 0 5分钟K线 
# 1 15分钟K线 
# 2 30分钟K线 
# 3 1小时K线 
# 4 日K线
# 5 周K线
# 6 月K线
# 7 1分钟
# 8 1分钟K线 
# 9 日K线
# 10 季K线
# 11 年K线
k_data = api.get_security_bars(9, 1, '600519', 0, 10)
print(k_data)

# 获取最近 10 条分笔成交数据
transaction_data = api.get_transaction_data(1, '600519', 0, 10)
print(api.to_df(transaction_data))

# 获取单只股票的实时行情
stock_data = api.get_security_quotes(0,'300328')
print(stock_data)

api.disconnect()

# 1 代表上海证券交易所 （6开头的股票，688、689 也填1）0 代表深证证券交易交所 （3、0 开头的股票）2 代表北京证券交易所 （4、8、9 开头的股票）