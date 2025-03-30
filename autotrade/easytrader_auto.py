import easytrader

# 2.初始化交易
# 接下来，您可以初始化交易账户。以下示例代码创建了一个交易客户端，并连接到默认的证券账户。
# 创建交易客户端
# user = easytrader.use('ths') # 使用同花颇客户满
user = easytrader.use('universal_client')


user.connect(r'D:\同花顺软件\同花顺\xiadan.exe') # 类似 r'C:\htzqzyb2\xiadan.exe'


from easytrader import grid_strategies

user.grid_strategy = grid_strategies.Xls
print(user.grid_strategy)
# 3. 查看账户信息
# 您可以使用以下代码查询账户信息，例如查询可用资金和当前持仓:
# 查询涨户信息
# account_info = user.get_account()
account_info = user.get_balance()
print(account_info)

# 4.买入股票
# 买入股票相对简单，您只需要指定股票代码和买入数量。例如，以下代码将会买入5股某股票(股票代码为'600000”):
# 买入股票
user.buy('600000',amount=5) # buy 5 shares of stock with code '600000

# 5. 卖出股票
# 卖出股票的过程与买入类似，只需调用 sel方法:
#，要出酸菜
user.sell('600000',amount=2)# sell 2 shares of stock with code '600000


