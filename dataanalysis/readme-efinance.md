简介
efinance 是一个用于获取股票、基金、期货、债券数据的免费开源 Python 库。从源代码可以看出，其主要基于东方财富网的api获取数据。虽然本身并没有提供数据服务，但其提供的功能已经比较多了，包括同时支持多股票请求，支持日k、周k、月k、60分钟、30分钟、15分钟、5分钟、1分钟k线数据等。
使用pip安装或更新：

pip install efinance
更新

pip install efinance --upgrade
安装后在命令行运行python -c "import efinance as ef” ，不报错则表明安装成功。

常见用法
使用前需要"import efinance as ef” 导入efinance库才能正常使用该库的各种功能

获取股票数据
获取A股历史日 K 线数据。支持同时获取多股票数据，支持日k、周k、月k、60分钟、30分钟、15分钟、5分钟、1分钟k线数据等
ef.stock.get_quote_history(stock_codes=['600519','300750'], beg='20220901', end='20221015', klt=60)
获取港美股的股票 K 线数据（支持输入股票名称以及代码）
ef.stock.get_quote_history('AAPL')
ef.stock.get_quote_history('微软')
ef.stock.get_quote_history('腾讯')
获取 ETF K 线数据，以中概互联网 ETF 为例说明
ef.stock.get_quote_history('513050')
沪深市场 A 股最新状况
ef.stock.get_realtime_quotes()
股票龙虎榜
ef.stock.get_daily_billboard()
沪深 A 股股票季度表现。默认为最新季度，也可以指定季度
ef.stock.get_all_company_performance()
股票历史单子流入数据(日级)
>>> import efinance as ef
>>> ef.stock.get_history_bill('300750')
股票最新一个交易日单子流入数据(分钟级)
>>> import efinance as ef
>>> ef.stock.get_today_bill('300750')
获取期货数据
获取交易所期货基本信息： ef.futures.get_futures_base_info()
获取期货历史行情： ef.futures.get_quote_history('115.ZCM')
获取基金数据
获取基金历史净值信息： ef.fund.get_quote_history('161725')
获取基金公开持仓信息： ef.fund.get_invest_position('161725')
同时获取多只基金基本信息： ef.fund.get_base_info(['161725','005827'])
获取可转债数据
获取可转债整体行情：ef.bond.get_realtime_quotes()
获取全部可转债信息：ef.bond.get_all_base_info()
获取指定可转债 K 线数据：ef.bond.get_quote_history('128053')
结论 & 交流
关注微信公众号：诸葛说talk，获取更多相关内容。同时还能获取邀请加入量化投资研讨群， 与众多从业者、技术大牛一起交流、切磋，名额有限，不要错过。

写文章不易，觉得本文对你有帮助的话，帮忙点赞转发赞赏，让笔者有坚持写好文章的动力。