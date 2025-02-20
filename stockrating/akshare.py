import akshare as ak
import pandas as pd

# 获取A股历史行情数据
df = ak.stock_zh_a_hist(symbol="000001",  # 股票代码（平安银行）
                        period="daily",  # 数据周期（日线）
                        start_date="20240101",  # 开始日期
                        end_date="20240228",  # 结束日期
                        adjust="qfq")  # 复权方式（前复权）

# 打印数据
print(df)


import akshare as ak
import pandas as pd

# 获取A股分钟级行情数据
df = ak.stock_zh_a_minute(symbol="000001",  # 股票代码（平安银行）
                          period="1",  # 分钟周期（1分钟）
                          adjust="qfq")  # 复权方式（前复权）

# 打印数据
print(df)