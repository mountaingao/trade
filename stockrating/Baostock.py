import baostock as bs
import pandas as pd

# 登录Baostock
lg = bs.login()
# 显示登录返回信息
print('登录返回信息：', lg.error_msg)

# 设置查询字段
fields = "date,code,open,high,low,close,preclose,volume,amount,turn,pctChg"

# 查询历史K线数据
rs = bs.query_history_k_data_plus(
    code="sh.000001",  # 上证指数
    fields=fields,
    start_date="2024-01-01",
    end_date="2024-02-28",
    frequency="d",  # 日K线
    adjustflag="3"  # 后复权
)

# 将查询结果转换为DataFrame
data = []
while (rs.error_code == '0') & rs.next():
    data.append(rs.get_row_data())
bs.logout()  # 登出

# 转换为DataFrame并打印
df = pd.DataFrame(data, columns=rs.fields)
print(df)