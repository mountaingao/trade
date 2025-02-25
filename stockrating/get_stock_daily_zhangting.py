import akshare as ak

# 获取涨停数据
date = "20250225"  # 替换为你需要的日期，格式为YYYYMMDD
# 跌停股数据
zt_data = ak.stock_zt_pool_dtgc_em(date=date)
print(zt_data)
zt_count = len(zt_data)
# //炸板股数据
zt_data = ak.stock_zt_pool_zbgc_em(date=date)
print(zt_data)
# //强势股数据
zt_data = ak.stock_zt_pool_strong_em(date=date)
print(zt_data)
# //涨停板股数据
zt_data = ak.stock_zt_pool_em(date=date)
print(zt_data)

# //涨停板股数据
zt_data = ak.stock_zt_pool_previous_em(date=date)
print(zt_data)

# //涨停板股数据
zt_data = ak.stock_zt_pool_sub_new_em(date=date)
print(zt_data)

# 获取跌停数据
spot_data = ak.stock_zh_a_spot_em()
dt_data = spot_data[spot_data["涨跌幅"] == -10]
dt_count = len(dt_data)

# 计算涨跌停总数
total_count = zt_count + dt_count
print(f"{date} 的跌停总数: {total_count}")