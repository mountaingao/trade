# 假设有一个包含每日成交量的列表 volume_data
volume_data = [100, 150, 200, 250, 300, 350, 400, 450, 500, 550, 600, 650, 700]

# 计算最近3日、5日、8日、13日的成交量之和
recent_3_days = sum(volume_data[-3:])
recent_5_days = sum(volume_data[-5:])
recent_8_days = sum(volume_data[-8:])
recent_13_days = sum(volume_data[-13:])

# 计算前3日、5日、8日、13日的成交量之和
previous_3_days = sum(volume_data[-6:-3])
previous_5_days = sum(volume_data[-10:-5])
previous_8_days = sum(volume_data[-16:-8])
previous_13_days = sum(volume_data[-26:-13])

# 计算放量比例
volume_ratio_3_days = recent_3_days / previous_3_days
volume_ratio_5_days = recent_5_days / previous_5_days
volume_ratio_8_days = recent_8_days / previous_8_days
volume_ratio_13_days = recent_13_days / previous_13_days

# 输出结果
print(f"最近3日放量比例: {volume_ratio_3_days}")
print(f"最近5日放量比例: {volume_ratio_5_days}")
print(f"最近8日放量比例: {volume_ratio_8_days}")
print(f"最近13日放量比例: {volume_ratio_13_days}")
