import pandas as pd

# 读取文件202503.xlsx，并转换成DataFrame
df = pd.read_excel('202503.xlsx')

# 将相关字段转换为数值类型
fields = ['1_day_close', '2_day_close', '3_day_close', '4_day_close', '5_day_close',
          '1_day_max', '2_day_max', '3_day_max', '4_day_max', '5_day_max',
          '1_day_min', '2_day_min', '3_day_min', '4_day_min', '5_day_min']

for field in fields:
    df[field] = pd.to_numeric(df[field], errors='coerce')

# 打印DataFrame的前5行
print(df.head())
# 统计各个字段大于0的延续性
fields = ['1_day_close', '2_day_close', '3_day_close', '4_day_close', '5_day_close',
          '1_day_max', '2_day_max', '3_day_max', '4_day_max', '5_day_max',
          '1_day_min', '2_day_min', '3_day_min', '4_day_min', '5_day_min']

for field in fields:
    if field.endswith('_close'):
        base_field = field
        for i in range(1, 5):
            next_field = f"{i+1}_day_close"
            if next_field in df.columns:
                ratio = df[df[base_field] > 0][next_field].gt(0).mean() * 100
                print(f"当{base_field}大于0时，{next_field}大于0的比例为: {ratio:.2f}%")
    elif field.endswith('_max'):
        base_field = field
        for i in range(1, 5):
            next_field = f"{i+1}_day_max"
            if next_field in df.columns:
                ratio = df[df[base_field] > 0][next_field].gt(0).mean() * 100
                print(f"当{base_field}大于0时，{next_field}大于0的比例为: {ratio:.2f}%")
    elif field.endswith('_min'):
        base_field = field
        for i in range(1, 5):
            next_field = f"{i+1}_day_min"
            if next_field in df.columns:
                ratio = df[df[base_field] > 0][next_field].gt(0).mean() * 100
                print(f"当{base_field}大于0时，{next_field}大于0的比例为: {ratio:.2f}%")