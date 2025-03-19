import pandas as pd

# 读取文件202503.xlsx，并转换成DataFrame
df = pd.read_excel('202501.xlsx')

# 得到所有的字段
print(df.columns)
# 将字段转换为数类型
# 将相关字段转换为数值类型
# fields = ['1_day_close', '2_day_close', '3_day_close', '4_day_close', '5_day_close',
#           '1_day_max', '2_day_max', '3_day_max', '4_day_max', '5_day_max',
#           '1_day_min', '2_day_min', '3_day_min', '4_day_min', '5_day_min']
fields = df.columns
for field in fields:
    df[field] = pd.to_numeric(df[field], errors='coerce')
print(df.count())
# 打印DataFrame的前5行
# print(df.head())
# 统计各个字段大于0的延续性
# 1、成交额大于1亿的继续下一步统计
df = df[df['0_amount'] <= 1e9]

print(df.head())
print(df.count())

# 需要统计当 1_day_close 大于0时，2_day_close 大于1_day_close的比例是多少,3_day_close 大于2_day_close的比例是多少,？以此类推，5_day_close 大于4_day_close的比例是多少？

def cal_day_ratio(df, fields):
    for field in fields:
        if field.endswith('_close'):
            # base_field = field
            for i in range(0, 5):
                base_field = f"{i}_day_close"
                next_field = f"{i+1}_day_close"

                if next_field in df.columns:
                    # 修改为统计 next_field 大于 base_field 的比例
                    ratio = df[df[base_field] > 0][next_field].gt(df[base_field]).mean() * 100
                    print(f"当{base_field}大于0时，{next_field}大于{base_field}的比例为: {ratio:.2f}%")
    return ratio


cal_day_ratio(df, fields)
exit()

#
# 当0_day_close大于0时，1_day_close大于0_day_close的比例为: 23.53%
# 当1_day_close大于0时，2_day_close大于1_day_close的比例为: 21.85%
# 当2_day_close大于0时，3_day_close大于2_day_close的比例为: 22.69%
# 当3_day_close大于0时，4_day_close大于3_day_close的比例为: 35.29%
# 当4_day_close大于0时，5_day_close大于4_day_close的比例为: 20.17%

# 10亿以下
# 当0_day_close大于0时，1_day_close大于0_day_close的比例为: 18.68%
# 当1_day_close大于0时，2_day_close大于1_day_close的比例为: 17.58%
# 当2_day_close大于0时，3_day_close大于2_day_close的比例为: 12.09%
# 当3_day_close大于0时，4_day_close大于3_day_close的比例为: 14.29%
# 当4_day_close大于0时，5_day_close大于4_day_close的比例为: 17.58%

for field in fields:
    if field.endswith('_close'):
        base_field = field
        for i in range(0, 5):
            next_field = f"{i+1}_day_close"
            if next_field in df.columns:
                # 修改为统计 next_field 大于 base_field 的比例
                ratio = df[df[base_field] > 0][next_field].gt(df[base_field]).mean() * 100
                print(f"当{base_field}大于0时，{next_field}大于{base_field}的比例为: {ratio:.2f}%")
    elif field.endswith('_max'):
        base_field = field
        for i in range(1, 5):
            next_field = f"{i+1}_day_max"
            if next_field in df.columns:
                # 修改为统计 next_field 大于 base_field 的比例
                ratio = df[df[base_field] > 0][next_field].gt(df[base_field]).mean() * 100
                print(f"当{base_field}大于0时，{next_field}大于{base_field}的比例为: {ratio:.2f}%")
    elif field.endswith('_min'):
        base_field = field
        for i in range(1, 5):
            next_field = f"{i+1}_day_min"
            if next_field in df.columns:
                # 修改为统计 next_field 大于 base_field 的比例
                ratio = df[df[base_field] > 0][next_field].gt(df[base_field]).mean() * 100
                print(f"当{base_field}大于0时，{next_field}大于{base_field}的比例为: {ratio:.2f}%")
    elif field.endswith('_open'):
        base_field = field
        for i in range(1, 5):
            next_field = f"{i+1}_day_min"
            if next_field in df.columns:
                # 修改为统计 next_field 大于 base_field 的比例
                ratio = df[df[base_field] > 0][next_field].gt(df[base_field]).mean() * 100
                print(f"当{base_field}大于0时，{next_field}大于{base_field}的比例为: {ratio:.2f}%")