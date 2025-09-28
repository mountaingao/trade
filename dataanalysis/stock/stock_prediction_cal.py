import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter

# 设置中文字体支持
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

def main():
    # 初始化预测器
    df = pd.read_excel("temp/0801-0923.xlsx")
    
    # 1. 增加两列计算结果
    # 假设原数据中F列对应'收盘价', X列对应'数量', Z列对应'涨幅'
    # ROUND(100000/(F544*100),0)*F544*100*X544/100
    df['计算列1'] = (100000 / (df['收盘价'] * 100)).round() * df['收盘价'] * 100 * df['数量'] / 100
    
    # ROUND(100000/(F544*100),0)*Z544*F544
    df['计算列2'] = (100000 / (df['收盘价'] * 100)).round() * df['涨幅'] * df['收盘价']
    
    # 2. 统计分析新增两列的总和，以及当日涨幅 <19.97 时的总和
    sum_col1 = df['计算列1'].sum()
    sum_col2 = df['计算列2'].sum()
    
    # 涨幅<19.97时的总和
    filtered_df = df[df['涨幅'] < 19.97]
    sum_col1_filtered = filtered_df['计算列1'].sum()
    sum_col2_filtered = filtered_df['计算列2'].sum()
    
    print(f"计算列1总和: {sum_col1}")
    print(f"计算列2总和: {sum_col2}")
    print(f"涨幅<19.97时计算列1总和: {sum_col1_filtered}")
    print(f"涨幅<19.97时计算列2总和: {sum_col2_filtered}")
    
    # 3. 按日期统计次数并打印出来
    date_counts = Counter(df['日期'])
    print("\n按日期统计次数:")
    for date, count in date_counts.items():
        print(f"{date}: {count}次")
    
    # 计算相邻两日之和最大的数据
    df_sorted = df.sort_values('日期')
    max_sum = 0
    max_dates = None
    
    dates = list(date_counts.keys())
    dates.sort()
    
    for i in range(len(dates) - 1):
        current_date = dates[i]
        next_date = dates[i + 1]
        current_sum = date_counts[current_date] + date_counts[next_date]
        if current_sum > max_sum:
            max_sum = current_sum
            max_dates = (current_date, next_date)
    
    if max_dates:
        print(f"\n相邻两日次数之和最大的是: {max_dates[0]} 和 {max_dates[1]}, 总次数为: {max_sum}")
    
    # 4. 所有数值统计并以图表形式展示
    # 绘制计算列的分布图
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # 计算列1分布
    axes[0, 0].hist(df['计算列1'], bins=30, alpha=0.7, color='blue')
    axes[0, 0].set_title('计算列1分布')
    axes[0, 0].set_xlabel('值')
    axes[0, 0].set_ylabel('频次')
    
    # 计算列2分布
    axes[0, 1].hist(df['计算列2'], bins=30, alpha=0.7, color='green')
    axes[0, 1].set_title('计算列2分布')
    axes[0, 1].set_xlabel('值')
    axes[0, 1].set_ylabel('频次')
    
    # 按日期统计的柱状图
    dates_list = list(date_counts.keys())
    counts_list = list(date_counts.values())
    axes[1, 0].bar(range(len(dates_list)), counts_list, color='orange')
    axes[1, 0].set_title('每日数据量统计')
    axes[1, 0].set_xlabel('日期')
    axes[1, 0].set_ylabel('次数')
    axes[1, 0].set_xticks(range(len(dates_list)))
    axes[1, 0].set_xticklabels([str(d) for d in dates_list], rotation=45)
    
    # 涨幅分布
    axes[1, 1].hist(df['涨幅'], bins=30, alpha=0.7, color='red')
    axes[1, 1].set_title('涨幅分布')
    axes[1, 1].set_xlabel('涨幅')
    axes[1, 1].set_ylabel('频次')
    
    plt.tight_layout()
    plt.show()
    
    # 保存处理后的数据
    df.to_excel("temp/0801-0923_processed.xlsx", index=False)
    print(len(df))