
import akshare as ak
import pandas as pd
import numpy as np
from dtaidistance import dtw
from tqdm import tqdm
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
# 解决中文乱码问题
plt.rcParams['font.sans-serif'] = ['STHeiti']  # 苹果系统字体
#plt.rcParams['font.sans-serif'] = ['yahei']
plt.rcParams['axes.unicode_minus'] = False
# --------------------------配置参数--------------------------
TARGET_SYMBOL = "600519"  # 目标股票代码（贵州茅台）
LOOKBACK_DAYS = 30  # 形态分析周期（单位：交易日）
PREDICT_DAYS = 5  # 预测未来天数
SIMILAR_NUM = 5  # 展示相似股票数量
START_DATE = "20250102"  # 数据起始日期（确保足够长的时间范围）
# --------------------------数据获取函数-----------------------
def get_stock_data(symbol):
    """获取前复权日线数据"""
    try:
        df = ak.stock_zh_a_hist(
            symbol=symbol,
            period="daily",
            start_date=START_DATE,
            adjust="hfq"  # 前复权处理
        )
        df['日期'] = pd.to_datetime(df['日期'])
        df = df.set_index('日期').sort_index()
        return df
    except Exception as e:
        print(f"获取{symbol}数据失败: {str(e)}")
        return None
def main():
    # 获取目标股票数据
    target_df = get_stock_data(TARGET_SYMBOL)
    if target_df is None:
        print(f"无法获取目标股票{TARGET_SYMBOL}数据")
        return
    # 提取最近的形态变化
    target_prices = target_df['收盘'].values[-LOOKBACK_DAYS:]
    scaler = MinMaxScaler()
    target_scaled = scaler.fit_transform(target_prices.reshape(-1, 1)).flatten()
    # 获取全市场股票列表
    all_stocks = ak.stock_info_a_code_name().code.tolist()
    print(f"开始分析{len(all_stocks)}支股票...")
    # 遍历所有股票查找相似形态
    similarities = []
    progress_bar = tqdm(all_stocks[:10], desc="Processing")  # 测试时限制前10支
    for symbol in progress_bar:
        try:
            df = get_stock_data(symbol)
            if df is None or len(df) < (LOOKBACK_DAYS + PREDICT_DAYS + 30):
                continue  # 确保足够历史数据
            # 滑动窗口遍历历史形态
            for i in range(len(df) - LOOKBACK_DAYS - PREDICT_DAYS):
                # 当前窗口数据
                window_prices = df['收盘'].values[i:i + LOOKBACK_DAYS]
                window_scaled = scaler.fit_transform(window_prices.reshape(-1, 1)).flatten()
                # 计算形态相似度
                dtw_distance = dtw.distance_fast(target_scaled, window_scaled)
                # 计算后续收益
                current_price = df['收盘'].values[i + LOOKBACK_DAYS - 1]  # 窗口最后一天收盘价
                future_price = df['收盘'].values[i + LOOKBACK_DAYS + PREDICT_DAYS - 1]  # 结束后第N天
                future_return = future_price / current_price - 1
                similarities.append({
                    'symbol': symbol,
                    'distance': dtw_distance,
                    'return': future_return,
                    'start_date': df.index[i],  # 形态开始日期
                    'end_date': df.index[i + LOOKBACK_DAYS - 1]  # 形态结束日期
                })
        except:
            continue
    # --------------------------结果分析--------------------------
    similar_df = pd.DataFrame(similarities)
    if similar_df.empty:
        print("未找到相似形态")
        return
    # 筛选前10%相似度案例
    top_samples = similar_df.nsmallest(int(len(similar_df) * 0.1), 'distance')
    print("\n=============== 分析结果 ===============")
    print(f"发现{len(top_samples)}个相似形态案例")
    print(f"未来{PREDICT_DAYS}日平均收益：{top_samples['return'].mean():.2%}")
    print(f"最大收益：{top_samples['return'].max():.2%}")
    print(f"最小收益：{top_samples['return'].min():.2%}")
    print("\n最具代表性股票：")
    print(top_samples.groupby('symbol').agg({
        'distance': 'mean',
        'return': 'mean'
    }).nsmallest(SIMILAR_NUM, 'distance'))
    # --------------------------结果可视化--------------------------
    best_case = top_samples.iloc[0]
    case_df = get_stock_data(best_case['symbol'])
    plt.figure(figsize=(12, 6))
    # 归一化显示
    plt.plot(np.linspace(0, 1, LOOKBACK_DAYS),
             scaler.fit_transform(target_prices.reshape(-1, 1)),
             label=f'目标形态 {TARGET_SYMBOL}', linewidth=2)
    best_window = case_df['收盘'].loc[best_case['start_date']:best_case['end_date']].values
    plt.plot(np.linspace(0, 1, LOOKBACK_DAYS),
             scaler.fit_transform(best_window.reshape(-1, 1)),
             '--',
             label=f'最佳匹配 {best_case["symbol"]} (收益:{best_case["return"]:.2%})')
    plt.title('股价形态匹配可视化')
    plt.xlabel('时间序列（归一化）')
    plt.ylabel('归一化价格')
    plt.legend()
    plt.grid(True)
    plt.show()
if __name__ == "__main__":
    main()