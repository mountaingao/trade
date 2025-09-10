import pandas as pd
import numpy as np
import os
from datetime import datetime
import warnings

warnings.filterwarnings('ignore')


#
# 请读取每一条数据，重新计算日线的Q和 boll29 上轨指标，如果符合条件，保存进新的文件中


# 导入原有数据读取函数
from data_prepare import prepare_all_data,get_dir_files_date,get_dir_files,prepare_prediction_data
from tdx_day_data import get_daily_data,get_stock_daily_data

# 计算单个股票的参数
def calculate_q_and_boll(code):
    """
    计算Q和boll29
    """
    data_daily = get_stock_daily_data(code)
    return data_daily


def get_history_accuracy_data(date_suffix='0827'):
    """
    主函数
    """

    # 如果需要实际使用，可以取消下面的注释
    df = prepare_all_data(date_suffix)
    if df is None:
        return

    # 循环读取数据，重新计算日线的Q和 boll29 上轨指标，如果符合条件 Q>2.5 和 boll29 > 1 和 流入>0，保存进新的文件中
    # 过滤掉数据中 当日资金流入<0 的数据
    df = df[df['当日资金流入'] > 0]
    # 过滤掉数据中 如果 Q有值，小于2.5 的数据，如果为空保留
    df = df[df['Q'].notna() | (df['Q'] >= 2.5)]

    print(f'数据量：{len(df)}')
    print(len(df['代码'].unique()))
    # exit()

    # 循环读取df数据，根据 代码 字段读取数据，计算Q和boll29
    for code in df['代码'].unique():
        code_df = df[df['代码'] == code]
        # 获取该股票的代码
        data = calculate_q_and_boll(code)
        # 得到df中该代码的所有数据
        code_df = df[df['代码'] == code]
        print(code, len(code_df))
        print(code_df.head(10))

        print( data.tail())

        
        # 提取data 中 满足 code_df 中日期的数据（多条）
        # 以code_df为基础，获取data中该日期的所有数据组成新的数组，并将结果保存为文件
        for index, row in code_df.iterrows():
            date = row['日期']
            # 获取该股票的收盘价
            # close = row['close']
            # 获取该股票的Q值
            # Q = row['Q']
            # 获取该股票的boll29值
            print(code, date)
            RESULT = data[data['date'] == str(date)]
            print(date, RESULT)
            # 获取该股票的流入值
            # money_flow = data[data['date'] == date]['money_flow'].values[0]
        print(code_df['日期'])
        print(data['date'])
        data = data[data['date'].isin(code_df['日期'])]
        print(len(data))

        # list_data = data[data['date'].isin(code_df['日期'])]
        # print(list_data.head())
        # print(len(list_data  ))

        exit()
        code = code_df['代码'].iloc[0]
        # 获取该股票的日期
        date = code_df['trade_date'].iloc[0]
        # 获取该股票的收盘价
        close = code_df['close'].iloc[0]
        # 获取该股票的Q值
        Q = code_df['Q'].iloc[0]
    # 计算BOLL指标（29日周期）
    df['ma29'] = df.groupby('ts_code')['close'].rolling(window=29, min_periods=1).mean().reset_index(0, drop=True)
    df['std29'] = df.groupby('ts_code')['close'].rolling(window=29, min_periods=1).std().reset_index(0, drop=True)
    df['boll29_upper'] = df['ma29'] + 2 * df['std29']

    # 计算Q指标（资金流入流出指标）
    # 假设Q指标是基于资金流量计算的，这里使用一个简化版本
    # 如果你有具体的Q指标计算方法，请替换下面的计算逻辑
    df['money_flow'] = df['close'] * df['amount']  # 资金流量 = 收盘价 * 成交额
    df['mf_avg'] = df.groupby('ts_code')['money_flow'].rolling(window=20, min_periods=1).mean().reset_index(0, drop=True)
    df['Q'] = df['money_flow'] / df['mf_avg']  # 简化版Q指标

    # 筛选条件: Q>2.5 和 boll29 > 1 和 流入>0
    # 假设"流入>0"是指资金流量为正
    condition = (df['Q'] > 2.5) & (df['boll29_upper'] > 1) & (df['money_flow'] > 0)
    filtered_df = df[condition]

    # 保存符合条件的数据到新文件中
    if not filtered_df.empty:
        output_filename = f'filtered_stocks_{date_suffix}.csv'
        filtered_df.to_csv(output_filename, index=False)
        print(f"已保存符合条件的股票数据到 {output_filename}，共 {len(filtered_df)} 条记录")

        # 按股票代码统计
        stock_counts = filtered_df['ts_code'].value_counts()
        print("各股票符合条件的记录数:")
        print(stock_counts)
    else:
        print("没有符合条件的股票数据")

    return filtered_df


def main():
    get_history_accuracy_data()


if __name__ == "__main__":

    main()
