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


def get_history_accuracy_data(date_suffix='0719'):
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
    # 过滤掉数据中 如果 Q有值，小于2.5 的数据，如果为空保留，如果Q不存在，保留
    # 过滤掉数据中 Q有值且小于2.5 的数据（如果Q列存在）
    if 'Q' in df.columns:
        df = df[df['Q'].isna() | (df['Q'] >= 2.5)]
    else:
        print("警告: 数据中不存在'Q'列，跳过Q值过滤")

    print(f'数据量：{len(df)}')
    print(len(df['代码'].unique()))
    # exit()
    new_data = pd.DataFrame ()
    # 循环读取df数据，根据 代码 字段读取数据，计算Q和boll29
    for code in df['代码'].unique():
        code_df = df[df['代码'] == code]
        # 获取该股票的代码
        data = calculate_q_and_boll(code)
        if data is None:
            print(f"{code} 数据为空")
            continue
        # 得到df中该代码的所有数据
        code_df = df[df['代码'] == code]
        code_df.loc[:, '日期'] = code_df['日期'][:10].astype(str)

        data.loc[:, 'date'] = data['date'][:10].astype(str)

        # 打印调试信息，查看日期格式
        print("code_df 日期格式示例:", code_df['日期'].head())
        print("data 日期格式示例:", data['date'].head())

        # 查找匹配的数据
        matched_data = data[data['date'].isin(code_df['日期'])]
        print("匹配的数据:")
        print(matched_data.tail(10))

        code_value = df[df['代码'] == code]
        # 打印code_value 的出所有的值
        print("code_value 所有值:")
        print(code_value)
        # 打印code_value 的所有列

        # 日期字段有两种数据格式，统一处理
        code_value['日期'] = code_value['日期'].apply(lambda x: str(x)[:10])
        print(code_value)


        # 合并数据
        code_value = code_value.merge(matched_data, left_on='日期', right_on='date', how='left')
        print(code_value)

        # 过滤掉数据中 Q有值，小于2.5 的数据，如果为空保留
        if 'Q' in code_value.columns:
            code_value = code_value[code_value['Q'].isna() | (code_value['Q'] >= 2.5)]
        else:
            print("警告: 数据中不存在'Q'列，跳过Q值过滤")

        exit()

        new_data = pd.concat([new_data, code_value], axis=0)

    # 保存到新的文件中
    # 目录不存在就建立
    if not os.path.exists('./temp/history_data'):
        os.makedirs('./temp/history_data')
    new_data.to_csv(f'./temp/history_data/{date_suffix}.csv', index=False)
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
