import pandas as pd
import numpy as np
import os
from datetime import datetime
import warnings

warnings.filterwarnings('ignore')


#
# 请读取每一条数据，重新计算日线的Q和 boll29 上轨指标，如果符合条件，保存进新的文件中


# 导入原有数据读取函数
from data_prepare import prepare_all_data,prepare_prediction_dir_data,get_dir_files,prepare_prediction_data
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

def get_existing_accuracy_data(hour='1600',date_suffix='0911'):
    """
    主函数
    """

    # 如果需要实际使用，可以取消下面的注释
    # df = prepare_all_data(date_suffix)

    # 指定目录数据
    df = prepare_prediction_dir_data(hour,"0717",date_suffix)
    if df is None:
        return

    print(f'数据量：{len(df)}')

    # 过滤掉数据中 次日涨幅为空的数据
    df = df[df['次日涨幅'].notna()]

    calculate_data_accuracy( df)

    # 过滤掉数据中 Q有值且小于2.5 的数据（如果Q列存在）
    if 'Q' in df.columns:
        df = df[(df['Q'] >= 2.5)]
    else:
        print("警告: 数据中不存在'Q'列，跳过Q值过滤")

    print(f'Q 数据量：{len(df)}')

    # 计算这时 次日最高涨幅 大于1的数量
    calculate_data_accuracy( df)

    # 针对df中的 信号天数字段，保留1-5的数字，大于5的数字只保留奇数 的行数
    # 修正：确保返回布尔值，添加异常处理
    def filter_signal_days(x):
        try:
            first_digit = int(str(x)[0])
            if first_digit <= 5:
                return True  # 保留1-5的数字
            else:
                return first_digit % 2 == 1  # 大于5的数字只保留奇数
        except (ValueError, IndexError):
            return False  # 如果转换失败，不保留该行

    # 使用布尔索引过滤数据
    df = df[df['信号天数'].apply(filter_signal_days)]

    print(f'信号天数 数据量：{len(df)}')
    calculate_data_accuracy( df)

    # df = df[df['信号天数'].apply(filter_signal_days)]
    df = df[(df['当日资金流入'] >= 0)]
    print(f'当日资金流入 数据量：{len(df)}')
    calculate_data_accuracy( df)

    # 将结果写入到临时文件excel中
    df.to_excel(f'./temp/{hour}_{date_suffix}_filter.xlsx', index=False)

def calculate_data_accuracy(df):
    # 确保相关列是数值类型，如果不是则进行转换
    numeric_columns = ['次日最高涨幅', '次日涨幅']
    df_copy = df.copy()

    for col in numeric_columns:
        if col in df_copy.columns:
            # 尝试将列转换为数值类型，无法转换的设置为NaN
            df_copy[col] = pd.to_numeric(df_copy[col], errors='coerce')

    print('次日最高涨幅大于1的数量：', len(df_copy[df_copy['次日最高涨幅'] > 1]),
          '比例：', len(df_copy[df_copy['次日最高涨幅'] > 1])/len(df_copy) if len(df_copy) > 0 else 0)

    # 计算 次日涨幅 大于1的数量
    print('次日涨幅大于1的数量：', len(df_copy[df_copy['次日涨幅'] > 1]),
          '比例：', len(df_copy[df_copy['次日涨幅'] > 1])/len(df_copy) if len(df_copy) > 0 else 0)

    # 计算 次日涨幅 大于5的数量
    print('次日涨幅大于5的数量：', len(df_copy[df_copy['次日涨幅'] > 5]),
          '比例：', len(df_copy[df_copy['次日涨幅'] > 5])/len(df_copy) if len(df_copy) > 0 else 0)

    # 计算 次日最高涨幅 大于5的数量
    print('次日最高涨幅大于5的数量：', len(df_copy[df_copy['次日最高涨幅'] > 5]),
          '比例：', len(df_copy[df_copy['次日最高涨幅'] > 5])/len(df_copy) if len(df_copy) > 0 else 0)




from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

def random_forest_analysis(df):

    # 准备数据
    X = df.drop('结果字段', axis=1)
    y = df['结果字段']

    # 训练随机森林模型
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)

    # 获取特征重要性
    feature_importance = pd.DataFrame({
        'feature': X.columns,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)

    print("特征重要性排序:")
    print(feature_importance)
def main():
    # 历史数据分析
    # get_history_accuracy_data()

    # 现有数据分析
    # get_existing_accuracy_data("0912")

    get_existing_accuracy_data('1000','0912')
    get_existing_accuracy_data('1200','0912')
    get_existing_accuracy_data('1400','0912')
    get_existing_accuracy_data('1600','0912')




if __name__ == "__main__":

    main()
