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

    calculate_data_accuracy_2( df)

    # 过滤掉数据中 Q有值且小于2.5 的数据（如果Q列存在）
    if 'Q' in df.columns:
        df = df[(df['Q'] >= 2.5)]
    else:
        print("警告: 数据中不存在'Q'列，跳过Q值过滤")

    print(f'Q 数据量：{len(df)}')

    # 计算这时 次日最高涨幅 大于1的数量
    calculate_data_accuracy_2( df)

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


def get_existing_accuracy_data_2(hour='1600',date_suffix='0911'):

    # 指定目录数据
    df = prepare_prediction_dir_data(hour,"0717",date_suffix)
    if df is None:
        return

    print(f'数据量：{len(df)}')

    # 过滤掉数据中 次日涨幅为空的数据
    df = df[df['次日涨幅'].notna()]

    calculate_data_accuracy_2( df)

    # 按日期、细分行业统计数量，筛选出数量大于2的组合
    df_grouped = df.groupby(['日期', '细分行业']).size().reset_index(name='count')
    df_filtered_groups = df_grouped[df_grouped['count'] > 2]
    print("按日期、细分行业分组数量大于2的组合:")
    print(df_filtered_groups)

    # 从原始数据中筛选出符合要求的记录（属于数量大于2的日期-行业组合）
    df_filtered = df.merge(df_filtered_groups[['日期', '细分行业']], on=['日期', '细分行业'])
    print(f"筛选后的数据量: {len(df_filtered)}")
    print(df_filtered.tail(10))

    # 挑选出按 日期 细分行业 中 Q值最大的哪条数据
    # 处理可能存在的NaN值问题
    df_filtered = df_filtered.dropna(subset=['Q'])
    if not df_filtered.empty:
        df = df_filtered.loc[df_filtered.groupby(['日期', '细分行业'])['Q'].idxmax()]
    else:
        df = df_filtered
    calculate_data_accuracy(df)

    # 过滤掉数据中 Q有值且小于2.5 的数据（如果Q列存在）
    if 'Q' in df.columns:
        df = df[(df['Q'] >= 2.5)]
    else:
        print("警告: 数据中不存在'Q'列，跳过Q值过滤")

    print(f'Q 数据量：{len(df)}')

    # 计算这时 次日最高涨幅 大于1的数量
    calculate_data_accuracy(df)

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

    # print(df_copy.tail(10))
    # 按日期统计每日数量
    df_copy_count = df_copy.groupby('日期').agg({col: 'count' for col in numeric_columns})
    print(df_copy_count)
    # # 按日期统计 细分行业 数量，并保留大于2的细分行业
    # df_copy_industry = df_copy.groupby('日期').agg({'细分行业': lambda x: x.value_counts().index[0]})
    # print(df_copy_industry)
    # # 从原始数据df 中，挑选出满足条件的数据：按照日期、细分行业 筛选大于2的所有数据
    # df_copy_industry = df_copy_industry[df_copy_industry['日期'].map(df_copy_industry.groupby('日期').size()) > 2]
    # print(df_copy_industry)
    #
    # df_copy_industry_count = df_copy.groupby(['日期', '细分行业']).agg({col: 'count' for col in numeric_columns})
    # print(df_copy_industry_count)
    # 保留按日期、细分行业

    # # 保留数量大于3的细分行业，并挑选其中Q值最大的一个保留
    # df_copy_industry_count = df_copy_industry_count.groupby('日期').apply(lambda x: x.nlargest(3, 'Q')).reset_index(drop=True)
    #

    df_copy_sum = df_copy.groupby('日期').agg({col: 'sum' for col in numeric_columns})
    print(df_copy_sum)
    # 打印出日期、次日涨幅、次日最高涨幅
    # 修改：重置索引以将日期作为列访问
    df_copy_sum_reset = df_copy_sum.reset_index()
    # print(df_copy_sum_reset[['日期', '次日涨幅', '次日最高涨幅']])
    print('次日涨幅：',df_copy_sum['次日涨幅'].sum(), df_copy_sum['次日涨幅'].mean())
    print('次日最高涨幅:',df_copy_sum['次日最高涨幅'].sum(), df_copy_sum['次日最高涨幅'].mean())

    # exit()
    # 和次日最高涨幅
    # df_copy['date'] = pd.to_datetime(df_copy['date'])
    # df_copy = df_copy.sort_values('date')
    # df_copy = df_copy.groupby('date').agg({col: 'sum' for col in numeric_columns})
    # df_copy = df_copy.reset_index()
    # df_copy = df_copy.dropna()


    for col in numeric_columns:
        if col in df_copy.columns:
            # 尝试将列转换为数值类型，无法转换的设置为NaN
            df_copy[col] = pd.to_numeric(df_copy[col], errors='coerce')

    # 计算 次日最高涨幅 之和 和 次日涨幅 之和
    sum_次日最高涨幅 = df_copy['次日最高涨幅'].sum()
    print('次日最高涨幅和：', sum_次日最高涨幅,
          '平均涨幅：', sum_次日最高涨幅/len(df_copy) if len(df_copy) > 0 else 0 )

    sum_次日涨幅 = df_copy['次日涨幅'].sum()
    print('次日涨幅和：', sum_次日涨幅,
          '平均涨幅：', sum_次日涨幅/len(df_copy) if len(df_copy) > 0 else 0)

    # 计算 次日涨幅 和当日涨幅《19.97的之和
    sum_次日涨幅_1997 = df_copy[df_copy['当日涨幅'] < 19.97]['次日涨幅'].sum()

    print('次日涨幅和且当日涨幅《19.97：', sum_次日涨幅_1997,
          '比例：', sum_次日涨幅_1997/len(df_copy) if len(df_copy) > 0 else 0)

    # 计算 次日最高涨幅 和当日涨幅《19.97的之和
    sum_次日最高涨幅_1997 = df_copy[df_copy['当日涨幅'] < 19.97]['次日最高涨幅'].sum()

    print('次日最高涨幅和且当日涨幅《19.97：', sum_次日最高涨幅_1997,
          '比例：', sum_次日最高涨幅_1997/len(df_copy) if len(df_copy) > 0 else 0)



    print('次日最高涨幅大于1的数量：', len(df_copy[df_copy['次日最高涨幅'] > 1]),
          '比例：', 100*len(df_copy[df_copy['次日最高涨幅'] > 1])/len(df_copy) if len(df_copy) > 0 else 0)

    # 计算 次日涨幅 大于1的数量
    print('次日涨幅大于1的数量：', len(df_copy[df_copy['次日涨幅'] > 1]),
          '比例：', 100*len(df_copy[df_copy['次日涨幅'] > 1])/len(df_copy) if len(df_copy) > 0 else 0)

    # 计算 次日涨幅 大于5的数量
    print('次日涨幅大于5的数量：', len(df_copy[df_copy['次日涨幅'] > 5]),
          '比例：', 100*len(df_copy[df_copy['次日涨幅'] > 5])/len(df_copy) if len(df_copy) > 0 else 0)

    # 计算 次日最高涨幅 大于5的数量
    print('次日最高涨幅大于5的数量：', len(df_copy[df_copy['次日最高涨幅'] > 5]),
          '比例：', 100*len(df_copy[df_copy['次日最高涨幅'] > 5])/len(df_copy) if len(df_copy) > 0 else 0)

def calculate_data_accuracy_2(df):
    # 确保相关列是数值类型，如果不是则进行转换
    numeric_columns = ['次日最高涨幅', '次日涨幅']
    df_copy = df.copy()

    # print(df_copy.tail(10))
    # 按日期统计每日数量
    df_copy_count = df_copy.groupby('日期').agg({col: 'count' for col in numeric_columns})
    print(df_copy_count)
    
    # 按日期、细分行业统计数量，筛选出数量大于2的组合
    df_grouped = df_copy.groupby(['日期', '细分行业']).size().reset_index(name='count')
    df_filtered_groups = df_grouped[df_grouped['count'] > 2]
    print("按日期、细分行业分组数量大于2的组合:")
    print(df_filtered_groups)
    
    # 从原始数据中筛选出符合要求的记录（属于数量大于2的日期-行业组合）
    df_filtered = df_copy.merge(df_filtered_groups[['日期', '细分行业']], on=['日期', '细分行业'])
    print(f"筛选后的数据量: {len(df_filtered)}")
    
    # 打印筛选后的统计数据
    df_filtered_sum = df_filtered.groupby('日期').agg({col: 'sum' for col in numeric_columns})
    print("筛选后按日期汇总:")
    print(df_filtered_sum)
    
    # 打印整体统计信息
    print('筛选后次日涨幅：', df_filtered_sum['次日涨幅'].sum(), df_filtered_sum['次日涨幅'].mean())
    print('筛选后次日最高涨幅:', df_filtered_sum['次日最高涨幅'].sum(), df_filtered_sum['次日最高涨幅'].mean())

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

    get_existing_accuracy_data_2('1000','0912')
    # get_existing_accuracy_data('1200','0912')
    # get_existing_accuracy_data('1400','0912')
    # get_existing_accuracy_data('1600','0912')




if __name__ == "__main__":

    main()
