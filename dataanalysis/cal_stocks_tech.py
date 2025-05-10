import datetime
import time
import os

from stockrating.tech import get_macd,sma_base,cal_ma_amount,cal_boll



def process_stock_data(input_file, output_file):
    """
    读取包含日期和股票代码的文件，逐行处理数据，并计算技术指标，最后将结果写入 Excel 文件。

    :param input_file: 输入文件路径，包含日期和股票代码
    :param output_file: 输出文件路径，用于保存计算结果
    """
    import pandas as pd
    from stockrating.read_local_info_tdx import get_stock_history_by_local

    # 读取输入文件
    df_input = pd.read_csv(input_file, sep='\t')

    # 创建一个空的 DataFrame 用于存储结果
    results = []
    print(df_input)

    # 逐行处理数据
    for index, row in df_input.iterrows():
        date = str(row['date'])
        code = str(row['code'])  # 将code转换为字符串
        print(row)

        # 获取股票历史数据
        data = get_stock_history_by_local(code)
        # date 转化为时间格式
        date = datetime.datetime.strptime(date, '%Y%m%d')
        # 计算技术指标

        data = sma_base(data, 6.5, 1) #上轨
        data = sma_base(data, 13.5, 1)  #下轨
        data = get_macd(data)
        print(data)

        ma_result = cal_ma_amount(data, date, 'amount')
        print(ma_result)
        boll_result = cal_boll(data, date)

        date_index = data.index.get_loc(date)
        date_data = data.iloc[date_index]
        print(date_data)
        exit;
        #比较date_data['close'] 和 date_data['sma'] ，大于0，则返回1，否则为0
        sma_result_up = 1 if date_data['close'] > date_data['sma-6.5'] else 0
        sma_result_down = 1 if date_data['close'] > date_data['sma-13.5'] else 0


    # 将结果添加到列表中
        results.append({
            'date': date,
            'code': code,
            'close': date_data['close'],
            'amount': date_data['amount'],
            'zhang': ma_result['zhang'],
            'zhen': ma_result['zhen'],
            'sma_up': sma_result_up,
            'sma_down': sma_result_down,
            'macd': date_data['MACD'],
            'boll': boll_result,
            'ma_amount_3_days_ratio': ma_result['3_days_ratio'],
            'ma_amount_5_days_ratio': ma_result['5_days_ratio'],
            'ma_amount_8_days_ratio': ma_result['8_days_ratio'],
            'ma_amount_11_days_ratio': ma_result['11_days_ratio']
        })

        # print(results)
        # exit()
    # 将结果转换为 DataFrame
    df_results = pd.DataFrame(results)

    # 将结果写入 Excel 文件
    df_results.to_excel(output_file, index=False)

# 筛选符合条件的行（每隔1分钟）
if __name__ == '__main__':

    file_path = r"20250501.txt"
    # output_file 为输入文件去除文件扩展名后增加后缀.xlsx
    base_name = os.path.splitext(file_path)[0]  # 去除文件扩展名
    output_file = base_name + '1.xlsx'  # 增加 .xlsx 后缀


    process_stock_data(file_path, output_file)


#分析结论：下面几种情况
# 1、 当 sma-up 小于0时， ma 必须在3倍以上，sma-down 必须大于0
# 2、 当 boll 为1 时，短期3日涨幅 或 10日涨幅大于100%的过滤掉，风险大
# 3、 上市周期大于453天均线，不满足不作为指标条件 BARSCOUNT(c)> 453 and c> ma(453)
# 应该形成一个闭环：通过前一天数据分析得到第二天结果，然后验证是否符合预期，不断循环完成核验
# 日内选股-》评分-》排序展示-》增加到自选股里
# 是否当前热门板块
