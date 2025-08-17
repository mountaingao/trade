import os
from datetime import datetime, timedelta
import json  # 假设文件是JSON格式，如果不是请替换相应的读取方式
from datetime import datetime, timedelta
from chinese_calendar import is_workday, is_holiday
import pandas as pd
import auto_shoupan
import model_xunlian
import time

def get_previous_trading_day(date):
    previous_date = date - timedelta(days=1)
    while not is_workday(previous_date) or is_holiday(previous_date):
        previous_date -= timedelta(days=1)
    return previous_date

'''
模型训练程序
1、每个目录训练一个模型
2、模型命名可以使用目录来区分
3、只有当日期小于今天的数据才能参与训练数据
4、模型保存在../data/models/目录下，可以采用目录进行区分，比如 1000/ 1200/下

'''
def process_prediction_files(base_dir="../data/predictions/"):
    # 1. 获取当前月日（例如：今天是7月8日，则得到 "0708"）
    # 上一个交易日的月日
    md = datetime.now().date()
    print(md)
    previous_mmdd = get_previous_trading_day(md).strftime("%m%d")
    print(previous_mmdd)
    # current_mmdd = datetime.now().strftime("%m%d")
    # previous_mmdd = '0716'
    # 2. 遍历base_dir下的所有文件夹
    for folder_name in os.listdir(base_dir):
        folder_path = os.path.join(base_dir, folder_name)

        # 确保是文件夹
        if not os.path.isdir(folder_path):
            continue

        print(f"正在处理文件夹: {folder_name}")
        # 数据集合
        dfs = []
        # 3. 遍历文件夹中的所有文件，读取文件内容
        for filename in os.listdir(folder_path):
            print(f"正在处理文件: {filename}")
            print(f"日期: {filename[:4]}")
            print(f"now: {previous_mmdd}")

            # 检查文件名前4位是否匹配当前月日
            if len(filename) >= 4 and filename[:4] < previous_mmdd:
                file_path = os.path.join(folder_path, filename)
                print(f"找到匹配文件: {file_path}")

                # try:
                # 4. 读取文件内容，组合数据以后进行训练
                df_pred = pd.read_excel(file_path)
                dfs.append(df_pred)
                # print(df_pred.head(10))
                print(len(df_pred))
                # 打印出dfs 总数
                print(f"dfs 总数: {len(dfs)}")

                # except Exception as e:
                #     print(f"处理文件 {filename} 时出错: {e}")

        # 训练模型
        # 5. 训练模型 数据
        if not dfs:
            print(f"文件夹 {folder_name} 中没有找到匹配日期的数据文件，跳过训练")
            continue
            
        df = pd.concat(dfs, ignore_index=True)
        
        # 检查数据是否为空
        if df.empty:
            print(f"文件夹 {folder_name} 的合并数据为空，跳过训练")
            continue
            
        print(f"文件夹 {folder_name} 的数据形状: {df.shape}")

        # 将模型写入目录中
        try:
            model_xunlian.generate_model_data(df,folder_name)
        except Exception as e:
            print(f"训练文件夹 {folder_name} 的模型时出错: {e}")

def process_prediction_files_from_files(input_files):
    model_xunlian.generate_model_data_from_files(input_files)

if __name__ == "__main__":
    # process_prediction_files()

    # 读取目录文件进行训练，并使用不同的参数来测试和计算结果
    df_other = model_xunlian.get_prediction_files_data()
    print(f'df_other数据量：{len(df_other)}')

    # 去掉重复数据
    df_other = df_other.drop_duplicates(subset=['日期', '代码', '当日涨幅'])

    # 去掉字段 次日涨幅 为空的数据
    df_other = df_other[df_other['次日涨幅'].notnull()]
    print(f'df_other数据量：{len(df_other)}')

    # //去掉字段 Q 为空的数据
    # df_other = df_other[df_other['Q'].notnull()]
    # print(f'df_other数据量：{len(df_other)}')

    # 去掉字段 日期为 08-14 的数据 以及 08-13日数据
    df_other = df_other[df_other['日期'] != '2025-08-13']
    df_other = df_other[df_other['日期'] != '2025-08-14']
    print(f'df_other数据量：{len(df_other)}')

    # 将df数据保存为文件，文件名为日期到秒
    file_root = f'../data/bak/{datetime.now().strftime("%Y%m%d%H%M%S")}'
    df_other.to_excel(file_root+".xlsx", index=False)

    # exit()

    # 提取有效字段 增加量比
    df_1 = df_other[['日期','代码', '当日涨幅', '量比','信号天数', '净额', '净流入', '当日资金流入', '是否领涨', '次日涨幅','次日最高涨幅']]
    # 第二种 增加总金额
    df_2 = df_other[['日期','代码', '当日涨幅', '量比','信号天数', '总金额','净额', '净流入', '当日资金流入', '是否领涨','次日涨幅','次日最高涨幅']]
    # 第三种，增加Q，Q1 Q3
    df_3 = df_other[['日期','代码', '当日涨幅', '量比','信号天数', 'Q','Q_1', 'Q3','总金额','净额', '净流入', '当日资金流入','是否领涨', '次日涨幅','次日最高涨幅']]
    # 第四种，再增加5分钟 band_width
    df_4 = df_other[['日期','代码', '当日涨幅', '量比','信号天数',  'Q','Q_1', 'Q3','band_width','总金额','净额', '净流入', '当日资金流入', '是否领涨','次日涨幅','次日最高涨幅']]


    features = [
        '量比','当日涨幅', '信号天数', '净额', '净流入', '当日资金流入'
    ]
    print(f'df_1数据量：{len(df_1)}')

# 训练
    model = model_xunlian.generate_model_data(df_1, '081601', features)

    # file=  "../data/predictions/1600/08151503_1505.xlsx"
    file=  "../data/predictions/1600/08141531_1533.xlsx"
    # 预测数据
    # 读取指定文件进行预测
    # 预测文件数据  features dataanalysis/data/predictions/1600/08151503_1505.xlsx
    model=     {
        'reg_weights': reg_filename,
        'clf_weights': clf_filename,
        'reg_model': reg_model_path,
        'clf_model': clf_model_path
    }

    model_xunlian.predictions_model_data_file(file,model,"../data/bak/stat/",features)
    time.sleep(1)

    exit()
    features = [
        '量比','总金额','当日涨幅', '信号天数', '净额', '净流入', '当日资金流入'
    ]
    # 训练
    model = model_xunlian.generate_model_data(df_2, '081602', features)

    # 预测数据
    # 读取指定文件进行预测
    # 预测文件数据  features dataanalysis/data/predictions/1600/08151503_1505.xlsx
    model_xunlian.predictions_model_data_file(file,model,"../data/bak/stat/",features)
    time.sleep(1)

    features = [
        '量比','总金额','Q','当日涨幅', '信号天数', '净额', '净流入', '当日资金流入'
    ]
    # 训练
    model = model_xunlian.generate_model_data(df_3, '081603', features)

    # 预测数据
    # 读取指定文件进行预测
    # 预测文件数据  features dataanalysis/data/predictions/1600/08151503_1505.xlsx
    model_xunlian.predictions_model_data_file(file,model,"../data/bak/stat/",features)
    time.sleep(1)

    features = [
        '量比','总金额','Q','band_width','当日涨幅', '信号天数', '净额', '净流入', '当日资金流入'
    ]
    # 训练
    model = model_xunlian.generate_model_data(df_4, '081604', features)

    # 预测数据
    # 读取指定文件进行预测
    # 预测文件数据  features dataanalysis/data/predictions/1600/08151503_1505.xlsx
    model_xunlian.predictions_model_data_file(file,model,"../data/bak/stat/",features)