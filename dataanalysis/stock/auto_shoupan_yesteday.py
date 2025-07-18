import os
from datetime import datetime, timedelta
import json  # 假设文件是JSON格式，如果不是请替换相应的读取方式
from datetime import datetime, timedelta
from chinese_calendar import is_workday, is_holiday
import pandas as pd
import auto_shoupan
def get_previous_trading_day(date):
    previous_date = date - timedelta(days=1)
    while not is_workday(previous_date) or is_holiday(previous_date):
        previous_date -= timedelta(days=1)
    return previous_date

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

        # 3. 遍历文件夹中的所有文件
        for filename in os.listdir(folder_path):
            print(f"正在处理文件: {filename}")
            print(f"4w: {filename[:4]}")

            # 检查文件名前4位是否匹配当前月日
            if len(filename) >= 4 and filename[:4] == previous_mmdd:
                file_path = os.path.join(folder_path, filename)
                print(f"找到匹配文件: {file_path}")
                filename_without_extension = os.path.splitext(filename)[0]
                # 另一个文件
                today_mmdd = datetime.now().strftime('%m%d')
                tdx_files = f"../data/tdx/{filename_without_extension}{today_mmdd}.xls"
                print("通达信导出文件"+filename_without_extension+today_mmdd)

                # tdx_files = f"../data/tdx/071520250717.xls"
                auto_shoupan.export_tdx_block_data(filename_without_extension+today_mmdd)
                # try:
                # 4. 读取文件内容
                df_pred = pd.read_excel(file_path)
                print(df_pred.head(100))


                print(tdx_files)
                df_today = pd.read_csv(tdx_files, encoding='GBK', sep='\t', header=0)
                # data_02 = pd.read_csv(file2,encoding='GBK',sep='\t',header=1)
                print(df_today)
                # 去掉最后一行
                df_today = df_today[:-1]
                # print(df_today)

                df_today['代码']  = df_today['代码'].str.split('=').str[1].str.replace('"', "")
                # df_today = df_today.sort_values(by=['代码'])
                df_today.sort_values(by='代码', inplace=True)
                df_today.reset_index(drop=True, inplace=True)
                # print(df_today)

            # 5. 获取需要的字段
                # 将df_today 中的字段 涨幅 最高 赋值到 df_pred 中的 次日涨幅	次日最高价 字段
                df_pred['次日涨幅'] = df_today['涨幅%']
                df_pred['次日最高价'] = df_today['最高']
                # print(df_pred)
                # 计算 次日最高涨幅 =  100*（次日最高价-现价）/现价
                df_pred['次日最高涨幅'] = 100 * (df_pred['次日最高价'] - df_pred['现价']) / df_pred['现价']
                df_pred['次日最高涨幅'] = df_pred['次日最高涨幅'].round(2)
                # 给 value 赋值 当 次日最高涨幅 >7 时为 1 ，否则为0
                df_pred['value'] = df_pred['次日最高涨幅'].apply(lambda x: 1 if x > 7 else 0)
                # 是否成功 字段 是当 AI预测 和 value 同时为1时为1，否则为0
                df_pred['是否成功'] = df_pred.apply(lambda x: 1 if x['value'] == 1 and x['AI预测'] == 1 else 0, axis=1)
                #预测成功 字段 是当 预测为是时 同时 value 为 1 时为1，否则为0
                df_pred['预测成功'] = df_pred.apply(lambda x: 1 if x['预测'] == "是" and x['value'] == 1 else 0, axis=1)
                print(df_pred)


                # //df_pred 内容保存回原文件
                file_path = os.path.join(folder_path,filename)

                # 将filename 先备份到 bak 目录中，然后再写回原文件
                os.rename(file_path, os.path.join("../data/bak/predictions/", filename))

                df_pred.to_excel(file_path, index=False)

                # except Exception as e:
                #     print(f"处理文件 {filename} 时出错: {e}")

if __name__ == "__main__":
    process_prediction_files()