import pandas as pd
import numpy as np
from typing import Dict, List, Optional
import warnings
import os
from datetime import datetime, timedelta
import model_xunlian
warnings.filterwarnings('ignore')



def get_data_from_files(input_files):
    """
根据输入文件列表生成模型数据并保存特征权重

参数:
    input_files (list): 输入Excel文件路径列表
"""
    # 读取并合并所有文件
    dfs = []
    for file in input_files:
        df_part = pd.read_excel(file)
        dfs.append(df_part)
    df = pd.concat(dfs, ignore_index=True)
    return df

def get_prediction_files_data(base_dir="../data/predictions/",start_mmddend = None,end_mmdd=None):
    # 1. 获取当前月日（例如：今天是7月8日，则得到 "0708"），可以指定开始和结束日期
    # 读取指定目录下所有子目录的文件的数据
    # 上一个交易日的月日
    if end_mmdd is not None:
        end_md = end_mmdd
    else:
        end_md = datetime.now().strftime("%m%d")

    if start_mmddend is not None:
        start_md = start_mmddend
    else:
        start_md = '0630'
    # md = datetime.now().date()
    # previous_mmdd = md.strftime("%m%d")
    previous_mmdd = end_md

    # 数据集合
    dfs = []
    # 2. 遍历base_dir下的所有文件夹
    for folder_name in os.listdir(base_dir):
        folder_path = os.path.join(base_dir, folder_name)

        # 确保是文件夹
        if not os.path.isdir(folder_path):
            continue

        print(f"正在处理文件夹: {folder_name}")

        # 3. 遍历文件夹中的所有文件，读取文件内容
        for filename in os.listdir(folder_path):
            # print(f"正在处理文件: {filename}")

            # 检查文件名前4位是否匹配当前月日
            if len(filename) >= 4 and filename[:4] >= start_md and filename[:4] < previous_mmdd:
                file_path = os.path.join(folder_path, filename)
                print(f"找到匹配文件: {file_path}")

                # try:
                # 4. 读取文件内容，组合数据以后进行训练
                df_pred = pd.read_excel(file_path)
                dfs.append(df_pred)
                # print(df_pred.head(10))
                # print(len(df_pred))
                # 打印出dfs 总数
                # print(f"dfs 总数: {len(dfs)}")

                # except Exception as e:
                #     print(f"处理文件 {filename} 时出错: {e}")
        # 训练模型
        # 5. 训练模型 数据
        if not dfs:
            print(f"文件夹 {folder_name} 中没有找到匹配日期的数据文件，跳过训练")
            continue

    if not dfs:
        print(f"文件夹 {folder_name} 中没有找到匹配日期的数据文件，跳过训练")
        return None

    df = pd.concat(dfs, ignore_index=True)
    # print(len(df))
    # 检查数据是否为空
    if df.empty:
        return None

    return df

def get_dir_files_date(dir_path: str,start_md: str,end_mmdd: str):
    """读取指定目录下的所有文件名的日期"""
    dates = []
    if end_mmdd is not None:
        end_mmdd = end_mmdd
    else:
        end_mmdd = datetime.now().strftime("%m%d")
    # 3. 遍历文件夹中的所有文件，读取文件内容
    for filename in os.listdir(dir_path):
        if len(filename) >= 4 and filename[:4] >= start_md and filename[:4] < end_mmdd:
            file_path = os.path.join(dir_path, filename)
            # print(f"读取文件: {file_path}")
            # 获取文件名中的日期
            date = filename[:4]
            dates.append(date)

    return dates

def get_dir_files(dir_path):
    """读取指定目录下的所有文件名的日期"""
    files = []
    # 3. 遍历文件夹中的所有文件，读取文件内容
    for filename in os.listdir(dir_path):
        file_path = os.path.join(dir_path, filename)
        print(f"读取文件: {file_path}")
        files.append(file_path)
    return files

def get_dir_files_data(dir_path: str,start_md: str,end_mmdd: str):
    """读取指定目录下的所有文件数据"""
    dfs = []
    # 3. 遍历文件夹中的所有文件，读取文件内容，只读取 xls或xlsx文件

    for filename in os.listdir(dir_path):
        # print(f"正在处理文件: {dir_path}")
        # 检查文件名是否以 .xls 或 .xlsx 结尾
        if filename.endswith('.xls') or filename.endswith('.xlsx'):
            # 检查文件名前4位是否匹配当前月日
            if len(filename) >= 4 and filename[:4] >= start_md and filename[:4] < end_mmdd:
                file_path = os.path.join(dir_path, filename)
                print(f"读取文件: {file_path}")

                # try:
                # 4. 读取文件内容，组合数据以后进行训练
                # 如果是xlsx文件，使用read_excel 读取
                # 如果是xls文件，使用read_csv 读取，并检查字符集
                if filename.endswith('.xlsx'):
                    df_pred = pd.read_excel(file_path)
                else:
                    df_pred = pd.read_csv(file_path,encoding='gbk')

                dfs.append(df_pred)
                # print(df_pred.head(10))
                # print(len(df_pred))
                # 打印出dfs 总数
                # print(f"dfs 总数: {len(dfs)}")

                # except Exception as e:
                #     print(f"处理文件 {filename} 时出错: {e}")
    if not dfs:
        print(f"文件夹 {dir_path} 中没有找到匹配日期的数据文件，跳过训练")

        return None
    df = pd.concat(dfs, ignore_index=True)
    return df


def prepare_prediction_data(start_md, end_md):
    """
    prediction 数据集
    """
    df = get_prediction_files_data("../data/predictions/",start_md,end_md)
    print(f'prediction 数据量：{len(df)}')

    return df

def prepare_all_data(end_mmdd: str):
    """
    准备数据集，包括历史数据和预测数据
    """
    # 检查临时文件是否存在,如果有end_mmdd 则使用end_mmdd 作为文件名最后部分，否则为all
    if end_mmdd is not None:
        temp_file_path = "cache/predictions_data_"+end_mmdd+".xlsx"
    else:
        temp_file_path = "cache/predictions_data_all.xlsx"

    if os.path.exists(temp_file_path):
        print("检测到临时文件，直接读取...")
        df = pd.read_excel(temp_file_path)
        print(f'历史数据量：{len(df)}')
        return df

    df = get_alert_files_data()
    print(len(df))

    prediction_df = get_prediction_files_data("../data/predictions/","0717",end_mmdd)
    # print(len(prediction_df))

    if prediction_df is not None and not prediction_df.empty:
        print(f'预测数据量：{len(prediction_df)}')
        # 合并数据
        df = pd.concat([df, prediction_df], ignore_index=True)

    print(f'总数据量：{len(df)}')
    # 将df写入临时文件，供下次使用
    df.to_excel(temp_file_path, index=False)
    return df

def prepare_prediction_dir_data(predir_path: "1000",start_md: str, end_md: str):
    """
    prediction 数据集
    """
    df = get_dir_files_data("../data/predictions/"+predir_path,start_md,end_md)
    print(f'prediction 数据量：{len(df)}')

    return df

def prepare_ths_data(start_md: str, end_md: str):
    """
    prediction 数据集
    """
    df = get_dir_files_data("../data/ths/",start_md,end_md)
    print(f'prediction 数据量：{len(df)}')

    return df

def get_alert_files_data(base_dir="../alert/",start_mmddend = None,end_mmdd=None):
    files= [
        "../alert/0630.xlsx",
        "../alert/0701.xlsx",
        "../alert/0702.xlsx",
        "../alert/0703.xlsx",
        "../alert/0704.xlsx",
        "../alert/0707.xlsx",
        "../alert/0708.xlsx",
        "../alert/0709.xlsx",
        "../alert/0710.xlsx",
        "../alert/0711.xlsx",
        "../alert/0714.xlsx",
        "../alert/0715.xlsx",
        "../alert/0716.xlsx",
    ]
    df = get_data_from_files(files)
    return df


# 使用示例
def main():
    """使用示例"""
    # 所有的数 据
    prepare_all_data(end_mmdd="0717")

    # prediction 某一段时间内的数据集
    prepare_prediction_data("0717","0720")

    # 得到某个时间段内的prediction 数据
    prepare_prediction_dir_data("1000","0717","0720")


if __name__ == "__main__":
    main()