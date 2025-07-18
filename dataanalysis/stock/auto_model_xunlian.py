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

        # 3. 遍历文件夹中的所有文件，读取文件内容
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
                # 4. 读取文件内容，组合数据以后进行训练
                df_pred = pd.read_excel(file_path)
                print(df_pred.head(100))
                # 5. 训练模型
                auto_shoupan.train_model(df_pred, filename_without_extension)
                # except Exception as e:
                #     print(f"处理文件 {filename} 时出错: {e}")

        # 训练模型

        # 将模型写入目录中





if __name__ == "__main__":
    process_prediction_files()