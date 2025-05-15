import os
import pandas as pd
import mysql.connector
from datetime import datetime
from stockrating.read_local_info_tdx import  calculate_stock_profit_from_date
from stockrating.stock_block_tdx import  get_tdx_custom_block_from_date
from dataanalysis.cal_excel_per import  calculate_positive_percentage
import json


# 新增代码：读取配置文件
config_path = os.path.join(os.path.dirname(__file__), '..', 'config', 'config.json')
with open(config_path, 'r', encoding='utf-8') as config_file:  # 修改编码为utf-8
    config = json.load(config_file)



def get_csv_files(directory):
    return [os.path.join(directory, file) for file in os.listdir(directory) if file.endswith(('.xls', '.xlsx'))]

# 从文件名中提取日期信息
def extract_date_from_filename(file_name):
    # 假设文件名格式为 "20250223_alert_data.csv"，提取日期部分
    date_str = os.path.splitext(file_name)[0].split('_')[0]  # 提取文件名中的日期部分并去除扩展名
    try:
        # 将日期字符串转换为日期对象
        date_obj = datetime.strptime(date_str, "%m%d").date()
        current_year = datetime.now().year
        final_date = date_obj.replace(year=current_year).strftime("%Y%m%d")
        return final_date
    except ValueError:
        print(f"无法从文件名中提取日期：{file_name}")
        return None

# 读取 CSV 文件内容
def read_csv(file_path):
    try:
        # 读取 CSV 文件，假设文件编码为 GBK，且没有标题行
        # df = pd.read_csv(file_path, encoding='GBK', header=None, sep='/t')  # 使用制表符作为分隔符
        df = pd.read_csv(file_path, encoding='GBK', header=None, sep=' ', engine='python')  # 使用制表符作为分隔符

        print(f"成功读取文件：{file_path}")
        return df
    except Exception as e:
        print(f"读取文件时出错：{file_path}，错误信息：{e}")
        return None

def cal_stock_profit(stockcode,date,time):
    return_data = calculate_stock_profit_from_date(stockcode,date,time)
    if return_data is None:
        print(f"无法计算收益率：{file_path}")
    return return_data

# 将数据导入数据库
def import_alert_data(df, file_date, db_config,file_name):
    try:
        # 连接数据库
        conn = mysql.connector.connect(**db_config)
        cursor = conn.cursor()
        print(df)
        result = {}
        # 遍历 DataFrame 并插入数据
        for _, row in df.iterrows():
            # 解析 row 内容
            row_data = row.iloc[0].split('\t')  # 使用制表符分割行数据
            if len(row_data) < 6:
                print(f"跳过行：{row_data}，数据列不足")
                continue

            stock_name = row_data[0].strip()  # 股票名称
            stock_code = row_data[1].strip()  # 股票代码
            alert_time_str = row_data[2].strip()  # 预警时间（仅时间部分）
            current_price = float(row_data[3].strip())  # 当前价格
            price_change = float(row_data[4].strip().rstrip('%'))  # 涨跌幅（去掉百分号）
            status = row_data[5].strip()  # 状态



            # 计算1-5天的收盘价和最高价相对于买入价的百分比
            # 将日期格式转换一下
            print(file_date)
            days = 5
            return_data = calculate_stock_profit_from_date(stock_code, file_date, current_price,days)
            if return_data is None:
                print(f"无法计算收益率：{stock_code}")
                continue

            return_data['stock_name'] = stock_name
            return_data['stock_code'] = stock_code
            return_data['alert_time'] = alert_time_str
            return_data['current_price'] = current_price
            return_data['alert_date'] = file_date
            print(return_data)


            result[stock_code] = return_data

        # 将 result 数据写入 Excel 文件
        result_df = pd.DataFrame.from_dict(result, orient='index')
        
        # 检查文件是否存在，如果存在则追加，否则创建新文件
        if os.path.exists(file_name):
            with pd.ExcelWriter(file_name, mode='a', engine='openpyxl', if_sheet_exists='overlay') as writer:
                result_df.to_excel(writer, index=False, header=False, startrow=writer.sheets['Sheet1'].max_row)
        else:
            result_df.to_excel(file_name, index=False)

        # 提交事务
        # conn.commit()
        print(result)
        print("数据写入成功！"+file_name)
    except Exception as e:
        print(f"导入数据时出错：{e}")
    finally:
        # 关闭连接
        cursor.close()
        conn.close()

# 读取所有的文件数据进行分析
def get_csv_file_stock_data(file_path):
    # 读取所有的提前选取的板块数据，获得当日的各项数据，并计算各个需要参考的技术指标，以便做出正确决策
    directory_path = config['directory_path']
    # 获取目录下的所有 CSV 文件
    csv_files = get_csv_files(directory_path)
    if not csv_files:
        print("未找到任何 CSV 文件，请检查目录路径！")
    else:
        for file_path in csv_files:
            # 提取文件名中的日期信息
            file_name = os.path.basename(file_path)
            file_date = extract_date_from_filename(file_name)
            if file_date is None:
                print(f"跳过文件：{file_name}，无法提取日期信息")
                continue

            # 读取 CSV 文件
            df = read_csv(file_path)
            xls_name = os.path.basename(directory_path)+".xlsx"
            if df is not None:
                # 计算数据
                import_alert_data(df, file_date, db_config,xls_name)
            # exit() b

def get_date_from_name(file_name):
    # 假设文件名格式为 "20250223_alert_data.csv"，提取日期部分
    date_str = os.path.splitext(file_name)[0].split('_')[0]  # 提取文件名中的日期部分并去除扩展名
    try:
        # 将日期字符串转换为日期对象
        date_obj = datetime.strptime(date_str, "%m%d").date()
        current_year = datetime.now().year
        final_date = date_obj.replace(year=current_year).strftime("%Y%m%d")
        return final_date
    except ValueError:
        print(f"无法从文件名中提取日期：{file_name}")
        return None

def get_tdx_block_pre_data(start,end):
    stocks = get_tdx_custom_block_from_date(start,end)
    print(stocks)
    result = {}
    for index, stock in stocks.iterrows():
        date = get_date_from_name(stock['blockname'])  # 日期需要完善
        stock_code = stock['code']
        print(f"Processing stock: {stock_code} {date} ")
        return_data = calculate_stock_profit_from_date(stock_code, date,0,3)
        if return_data is not None:
            result[f"{stock_code}-{date}"] = return_data
        else:
            print(f"无法计算收益率：{stock_code}")

        # print(result)
        # exit()
        # print(result)

    # 将 result 数据写入 Excel 文件
    result_df = pd.DataFrame.from_dict(result, orient='index')
    # 文件名为固定+日期+后缀
    xls_name = f"tdx_block_pre_data_{start}-{end}.xlsx"
    result_df.to_excel(xls_name, index=False)
    print(f"数据写入成功！{xls_name}")

    #统计数据
    calculate_positive_percentage(xls_name)

# 主程序
if __name__ == "__main__":

    # get_csv_file_stock_data(directory_path)
    #1、通达信的板块数据
    get_tdx_block_pre_data(401,430)

    # 可以根据条件过滤掉部分无效数据以后再进行数据分析和判断，如上轨以上，放量的比较
    # 2、本地的日期+代码数据
    # 2、本地的日期+代码数据