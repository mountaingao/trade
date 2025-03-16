import os
import pandas as pd
import mysql.connector
from datetime import datetime
from stockrating.read_local_info_tdx import  calculate_stock_profit_from_date

# 数据库连接配置
db_config = {
    "host": "localhost",  # 数据库主机地址
    "user": "root",  # 数据库用户名
    "password": "111111",  # 数据库密码
    "database": "trade"  # 数据库名称
}

# 目标目录路径
directory_path = "F:/baidu/BaiduSyncdisk/个人/通达信/202503"  # 替换为你的文件所在目录
# directory_path = "D:/BaiduSyncdisk/个人/通达信/202502"
# 读取目录下的所有 CSV 文件
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
def import_alert_data(df, file_date, db_config):
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

            print(return_data)
            return_data['stock_name'] = stock_name
            return_data['stock_code'] = stock_code
            return_data['alert_time'] = alert_time_str
            return_data['current_price'] = current_price
            return_data['alert_date'] = file_date

            result[stock_code] = return_data

        # 将 result 数据写入 Excel 文件
        result_df = pd.DataFrame.from_dict(result, orient='index')
        
        # 检查文件是否存在，如果存在则追加，否则创建新文件
        if os.path.exists("alert_data_all.xlsx"):
            with pd.ExcelWriter("alert_data_all.xlsx", mode='a', engine='openpyxl', if_sheet_exists='overlay') as writer:
                result_df.to_excel(writer, index=False, header=False, startrow=writer.sheets['Sheet1'].max_row)
        else:
            result_df.to_excel("alert_data_all.xlsx", index=False)

        # 提交事务
        # conn.commit()
        print(result)
        print("数据导入成功！")
    except Exception as e:
        print(f"导入数据时出错：{e}")
    finally:
        # 关闭连接
        cursor.close()
        conn.close()

# 主程序
if __name__ == "__main__":
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
            if df is not None:
                # 导入数据到数据库
                import_alert_data(df, file_date, db_config)
            # exit() b