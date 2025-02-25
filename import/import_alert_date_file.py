import os
import pandas as pd
import mysql.connector
from datetime import datetime

# 数据库连接配置
db_config = {
    "host": "localhost",  # 数据库主机地址
    "user": "root",  # 数据库用户名
    "password": "111111",  # 数据库密码
    "database": "trade"  # 数据库名称
}

# 目标目录路径
directory_path = "F:/baidu/BaiduSyncdisk/个人/通达信/202502"  # 替换为你的文件所在目录
directory_path = "D:/BaiduSyncdisk/个人/通达信/202502"
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
        final_date = date_obj.replace(year=current_year)
        return final_date
    except ValueError:
        print(f"无法从文件名中提取日期：{file_name}")
        return None

# 读取 CSV 文件内容
def read_csv(file_path):
    try:
        # 读取 CSV 文件，假设文件编码为 GBK，且没有标题行
        df = pd.read_csv(file_path, encoding='GBK', header=None, sep='/t')  # 使用制表符作为分隔符
        print(f"成功读取文件：{file_path}")
        return df
    except Exception as e:
        print(f"读取文件时出错：{file_path}，错误信息：{e}")
        return None

# 将数据导入数据库
def import_to_database(df, file_date, db_config):
    try:
        # 连接数据库
        conn = mysql.connector.connect(**db_config)
        cursor = conn.cursor()

        # 遍历 DataFrame 并插入数据
        for _, row in df.iterrows():
            print(row)
            stock_name = row.iloc[0].strip()  # 股票名称
            stock_code = str(row.iloc[1]).strip()  # 股票代码
            print(stock_code)

            alert_time_str = row.iloc[2].strip()  # 预警时间（仅时间部分）
            current_price = float(str(row.iloc[3]).strip())  # 当前价格，确保转换为字符串
            price_change = float(str(row.iloc[4]).strip().rstrip('%'))  # 涨跌幅（去掉百分号）
            status = row.iloc[5].strip()  # 状态
            # 解析预警时间（仅时间部分）
            alert_time = datetime.strptime(alert_time_str, "%H:%M").time()
            print(alert_time)
            # 构造 SQL 插入语句
            insert_query = """
            INSERT INTO AlertData (stock_code, stock_name, alert_time, current_price, price_change, status, date)
            VALUES (%s, %s, %s, %s, %s, %s, %s)
            """
            values = (stock_code, stock_name, alert_time, current_price, price_change, status, file_date)
            print(insert_query, values)
            # 执行插入操作
            cursor.execute(insert_query, values)

        # 提交事务
        conn.commit()
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
                import_to_database(df, file_date, db_config)