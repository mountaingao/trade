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
directory_path = "F:/baidu/BaiduSyncdisk/个人/通达信/202502"  # 替换为你的 Excel 文件所在目录

# 读取目录下的所有 Excel 文件
def get_excel_files(directory):
    return [os.path.join(directory, file) for file in os.listdir(directory) if file.endswith(('.xls', '.xlsx'))]

# 读取 Excel 文件内容
def read_excel(file_path):
    try:
        df = pd.read_csv(file_path, encoding='GBK')  # 读取 Excel 文件，确保编码为 UTF-8
        print(f"成功读取文件：{file_path}")
        return df
    except Exception as e:
        print(f"读取文件时出错：{file_path}，错误信息：{e}")
        return None

# 将数据导入数据库
def import_to_database(df, db_config):
    try:
        # 连接数据库
        conn = mysql.connector.connect(**db_config)
        cursor = conn.cursor()

        # 遍历 DataFrame 并插入数据
        for _, row in df.iterrows():
            print(row)
            stock_name = row.iloc[0]  # 使用 .iloc[] 访问数据
            stock_code = row.iloc[1]
            alert_time = datetime.strptime(row.iloc[2], "%H:%M").time()  # 假设时间格式为 "HH:MM"
            current_price = row.iloc[3]
            price_change = float(row.iloc[4].strip('%'))  # 去掉百分号并转换为浮点数
            status = row.iloc[5]
            print(stock_name)
            # 构造 SQL 插入语句
            insert_query = """
            INSERT INTO AlertData (stock_code, stock_name, alert_time, current_price, price_change, status)
            VALUES (%s, %s, %s, %s, %s, %s)
            """
            values = (stock_code, stock_name, alert_time, current_price, price_change, status)

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
    # 获取目录下的所有 Excel 文件
    excel_files = get_excel_files(directory_path)
    if not excel_files:
        print("未找到任何 Excel 文件，请检查目录路径！")
    else:
        for file_path in excel_files:
            # 读取 Excel 文件
            df = read_excel(file_path)
            if df is not None:
                # 导入数据到数据库
                import_to_database(df, db_config)