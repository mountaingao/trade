
import os
import time
import tkinter as tk
from tkinter import messagebox
from playsound import playsound
import mysql.connector
import datetime
import pandas as pd
import logging
from configparser import ConfigParser


# 数据库连接配置
db_config = {
    "host": "localhost",  # 数据库主机地址
    "user": "root",  # 数据库用户名
    "password": "111111",  # 数据库密码
    "database": "trade"  # 数据库名称
}


def import_to_database(data, db_config):
    try:
        conn = mysql.connector.connect(**db_config)
        cursor = conn.cursor()
        insert_query = """
        INSERT INTO AlertData (stock_code, stock_name, alert_time, current_price, price_change, status, date)
        VALUES (%s, %s, %s, %s, %s, %s, %s)
        """
        for row in data:
            try:
                stock_code = row[0].strip()
                stock_name = str(row[1]).strip()
                alert_datetime_str = row[2].strip()  # 包含日期和时间的字符串
                current_price = float(str(row[3]).strip())
                price_change = float(str(row[4]).strip().rstrip('%'))
                status = row[6].strip()

                # 解析包含日期和时间的字符串
                alert_datetime = datetime.datetime.strptime(alert_datetime_str, "%Y-%m-%d %H:%M")
                alert_time = alert_datetime.time()  # 提取时间部分
                alert_date = alert_datetime.date()  # 提取日期部分

                values = (stock_code, stock_name, alert_time, current_price, price_change, status, alert_date)
                cursor.execute(insert_query, values)
                print(f"成功插入数据: {values}")
            except Exception as e:
                print(f"处理行数据时出错：{e}, 数据行: {row}")
        conn.commit()
        print("数据导入成功！")
    except Exception as e:
        print(f"导入数据时出错：{e}")
    finally:
        cursor.close()
        conn.close()


if __name__ == "__main__":
    test_data = [
        ['300911', '亿田智能', '2025-02-21 09:21', '40.95', ' 0.00%', '    0', '开盘'],
        ['300912', '其他股票', '2025-02-21 09:22', '50.00', ' 1.00%', '    1', '开盘']
    ]
    import_to_database(test_data, db_config)
