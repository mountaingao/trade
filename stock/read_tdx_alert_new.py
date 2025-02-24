import os
import time
import logging
import threading
import tkinter as tk
from tkinter import messagebox
from playsound import playsound
import mysql.connector
import datetime
import pandas as pd
import configparser
import watchdog.observers
import watchdog.events
import os

# 获取当前脚本文件的目录
script_dir = os.path.dirname(os.path.abspath(__file__))

# # 构建配置文件的路径
# config_file = os.path.join(script_dir, '..', 'config.ini')
#
# # 加载配置文件
# config = configparser.ConfigParser()
# # 指定文件编码为 utf-8
# config.read('config_file_path', encoding='utf-8')
# print(config)
# 文件路径
# file_path = config.get('Paths', 'file_path')

file_path = r"alert1.txt"

# 数据库连接配置
db_config = {
    "host": "localhost",  # 数据库主机地址
    "user": "root",  # 数据库用户名
    "password": "111111",  # 数据库密码
    "database": "trade"  # 数据库名称
}

# 音频文件路径
# mp3_path = os.path.join(script_dir, config.get('Paths', 'mp3_path'))
mp3_path = os.path.join(script_dir, "mp3", "alarm.mp3")

# 数据库连接配置
db_config = {
    "host": "localhost",  # 数据库主机地址
    "user": "root",  # 数据库用户名
    "password": "111111",  # 数据库密码
    "database": "trade"  # 数据库名称
}
# 设置日志记录
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def show_alert(new_content):
    if not new_content.strip():
        return

    def play_sound_and_show_window():
        root = tk.Tk()
        root.title("提醒")
        root.attributes('-topmost', True)
        message = new_content
        label = tk.Label(root, text=message, wraplength=280, justify="center")
        label.pack(expand=True)
        playsound(mp3_path)
        root.after(5000, root.destroy)
        root.mainloop()

    # 使用多线程播放声音并显示窗口
    threading.Thread(target=play_sound_and_show_window).start()

def import_to_database(df, db_config, file_date):
    try:
        conn = mysql.connector.connect(**db_config)
        cursor = conn.cursor()

        for _, row in df.iterrows():
            stock_name = row.iloc[0].strip()
            stock_code = str(row.iloc[1]).strip()
            alert_time_str = row.iloc[2].strip()
            current_price = float(str(row.iloc[3]).strip())
            price_change = float(str(row.iloc[4]).strip().rstrip('%'))
            status = row.iloc[5].strip()
            alert_time = datetime.datetime.strptime(alert_time_str, "%H:%M").time()

            insert_query = """
            INSERT INTO AlertData (stock_code, stock_name, alert_time, current_price, price_change, status, date)
            VALUES (%s, %s, %s, %s, %s, %s, %s)
            """
            values = (stock_code, stock_name, alert_time, current_price, price_change, status, file_date)
            cursor.execute(insert_query, values)

        conn.commit()
        logging.info("数据导入成功！")
    except Exception as e:
        logging.error(f"导入数据时出错：{e}")
    finally:
        cursor.close()
        conn.close()

class FileModifiedHandler(watchdog.events.FileSystemEventHandler):
    def __init__(self, mp3_path, db_config):
        self.mp3_path = mp3_path
        self.db_config = db_config
        self.last_content = ""

    def on_modified(self, event):
        if event.src_path == file_path:
            try:
                with open(file_path, 'r', encoding='GBK') as file:
                    current_content = file.read()

                added_content = current_content[len(self.last_content):].strip()
                self.last_content = current_content

                if added_content:
                    df = pd.DataFrame([["示例数据"]])  # 示例数据，实际应从文件中读取
                    file_date = datetime.datetime.now().strftime("%Y-%m-%d")
                    import_to_database(df, self.db_config, file_date)
                    show_alert(added_content)
            except Exception as e:
                logging.error(f"文件读取或处理出错：{e}")

def monitor_file(mp3_path, db_config):
    event_handler = FileModifiedHandler(mp3_path, db_config)
    observer = watchdog.observers.Observer()
    observer.schedule(event_handler, path=os.path.dirname(file_path), recursive=False)
    observer.start()
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()
    observer.join()

if __name__ == "__main__":
    monitor_file(mp3_path, db_config)
