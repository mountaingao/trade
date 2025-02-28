import os
import time
import tkinter as tk
from tkinter import messagebox
from playsound import playsound
from pydub import AudioSegment
from pydub.playback import play
import mysql.connector
import datetime
import pandas as pd
from datetime import datetime

# 文件路径
file_path = r"alert1.txt"
file_path = r"D:/BaiduSyncdisk/个人/通达信/ALERT/ALERT.txt"
# file_path = r"F:/baidu/BaiduSyncdisk/个人/通达信/ALERT/ALERT.txt"
# 记录文件的最后修改时间和内容
last_modified_time = os.path.getmtime(file_path)
with open(file_path, 'r', encoding='GBK') as file:
    last_content = file.read()  # 读取初始文件内容

# 数据库连接配置
db_config = {
    "host": "localhost",  # 数据库主机地址
    "user": "root",  # 数据库用户名
    "password": "111111",  # 数据库密码
    "database": "trade"  # 数据库名称
}

def show_alert(new_content, mp3_path):
    root = tk.Tk()
    root.title("提醒")  # 设置窗口标题
    root.attributes('-topmost', True)  # 确保窗口始终在最前面
    # 显示消息内容
    message = new_content
    label = tk.Label(root, text=message, wraplength=420, justify="center",padx=20,  # 内部水平填充
                     pady=20,  # 内部垂直填充
                     borderwidth=2,  # 边框宽度
                     relief="groove")  # 边框样式)
    label.pack(expand=True, padx=20, pady=20)
    # messagebox.showinfo("提醒", f"文件内容已更新！\n\n新增内容：\n{new_content}")
    # playsound("alarm.mp3")

    # 设置定时器，5秒后关闭窗口
    root.after(8000, root.destroy)

    # 播放音频
    sound = AudioSegment.from_mp3(mp3_path)
    play(sound)
    # playsound(mp3_path)

    # 阻止窗口关闭按钮关闭窗口
    # root.protocol('WM_DELETE_WINDOW', lambda: None)

    # 运行主循环
    root.mainloop()

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
                alert_datetime = datetime.strptime(alert_datetime_str, "%Y-%m-%d %H:%M")
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

def format_result(result):
    """格式化 result 列表，提取每行的第一个和第二个字段，并用空格分隔，多条记录用换行符分隔"""
    formatted_lines = []
    for item in result:
        if len(item) >= 2:  # 确保每行至少有两个字段
            formatted_line = f"{item[0].strip()} {item[1].strip()} {item[2].strip()} {item[3].strip()} {item[4].strip()} {item[6].strip()} "
            formatted_lines.append(formatted_line)
    return "\n".join(formatted_lines)
def monitor_file(mp3_path,db_config):
    global last_modified_time, last_content
    while True:
        # 获取文件的当前修改时间和内容

        current_modified_time = os.path.getmtime(file_path)
        formatted_time = datetime.fromtimestamp(current_modified_time).strftime('%Y-%m-%d %H:%M:%S')
        print(formatted_time)
        # print(last_modified_time)
        # print(last_content)
        with open(file_path, 'r', encoding='GBK') as file:
            current_content = file.read()

        # 如果文件被修改
        if current_modified_time != last_modified_time or current_content != last_content:
            last_modified_time = current_modified_time
            # 计算新增的内容
            added_content = current_content[len(last_content):].strip()
            last_content = current_content  # 更新记录的内容
            # 301396	宏景科技	2025-02-21 09:20	55.20	 0.00%	    0	开盘
            print(added_content)

            # 如果有新增内容，显示提醒
            if added_content:
                # 插入到数据库中
                # 将数据按行分割
                lines = added_content.strip().split("\n")

                # 解析每行数据
                result = []
                for line in lines:
                    fields = line.split("\t")  # 按制表符分割字段
                    result.append(fields)

                # 弹出提示信息
                show_alert(format_result(result),mp3_path)

                # print(df)
                import_to_database(result,  db_config)

        # 每隔1秒检查一次
        time.sleep(2)

if __name__ == "__main__":
    # 获取脚本所在目录的上一级目录
    script_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    # 构造音频文件的完整路径
    mp3_path = os.path.join(script_dir, "mp3", "alarm.mp3")
    print(mp3_path)

    monitor_file(mp3_path,db_config)
