import os
import time
import tkinter as tk
from tkinter import messagebox
from playsound import playsound

# 文件路径
# file_path = r"D:\BaiduSyncdisk\个人\通达信\ALERT\ALERT.txt"
file_path = r"alert1.txt"

# 记录文件的最后修改时间
last_modified_time = os.path.getmtime(file_path)

def show_alert():
    root = tk.Tk()
    root.withdraw()  # 隐藏主窗口
    messagebox.showinfo("提醒", "这是一个提示框！")
    # 播放声音
    playsound("D:/project/trades/mp3/alert.mp3")

def monitor_file():
    global last_modified_time
    while True:
        # 获取文件的当前修改时间
        current_modified_time = os.path.getmtime(file_path)

        # 如果文件被修改
        if current_modified_time != last_modified_time:
            last_modified_time = current_modified_time

            # 读取文件内容
            with open(file_path, 'r', encoding='GBK') as file:
                content = file.read()

            show_alert()

        # 每隔1秒检查一次
        time.sleep(1)


if __name__ == "__main__":
    monitor_file()
