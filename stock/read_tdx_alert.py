import os
import time
import tkinter as tk
from tkinter import messagebox
from playsound import playsound

# 文件路径
file_path = r"alert1.txt"

# 记录文件的最后修改时间和内容
last_modified_time = os.path.getmtime(file_path)
with open(file_path, 'r', encoding='GBK') as file:
    last_content = file.read()  # 读取初始文件内容

def show_alert(new_content):
    root = tk.Tk()
    root.withdraw()  # 隐藏主窗口
    messagebox.showinfo("提醒", f"文件内容已更新！\n\n新增内容：\n{new_content}")
    # 播放声音
    # playsound("D:/project/trades/mp3/Alarm03.wav")

def monitor_file():
    global last_modified_time, last_content
    while True:
        # 获取文件的当前修改时间和内容

        current_modified_time = os.path.getmtime(file_path)
        print(current_modified_time)
        print(last_modified_time)
        # print(last_content)
        with open(file_path, 'r', encoding='GBK') as file:
            current_content = file.read()

        # 如果文件被修改
        if current_modified_time != last_modified_time or current_content != last_content:
            last_modified_time = current_modified_time
            # 计算新增的内容
            added_content = current_content[len(last_content):].strip()
            last_content = current_content  # 更新记录的内容

            # 如果有新增内容，显示提醒
            if added_content:
                show_alert(added_content)

        # 每隔1秒检查一次
        time.sleep(1)

if __name__ == "__main__":
    monitor_file()