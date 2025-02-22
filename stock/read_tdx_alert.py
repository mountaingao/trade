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

def show_alert(new_content, mp3_path):
    root = tk.Tk()
    root.title("提醒")  # 设置窗口标题
    root.attributes('-topmost', True)  # 确保窗口始终在最前面
    # 显示消息内容
    message = new_content
    label = tk.Label(root, text=message, wraplength=280, justify="center")
    label.pack(expand=True)
    playsound(mp3_path)
    # messagebox.showinfo("提醒", f"文件内容已更新！\n\n新增内容：\n{new_content}")

    # 设置定时器，5秒后关闭窗口
    root.after(5000, root.destroy)

    # 阻止窗口关闭按钮关闭窗口
    # root.protocol('WM_DELETE_WINDOW', lambda: None)

    # 运行主循环
    root.mainloop()

def monitor_file(mp3_path):
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
                show_alert(added_content,mp3_path)

        # 每隔1秒检查一次
        time.sleep(2)

if __name__ == "__main__":
    # 获取脚本所在目录的上一级目录
    script_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    # 构造音频文件的完整路径
    mp3_path = os.path.join(script_dir, "mp3", "alarm.mp3")
    print(mp3_path)

    monitor_file(mp3_path)