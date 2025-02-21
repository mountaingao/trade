
import tkinter as tk
from tkinter import messagebox
from playsound import playsound

# 创建提示框
def show_alert():
    root = tk.Tk()
    root.withdraw()  # 隐藏主窗口
    messagebox.showinfo("提醒", "这是一个提示框！")
    # 播放声音
    playsound("D:/project/trade/mp3/alert.mp3")
# 调用函数
show_alert()
