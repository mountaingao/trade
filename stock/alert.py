import tkinter as tk
from playsound import playsound
from pydub import AudioSegment
from pydub.playback import play
import os
def show_alert():
    root = tk.Tk()
    root.title("提醒")  # 设置窗口标题
    root.attributes('-topmost', True)  # 确保窗口始终在最前面
    root.geometry("300x150")  # 设置窗口大小

    # 显示消息内容
    message = "这是一个提示框！"
    label = tk.Label(root, text=message, wraplength=280, justify="center")
    label.pack(expand=True)

    # 获取脚本所在目录的上一级目录
    script_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    # 构造音频文件的完整路径
    mp3_path = os.path.join(script_dir, "mp3", "alarm.mp3")
    print(mp3_path)
    # 播放声音
    # playsound(mp3_path)
    playsound("../mp3/alarm.mp3")
    # 播放音频
    sound = AudioSegment.from_mp3(mp3_path)
    play(sound)
    # 播放声音
    # playsound("../mp3/alert.mp3", block=False)

    # 设置定时器，5秒后关闭窗口
    root.after(5000, root.destroy)

    # 阻止窗口关闭按钮关闭窗口
    # root.protocol('WM_DELETE_WINDOW', lambda: None)

    # 运行主循环
    root.mainloop()

# 调用函数
show_alert()