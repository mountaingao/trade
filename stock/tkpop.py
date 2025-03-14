import tkinter as tk

def create_custom_popup():
    # 创建一个顶级窗口
    popup = tk.Toplevel(root)
    popup.title("自定义弹窗")
    # 设置窗口大小
    popup.geometry("300x200")

    # 创建一个标签并设置字体大小和颜色
    label1 = tk.Label(popup, text="海兰信 300065", font=("Arial", 14, "bold"), fg="blue")
    label1.pack(pady=10)

    label2 = tk.Label(popup, text="开盘：63.0", font=("Arial", 12), fg="green")
    label2.pack(pady=5)

    label3 = tk.Label(popup, text="当前价格：12.85 (0.00%)", font=("Arial", 10), fg="red")
    label3.pack(pady=5)

    label4 = tk.Label(popup, text="预计成交额：62.75亿", font=("Arial", 12, "bold"), fg="purple")
    label4.pack(pady=10)

    # 添加一个关闭按钮
    close_button = tk.Button(popup, text="关闭", command=popup.destroy)
    close_button.pack(pady=20)

    # 设置定时器，10秒后关闭窗口
    popup.after(10000, popup.destroy)

# 创建主窗口，但不显示
root = tk.Tk()
root.withdraw()  # 隐藏主窗口

# 直接创建弹窗
create_custom_popup()

# 运行主循环，保持程序运行直到弹窗关闭
root.mainloop()