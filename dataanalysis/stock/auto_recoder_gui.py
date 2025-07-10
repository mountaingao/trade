import tkinter as tk
from tkinter import ttk, messagebox
from auto_screen_point_recorder import ScreenPointRecorder
import time
import threading

class RecorderGUI:
    def __init__(self, recorder):
        self.recorder = recorder
        self.root = tk.Tk()
        self.root.title("屏幕取点记录器")

        self.create_widgets()

        # 更新GUI的线程
        self.update_thread = threading.Thread(target=self.update_gui)
        self.update_thread.daemon = True
        self.update_thread.start()

    def create_widgets(self):
        # 状态显示
        ttk.Label(self.root, text="记录状态:").grid(row=0, column=0, padx=5, pady=5)
        self.status_var = tk.StringVar(value="已停止")
        ttk.Label(self.root, textvariable=self.status_var).grid(row=0, column=1, padx=5, pady=5)

        # 控制按钮
        ttk.Button(self.root, text="开始/停止记录", command=self.recorder.toggle_recording).grid(row=1, column=0, padx=5, pady=5)
        ttk.Button(self.root, text="保存记录", command=self.recorder.save_points).grid(row=1, column=1, padx=5, pady=5)
        ttk.Button(self.root, text="回放记录", command=self.recorder.replay_points).grid(row=1, column=2, padx=5, pady=5)

        # 记录点列表
        columns = ("#", "X坐标", "Y坐标", "按钮", "时间")
        self.points_tree = ttk.Treeview(self.root, columns=columns, show="headings")
        for col in columns:
            self.points_tree.heading(col, text=col)
        self.points_tree.grid(row=2, column=0, columnspan=3, padx=5, pady=5)

        # 鼠标位置显示
        self.position_var = tk.StringVar(value="鼠标位置: (0, 0)")
        ttk.Label(self.root, textvariable=self.position_var).grid(row=3, column=0, columnspan=3, padx=5, pady=5)

    def update_gui(self):
        """更新GUI界面"""
        while True:
            try:
                # 更新状态
                self.status_var.set("正在记录..." if self.recorder.is_recording else "已停止")

                # 更新鼠标位置
                if self.recorder.current_position:
                    x, y = self.recorder.current_position
                    self.position_var.set(f"鼠标位置: ({x}, {y})")

                # 更新点列表
                self.points_tree.delete(*self.points_tree.get_children())
                for i, point in enumerate(self.recorder.recorded_points):
                    self.points_tree.insert("", "end", values=(
                        i+1,
                        point["x"],
                        point["y"],
                        point["button"],
                        point["timestamp"]
                    ))

                time.sleep(0.1)
            except:
                break

# 在主程序中使用
if __name__ == "__main__":
    recorder = ScreenPointRecorder()
    gui = RecorderGUI(recorder)
    recorder.start()  # 需要在GUI线程之外启动记录器
    gui.root.mainloop()