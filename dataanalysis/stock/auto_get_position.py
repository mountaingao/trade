import pyautogui
import time
import json
import os
from pynput import mouse, keyboard
import threading
import datetime
# 记录点 1: (193, 333) - Button.left - 09:11:57.830
# 记录点 2: (425, 189) - Button.right - 09:11:59.054
# 记录点 3: (740, 664) - Button.left - 09:12:05.053
# 记录点 4: (1506, 556) - Button.left - 09:12:10.782
# 记录点 5: (1971, 1077) - Button.left - 09:12:17.038
# 记录点 6: (1972, 1086) - Button.left - 09:12:17.758
# 记录点 7: (1414, 914) - Button.left - 09:12:19.309
# 记录点 8: (1407, 914) - Button.left - 09:12:20.838
# 记录点 9: (1407, 914) - Button.left - 09:12:22.270
# 记录点 10: (474, 1415) - Button.left - 09:12:25.046



class ScreenPointRecorder:
    def __init__(self):
        self.recorded_points = []  # 存储记录的点位
        self.is_recording = False  # 记录状态标志
        self.current_position = None  # 当前鼠标位置
        self.listener = None  # 鼠标监听器
        self.keyboard_listener = None  # 键盘监听器
        self.output_dir = "screen_points"  # 输出目录

        # 确保输出目录存在
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

    def on_move(self, x, y):
        """更新当前鼠标位置"""
        self.current_position = (x, y)

    def on_click(self, x, y, button, pressed):
        """处理鼠标点击事件"""
        if not pressed or not self.is_recording:
            return

        # 记录点击位置和时间戳
        timestamp = datetime.datetime.now().strftime("%H:%M:%S.%f")[:-3]
        point = {
            "x": x,
            "y": y,
            "button": str(button).split('.')[-1],
            "timestamp": timestamp,
            "screen_size": pyautogui.size()
        }

        self.recorded_points.append(point)
        print(f"记录点 {len(self.recorded_points)}: ({x}, {y}) - {button} - {timestamp}")

        return True

    def on_key_release(self, key):
        """处理键盘事件"""
        try:
            # 开始/停止记录
            if key == keyboard.Key.f1:
                self.toggle_recording()
            # 保存记录
            elif key == keyboard.Key.f2:
                self.save_points()
            # 退出程序
            elif key == keyboard.Key.f3:
                self.exit_program()
        except AttributeError:
            pass

    def toggle_recording(self):
        """切换记录状态"""
        self.is_recording = not self.is_recording

        if self.is_recording:
            print("开始记录屏幕点击位置... (按F1停止)")
            self.recorded_points = []  # 开始新记录时清空之前的数据
        else:
            print("记录已停止")

    def save_points(self):
        """保存记录的点位到文件"""
        if not self.recorded_points:
            print("没有可保存的记录点")
            return

        # 生成带时间戳的文件名
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = os.path.join(self.output_dir, f"screen_points_{timestamp}.json")

        # 保存为JSON文件
        with open(filename, 'w') as f:
            json.dump(self.recorded_points, f, indent=4)

        print(f"已保存 {len(self.recorded_points)} 个记录点到 {filename}")

    def replay_points(self, filename=None):
        """回放记录的点位"""
        if not filename:
            # 获取最新的记录文件
            files = sorted(
                [f for f in os.listdir(self.output_dir) if f.startswith("screen_points_")],
                reverse=True
            )
            if not files:
                print("没有找到记录文件")
                return
            filename = os.path.join(self.output_dir, files[0])

        # 读取记录文件
        with open(filename, 'r') as f:
            points = json.load(f)

        print(f"开始回放 {len(points)} 个点...")

        # 检查屏幕尺寸是否一致
        current_screen = pyautogui.size()
        recorded_screen = tuple(points[0]["screen_size"])

        if current_screen != recorded_screen:
            print(f"警告: 记录时的屏幕尺寸为 {recorded_screen}，当前为 {current_screen}")
            if input("继续回放? (y/n) ").lower() != 'y':
                return

        # 回放每个点
        for i, point in enumerate(points):
            # 移动鼠标到该位置
            pyautogui.moveTo(point['x'], point['y'], duration=0.2)

            # 模拟点击
            button = point['button'].lower()
            if button == 'left':
                pyautogui.click()
            elif button == 'right':
                pyautogui.rightClick()
            elif button == 'middle':
                pyautogui.middleClick()

            print(f"回放点 {i+1}/{len(points)}: ({point['x']}, {point['y']})")

            # 添加延迟（可选）
            time.sleep(0.1)

        print("回放完成")

    def start(self):
        """启动记录器"""
        print("屏幕取点记录器已启动")
        print("使用说明:")
        print("  F1 键 - 开始/停止记录点击位置")
        print("  F2 键 - 保存当前记录的点位")
        print("  F3 键 - 退出程序")

        # 启动鼠标监听
        self.listener = mouse.Listener(
            on_move=self.on_move,
            on_click=self.on_click
        )
        self.listener.start()

        # 启动键盘监听
        self.keyboard_listener = keyboard.Listener(on_release=self.on_key_release)
        self.keyboard_listener.start()

        # 显示当前鼠标位置
        self.display_current_position()

    def display_current_position(self):
        """实时显示鼠标位置"""
        try:
            while self.listener.is_alive():
                if self.current_position:
                    x, y = self.current_position
                    # 使用ANSI转义序列更新控制台输出
                    print(f"\033[K当前鼠标位置: ({x}, {y})", end='\r')
                time.sleep(0.05)
        except KeyboardInterrupt:
            pass

    def exit_program(self):
        """退出程序"""
        print("\n正在退出程序...")
        self.is_recording = False

        if self.listener:
            self.listener.stop()
        if self.keyboard_listener:
            self.keyboard_listener.stop()

        # 询问是否保存未保存的点位
        if self.recorded_points:
            if input("\n是否保存当前记录的点位? (y/n) ").lower() == 'y':
                self.save_points()

        print("程序已退出")
        os._exit(0)

if __name__ == "__main__":
    recorder = ScreenPointRecorder()
    recorder.start()

    # 等待程序结束
    try:
        while recorder.listener.is_alive():
            time.sleep(0.1)
    except KeyboardInterrupt:
        recorder.exit_program()