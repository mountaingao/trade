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
        self.recorded_keys = []    # 存储记录的键盘输入
        self.recorded_moves = []   # 存储记录的鼠标移动轨迹
        self.is_recording = False  # 记录状态标志
        self.current_position = None  # 当前鼠标位置
        self.listener = None  # 鼠标监听器
        self.keyboard_listener = None  # 键盘监听器
        self.output_dir = "screen_points"  # 输出目录
        self.is_between_clicks = False  # 新增状态标志，标记是否在两个点击事件之间

        # 确保输出目录存在
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

    def on_move(self, x, y):
        """更新当前鼠标位置并记录移动轨迹"""
        self.current_position = (x, y)
        
        # 只在两个点击事件之间记录移动轨迹
        if self.is_recording and self.is_between_clicks:
            timestamp = datetime.datetime.now().strftime("%H:%M:%S.%f")[:-3]
            move_point = {
                "x": x,
                "y": y,
                "timestamp": timestamp
            }
            self.recorded_moves.append(move_point)

    def on_click(self, x, y, button, pressed):

        # 记录点击位置和时间戳
        timestamp = datetime.datetime.now().strftime("%H:%M:%S.%f")[:-3]
        point = {
            "x": x,
            "y": y,
            "button": str(button).split('.')[-1],
            "timestamp": timestamp,
            "screen_size": pyautogui.size()
        }
        # 记录与前一点的间隔时间
        if self.recorded_points:
            last_time = datetime.datetime.strptime(
                self.recorded_points[-1]["timestamp"], "%H:%M:%S.%f"
            )
            current_time = datetime.datetime.strptime(point["timestamp"], "%H:%M:%S.%f")
            delay = (current_time - last_time).total_seconds()
            point["delay_since_last"] = delay

        """处理鼠标点击事件"""
        if not pressed or not self.is_recording:
            return
        
        # 更新点击事件之间的状态标志
        if not self.recorded_points:  # 第一个点击事件
            self.is_between_clicks = True
        else:  # 后续点击事件
            # 结束前一段移动记录，开始新一段移动记录
            self.is_between_clicks = True
            
        self.recorded_points.append(point)
        print(f"记录点 {len(self.recorded_points)}: ({x}, {y}) - {button} - {timestamp}")

        return True

    # 新增键盘按下事件处理函数
    def on_key_press(self, key):
        """处理键盘按下事件"""
        if not self.is_recording:
            return
            
        try:
            # 记录按键事件
            timestamp = datetime.datetime.now().strftime("%H:%M:%S.%f")[:-3]
            key_event = {
                "key": key.char if hasattr(key, 'char') else str(key),
                "timestamp": timestamp,
                "action": "press"
            }
            self.recorded_keys.append(key_event)
        except AttributeError:
            pass

    # 新增键盘释放事件处理函数
    def on_key_release(self, key):
        """处理键盘释放事件"""
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

        if self.is_recording:
            try:
                # 记录按键释放事件
                timestamp = datetime.datetime.now().strftime("%H:%M:%S.%f")[:-3]
                key_event = {
                    "key": key.char if hasattr(key, 'char') else str(key),
                    "timestamp": timestamp,
                    "action": "release"
                }
                self.recorded_keys.append(key_event)
            except AttributeError:
                pass

    def toggle_recording(self):
        """切换记录状态"""
        self.is_recording = not self.is_recording
        self.is_between_clicks = False  # 重置状态标志

        if self.is_recording:
            print("开始记录屏幕点击位置... (按F1停止)")
            self.recorded_points = []  # 开始新记录时清空之前的数据
            self.recorded_moves = []   # 清空移动轨迹
            self.recorded_keys = []    # 清空键盘记录
        else:
            self.is_between_clicks = False  # 停止记录时重置状态
            print("记录已停止")

    def save_points(self):
        """保存记录的点位、键盘输入和鼠标轨迹到文件"""
        if not self.recorded_points and not self.recorded_keys and not self.recorded_moves:
            print("没有可保存的记录")
            return

        # 生成带时间戳的文件名
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = os.path.join(self.output_dir, f"screen_points_{timestamp}.json")

        # 保存为JSON文件，包含所有记录
        with open(filename, 'w') as f:
            json.dump({
                "points": self.recorded_points,
                "keys": self.recorded_keys,
                "moves": self.recorded_moves,
                "screen_size": pyautogui.size()
            }, f, indent=4)

        print(f"已保存 {len(self.recorded_points)} 个点位, {len(self.recorded_keys)} 个键盘事件, {len(self.recorded_moves)} 个移动点到 {filename}")

    def replay_points(self, filename=None):
        """回放记录的点位、键盘输入和鼠标轨迹"""
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
            data = json.load(f)

        points = data.get("points", [])
        keys = data.get("keys", [])
        moves = data.get("moves", [])
        recorded_screen = tuple(data.get("screen_size", (0, 0)))

        # 检查屏幕尺寸是否一致
        current_screen = pyautogui.size()
        recorded_screen = tuple(points[0]["screen_size"])

        if current_screen != recorded_screen:
            print(f"警告: 记录时的屏幕尺寸为 {recorded_screen}，当前为 {current_screen}")
            if input("继续回放? (y/n) ").lower() != 'y':
                return

        print(f"开始回放 {len(points)} 个点, {len(keys)} 个键盘事件, {len(moves)} 个移动点...")

        # 合并所有事件并按时间排序
        all_events = []
        for point in points:
            point["type"] = "click"
            all_events.append(point)
        for key_event in keys:
            key_event["type"] = "key"
            all_events.append(key_event)
        for move in moves:
            move["type"] = "move"
            all_events.append(move)
            
        # 按时间戳排序
        all_events.sort(key=lambda e: e["timestamp"])

        # 回放所有事件
        start_time = datetime.datetime.now()
        for i, event in enumerate(all_events):
            event_time = datetime.datetime.strptime(event["timestamp"], "%H:%M:%S.%f")
            elapsed = (event_time - start_time).total_seconds()
            time.sleep(max(0, elapsed - (datetime.datetime.now() - start_time).total_seconds()))
            
            if event["type"] == "click":
                # 回放点击事件
                x = event["x"]
                y = event["y"]
                button = event["button"].lower()
                pyautogui.moveTo(x, y, duration=0.1)
                if button == 'left':
                    pyautogui.click()
                elif button == 'right':
                    pyautogui.rightClick()
                elif button == 'middle':
                    pyautogui.middleClick()
                print(f"回放点 {i+1}/{len(all_events)}: ({x}, {y})")
                
            elif event["type"] == "key":
                # 回放键盘事件
                key = event["key"]
                action = event["action"]
                if action == "press":
                    pyautogui.keyDown(key)
                else:
                    pyautogui.keyUp(key)
                print(f"回放键盘事件 {i+1}/{len(all_events)}: {key} {action}")
                
            elif event["type"] == "move":
                # 回放鼠标移动
                x = event["x"]
                y = event["y"]
                pyautogui.moveTo(x, y, duration=0.05)
                print(f"回放移动 {i+1}/{len(all_events)}: ({x}, {y})")

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

        # 启动键盘监听器，添加press事件监听
        self.keyboard_listener = keyboard.Listener(
            on_press=self.on_key_press,
            on_release=self.on_key_release
        )
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

    def capture_screen_around_point(self, x, y, radius=100, filename=None):
        """捕获点击位置周围的屏幕区域"""
        screen = pyautogui.screenshot()

        # 计算捕获区域
        left = max(0, x - radius)
        top = max(0, y - radius)
        right = min(screen.width, x + radius)
        bottom = min(screen.height, y + radius)

        # 截取区域
        region = screen.crop((left, top, right, bottom))

        # 生成文件名
        if not filename:
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"click_{x}_{y}_{timestamp}.png"

        # 保存图片
        region.save(os.path.join(self.output_dir, filename))
        print(f"已保存点击位置截图: {filename}")

if __name__ == "__main__":
    recorder = ScreenPointRecorder()
    recorder.start()

    # 等待程序结束
    try:
        while recorder.listener.is_alive():
            time.sleep(0.1)
    except KeyboardInterrupt:
        recorder.exit_program()


    # 在主程序末尾添加
    recorder.replay_points()