import pyautogui
import time
import json
import os
import threading
import keyboard
from pynput import mouse
import datetime

class ScreenAutomator:
    def __init__(self):
        self.recorded_actions = []  # 存储录制的动作1
        self.is_recording = False   # 录制状态
        self.is_playing = False     # 回放状态
        self.current_position = None
        self.mouse_listener = None
        self.keyboard_listener = None
        self.data_dir = "automation_data"
        self.playback_speed = 1.0   # 回放速度
        self.playback_count = 1     # 回放次数
        self.current_playback = 0   # 当前回放次数

        # 创建数据目录
        if not os.path.exists(self.data_dir):
            os.makedirs(self.data_dir)

    def start_recording(self):
        """开始录制操作"""
        if self.is_recording:
            print("已经在录制中")
            return

        print("开始录制操作...")
        print("操作说明:")
        print("  F1 - 记录左键点击")
        print("  F2 - 记录右键点击")
        print("  F3 - 记录中键点击")
        print("  F4 - 记录键盘输入")
        print("  F5 - 停止录制")
        print("  F6 - 保存录制")

        self.recorded_actions = []
        self.is_recording = True

        # 启动鼠标监听
        self.mouse_listener = mouse.Listener(
            on_move=self.on_move,
            on_click=self.on_click
        )
        self.mouse_listener.start()

        # 启动键盘监听
        self.keyboard_listener = keyboard.GlobalHotKeys({
            '<f1>': self.record_left_click,
            '<f2>': self.record_right_click,
            '<f3>': self.record_middle_click,
            '<f4>': self.record_keyboard_input,
            '<f5>': self.stop_recording,
            '<f6>': self.save_recording
        })
        self.keyboard_listener.start()

        # 启动位置显示线程
        threading.Thread(target=self.display_position, daemon=True).start()

    def stop_recording(self):
        """停止录制"""
        if not self.is_recording:
            print("未在录制中")
            return

        print("录制已停止")
        self.is_recording = False

        if self.mouse_listener:
            self.mouse_listener.stop()
        if self.keyboard_listener:
            self.keyboard_listener.stop()

    def save_recording(self, filename=None):
        """保存录制的操作"""
        if not self.recorded_actions:
            print("没有可保存的操作")
            return

        if not filename:
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"automation_{timestamp}.json"

        filepath = os.path.join(self.data_dir, filename)

        with open(filepath, 'w') as f:
            json.dump({
                "actions": self.recorded_actions,
                "screen_size": pyautogui.size(),
                "created_at": datetime.datetime.now().isoformat()
            }, f, indent=4)

        print(f"已保存 {len(self.recorded_actions)} 个操作到 {filepath}")
        return filepath

    def load_recording(self, filename=None):
        """加载录制的操作"""
        if not filename:
            # 获取最新的录制文件
            files = sorted(
                [f for f in os.listdir(self.data_dir) if f.startswith("automation_") and f.endswith(".json")],
                reverse=True
            )
            if not files:
                print("没有找到录制文件")
                return None
            filename = files[0]

        filepath = os.path.join(self.data_dir, filename)

        try:
            with open(filepath, 'r') as f:
                data = json.load(f)
                print(f"已加载录制文件: {filename} ({len(data['actions'])} 个操作)")
                return data
        except Exception as e:
            print(f"加载录制文件失败: {e}")
            return None

    def play_recording(self, filename=None, speed=1.0, count=1):
        """回放录制的操作"""
        if self.is_playing:
            print("已经在回放中")
            return

        data = self.load_recording(filename)
        if not data:
            return

        self.playback_speed = speed
        self.playback_count = count
        self.current_playback = 0

        # 检查屏幕尺寸
        current_screen = pyautogui.size()
        recorded_screen = tuple(data["screen_size"])

        if current_screen != recorded_screen:
            print(f"警告: 录制时的屏幕尺寸为 {recorded_screen}，当前为 {current_screen}")
            if input("继续回放? (y/n) ").lower() != 'y':
                return

        print(f"开始回放录制操作 (速度: {speed}x, 次数: {count})...")
        print("按 ESC 键可随时中断回放")

        self.is_playing = True

        # 在单独线程中运行回放
        threading.Thread(target=self._run_playback, args=(data["actions"],), daemon=True).start()

    def _run_playback(self, actions):
        """实际执行回放操作的内部方法"""
        try:
            for i in range(self.playback_count):
                self.current_playback = i + 1
                print(f"\n--- 开始第 {self.current_playback}/{self.playback_count} 次回放 ---")

                # 执行每个动作
                for idx, action in enumerate(actions):
                    if not self.is_playing:  # 检查是否中断
                        break

                    # 处理延迟
                    if "delay" in action and action["delay"] > 0:
                        time.sleep(action["delay"] / self.playback_speed)

                    # 执行不同类型的动作
                    if action["type"] == "click":
                        self._perform_click(action)
                    elif action["type"] == "keyboard":
                        self._perform_keyboard(action)

                    print(f"执行动作 {idx+1}/{len(actions)}: {action['type']}")

                if not self.is_playing:
                    break
        except Exception as e:
            print(f"回放出错: {e}")
        finally:
            self.is_playing = False
            print("回放完成")

    def _perform_click(self, action):
        """执行点击动作"""
        # 移动鼠标
        pyautogui.moveTo(action["x"], action["y"], duration=0.1 / self.playback_speed)

        # 执行点击
        if action["button"] == "left":
            pyautogui.click()
        elif action["button"] == "right":
            pyautogui.rightClick()
        elif action["button"] == "middle":
            pyautogui.middleClick()

    def _perform_keyboard(self, action):
        """执行键盘动作"""
        # 处理特殊键
        if action["input"] == "enter":
            pyautogui.press("enter")
        elif action["input"] == "tab":
            pyautogui.press("tab")
        elif action["input"] == "space":
            pyautogui.press("space")
        elif action["input"] == "backspace":
            pyautogui.press("backspace")
        elif action["input"] == "escape":
            pyautogui.press("esc")
        # 处理组合键
        elif "+" in action["input"]:
            keys = action["input"].split("+")
            pyautogui.hotkey(*keys)
        # 处理普通文本
        else:
            pyautogui.write(action["input"])

    def stop_playback(self):
        """停止回放"""
        if self.is_playing:
            print("正在停止回放...")
            self.is_playing = False

    def on_move(self, x, y):
        """更新鼠标位置"""
        self.current_position = (x, y)

    def on_click(self, x, y, button, pressed):
        """处理鼠标点击事件"""
        # 只处理按下事件
        if not pressed or not self.is_recording:
            return

        # 记录点击位置
        button_name = str(button).split('.')[-1]
        print(f"记录点击: ({x}, {y}) - {button_name}")

        # 添加延迟（如果之前有动作）
        delay = 0
        if self.recorded_actions:
            last_time = self.recorded_actions[-1]["time"]
            delay = time.time() - last_time

        # 添加动作
        self.recorded_actions.append({
            "type": "click",
            "x": x,
            "y": y,
            "button": button_name,
            "delay": delay,
            "time": time.time()
        })

    def record_left_click(self):
        """记录当前鼠标位置的左键点击"""
        if not self.is_recording or not self.current_position:
            return

        x, y = self.current_position
        print(f"记录左键点击: ({x}, {y})")

        # 添加延迟
        delay = 0
        if self.recorded_actions:
            last_time = self.recorded_actions[-1]["time"]
            delay = time.time() - last_time

        # 添加动作
        self.recorded_actions.append({
            "type": "click",
            "x": x,
            "y": y,
            "button": "left",
            "delay": delay,
            "time": time.time()
        })

    def record_right_click(self):
        """记录当前鼠标位置的右键点击"""
        if not self.is_recording or not self.current_position:
            return

        x, y = self.current_position
        print(f"记录右键点击: ({x}, {y})")

        # 添加延迟
        delay = 0
        if self.recorded_actions:
            last_time = self.recorded_actions[-1]["time"]
            delay = time.time() - last_time

        # 添加动作
        self.recorded_actions.append({
            "type": "click",
            "x": x,
            "y": y,
            "button": "right",
            "delay": delay,
            "time": time.time()
        })

    def record_middle_click(self):
        """记录当前鼠标位置的中键点击"""
        if not self.is_recording or not self.current_position:
            return

        x, y = self.current_position
        print(f"记录中键点击: ({x}, {y})")

        # 添加延迟
        delay = 0
        if self.recorded_actions:
            last_time = self.recorded_actions[-1]["time"]
            delay = time.time() - last_time

        # 添加动作
        self.recorded_actions.append({
            "type": "click",
            "x": x,
            "y": y,
            "button": "middle",
            "delay": delay,
            "time": time.time()
        })

    def record_keyboard_input(self):
        """记录键盘输入"""
        if not self.is_recording:
            return

        # 获取用户输入
        input_text = input("请输入要记录的键盘输入 (或按Enter跳过): ")
        if not input_text:
            return

        print(f"记录键盘输入: {input_text}")

        # 添加延迟
        delay = 0
        if self.recorded_actions:
            last_time = self.recorded_actions[-1]["time"]
            delay = time.time() - last_time

        # 添加动作
        self.recorded_actions.append({
            "type": "keyboard",
            "input": input_text,
            "delay": delay,
            "time": time.time()
        })

    def display_position(self):
        """实时显示鼠标位置"""
        try:
            while self.is_recording:
                if self.current_position:
                    x, y = self.current_position
                    print(f"\033[K当前鼠标位置: ({x}, {y})", end='\r')
                time.sleep(0.05)
            print("\n")  # 清除最后的位置显示
        except Exception as e:
            print(f"位置显示错误: {e}")

# 使用示例
if __name__ == "__main__":
    automator = ScreenAutomator()

    # 简单命令行界面
    while True:
        print("\n===== 屏幕操作自动化工具 =====")
        print("1. 开始录制操作")
        print("2. 停止录制操作")
        print("3. 保存录制操作")
        print("4. 回放录制操作")
        print("5. 设置回放速度")
        print("6. 设置回放次数")
        print("7. 停止回放")
        print("8. 退出")

        choice = input("请选择操作: ")

        if choice == "1":
            automator.start_recording()
        elif choice == "2":
            automator.stop_recording()
        elif choice == "3":
            filename = input("输入保存文件名 (留空使用默认): ")
            if filename:
                automator.save_recording(filename)
            else:
                automator.save_recording()
        elif choice == "4":
            filename = input("输入要回放的文件名 (留空使用最新): ")
            speed = float(input("回放速度 (例如 1.0): ") or "1.0")
            count = int(input("回放次数 (例如 1): ") or "1")
            automator.play_recording(filename if filename else None, speed, count)
        elif choice == "5":
            speed = float(input("设置回放速度 (例如 0.5 慢速, 2.0 快速): ") or "1.0")
            automator.playback_speed = speed
            print(f"已设置回放速度为: {speed}x")
        elif choice == "6":
            count = int(input("设置回放次数: ") or "1")
            automator.playback_count = count
            print(f"已设置回放次数为: {count}次")
        elif choice == "7":
            automator.stop_playback()
        elif choice == "8":
            if automator.is_playing:
                automator.stop_playback()
            break
        else:
            print("无效选择，请重新输入")