import pyautogui
import time

import pyautogui
import time

# 启动同花顺或通达信
# 假设它们的图标在桌面上，你可以通过图标位置来启动
pyautogui.click(x=100, y=100)  # 修改坐标以匹配你的桌面图标位置
time.sleep(2)  # 等待软件启动

# 切换到软件窗口（如果它在后台）
pyautogui.hotkey('alt', 'z')

time.sleep(1)
pyautogui.hotkey('alt', 'tab')
# pyautogui.press('tab')  # 选择同花顺或通达信窗口
time.sleep(1)
pyautogui.typewrite('000001')
# pyautogui.hotkey('enter')  # 切换到窗口

# 打开一个股票的K线图
# 假设快捷键是Ctrl+K，你可以通过查找快捷键或使用鼠标点击来实现
# pyautogui.hotkey('ctrl', 'k')
time.sleep(2)

# 执行其他操作，例如切换到另一个股票的K线图
# 这里你可以使用鼠标点击或键盘快捷键来切换股票
# pyautogui.click(x=500, y=300)  # 示例点击位置
exit()

# print(pyautogui.displayMousePosition())
# 等待同花顺客户端打开
# time.sleep(5)

pyautogui.hotkey('alt', 'z')  # 关闭当前窗口（如果需要）

# 1. 点击股票代码输入框
# 假设股票代码输入框的坐标为 (x1, y1)
x1, y1 = 100, 150
pyautogui.click(x1, y1)

# 2. 输入股票代码
# 假设要添加的股票代码为 '000001'
pyautogui.typewrite('000001')

# 3. 按下回车键
pyautogui.press('enter')

# 4. 等待股票信息加载
time.sleep(2)

# 5. 点击自定义板块
# 假设自定义板块的坐标为 (x2, y2)
x2, y2 = 200, 250
pyautogui.click(x2, y2)

# 6. 点击添加到自定义板块的按钮
# 假设添加到自定义板块的按钮坐标为 (x3, y3)
x3, y3 = 300, 350
pyautogui.click(x3, y3)

# 7. 等待操作完成
time.sleep(1)

exit()


def login_to_tonghuashun():
    # 打开同花顺客户端
    pyautogui.hotkey('winleft', 'd')  # 切换到桌面
    time.sleep(1)
    pyautogui.press('enter')  # 启动同花顺
    time.sleep(10)  # 等待同花顺加载完成

def buy_stock(stock_code, price, quantity):
    # 切换到交易界面
    pyautogui.hotkey('alt', 'f4')  # 关闭当前窗口（如果需要）
    time.sleep(1)
    pyautogui.hotkey('winleft', 'd')  # 切换到桌面
    time.sleep(1)
    pyautogui.press('enter')  # 进入同花顺交易界面
    time.sleep(5)

    # 输入股票代码
    pyautogui.click(100, 200)  # 点击股票代码输入框
    time.sleep(1)
    pyautogui.typewrite(stock_code)
    time.sleep(1)

    # 点击买入按钮
    pyautogui.click(300, 200)
    time.sleep(1)

    # 输入价格
    pyautogui.click(400, 200)  # 点击价格输入框
    time.sleep(1)
    pyautogui.typewrite(str(price))
    time.sleep(1)

    # 输入数量
    pyautogui.click(500, 200)  # 点击数量输入框
    time.sleep(1)
    pyautogui.typewrite(str(quantity))
    time.sleep(1)

    # 点击确认下单
    pyautogui.click(600, 200)
    time.sleep(1)

def main():
    print(pyautogui.position())  # 返回当前鼠标位置的坐标 (x, y)
    print(pyautogui.size())
    # 登录同花顺
    login_to_tonghuashun()
    time.sleep(10)

    # 下单示例
    buy_stock('600000', 10.0, 100)  # 买入股票代码为600000，价格为10.0，数量为100

if __name__ == '__main__':
    main()