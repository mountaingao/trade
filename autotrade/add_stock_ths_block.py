import pyautogui
import time
import json
import os

# 新增代码：读取配置文件
config_path = os.path.join(os.path.dirname(__file__), '..', 'config', 'config.json')
with open(config_path, 'r', encoding='utf-8') as config_file:  # 修改编码为utf-8
    config = json.load(config_file)

ths_positon = config['ths_positon']

# 假设它们的图标在桌面上，你可以通过图标位置来启动
# pyautogui.click(x=1076, y=1411)  # 修改坐标以匹配你的桌面图标位置
# time.sleep(2)  # 等待软件启动
def start_open_ths(stock_codes):
    # 假设你知道应用窗口的某个特定图标的位置
    try:
        icon_location = pyautogui.locateOnScreen('../image/ths_icon.png')
        if icon_location:
            print("应用已经打开")
        else:
            print("应用未打开")
            # 切换到软件窗口（如果它在后台）300001
            pyautogui.hotkey('alt', 'z')
    except pyautogui.ImageNotFoundException:
        print("未找到应用图标，请检查路径或图标是否匹配")
        # 切换到软件窗口（如果它在后台）
        pyautogui.hotkey('alt', 'z')


    time.sleep(2)
    print(ths_positon)
    pyautogui.click(x=ths_positon['x'], y=ths_positon['y'])  # 修改坐标以匹配你的桌面图标位置
    time.sleep(2)  # 等待软件启动

def add_stocks_to_ths_block(stock_codes):
    if not stock_codes:
        print("没有要添加的股票代码")
        return
    print(stock_codes)
    start_open_ths(stock_codes)
    add_stocks_to_block(stock_codes)
    pyautogui.hotkey('alt', 'z')

def add_stocks_to_block(stock_codes):
    for stock_code in stock_codes:
        pyautogui.typewrite(stock_code)
        time.sleep(1)
        pyautogui.hotkey('enter')  # 切换到窗口
        time.sleep(1)
        pyautogui.click(x=ths_positon['c_x'], y=ths_positon['c_y'])  # 修改坐标以匹配你的桌面图标位置
        time.sleep(1)
        pyautogui.hotkey('insert')  # 加入
        time.sleep(1)



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

    # 启动同花顺或通达信
    time.sleep(3)
    x, y = pyautogui.position()
    print(x, y)
    # 屏幕
    x, y = pyautogui.size()
    print(x, y)



    # 实时显示鼠标位置
    # pyautogui.displayMousePosition()
    # 使用示例
    stock_codes = ['600697', '300001']
    add_stocks_to_ths_block(stock_codes)


