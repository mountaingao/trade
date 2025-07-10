import pyautogui
import time
import json
import os
import pandas as pd


# 新增代码：读取配置文件
config_path = os.path.join(os.path.dirname(__file__), '../../', 'config', 'config.json')
with open(config_path, 'r', encoding='utf-8') as config_file:  # 修改编码为utf-8
    config = json.load(config_file)

ths_positon = config['ths_positon']

# 假设它们的图标在桌面上，你可以通过图标位置来启动
# pyautogui.click(x=1076, y=1411)  # 修改坐标以匹配你的桌面图标位置
# time.sleep(2)  # 等待软件启动
def start_open_ths(stock_codes):
    # 假设你知道应用窗口的某个特定图标的位置
    try:
        icon_location = pyautogui.locateOnScreen('../../image/ths_icon.png')
        if icon_location:
            print("应用已经打开")
        else:
            print("应用未打开")
            # 切换到软件窗口（如果它在后台）300001
            # pyautogui.hotkey('alt', 'z')
    except pyautogui.ImageNotFoundException:
        print("未找到应用图标，请检查路径或图标是否匹配")
        # 切换到软件窗口（如果它在后台）
        # pyautogui.hotkey('alt', 'z')


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

    add_stocks_to_ths_zxg(stock_codes)
    # pyautogui.hotkey('alt', 'z')

def add_stocks_to_ths_zxg(stock_codes):
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

def export_stock_data(stock):
    # 使用全局配置的坐标
    global ths_positon
    
    # 屏幕分辨率检测
    x, y = pyautogui.size()
    print(x, y)
    
    # 如果配置中没有当前分辨率的设置，使用默认值并更新配置
    resolution_key = f"{x}x{y}"
    if resolution_key not in ths_positon:
        print(f"未找到{resolution_key}分辨率的配置，使用默认值")
        ths_positon[resolution_key] = {
            'self_selection': {'x': 195, 'y': 705},
            'content_panel': {'x': 320, 'y': 129},
            'menu_item1': {'x': 477, 'y': 530},
            'menu_item2': {'x': 638, 'y': 530},
            'dialog_confirm': {'x': 1178, 'y': 374},
            'select_button': {'x': 1449, 'y': 705},
            'save_button': {'x': 1086, 'y': 723}
        }
        # 更新配置文件
        config['ths_positon'] = ths_positon
        with open(config_path, 'w', encoding='utf-8') as config_file:
            json.dump(config, config_file, indent=4, ensure_ascii=False)
    
    # 获取当前分辨率的配置
    res_config = ths_positon[resolution_key]
    
    # 点击屏幕上的自选股坐标2507090030
    pyautogui.click(res_config['self_selection']['x'], res_config['self_selection']['y'])
    time.sleep(1)

    # 点击内容板块
    pyautogui.click(res_config['content_panel']['x'], res_config['content_panel']['y'])
    time.sleep(0.5)
    
    # 右键弹出菜单
    pyautogui.rightClick(res_config['content_panel']['x'], res_config['content_panel']['y'])
    time.sleep(1)
    
    # 滑动鼠标
    pyautogui.moveTo(res_config['menu_item1']['x'], res_config['menu_item1']['y'], duration=0.5)
    time.sleep(1)
    
    pyautogui.moveTo(res_config['menu_item2']['x'], res_config['menu_item2']['y'], duration=0.5)
    time.sleep(1)

    pyautogui.click(res_config['menu_item2']['x'], res_config['menu_item2']['y'])
    # 弹出对话框
    pyautogui.click(res_config['dialog_confirm']['x'], res_config['dialog_confirm']['y'])
    time.sleep(0.5)
    # 输入文件名
    filename= pd.Timestamp.now().strftime("%y%m%d%H%M")
    print(filename)
    pyautogui.typewrite(filename)

    pyautogui.click(res_config['select_button']['x'], res_config['select_button']['y'])

    pyautogui.click(res_config['save_button']['x'], res_config['save_button']['y'])
    time.sleep(0.5)

    pyautogui.click(res_config['save_button']['x'], res_config['save_button']['y'])
    time.sleep(0.5)

    pyautogui.click(res_config['save_button']['x'], res_config['save_button']['y'])

    return filename+'.xls'




# stock_data = export_stock_data(stock)
    # return stock_data

def main():
    print(pyautogui.position())  # 返回当前鼠标位置的坐标 (x, y)
    print(pyautogui.size())
    # 登录同花顺
    login_to_tonghuashun()
    time.sleep(10)

    # 下单示例
    buy_stock('600000', 10.0, 100)  # 买入股票代码为600000，价格为10.0，数量为100

if __name__ == '__main__':

    # 收盘执行选股，导出数据，保存数据，并获取关键数据
    # 1、启动通达信
    # 启动同花顺或通达信
    pyautogui.FAILSAFE = True
    time.sleep(3)
    x, y = pyautogui.position()
    print(x, y)
    # 屏幕
    x, y = pyautogui.size()
    print(x, y)

    # 2、执行选股步骤，填写导出文件目录
    ctrl_t = pyautogui.hotkey('ctrl', 't')
    time.sleep(0.5)
    # 点击加入条件
    pyautogui.click('ctrl', 't')
    # 点击选股入板块
    pyautogui.click('ctrl', 't')
    time.sleep(0.2)
    # 点击新建板块
    pyautogui.click('ctrl', 't')
    time.sleep(0.2)
    # 输入文件名
    blockname = pd.Timestamp.now().strftime("%m%d%H%M")
    print(blockname)
    pyautogui.typewrite(blockname)
    # 点击确定
    pyautogui.click('ctrl', 't')
    time.sleep(0.5)
    # 再点击确定，等待结果
    pyautogui.click('ctrl', 't')
    # 等10s 或者等待确认键
    time.sleep(10)

    # 导出当前文件内容  tdx 导出功能
    pyautogui.typewrite('34')
    # 选择中间项-所有数据
    pyautogui.click('ctrl', 't')
    # 点击浏览按钮
    pyautogui.click('ctrl', 't')
    # 输入文件名
    pyautogui.typewrite(blockname)
    # 点击确定
    pyautogui.click('ctrl', 't')
    time.sleep(0.5)
    # 取消 不打开
    pyautogui.click('ctrl', 't')
    # 点击右键
    pyautogui.rightClick('ctrl', 't')
    # 移动鼠标
    pyautogui.moveTo(x, y)
    pyautogui.click('ctrl', 't') # 点击
    time.sleep(1)
    # 继续执行导出 tdx 导出功能

    # 合并数据

# 3、保存文件到临时文件夹，可以通过某个按钮开始执行下面的动作

    # 4、打开同花顺，创建新的板块，并导入临时文件夹中的文件，可以手工操作，也可以通过代码实现

    # 5、打开同花顺，导出实时数据，保存下来

    # 6、 分析和整合数据，生成需要的数据内容

    # 7、调用模型，并预测结果，将结果输出到文件中，并返回合适的结果

    stock_codes = ['300123']

    add_stocks_to_ths_block(stock_codes)


    # 得到预警股票，综合评价所有数据，给出结果
    export_file = export_stock_data(stock_codes)
    file = "F:/stock/data/"+export_file

    # 读取数据，得到需要的那条数据
    # file = "F:/stock/data/2507082357.xls"

    df = pd.read_csv(file,encoding='GBK',sep='\t')
    # 修改：将'代码'列的值截取后6位
    # df['代码'] = df['代码'].astype(str).str[-6:]
    
    print(df)
    print(df.columns)
    # 获取字段”代码“ 为stock_code的行
    row = df[df['代码'].astype(str).str[-6:] == '688291']
    print(row)
    print(row['量比'])
    print(row['主力净额'])
    print(row['净流入'])
    print(row['金额'])
    print(row['换手(实)'])
    print(row['主力净量'])


   # 根据数据调用大模型，得到结果  弹窗提示
# 导出数据，保存数据，并获取关键数据



