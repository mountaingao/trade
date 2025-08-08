import pyautogui
import time
import json
import os
import pandas as pd
from model_xunlian_alert_1 import predictions_model_data_file,predictions_model_data
import keyboard
import numpy as np
import datetime  # 新增导入datetime模块
from tdx_mini_data import process_multiple_codes

# 新增代码：读取配置文件
config_path = os.path.join(os.path.dirname(__file__), '../../', 'config', 'config.json')
with open(config_path, 'r', encoding='utf-8') as config_file:  # 修改编码为utf-8
    config = json.load(config_file)

ths_positon = config['ths_positon']

STATUS_FILE = "../status/status.json"

def save_status(step_name, data=None):
    status = {"step": step_name, "data": data or {}}
    with open(STATUS_FILE, 'w', encoding='utf-8') as f:
        json.dump(status, f, ensure_ascii=False, indent=2)

def load_status():
    try:
        with open(STATUS_FILE, 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        return {"step": "start", "data": {}}

# 假设它们的图标在桌面上，你可以通过图标位置来启动
# pyautogui.click(x=1076, y=1411)  # 修改坐标以匹配你的桌面图标位置
# time.sleep(2)  # 等待软件启动

def wait_for_keypress(key='pause'):
    # insert 键


    print(f"请按下 '{key}' 键继续...")
    keyboard.wait(key)
    print("已检测到按键，继续执行...")

def manual_confirm():
    print("请确认操作已完成，按下任意键继续...")
    keyboard.read_event()  # 等待任意按键
    print("继续执行...")
# manual_confirm()
def start_open_ths(stock_codes):
    # 假设你知道应用窗口的某个特定图标的位置07141706
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

# todo  支持不同分辨率的点位参数 ；先完善创业板，下一步北京板单独方法，以及深圳和上海主板，建立不同的模型和数据来分别进行分析和建模，做到准确，有效，
# 完成自动下单，手工确认，自动化交易
# 通达信选股
def select_tdx_block_list():
    # 获取当前屏幕分辨率
    screen_width, screen_height = pyautogui.size()
    resolution_key = f"{screen_width}x{screen_height}"
    
    # 默认坐标配置
    default_tdx_positions = {
        '1920x1080': {
            'join_condition': (1251, 568),
            'select_to_block': (1431, 860),
            'new_block': (1497, 570),
            'confirm_new_block': (1250, 760),
            'confirm_selection': (1478, 815),
            'finish_selection': (1509, 884)
        },
        '2560x1440': {
            'join_condition': (1251, 568),
            'select_to_block': (1431, 860),
            'new_block': (1497, 570),
            'confirm_new_block': (1250, 760),
            'confirm_selection': (1478, 815),
            'finish_selection': (1509, 884)
        },
        '3840x2160': {
            'join_condition': (1881, 804),
            'select_to_block': (2169, 1323),
            'new_block': (2273, 809),
            'confirm_new_block': (1831, 1150),
            'confirm_selection': (2268, 1242),
            'finish_selection': (2320, 1371)
        }
    }
    
    # 如果当前分辨率没有配置，则使用默认1920x1080的配置并缩放
    if resolution_key not in default_tdx_positions:
        print(f"未找到{resolution_key}分辨率的配置，使用1920x1080配置并进行缩放")
        base_positions = default_tdx_positions['1920x1080']
        scale_x = screen_width / 1920
        scale_y = screen_height / 1080
        tdx_positions = {}
        for key, (x, y) in base_positions.items():
            tdx_positions[key] = (int(x * scale_x), int(y * scale_y))
    else:
        tdx_positions = default_tdx_positions[resolution_key]

# 记录点 1: (1251, 568) - Button.left - 15:38:28.446
# 记录点 2: (1431, 860) - Button.left - 15:38:30.269
# 记录点 3: (1497, 570) - Button.left - 15:38:32.246
# 记录点 4: (1250, 760) - Button.left - 15:38:37.950
# 记录点 5: (1478, 815) - Button.left - 15:38:39.415
# 记录点 6: (472, 1412) - Button.left - 15:38:43.030
# 记录点 1: (1509, 884) - Button.left - 16:05:21.826
#     time.sleep(5)
    # 2、执行选股步骤，填写导出文件目录07111646
    ctrl_t = pyautogui.hotkey('ctrl', 't')
    print("请选择指标...")
    print(tdx_positions)
    wait_for_keypress()

    time.sleep(0.5)
    # 点击加入条件
    pyautogui.click(*tdx_positions['join_condition'])
    # 点击选股入板块
    pyautogui.click(*tdx_positions['select_to_block'])
    time.sleep(0.2)
    # 点击新建板块
    pyautogui.click(*tdx_positions['new_block'])
    time.sleep(0.5)
    # 输入文件名
    blockname = pd.Timestamp.now().strftime("%m%d%H%M")
    print(blockname)
    pyautogui.typewrite(blockname)
    # 点击确定
    pyautogui.click(*tdx_positions['confirm_new_block'])
    time.sleep(0.5)
    # 再点击确定，等待结果
    pyautogui.click(*tdx_positions['confirm_selection'])
    # 等10s 或者等待确认键

    # time.sleep(15)
    # 修改为：
    print("请确认操作已完成，按回车键继续...")
    wait_for_keypress()
    pyautogui.click(*tdx_positions['finish_selection'])

    return blockname
# 通达信数据导出
def export_tdx_block_data(blockname):
    # 获取当前屏幕分辨率
    screen_width, screen_height = pyautogui.size()
    resolution_key = f"{screen_width}x{screen_height}"
    
    # 默认坐标配置
    default_export_positions = {
        '1920x1080': {
            'all_data': (1200, 738),
            'browse_button': (1462, 788),
            'save_button': (1999, 1106),
            'cancel_open': (1402, 828),
            'final_confirm': (1327, 748)
        },
        '2560x1440': {
            'all_data': (1200, 738),
            'browse_button': (1462, 788),
            'save_button': (1999, 1106),
            'cancel_open': (1402, 828),
            'final_confirm': (1327, 748)
        },
        '3840x2160': {
            'all_data': (1786, 1106),
            'browse_button': (2271, 1199),
            'save_button': (2758, 1440),
            'cancel_open': (2114, 1254),
            'final_confirm': (2009, 1124)
        }
    }
    
    # 如果当前分辨率没有配置，则使用默认1920x1080的配置并缩放
    if resolution_key not in default_export_positions:
        print(f"未找到{resolution_key}分辨率的配置，使用1920x1080配置并进行缩放")
        base_positions = default_export_positions['1920x1080']
        scale_x = screen_width / 1920
        scale_y = screen_height / 1080
        export_positions = {}
        for key, (x, y) in base_positions.items():
            export_positions[key] = (int(x * scale_x), int(y * scale_y))
    else:
        export_positions = default_export_positions[resolution_key]

    #
#     # 导出当前文件内容  tdx 导出功能
#     开始记录屏幕点击位置... (按F1停止)
# 记录点 1: (1197, 738) - Button.left - 15:51:20.091
# 记录点 2: (1462, 788) - Button.left - 15:51:22.172
# 记录点 3: (1999, 1106) - Button.left - 15:51:29.611
# 记录点 4: (1402, 828) - Button.left - 15:51:31.852
# 记录点 5: (1327, 748) - Button.left - 15:51:37.292
    wait_for_keypress()
    pyautogui.typewrite('34')
    # 回车
    pyautogui.press('enter')
    time.sleep(0.5)
    # 选择中间项-所有数据
    pyautogui.click(*export_positions['all_data'])
    time.sleep(0.5)
    # 点击浏览按钮
    pyautogui.click(*export_positions['browse_button'])
    time.sleep(0.5)
    # 输入文件名
    pyautogui.typewrite(blockname)
    # 点击确定
    pyautogui.click(*export_positions['save_button'])
    time.sleep(0.5)
    # 取消 不打开
    pyautogui.click(*export_positions['cancel_open'])
    # time.sleep(12)
    wait_for_keypress()
    print("请确认操作已完成，按键继续...")
    pyautogui.click(*export_positions['final_confirm'])
    time.sleep(0.3)
def tdx_change_history_list():
    # 获取当前屏幕分辨率
    screen_width, screen_height = pyautogui.size()
    resolution_key = f"{screen_width}x{screen_height}"
    
    # 默认坐标配置
    default_history_positions = {
        '1920x1080': {
            'right_click_area': (652, 597),
            'history_menu': (725, 779)
        },
        '2560x1440': {
            'right_click_area': (652, 597),
            'history_menu': (725, 779)
        },
        '3840x2160': {
            'right_click_area': (1131, 1168),
            'history_menu': (1358, 535)
        }
    }
    
    # 如果当前分辨率没有配置，则使用默认1920x1080的配置并缩放
    if resolution_key not in default_history_positions:
        print(f"未找到{resolution_key}分辨率的配置，使用1920x1080配置并进行缩放")
        base_positions = default_history_positions['1920x1080']
        scale_x = screen_width / 1920
        scale_y = screen_height / 1080
        history_positions = {}
        for key, (x, y) in base_positions.items():
            history_positions[key] = (int(x * scale_x), int(y * scale_y))
    else:
        history_positions = default_history_positions[resolution_key]

    # 右键切换到列表
    #     记录点 2: (1091, 1427) - Button.left - 15:47:30.412
    # 记录点 3: (652, 597) - Button.right - 15:47:33.380
    # 记录点 4: (725, 779) - Button.left - 15:47:37.909
    # 点击右键
    pyautogui.rightClick(*history_positions['right_click_area'])
    time.sleep(0.5)
    # 点击右键
    pyautogui.moveTo(*history_positions['history_menu'])
    # 点击历史
    pyautogui.click(*history_positions['history_menu'])
    time.sleep(0.5)
# 同花顺数据导出
def export_ths_block_data(blockname):
    # 获取当前屏幕分辨率
    screen_width, screen_height = pyautogui.size()
    resolution_key = f"{screen_width}x{screen_height}"
    
    # 默认坐标配置
    default_ths_export_positions = {
        '1920x1080': {
            'pos1': (290, 337),
            'pos2': (349, 56),
            'pos21': (13, 109),
            'pos22': (365, 215),
            'pos3': (440, 686),
            'pos4': (691, 684),
            'pos5': (539, 594),
            'pos6': (1513, 558),
            'pos7': (1210, 992),
            'pos8': (1971, 1092),
        },
        '2560x1440': {
            'pos1': (303, 337),
            # 'pos2': (243, 65), {上方}  下面是左上角坐标
            'pos2': (120, 59),
            'pos21': (19, 107),
            'pos22': (362, 217),
            'pos3': (444, 687),
            'pos4': (700, 697),
            'pos5': (1498, 561),
            'pos6': (1664, 435),
            'pos7': (1981, 1093),
            'pos8': (1407, 916)
        },
        '3840x2160': {
            'pos1': (525, 604),
            'pos2': (157, 107),
            'pos21': (25, 189),
            'pos22': (671, 349),
            'pos3': (860, 1171),
            'pos4': (1232, 1165),
            'pos5': (1002, 1215),
            'pos6': (2290, 791),
            'pos7': (2737, 1419),
            'pos8': (2170, 1414)
        }
    }
    
    # 如果当前分辨率没有配置，则使用默认1920x1080的配置并缩放
    if resolution_key not in default_ths_export_positions:
        print(f"未找到{resolution_key}分辨率的配置，使用1920x1080配置并进行缩放")
        base_positions = default_ths_export_positions['1920x1080']
        scale_x = screen_width / 1920
        scale_y = screen_height / 1080
        ths_export_positions = {}
        for key, (x, y) in base_positions.items():
            ths_export_positions[key] = (int(x * scale_x), int(y * scale_y))
    else:
        ths_export_positions = default_ths_export_positions[resolution_key]

    # 导出同花顺板块数据
# 记录点 1: (290, 337) - Button.left - 15:26:20.519
# 记录点 2: (346, 56) - Button.left - 15:26:23.799
# 记录点 3: (229, 209) - Button.right - 15:26:25.111
# 记录点 4: (350, 596) - Button.left - 15:26:27.367
# 记录点 5: (539, 594) - Button.left - 15:26:30.143
# 记录点 6: (1513, 558) - Button.left - 15:26:32.774
# 记录点 7: (1210, 992) - Button.left - 15:26:38.159
# 记录点 8: (1971, 1092) - Button.left - 15:26:49.263
# 记录点 9: (1407, 916) - Button.left - 15:26:50.838
# 记录点 10: (1407, 916) - Button.left - 15:26:51.526
# 记录点 11: (1407, 916) - Button.left - 15:26:53.095

# 3840x2160
# 记录点 1: (526, 604) - Button.left - 16:42:50.812
# 记录点 2: (127, 99) - Button.left - 16:42:52.944
# 记录点 3: (18, 193) - Button.left - 16:42:56.264
# 记录点 4: (482, 399) - Button.right - 16:43:00.355
# 记录点 5: (695, 780) - Button.left - 16:43:02.658
# 记录点 6: (936, 779) - Button.left - 16:43:05.088
# 记录点 7: (2307, 791) - Button.left - 16:43:08.024
# 记录点 8: (2713, 1417) - Button.left - 16:43:16.798
# 记录点 9: (2159, 1408) - Button.left - 16:43:18.606
# 记录点 10: (2159, 1408) - Button.left - 16:43:19.858
# 记录点 11: (2159, 1408) - Button.left - 16:43:21.840
# 创建同花顺板块
#     选择板块并排序
    pyautogui.click(*ths_export_positions['pos1'])
    time.sleep(0.5)
    pyautogui.moveTo(*ths_export_positions['pos2'])
    time.sleep(0.5)
    pyautogui.click(*ths_export_positions['pos2'])
    time.sleep(0.5)
    pyautogui.click(*ths_export_positions['pos21'])
    # 移动并右键
    time.sleep(0.5)
    pyautogui.click(*ths_export_positions['pos22'])
    time.sleep(0.5)
    pyautogui.rightClick(*ths_export_positions['pos22'])
    time.sleep(0.5)
    # wait_for_keypress()
    pyautogui.moveTo(*ths_export_positions['pos3'])
    time.sleep(0.5)
    pyautogui.click(*ths_export_positions['pos4'])
    time.sleep(0.5)
    pyautogui.click(*ths_export_positions['pos5'])
    # wait_for_keypress()
    time.sleep(0.5)
    pyautogui.click(*ths_export_positions['pos6'])
    time.sleep(0.5)
    pyautogui.typewrite(blockname)
    # 确定板块
    # wait_for_keypress()
    time.sleep(0.5)
    pyautogui.click(*ths_export_positions['pos7'])
    # pyautogui.click(*ths_export_positions['pos8'])
    time.sleep(0.5)
    pyautogui.click(*ths_export_positions['pos8'])
    time.sleep(0.5)
    pyautogui.click(*ths_export_positions['pos8'])
    time.sleep(0.5)
    pyautogui.click(*ths_export_positions['pos8'])
# 同花顺创建板块
def create_ths_block_from_file(blockname):
    # 获取当前屏幕分辨率
    screen_width, screen_height = pyautogui.size()
    resolution_key = f"{screen_width}x{screen_height}"
    
    # 默认坐标配置
    default_ths_create_positions = {
        '1920x1080': {
            'pos1': (289, 36),
            'pos2': (337, 316),
            'pos3': (1550, 531),
            'pos4': (1330, 752),
            'pos5': (1559, 657),
            'pos6': (1589, 684),
            'pos7': (2035, 1060),
            'pos8': (2026, 1102),
            'pos9': (1185, 713),
            'pos10': (1991, 1090),
            'pos11': (1381, 834),
            'pos12': (1468, 914)
        },
        '2560x1440': {
            'pos1': (289, 36),
            'pos2': (337, 316),
            'pos3': (1550, 531),
            'pos4': (1330, 752),
            'pos5': (1559, 657),
            'pos6': (1589, 684),
            'pos7': (2035, 1060),
            'pos8': (2026, 1102),
            'pos9': (1355, 568),
            'pos10': (1991, 1090),
            'pos11': (1381, 834),
            'pos12': (1468, 914)
        },
        '3840x2160': {
            'pos1': (508, 65),
            'pos2': (625, 545),
            'pos3': (2405, 738),
            'pos4': (1980, 1148),
            'pos5': (2400, 974),
            'pos6': (2403, 1010),
            'pos7': (2698, 1358),
            'pos8': (2713, 1425),
            # 'pos8': (2195, 763), 排序
            'pos9': (1782, 822),
            'pos10': (2717, 1417),
            'pos11': (2100, 1274),
            'pos12': (2240, 1418)
        }
    }
    
    # 如果当前分辨率没有配置，则使用默认1920x1080的配置并缩放
    if resolution_key not in default_ths_create_positions:
        print(f"未找到{resolution_key}分辨率的配置，使用1920x1080配置并进行缩放")
        base_positions = default_ths_create_positions['1920x1080']
        scale_x = screen_width / 1920
        scale_y = screen_height / 1080
        ths_create_positions = {}
        for key, (x, y) in base_positions.items():
            ths_create_positions[key] = (int(x * scale_x), int(y * scale_y))
    else:
        ths_create_positions = default_ths_create_positions[resolution_key]

    #导入同花顺板块文件
#     开始记录屏幕点击位置... (按F1停止)
# 记录点 1: (289, 36) - Button.left - 17:27:28.711
# 记录点 2: (337, 316) - Button.left - 17:27:31.062
# 记录点 3: (1550, 531) - Button.left - 17:27:34.727
# 记录点 4: (1330, 752) - Button.left - 17:27:44.335
# 记录点 5: (1559, 657) - Button.left - 17:27:47.559
# 记录点 6: (1589, 684) - Button.left - 17:27:50.391
# 记录点 7: (2035, 1060) - Button.left - 17:27:53.903
# 记录点 8: (2026, 1102) - Button.left - 17:27:55.582
# 记录点 9: (1185, 713) - Button.left - 17:28:08.014
# 记录点 10: (1991, 1090) - Button.left - 17:28:09.911
# 记录点 11: (1381, 834) - Button.left - 17:28:12.958
# 记录点 12: (1468, 914) - Button.left - 17:28:16.830
# 记录已停止
#     创建板块
    pyautogui.click(*ths_create_positions['pos1'])
    time.sleep(0.5)
    pyautogui.click(*ths_create_positions['pos2'])
    time.sleep(0.5)
    pyautogui.click(*ths_create_positions['pos3'])
    time.sleep(0.5)
    # 输入文件名
    pyautogui.typewrite(blockname)
    time.sleep(0.5)
    pyautogui.click(*ths_create_positions['pos4'])
    time.sleep(0.5)
    # 导入数据
    pyautogui.click(*ths_create_positions['pos5'])
    time.sleep(0.5)
    pyautogui.moveTo(*ths_create_positions['pos6'])
    pyautogui.click(*ths_create_positions['pos6'])
    time.sleep(0.5)
    pyautogui.click(*ths_create_positions['pos7'])
    time.sleep(0.5)
    pyautogui.moveTo(*ths_create_positions['pos8'])
    time.sleep(0.5)
    pyautogui.click(*ths_create_positions['pos8'])
    time.sleep(0.5)
    # 选择文件
    pyautogui.click(*ths_create_positions['pos9'])
    time.sleep(0.5)
    pyautogui.click(*ths_create_positions['pos10'])
    time.sleep(0.5)
    pyautogui.click(*ths_create_positions['pos11'])
    time.sleep(0.5)
    pyautogui.click(*ths_create_positions['pos12'])
    time.sleep(0.5)
# stock_data = export_stock_data(stock)
    # return stock_data


# 通达信数据处理
def tdx_get_block_data():
    # 2、执行选股步骤，填写导出文件目录
    blockname = select_tdx_block_list()

    # 可以先选择导出数据
    export_tdx_block_data(blockname)
    print("导出数据已完成！")
    #右键选择历史数据
    tdx_change_history_list()
    # 避免中文输入法的问题
    blockname_01 = blockname+'001'
    # 导出数据
    export_tdx_block_data(blockname_01)
    print("导出历史数据已完成！")

    # 合并数据
    tdx_merge_data(blockname,blockname_01)
    return blockname

def tdx_merge_data(blockname,blockname_01):
    # 读取文件，代码写入到 txt中
    file1 = "../data/tdx/"+blockname+'.xls'
    # file1 = "../data/tdx/07111619.xls"

    data = pd.read_csv(file1,encoding='GBK',sep='\t')
    # 去掉最后一行数据
    data = data[:-1]
    # 取列的名称

    # 去掉列名中的括号和数字
    data.columns = data.columns.str.replace(r'\(\d+\)', '', regex=True)

    columns = data.columns
    # print( columns)
    # 提取有用的字段并准备和另一个进行合并 代码	名称(83)	涨幅%	现价	最高	量比	总金额	细分行业
    new_data = data[['代码', '名称', '涨幅%', '现价', '最高', '量比', '总金额', '细分行业']]
    # print( new_data.head(100))

    # new_data['代码']  = new_data['代码'].str.split('=').str[1].str.replace('"', "")

    file2 = "../data/tdx/"+blockname_01+'.xls'
    # file2 = "../data/tdx/0711161920250711.xls"
    # 读取数据去掉第一行，第二行是列名
    data_02 = pd.read_csv(file2,encoding='GBK',sep='\t',header=1)
    data_02 = data_02[:-1]
    data_02.columns = data_02.columns.str.replace(' ', '')
    data_02['代码']  = data_02['代码'].str.split('=').str[1].str.replace('"', "")

    data_02.sort_values(by=['代码'], inplace=True)
    data_02.reset_index(drop=True, inplace=True)

    print( data_02.head(10))
    # 合并new_data和data_02的数据，按照T列进行合并
    # data_02 按字段 代码 排序
    # data_02 = data_02.sort_values(by=['代码'])
    # print(  data_02[['T', 'T1']])
    # data_t = data_02[['代码','T', 'T1']].copy()
    # print("排序前:")
    # print(data_t.head())
    #
    # # 按'代码'列排序并重置索引
    # data_t.sort_values(by='代码', inplace=True)
    # data_t.reset_index(drop=True, inplace=True)
    #
    # print("\n排序后:")
    # print(data_t.head())
    #
    # print(  data_t)
    # 合并时按照排序后的值进行合并 删除索引后重建

    # 按照 代码 进行合并 Q	Q_1	Q3
    # merged_data = pd.merge(new_data, data_t, left_on='代码', right_on='T')
    merged_data = pd.merge(new_data, data_02[['T', 'Q', 'Q_1','Q3']], left_index=True, right_index=True)
    # 在合并前处理代码格式
    # new_data['代码'] = new_data['代码'].str.split('=').str[1].str.replace('"', "")
    # data_02['代码'] = data_02['代码'].astype(str)  # 确保类型一致
    # 替代当前合并方式
    # merged_data = pd.merge(
    #     new_data,
    #     data_t[['T', 'T1']],
    #     left_on='代码',
    #     right_on='代码',  # 确保列名一致
    #     how='left'
    # )
    # merged_data = pd.merge(new_data, data_02[['T', 'T1']], left_index=True, right_index=True)

    # print( merged_data.head(100))
    merged_data.insert(len(merged_data.columns), '是否领涨', '否')

    # merged_data['代码']  = merged_data['代码'].str.split('=').str[1].str.replace("'", "")
    # 去掉双引号
    merged_data['代码']  = merged_data['代码'].str.split('=').str[1].str.replace('"', "")
    # 保存中间文件，供后续使用
    merged_data.to_excel(f"../data/tdx/{blockname}_data.xlsx", index=False)

    # 取出代码字段，并且去掉等号和引号
    stock_codes = merged_data['代码']

    # 写入 txt 文件，每行一个代码# 3、保存文件到临时文件夹，可以通过某个按钮开始执行下面的动作
    with open(f"../data/ths/{blockname}.txt", "w", encoding="utf-8") as f:
        for code in stock_codes:
            f.write(code + "\n")

    # 给个提示，打开同花顺，按f1键开始执行下面的动作
    # print("请打开同花顺，按F1键开始执行下面的动作")
    # time.sleep(5)
    # input("请按回车键继续...")
    return blockname


# 同花顺数据处理
def ths_get_block_data(blockname):
    # ths导入
    wait_for_keypress()
    # 4、打开同花顺，创建新的板块，并导入临时文件夹中的文件，可以手工操作，也可以通过代码实现
    create_ths_block_from_file(blockname)
    print("同花顺板块创建已完成！")

    wait_for_keypress()
    # 5、打开同花顺，导出实时数据，保存下来
    export_ths_block_data(blockname)
    print("同花顺数据导出已完成！")

def get_minite_band_width(df):
    return process_multiple_codes(df)
# 合并两个文件的数据，并返回需要的格式的数据
def merge_block_data(blockname):
    # 判断两个文件是否存在
    if not os.path.exists("../data/tdx/"+blockname+'_data.xlsx'):
        print("../data/tdx/"+blockname+'_data.xlsx'+" 文件不存在")
        return None
    if not os.path.exists("../data/ths/"+blockname+'.xls'):
        print("../data/ths/"+blockname+'.xls'+" 文件不存在")
        return None

    tdx_data = pd.read_excel("../data/tdx/"+blockname+'_data.xlsx')
    codes = tdx_data['代码']
    # 转为数组
    codes = codes.to_numpy()
    # 获取5分钟数据
    minite_data = get_minite_band_width( codes)
    minite_data = pd.DataFrame(minite_data)
    # minite_data 和 tdx_data 合并 根据代码和code 字段进行，或者顺序直接合并
    tdx_data = pd.merge(tdx_data, minite_data, left_index=True, right_index=True)
    print(tdx_data.head(5))


    print( tdx_data.head(5))

    ths_data = pd.read_csv("../data/ths/"+blockname+'.xls',encoding='GBK',sep='\t')
    # 如果 '净额',  '净量' 字段不存在，则将  '主力净额',  '主力净量'更改为  '净额',  '净量'
    if '主力净额' in ths_data.columns:
        ths_data['净额'] = ths_data['主力净额']
    # 将   '主力净量'更改为   '净量'
    if '主力净量' in ths_data.columns:
        ths_data['净量'] = ths_data['主力净量']

    data = pd.merge(tdx_data, ths_data[['净额', '净流入', '净量']], left_index=True, right_index=True)
    print( data.head(10))
    # 打印 columns
    # print(data.columns)
    # 将columns 按照自己的需求进行排序
    # '代码', '名称', '涨幅%', '现价', '最高', '量比', '总金额', '细分行业', 'T', 'T1', '净额','净流入', '净量'
    # 在data最前面增加两个字段：序号	日期
    data.insert(0, '序号', range(1, len(data) + 1))
    data.insert(1, '日期', pd.Timestamp.now().strftime("%Y-%m-%d"))

    # 后面增加字段 '次日涨幅', '次日最高价', '次日最高涨幅', '概念', '说明','是否领涨', '预测', '是否成功', '最高价', 'AI预测', 'AI幅度', '重合'，值均为空
    data.insert(len(data.columns), '次日涨幅', '')
    data.insert(len(data.columns), '次日最高价', '')
    data.insert(len(data.columns), '次日最高涨幅', '')
    data.insert(len(data.columns), '概念', '')
    data.insert(len(data.columns), '说明', '')

    data.insert(len(data.columns), '预测', '')
    data.insert(len(data.columns), '是否成功', '')

    # 将字段 T更名为信号天数 净量更名为 当日资金流入
    data = data.rename(columns={'T': '信号天数', '净量': '当日资金流入', '涨幅%': '当日涨幅', '最高': '最高价'})
    # 预测的值 = =IF(OR(AND(M1104<=3,P1104>0.2),P1104>2),"是","否")  当 （信号天数 <=3 and 净量 > 0.2） or 净量 > 2 时为是；其余为否

    # 07170948 把这个时间加上年 和后面的秒 如 20250717084800  然后转换成time 传给函数
    # 获取当前年份（动态）
    current_year = datetime.datetime.now().year  # 例如 2025

    # 组合完整时间字符串
    full_time_str = f"{current_year}{blockname}00"  # 添加年份和秒数

    print('full_time_str' +full_time_str)
    # 转换为 datetime 对象
    dt = datetime.datetime.strptime(full_time_str, "%Y%m%d%H%M%S")

    hour = get_time_directory(dt.time())

    # 确保关键列的数据类型正确
    numeric_columns = ['信号天数', '当日资金流入']

    for col in numeric_columns:
        if col in data.columns:
            # 转换为数值类型，无法转换的设置为 NaN
            data[col] = pd.to_numeric(data[col], errors='coerce')
            # 可选：用0填充NaN值
            data[col] = data[col].fillna(0)

    # print('hour' +hour)
    if hour == 1000:    # 早盘预测 放量滞涨为否，当日资金流入大于0，需要重点关注后续走势，前面是阴线的有爆发力 0717
        data['预测'] = np.where(
            ((data['信号天数'] <= 10) & (data['当日资金流入'] > 0)) | (data['当日资金流入'] > 2), "是", "否")
    elif hour == 1200:    # 中午预测 基本要看头部的几个票
        data['预测'] = np.where(
            ((data['信号天数'] <= 3) & (data['当日资金流入'] > 0.2)) | (data['当日资金流入'] > 2), "是", "否")
    else:
        data['预测'] = np.where(
            ((data['信号天数'] <= 3) & (data['当日资金流入'] > 0.2)) | (data['当日资金流入'] > 2), "是", "否")

    # 下午预测 老票不会有太多机会，看低位，或者冲高后横盘 # 收盘预测和下午一致  ，关注新出现的标的



    # 序号	日期	代码	名称	当日涨幅	现价	细分行业	次日涨幅	次日最高价	次日最高涨幅	概念	说明	信号天数	净额	净流入	当日资金流入	是否领涨	预测	是否成功	最高价	AI预测	AI幅度	重合
    # data = data[['序号', '日期', '代码', '名称', '当日涨幅', '现价', '细分行业', '次日涨幅', '次日最高价', '次日最高涨幅', '概念', '说明', '信号天数', '净额', '净流入', '当日资金流入', '是否领涨', '预测', '是否成功', '最高价', 'AI预测', 'AI幅度', '重合']]
    # 没有的字段均为空置
    # 合并数据 净额	净流入	换手(实)	总金额	净量
    # merged_data = pd.merge(merged_data, data_03[['净额', '净流入', '净量']], left_index=True, right_index=True)
    # print( merged_data.head(100))
    # 增加最高价字段
    # data['最高价'] = data['最高'].std.replace(',', '').astype(float)
    # print(data.columns)
    # 保存文件
    # result_df.to_excel(output_file, index=False)
    print( data.head(10))
    data.to_excel(f"../alert/{blockname}.xlsx", index=False)
    return data

# 新增函数：根据当前时间获取目录名
def get_time_directory(now=datetime.datetime.now().time()):
    # now = datetime.datetime.now().time()
    if datetime.time(9, 30) <= now <= datetime.time(11, 0):
        return '1000'
    elif datetime.time(11, 30) <= now <= datetime.time(13, 30):
        return '1200'
    elif datetime.time(13, 30) <= now <= datetime.time(15, 0):
        return '1400'
    elif datetime.time(15, 0) <= now <= datetime.time(22, 30):
        return '1600'
    else:
        return '250721'

# 推理模型
def predict_block_data(blockname,date='250721'):
    # 7、调用模型，并预测结果，将结果输出到文件中，并返回合适的结果
    # 使用多个数据集训练并生成模型 , 需要按照日期更新到最新的模型
    model = {
        'reg_weights': f"../models/{date}_feature_weights_reg.csv",
        'clf_weights': f"../models/{date}_feature_weights_clf.csv",
        'reg_model': f"../models/{date}_model_reg.json",
        'clf_model': f"../models/{date}_model_clf.json"}

    print( model)
    #预测文件中的数据
    # 按照时间确定目录名
    time_dir = get_time_directory()  # 获取时间目录
    target_dir = f"../data/predictions/{time_dir}"  # 目标目录路径
    os.makedirs(target_dir, exist_ok=True)  # 创建目录（如果不存在）
    # output_file = f'../data/predictions/{file_root}_{pd.Timestamp.now().strftime("%H%M")}{file_ext}'
    #预测文件中的数据
    predictions_file = predictions_model_data_file(f"../alert/{blockname}.xlsx",model,target_dir)
    predictions_data = pd.read_excel(predictions_file)
    # 将字段 T更名为信号天数 净量更名为 当日资金流入
    data = predictions_data.rename(columns={'分类预测_y_pred_clf': 'AI预测', '回归预测_y_pred_reg': 'AI幅度'})

    data['重合'] = np.where(
        (data['预测'] == "是") & (data['AI预测'] == 1), "1", "0")

    # 打印数据 重合值为1的数据 预测 为是或 AI预测 为1 的 数据

    # print(data[data['重合'] == '1'] | data[data['AI预测'] == 1] | data[data['预测'] == 1])

    print(data[data['重合'] == '1'])

    print(data[data['预测'] == '1'])

    print(data[data['AI预测'] == 1])

    # 保存文件
    data.to_excel(predictions_file, index=False)
    return predictions_file

def no_step_shoupan():
    x, y = pyautogui.position()
    print(x, y)
    # 屏幕分辨率
    x, y = pyautogui.size()
    print(x, y)
    wait_for_keypress()
    # 通达信数据获取，导出和保存
    blockname = tdx_get_block_data()
    # 同花顺数据获取，导出和保存
    ths_get_block_data(blockname)
    print("数据已经生成，请修改是否领涨字段的值，然后保存文件，然后按空格键继续...")

    #
    wait_for_keypress()
    # # 合并两个数据
    # # 6、 分析和整合数据，生成需要的数据内容
    # # new_data = data_03[['代码', '名称', '净额', '净流入', '净量']]2507111818
    file = merge_block_data(blockname)
    #07211414

    # model_name = get_time_directory()
    # # 计算结果，返回符合条件的股票代码
    # predict_block_data(blockname,model_name)
    predict_block_data(blockname)

def step_by_step_shoupan():
    status = load_status()
    current_step = status["step"]
    step_data = status["data"]
    # 收盘执行选股，导出数据，保存数据，并获取关键数据
    # 1、启动通达信
    # 启动同花顺或通达信
    pyautogui.FAILSAFE = True
    # # 增加一个按钮 按 esc 以后程序退出运行
    # pyautogui.hotkey('alt', 'esc')
    # print('按 esc 退出程序')
    # while True:
    #     if pyautogui.press('esc'):
    #         break
    #     time.sleep(1)
    #位置：
    x, y = pyautogui.position()
    print(x, y)
    # 屏幕分辨率
    x, y = pyautogui.size()
    print(x, y)
    wait_for_keypress()
    # 步骤1: 通达信数据处理
    if current_step in ["start", "tdx_get_block_data"]:
        blockname = step_data.get("blockname") or tdx_get_block_data()
        save_status("ths_get_block_data", {"blockname": blockname})
        # blockname = tdx_get_block_data()

    # 步骤2: 同花顺数据处理
    if current_step in ["ths_get_block_data", "merge_block_data"]:
        blockname = step_data["blockname"]
        ths_get_block_data(blockname)
        save_status("merge_block_data", {"blockname": blockname})

    wait_for_keypress()
    # 步骤3: 数据合并
    if current_step in ["merge_block_data", "predict_block_data"]:
        blockname = step_data["blockname"]
        merge_block_data(blockname)
        save_status("predict_block_data", {"blockname": blockname})

    wait_for_keypress()
    # 步骤4: 预测执行
    if current_step == "predict_block_data":
        blockname = step_data["blockname"]
        time_dir = get_time_directory()
        blockname = '07211543'

        predict_block_data(blockname, time_dir)
        save_status("completed")

def main():
    print(pyautogui.position())  # 返回当前鼠标位置的坐标 (x, y)
    print(pyautogui.size())
    # 登录同花顺
    login_to_tonghuashun()
    time.sleep(10)

    # 下单示例
    buy_stock('600000', 10.0, 100)  # 买入股票代码为600000，价格为10.0，数量为100

if __name__ == '__main__':

    no_step_shoupan()

    # step_by_step_shoupan()

# 导出数据，保存数据，并获取关键数据
# 每日运行4次：9：45 - 10：30 - 11：30 - 14：40 - 15：10
# 得到通达信的情绪温度数据
# 通达信实时数据 880005