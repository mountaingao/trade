import pyautogui
import time
import json
import os
import pandas as pd
from model_xunlian_alert_1 import predictions_model_data_file, predictions_model_data
import keyboard
from typing import Dict, List, Optional, Tuple

# 新增代码：读取配置文件
config_path = os.path.join(os.path.dirname(__file__), '../../', 'config', 'config.json')
with open(config_path, 'r', encoding='utf-8') as config_file:  # 修改编码为utf-8
    config = json.load(config_file)

# 新增ConfigManager类实现
class ConfigManager:
    def __init__(self, config: dict):
        self.config = config
        # 获取当前屏幕分辨率
        screen_width, screen_height = pyautogui.size()
        self.current_resolution = f"{screen_width}x{screen_height}"
    
    def get_coordinate(self, key: str) -> Tuple[int, int]:
        """根据键名获取坐标"""
        coordinates = self.config.get('coordinates', {})
        if key in coordinates:
            coord = coordinates[key]
            return coord[0], coord[1]
        # 如果找不到，返回(0,0)并警告
        print(f"Warning: Coordinate key '{key}' not found in config")
        return 0, 0
    
    def get_resolution_config(self) -> dict:
        """获取当前分辨率的配置"""
        resolutions = self.config.get('resolutions', {})
        # 尝试获取当前分辨率的配置
        if self.current_resolution in resolutions:
            return resolutions[self.current_resolution]
        # 如果当前分辨率不在配置中，尝试获取第一个配置
        if resolutions:
            first_res = next(iter(resolutions.values()))
            print(f"Warning: Resolution {self.current_resolution} not found, using first available")
            return first_res
        # 如果都没有，返回空字典
        print("Warning: No resolution config found")
        return {}

# 创建ConfigManager实例
config_manager = ConfigManager(config)
ths_positon = config['ths_positon']

# 假设它们的图标在桌面上，你可以通过图标位置来启动
# pyautogui.click(x=1076, y=1411)  # 修改坐标以匹配你的桌面图标位置
# time.sleep(2)  # 等待软件启动

def wait_for_keypress(key='insert'):
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

def add_stocks_to_ths_zxg(stock_codes: List[str]):
    """添加股票到自选股板块"""
    for stock_code in stock_codes:
        pyautogui.typewrite(stock_code)
        time.sleep(1)
        pyautogui.hotkey('enter')
        time.sleep(1)
        
        # 使用配置获取坐标
        c_x, c_y = config_manager.get_coordinate('add_stock')
        pyautogui.click(x=c_x, y=c_y)
        time.sleep(1)
        
        pyautogui.hotkey('insert')
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
    """导出股票数据"""
    # 获取当前分辨率的配置
    res_config = config_manager.get_resolution_config()
    
    # 使用配置中的坐标进行操作
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

def select_tdx_block_list() -> str:
    """通达信选股操作"""
    # 使用配置获取坐标
    step1_x, step1_y = config_manager.get_coordinate('tdx_step1')
    step2_x, step2_y = config_manager.get_coordinate('tdx_step2')
    # ... 其他步骤坐标 ...
    
    pyautogui.hotkey('ctrl', 't')
    time.sleep(0.5)
    
    pyautogui.click(step1_x, step1_y)
    pyautogui.click(step2_x, step2_y)
    # ... 其他使用配置坐标的操作 ...

    return blockname

def export_tdx_block_data(blockname):
    #
#     # 导出当前文件内容  tdx 导出功能
#     开始记录屏幕点击位置... (按F1停止)
# 记录点 1: (1197, 738) - Button.left - 15:51:20.091
# 记录点 2: (1462, 788) - Button.left - 15:51:22.172
# 记录点 3: (1999, 1106) - Button.left - 15:51:29.611
# 记录点 4: (1402, 828) - Button.left - 15:51:31.852
# 记录点 5: (1327, 748) - Button.left - 15:51:37.292
    pyautogui.typewrite('34')
    # 回车
    pyautogui.press('enter')
    time.sleep(0.5)
    # 选择中间项-所有数据
    pyautogui.click(1200, 738)
    time.sleep(0.5)
    # 点击浏览按钮
    pyautogui.click(1462, 788)
    time.sleep(0.5)
    # 输入文件名
    pyautogui.typewrite(blockname)
    # 点击确定
    pyautogui.click(1999, 1106)
    time.sleep(0.5)
    # 取消 不打开
    pyautogui.click(1402, 828)
    # time.sleep(12)
    print("请确认操作已完成，按回车键继续...")
    wait_for_keypress()
    pyautogui.click(1327, 748)
    time.sleep(1)
def change_history_list():
    # 右键切换到列表
    #     记录点 2: (1091, 1427) - Button.left - 15:47:30.412
    # 记录点 3: (652, 597) - Button.right - 15:47:33.380
    # 记录点 4: (725, 779) - Button.left - 15:47:37.909
    # 点击右键
    pyautogui.rightClick(652, 597)
    time.sleep(0.5)
    # 点击右键
    pyautogui.moveTo(725, 779)
    # 点击历史
    pyautogui.click(725,779)
    time.sleep(1)

def export_ths_block_data(blockname):
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
# 创建同花顺板块
    pyautogui.click(300, 337)
    time.sleep(0.5)
    pyautogui.moveTo(360, 66)
    time.sleep(0.5)
    pyautogui.click(360, 66)
    time.sleep(0.5)
    pyautogui.click(13, 109)
    time.sleep(0.5)
    pyautogui.click(365, 215)
    time.sleep(0.5)
    pyautogui.rightClick(365, 215)
    time.sleep(0.5)
    pyautogui.moveTo(440, 686)
    time.sleep(0.5)
    pyautogui.click(691, 684)
    time.sleep(0.5)
    pyautogui.click(539, 594)
    time.sleep(0.5)
    pyautogui.click(1513, 558)
    time.sleep(0.5)
    pyautogui.typewrite(blockname)
    pyautogui.click(1210, 992)
    pyautogui.click(1971, 1092)
    time.sleep(0.5)
    pyautogui.click(1407, 916)
    time.sleep(0.5)
    pyautogui.click(1407, 916)
    time.sleep(0.5)
    pyautogui.click(1407, 916)

def create_ths_block_from_file(blockname):
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
    pyautogui.click(289, 36)
    time.sleep(0.5)
    pyautogui.click(337, 316)
    time.sleep(0.5)
    pyautogui.click(1550, 531)
    # 输入文件名
    pyautogui.typewrite(blockname)
    time.sleep(0.5)
    pyautogui.click(1330, 752)
    time.sleep(0.5)
    pyautogui.click(1559, 657)
    time.sleep(0.5)
    pyautogui.moveTo(1589, 684)
    pyautogui.click(1589, 684)
    time.sleep(0.5)
    pyautogui.click(1955, 1061)
    time.sleep(0.5)
    pyautogui.click(1971, 1099)
    time.sleep(0.5)
    pyautogui.click(1250, 578)
    time.sleep(0.5)
    pyautogui.click(1972, 1093)
    time.sleep(0.5)
    pyautogui.click(1369, 834)
    time.sleep(0.5)
    pyautogui.click(1457, 917)
    time.sleep(0.5)
# stock_data = export_stock_data(stock)
    # return stock_data

def main():
    # 初始化分辨率配置
    config_manager.get_resolution_config()
    
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
    # 屏幕
    x, y = pyautogui.size()
    print(x, y)
    wait_for_keypress()
    blockname='07141725'
    # 2、执行选股步骤，填写导出文件目录
    blockname = select_tdx_block_list()

    # 可以先选择导出数据
    export_tdx_block_data(blockname)

    #右键选择历史数据
    change_history_list()

    blockname_01 = blockname+'01'
    # 导出数据
    export_tdx_block_data(blockname_01)


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
    print( columns)
    # 提取有用的字段并准备和另一个进行合并 代码	名称(83)	涨幅%	现价	最高	量比	总金额	细分行业
    new_data = data[['代码', '名称', '涨幅%', '现价', '最高', '量比', '总金额', '细分行业']]
    print( new_data.head(100))

    file2 = "../data/tdx/"+blockname_01+'.xls'
    # file2 = "../data/tdx/0711161920250711.xls"
    # 读取数据去掉第一行，第二行是列名
    data_02 = pd.read_csv(file2,encoding='GBK',sep='\t',header=1)
    print( data_02.head(100))

    # 合并new_data和data_02的数据，按照T列进行合并
    merged_data = pd.merge(new_data, data_02[['T', 'T1']], left_index=True, right_index=True)
    print( merged_data.head(100))

    # 保存中间文件，供后续使用
    merged_data.to_excel(f"../data/tdx/{blockname}_data.xlsx", index=False)

    # 取出代码字段，并且去掉等号和引号
    stock_codes = merged_data['代码'].str.split('=').str[1].str.replace("'", "")

    # 写入 txt 文件，每行一个代码# 3、保存文件到临时文件夹，可以通过某个按钮开始执行下面的动作
    with open(f"../data/ths/{blockname}.txt", "w", encoding="utf-8") as f:
        for code in stock_codes:
            f.write(code + "\n")

    # 给个提示，打开同花顺，按f1键开始执行下面的动作
    # print("请打开同花顺，按F1键开始执行下面的动作")
    # time.sleep(5)
    # input("请按回车键继续...")

    # time.sleep(3)
    print("请确认操作已完成，按回车键继续...")
    wait_for_keypress()
    # ths导入
    # 4、打开同花顺，创建新的板块，并导入临时文件夹中的文件，可以手工操作，也可以通过代码实现
    create_ths_block_from_file(blockname)
    # 5、打开同花顺，导出实时数据，保存下来
    export_ths_block_data(blockname)
    # 合并数据
    file3 = "../data/ths/"+blockname+'.xls'


    # file3 = "../data/ths/07110.xls"
    data_03 = pd.read_csv(file3,encoding='GBK',sep='\t')
    print( data_03.head(100))

    # exit()
    # 6、 分析和整合数据，生成需要的数据内容
# new_data = data_03[['代码', '名称', '净额', '净流入', '净量']]2507111818
    # 合并数据 净额	净流入	换手(实)	总金额	净量
    merged_data = pd.merge(merged_data, data_03[['净额', '净流入', '净量']], left_index=True, right_index=True)
    print( merged_data.head(100))
    # 增加最高价字段
    merged_data['最高价'] = merged_data['最高'].std.replace(',', '').astype(float)

    # 保存文件
    # result_df.to_excel(output_file, index=False)
    merged_data.to_excel(f"../alert/{blockname}.xlsx", index=False)

    # 7、调用模型，并预测结果，将结果输出到文件中，并返回合适的结果
    # 使用多个数据集训练并生成模型 , 需要按照日期更新到最新的模型
    model = {
        'reg_weights': '../models/250709_feature_weights_reg.csv',
        'clf_weights': '../models/250709_feature_weights_clf.csv',
        'reg_model': '../models/250709_model_reg.json',
        'clf_model': '../models/250709_model_clf.json'}

    #预测文件中的数据
    predictions_model_data_file(f"../alert/{blockname}.xlsx",model)

    # 计算结果，返回符合条件的股票代码


# 导出数据，保存数据，并获取关键数据



