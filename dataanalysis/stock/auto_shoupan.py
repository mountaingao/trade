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

def select_tdx_block_list():

# 记录点 1: (1251, 568) - Button.left - 15:38:28.446
# 记录点 2: (1431, 860) - Button.left - 15:38:30.269
# 记录点 3: (1497, 570) - Button.left - 15:38:32.246
# 记录点 4: (1250, 760) - Button.left - 15:38:37.950
# 记录点 5: (1478, 815) - Button.left - 15:38:39.415
# 记录点 6: (472, 1412) - Button.left - 15:38:43.030
# 记录点 1: (1509, 884) - Button.left - 16:05:21.826
    time.sleep(5)
    # 2、执行选股步骤，填写导出文件目录07111646
    ctrl_t = pyautogui.hotkey('ctrl', 't')
    time.sleep(0.5)
    # 点击加入条件
    pyautogui.click(1251, 568)
    # 点击选股入板块
    pyautogui.click(1431, 860)
    time.sleep(0.2)
    # 点击新建板块
    pyautogui.click(1497, 570)
    time.sleep(0.2)
    # 输入文件名
    blockname = pd.Timestamp.now().strftime("%m%d%H%M")
    print(blockname)
    pyautogui.typewrite(blockname)
    # 点击确定
    pyautogui.click(1250, 760)
    time.sleep(0.5)
    # 再点击确定，等待结果
    pyautogui.click(1478, 815)
    # 等10s 或者等待确认键
    time.sleep(8)
    pyautogui.click(1509, 884)

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
    # 选择中间项-所有数据
    pyautogui.click(1200, 738)
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
    time.sleep(0.5)
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
# 记录点 1: (299, 176) - Button.right - 17:33:22.806
# 记录点 2: (341, 650) - Button.left - 17:33:29.806
# 记录点 3: (632, 644) - Button.left - 17:33:32.646
# 记录点 4: (1513, 557) - Button.left - 17:33:35.862
# 记录点 5: (1984, 1087) - Button.left - 17:33:54.709
# 记录点 6: (1412, 911) - Button.left - 17:33:56.926
# 记录点 7: (1412, 911) - Button.left - 17:33:58.165
# 记录点 8: (1412, 911) - Button.left - 17:33:59.678
# 创建同花顺板块
    pyautogui.click(299, 176)
    pyautogui.rightClick(299, 176)
    time.sleep(0.5)
    pyautogui.moveTo(341, 650)
    time.sleep(0.5)
    pyautogui.click(632, 650)
    time.sleep(0.5)
    pyautogui.click(1513, 557)
    time.sleep(0.5)
    pyautogui.typewrite(blockname)
    pyautogui.click(1984, 1087)
    time.sleep(0.5)
    pyautogui.click(1412, 911)
    time.sleep(0.5)
    pyautogui.click(1412, 911)
    time.sleep(0.5)
    pyautogui.click(1412, 911)

def create_ths_block_from_file():
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
    time.sleep(0.5)
    pyautogui.click(1330, 752)
    time.sleep(0.5)
    pyautogui.click(1559, 657)
    time.sleep(0.5)
    pyautogui.click(1589, 684)
    time.sleep(0.5)
    pyautogui.click(2035, 1060)
    time.sleep(0.5)
    pyautogui.click(2026, 1102)
    time.sleep(0.5)
    pyautogui.click(1185, 713)
    time.sleep(0.5)
    pyautogui.click(1991, 1090)
    time.sleep(0.5)
    pyautogui.click(1381, 834)
    time.sleep(0.5)
    pyautogui.click(1468, 914)
    time.sleep(0.5)
    # 输入文件名
    pyautogui.typewrite(blockname)

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
    #
    x, y = pyautogui.position()
    print(x, y)
    # 屏幕
    x, y = pyautogui.size()
    print(x, y)

    blockname='600000'
    # # 2、执行选股步骤，填写导出文件目录
    # blockname = select_tdx_block_list()
    #
    # # 可以先选择导出数据
    # export_tdx_block_data(blockname)
    #
    # #右键选择历史数据
    # change_history_list()
    #
    # blockname_01 = blockname+'01'
    # # 导出数据
    # export_tdx_block_data(blockname_01)


    # 读取文件，代码写入到 txt中
    # file1 = "../data/tdx/"+blockname+'.xls'
    file1 = "../data/tdx/07111619.xls"

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

    # file2 = "../data/tdx/"+blockname01+'.xls'
    file2 = "../data/tdx/0711161920250711.xls"
    # 读取数据去掉第一行，第二行是列名
    data_02 = pd.read_csv(file2,encoding='GBK',sep='\t',header=1)
    print( data_02.head(100))

    # 合并new_data和data_02的数据，按照T列进行合并
    merged_data = pd.merge(new_data, data_02[['T', 'T1']], left_index=True, right_index=True)
    print( merged_data.head(100))

    # 取出代码字段，并且去掉等号和引号
    stock_codes = merged_data['代码'].str.split('=').str[1].str.replace("'", "")

    # 写入 txt 文件，每行一个代码
    with open(f"../data/ths/{blockname}.txt", "w", encoding="utf-8") as f:
        for code in stock_codes:
            f.write(code + "\n")


    exit()
    # ths导入
    create_ths_block_from_file(blockname)

    export_ths_block_data(blockname)
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



