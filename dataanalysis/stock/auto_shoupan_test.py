
import auto_shoupan
import os
import pandas as pd
import numpy as np
import pyautogui
import time
from data_prepare import get_data_from_files
from sqllite_block_manager import add_blockname_data


def merge_test_data(blocknames):
    """重新合并数据"""
    # 这是一个数组，逐个去读并合并数据
    for blockname in blocknames:
        blockname_01 = blockname + "001"
        auto_shoupan.tdx_merge_data(blockname, blockname_01)

        auto_shoupan.merge_block_data(blockname)

        auto_shoupan.predict_block_data(blockname)

def merge_tdx_data(blocknames):
    """重新合并数据"""
    # 这是一个数组，逐个去读并合并数据
    for blockname in blocknames:
        blockname_01 = blockname + "001"
        auto_shoupan.tdx_merge_data(blockname, blockname_01)


# 同花顺往后执行
def start_ths_data(blocknames):
    """重新合并数据"""
    # 这是一个数组，逐个去读并合并数据
    for blockname in blocknames:
        blockname_01 = blockname + "001"
        auto_shoupan.ths_get_block_data(blockname)

        auto_shoupan.merge_block_data(blockname)

        auto_shoupan.predict_block_data(blockname)

def start_merge_and_predict_data(blocknames):
    """重新合并数据"""
    # 这是一个数组，逐个去读并合并数据
    for blockname in blocknames:

        auto_shoupan.merge_block_data(blockname)
        auto_shoupan.predict_block_data(blockname)

def start_predict_data(blocknames):
    """预测结果"""
    # 这是一个数组，逐个去读并合并数据
    for blockname in blocknames:

        auto_shoupan.predict_block_data(blockname)

def start_predict_date_data(blocknames):
    """重新合并数据"""
    # 这是一个数组，逐个去读并合并数据
    for blockname in blocknames:

        auto_shoupan.predict_block_data_from_block_name(blockname)

def start_cal_predict_data_selected(file_path):
    """重新合并数据"""
    auto_shoupan.cal_predict_data_selected(file_path)

    # auto_shoupan.cal_predict_data_selected('../data/predictions/1000/09120952_0954.xlsx')
    # cal_predict_data_selected('../data/predictions/1200/09121136_1137.xlsx')
    # cal_predict_data_selected('../data/predictions/1400/09121440_1442.xlsx')
    # cal_predict_data_selected('../data/predictions/1600/09121517_1522.xlsx')


    # 15日数据
    # cal_predict_data_selected('../data/predictions/1000/09150954_0956.xlsx')
    # cal_predict_data_selected('../data/predictions/1200/09151132_1133.xlsx')
    # cal_predict_data_selected('../data/predictions/1400/09151359_1401.xlsx')
    # cal_predict_data_selected('../data/predictions/1600/09151506_1507.xlsx')

    # 16日数据
    # cal_predict_data_selected('../data/predictions/1000/09160943_0945.xlsx')
    # cal_predict_data_selected('../data/predictions/1200/09161142_1144.xlsx')
    # cal_predict_data_selected('../data/predictions/1400/09161428_1431.xlsx')
    # cal_predict_data_selected('../data/predictions/1600/09161509_1510.xlsx')

    # 17日数据
    # auto_shoupan.cal_predict_data_selected('../data/predictions/1000/09170940_0942.xlsx')
    # auto_shoupan.cal_predict_data_selected('../data/predictions/1200/09171143_1145.xlsx')
    # auto_shoupan.cal_predict_data_selected('../data/predictions/1400/09171416_1418.xlsx')
    # auto_shoupan.cal_predict_data_selected('../data/predictions/1600/09171504_1506.xlsx')

def cal_daily_canshu_data():
    """重新合并数据"""
    # 这是一个数组，逐个去读并合并数据

    files= [

        # '../data/predictions/1000/09120939_0942.xlsx',
        # '../data/predictions/1200/09121136_1137.xlsx',
        # '../data/predictions/1400/09121440_1442.xlsx',
        # '../data/predictions/1600/09121517_1522.xlsx',
        # '../data/predictions/1000/09150954_0956.xlsx',
        # '../data/predictions/1200/09151132_1133.xlsx',
        # '../data/predictions/1400/09151359_1401.xlsx',
        # '../data/predictions/1600/09151506_1507.xlsx',
        # '../data/predictions/1000/09160943_0945.xlsx',
        # '../data/predictions/1200/09161142_1144.xlsx',
        # '../data/predictions/1400/09161428_1431.xlsx',
        # '../data/predictions/1600/09161509_1510.xlsx',
        # '../data/predictions/1000/09170940_0942.xlsx',
        # '../data/predictions/1200/09171143_1145.xlsx',
        # '../data/predictions/1400/09171416_1418.xlsx',
        # '../data/predictions/1600/09171504_1506.xlsx',

        # '../data/predictions/1000/09180942_0944.xlsx',
        # '../data/predictions/1200/09181142_1144.xlsx',
        # '../data/predictions/1400/09181412_1414.xlsx',
        # '../data/predictions/1600/09181507_1509.xlsx',

        '../data/predictions/1000/09190945_0948.xlsx',
        # "../data/predictions/1200/09191239_1242.xlsx"""
        # '../data/predictions/1400/09191357_1359.xlsx',
        # '../data/predictions/1600/09191522_1526.xlsx',
        # '../data/predictions/1600/09221516_1518.xlsx',
    ]
    df = get_data_from_files(files)

    df = add_blockname_data(df)
    # 如果概念为空，获取概念
# 17  688028   沃尔德  -2.67  0.75  2.35  3.56  3.79    0.01   16.43  13.79
# 11  688028   沃尔德 -3.45   3.84  2.25  3.56  3.76   -0.03   17.38  13.79
    df['概念'] = df['概念'].apply(lambda x: x.split(' ')[0] if isinstance(x, str) else x)
    df['概念'] = df['概念'].apply(lambda x: x.split('+')[0] if isinstance(x, str) else x)

    # df['概念'] = df['概念'].apply(lambda x: x.split('+')[0])

    auto_shoupan.select_from_block_data(df)

def main():
    blocknames = [
        # "07170948",
        # "07171244",
        # "07171431",
        # "07181144",
        # "07181435",
        # "07210943",
        # "07211131",
        # "07211414",
        # "07211543",
        # "07220947",
        # "07221140",
        # "07221442",
        # "07181529",
        # "07291854",
        # "08120952",
        # "08121157",
        # "08121444",
        # "08121516",
        "09160938",
    ]
    # 重新合并数据并计算07211131
    merge_test_data(blocknames)


    # merge_tdx_data(blocknames)


    # 从同花顺开始执行程序
    # blockname = "07181144"
    #
    # start_ths_data(blocknames)


    # start_merge_and_predict_data(blocknames)

    # start_predict_data(blocknames)

    # start_predict_date_data(blocknames)

    # minite_data = get_minite_band_width( blocknames)



if __name__ == '__main__':
    # 基本测试
    # main()

    # cal_daily_canshu_data()

    # 推测目录下的数据
    # start_cal_predict_data_selected("../data/predictions/")
    # start_cal_predict_data_selected("../data/predictions/1000/09170940_0942.xlsx")
    # start_cal_predict_data_selected("../data/predictions/1000/09190945_0948.xlsx")
    # start_cal_predict_data_selected("../data/predictions/1200/09191239_1242.xlsx")
    # start_cal_predict_data_selected("../data/predictions/1200/09231132_1134.xlsx")
    # start_cal_predict_data_selected("../data/predictions/1400/09231409_1411.xlsx")
    # start_cal_predict_data_selected("../data/predictions/1600/09231503_1505.xlsx")
    # start_cal_predict_data_selected("../data/predictions/1400/09241351_1353.xlsx")
    # start_cal_predict_data_selected("../data/predictions/1200/09251133_1135.xlsx")
    # start_cal_predict_data_selected("../data/predictions/1000/09250948_0950.xlsx")
    # start_cal_predict_data_selected("../data/predictions/1000/09260949_0950.xlsx")
    # start_cal_predict_data_selected("../data/predictions/1200/09261140_1144.xlsx")
    start_cal_predict_data_selected("../data/predictions/1400/09261436_1437.xlsx")
    # start_cal_predict_data_selected("../data/predictions/1000/09290941_0942.xlsx")

