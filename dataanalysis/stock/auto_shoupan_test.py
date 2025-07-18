
import auto_shoupan
import os
import pandas as pd
import numpy as np
import pyautogui
import time



def merge_test_data(blocknames):
    """重新合并数据"""
    # 这是一个数组，逐个去读并合并数据
    for blockname in blocknames:
        blockname_01 = blockname + "_01"
        auto_shoupan.tdx_merge_data(blockname, blockname_01)

        auto_shoupan.merge_block_data(blockname)

        auto_shoupan.predict_block_data(blockname)

# 同花顺往后执行
def start_ths_data(blocknames):
    """重新合并数据"""
    # 这是一个数组，逐个去读并合并数据
    for blockname in blocknames:
        blockname_01 = blockname + "_01"
        auto_shoupan.ths_get_block_data(blockname)

        auto_shoupan.merge_block_data(blockname)

        auto_shoupan.predict_block_data(blockname)



if __name__ == '__main__':

    blocknames = [
        # "07170948",
        # "07171244",
        # "07171431",
        # "07181144",
        "07181435",
    ]
    # 重新合并数据并计算
    merge_test_data(blocknames)

    # 从同花顺开始执行程序
    # blockname = "07181144"
    #
    # start_ths_data([blockname])
