# 1、读取通达信的0327yu板块的数据，得到股票列表
# 2、计算每个股票的积分，标识为预先分析，并入库
# 3、得到50分以上的数据保存为板块名0327.txt,内容为名称，代码，积分，评估时间

import os
import datetime
import pandas as pd
from stockrating.stock_rating_ds import evaluate_stock
from autotrade.add_stock_ths_block import add_stocks_to_ths_block
from stockrating.get_stock_block import process_stock_concept_data
from mootdx.reader import Reader
import json
# 初始化板块对象
from mootdx.tools.customize import Customize

# 新增代码：读取配置文件
config_path = os.path.join(os.path.dirname(__file__), '..', 'config', 'config.json')
with open(config_path, 'r', encoding='utf-8') as config_file:  # 修改编码为utf-8
    config = json.load(config_file)


# 修改代码：从配置文件中获取tdxdir的值
custom = Customize(tdxdir=config['tdxdir'])

def read_tdx_block_data(block_name):
    print(block_name)
    # 读取通达信板块数据，返回股票列表
    # 读取板块数据
    block_data = custom.search(block_name)
    # 打印板块数据
    print(block_data)
    # 这里假设板块数据已经以某种方式存储，具体实现需要根据实际情况调整
    return block_data

# def read_tdx_block_data(block_name):
#     print(block_name)
#     # 修改代码：从配置文件中获取tdxdir的值
#     custom = Customize(tdxdir=config['tdxdir'])
#     # 读取通达信板块数据，返回股票列表
#     # 读取板块数据
#     block_data = custom.search(block_name)300099603109300466605068300339


#     # 打印板块数据
#     print(block_data)
#     # 这里假设板块数据已经以某种方式存储，具体实现需要根据实际情况调整
#     return block_data

def save_tdx_block_data(block_name, stock_list):
    if custom.search(block_name):
        custom.update(block_name, symbol=stock_list)
        return True
    # 新建自定义板块
    return custom.create(name=block_name, symbol=stock_list)


def calculate_and_save_block_scores(block_name):
    # 读取板块数据
    stock_list = read_tdx_block_data(block_name)
    if not stock_list:
        print(f"板块 {block_name} 数据为空，无法计算积分")
        return
    # 计算每个股票的积分并保存
    result = []
    for stock_code in stock_list:
        score = evaluate_stock(stock_code)
        if score >= 50:
            result.append((stock_code, score, datetime.datetime.now().strftime("%Y-%m-%d %H:%M")))

    # 修改代码：只取 result 的第一列数据（股票代码）300663

    stock_codes = [item[0] for item in result]
    save_tdx_block_data(f"{block_name}01", stock_codes)
    # 保存到文件
    output_file = f"{block_name}.txt"
    with open(output_file, 'w', encoding='GBK') as f:  # 修改编码为GBK
        # 直接写入整个 result 列表
        f.writelines([f"{item[0]}\t{item[1]}\t{item[2]}\n" for item in result])

     #保存到同花顺自选股
    # add_stocks_to_ths_block(stock_codes)

if __name__ == "__main__":

    # 获取当前日期并格式化为"MMdd"形式
    current_date = datetime.datetime.now().strftime("%m%d")

    block_name = f"{current_date}"
    # block_name = "0519"
    # block_name = "1Y".encode('GBK')  # 修改字符集为GBK
    # block_name = "0513"
    calculate_and_save_block_scores(block_name)

    # 读取当日的评分数据，倒序展示300322
