
import os
import datetime
import pandas as pd
from stockrating.stock_rating_ds import evaluate_stock
from autotrade.add_stock_ths_block import add_stocks_to_ths_block
from stockrating.get_stock_block import process_stock_concept_data
from mootdx.reader import Reader
import json

# 通达信板块操作函数
# 初始化板块对象
from mootdx.tools.customize import Customize

# 新增代码：读取配置文件
config_path = os.path.join(os.path.dirname(__file__), '..', 'config', 'config.json')
with open(config_path, 'r', encoding='utf-8') as config_file:  # 修改编码为utf-8
    config = json.load(config_file)


# 修改代码：从配置文件中获取tdxdir的值
custom = Customize(tdxdir=config['tdxdir'])
reader = Reader.factory(market='std',tdxdir=config['tdxdir'])
def read_tdx_block_data(block_name):
    print(block_name)
    # 读取通达信板块数据，返回股票列表
    # 读取板块数据
    block_data = custom.search(block_name)
    # 打印板块数据
    print(block_data)
    return block_data


def get_tdx_custom_block():
    # 默认扁平格式
    block_data = reader.block_new(False)
    return block_data

def save_tdx_block_data(block_name, stock_list):
    if custom.search(block_name):
        custom.update(block_name, symbol=stock_list)
        return True
    # 新建自定义板块
    return custom.create(name=block_name, symbol=stock_list)


def filter_block_names(df,start=401,end=430):
    # 过滤 blockname 为4位数字且大于0401的数据
    # 先确保 blockname 是纯数字
    filtered_df = df[(df['blockname'].str.len() == 4) & (df['blockname'].str.isdigit())]
    # 修改比较范围，确保 blockname 在 0401 到 0430 之间
    filtered_df = filtered_df[(filtered_df['blockname'].astype(int) >= start) & (filtered_df['blockname'].astype(int) <= end)]
    return filtered_df

# 得到所有的符合条件的个股数据
def get_tdx_custom_block_from_date(start=101,end=1231):
    # 获取当前日期并格式化为"MMdd"形式
    block_names = get_tdx_custom_block()
    # print(block_names)
    # 提取第二个和第四个字段
    df_filtered = block_names[['blockname', 'code']]
    # print(df_filtered)
    # 过滤 blockname
    filtered_df = filter_block_names(df_filtered,start,end)

    # print(filtered_df)
    return filtered_df

if __name__ == "__main__":

    get_tdx_custom_block_from_date(401,430)

    # 读取当日的评分数据，倒序展示