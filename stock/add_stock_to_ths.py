import json
import re
import os
from datetime import datetime

def update_ths_mfb_selfstock(stock_codes, file_path=r"D:\同花顺软件\同花顺\mx_550967754\SelfStockInfo.json", update_type='append', head=False):
    """
    将指定股票代码自动添加到同花顺免费版自选股。
    :param stock_codes: list, 需添加的股票代码，如 ['600519.SH', '000001.SZ']
    :param file_path: str, SelfStockInfo.json 文件路径
    :param update_type: str, 更新方式（'append' 或 'overwrite'）
    :param head: bool, 是否将新股票添加到列表头部
    :return: bool, 更新成功返回 True，失败返回 False
    """
    if update_type not in ['append', 'overwrite']:
        return False
    if not os.path.exists(file_path):
        return False

    update_stock_codes = []
    update_market_codes = []

    with open(file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)
        origin_stock_codes = [d["C"] for d in data]
        origin_market_codes = [d["M"] for d in data]

    for stock_code in stock_codes:
        new_market_code = ''
        if re.match(r'([0-9]{6})(.SH)$', stock_code):
            new_market_code = '17'
        elif re.match(r'([0-9]{6})(.SZ)$', stock_code):
            new_market_code = '33'
        elif re.match(r'([0-9]{6})(.BJ)$', stock_code):
            new_market_code = '151'
        new_stock_code = re.sub(r'(.SH)|(.SZ)|(.BJ)$', '', stock_code)

        if update_type == 'overwrite' or (update_type == 'append' and new_stock_code not in origin_stock_codes):
            update_stock_codes.append(new_stock_code)
            update_market_codes.append(new_market_code)

    if update_type == 'append':
        new_stock_codes = update_stock_codes + origin_stock_codes if head else origin_stock_codes + update_stock_codes
        new_market_codes = update_market_codes + origin_market_codes if head else origin_market_codes + update_market_codes
    else:
        new_stock_codes = update_stock_codes
        new_market_codes = update_market_codes

    new_dates = [datetime.now().strftime("%Y%m%d") for _ in range(len(new_stock_codes))]
    new_prices = ['0' for _ in range(len(new_stock_codes))]
    stock_infos = zip(new_stock_codes, new_market_codes, new_prices, new_dates)
    new_stock_infos = [{'C': stock_info[0], 'M': stock_info[1], 'P': stock_info[2], 'T': stock_info[3]} for stock_info in stock_infos]

    with open(file_path, 'w', encoding='utf-8') as file:
        json.dump(new_stock_infos, file, ensure_ascii=False)

    return True

# 示例：添加股票
stock_codes = ['600519.SH', '000001.SZ']
update_ths_mfb_selfstock(stock_codes)