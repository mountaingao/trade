#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
Date: 2024/8/28 15:00
Desc: 同花顺-盈利预测
https://basic.10jqka.com.cn/new/600519/
"""

from io import StringIO

import pandas as pd
import requests

from akshare.utils.cons import headers
from get_stock_block_ths import stock_extract_concept_ranking


def stock_profit_forecast_ths(
        symbol: str = "600519", indicator: str = "预测年报每股收益"
) -> pd.DataFrame:
    """
    同花顺-盈利预测
    https://basic.10jqka.com.cn/new/600519/
    :param symbol: 股票代码
    :type symbol: str
    :param indicator: choice of {"预测年报每股收益", "预测年报净利润", "业绩预测详表-机构", "业绩预测详表-详细指标预测"}
    :type indicator: str
    :return: 盈利预测
    :rtype: pandas.DataFrame
    """
    url = f"https://basic.10jqka.com.cn/new/{symbol}/"
    r = requests.get(url, headers=headers)
    r.encoding = "gbk"
    # 得到 概念贴合度排名
    # print(r.text)

    temp_df = stock_extract_concept_ranking(r.text)
    return temp_df


if __name__ == "__main__":
    for _item in [
        "概念贴合度排名",
    ]:
        stock_profit_forecast_ths_df = stock_profit_forecast_ths(
            symbol="300718", indicator=_item
        )
        print(stock_profit_forecast_ths_df)

