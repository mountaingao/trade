
import pandas as pd
import json
import os


def EMA(DF, N):
    return pd.Series(DF).ewm(alpha=2/(N+1), adjust=True).mean().round(2)

def SMA(DF, N, M):
    return pd.Series(DF).ewm(alpha=M / N, adjust=True).mean().round(2)


# def ema1(data,val,window=5):
#     # ma_5 = data[f"{val}"].rolling(window).mean()
#     ema = data['close'].ewm(span=window, adjust=False).mean().round(2)
#     return ema


if __name__ == '__main__':


    # 新增代码：读取配置文件
    # config_path = os.path.join(os.path.dirname(__file__), '../../', 'config', 'config.json')
    # with open(config_path, 'r', encoding='utf-8') as config_file:  # 修改编码为utf-8
    #     config = json.load(config_file)

    from stockrating.read_local_info_tdx import get_stock_history_by_local
    data = get_stock_history_by_local('300386')

    print(data.head(5))
    print(data.tail(5))
    # 新增代码：从配置文件中获取参数
    ema = EMA(data['close'], 7)
    # ema = ema1(data, 7)

    print(ema)