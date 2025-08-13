import os
import pandas as pd
import numpy as np
import pickle
from datetime import datetime
import time
from typing import List, Dict, Optional
from pytdx.hq import TdxHq_API
from pytdx.params import TDXParams
import json
import struct
# 新增代码：读取配置文件
config_path = os.path.join(os.path.dirname(__file__), '../../', 'config', 'config.json')
with open(config_path, 'r', encoding='utf-8') as config_file:  # 修改编码为utf-8
    config = json.load(config_file)

# 假设已有以下模块或函数（需根据实际项目结构调整）
# from dataanalysis.stock.tdx_runtime_data import get_minute_data
# from dataanalysis.stock.utils import qfq  # 前复权函数

# 示例占位函数，实际应替换为真实的数据获取和复权逻辑
def get_market_code(code: str) -> int:
    """根据股票代码判断市场代码"""
    # 确保code是字符串类型
    code = str(code)
    if code.startswith('sh') or code.startswith('6'):
        return 1  # 上海
    elif code.startswith('sz') or code.startswith(('0', '3', '15', '16', '20', '30')):
        return 0  # 深圳
    else:
        raise ValueError(f"Unknown market for code: {code}")

def get_stock_code(code: str) -> str:
    """提取纯股票代码"""
    # 确保code是字符串类型
    code = str(code)
    if code.startswith(('sh', 'sz')):
        return code[2:]
    return code

def get_minute_data(code: str) -> pd.DataFrame:
    """使用pytdx获取5分钟K线数据"""
    api = TdxHq_API()
    try:
        # 连接通达信服务器
        # if not api.connect('14.215.128.18', 7709):  # 免费服务器，可替换为其他服务器
        if not api.connect('123.125.108.90', 7709):  # 免费服务器，可替换为其他服务器
        # if not api.connect('183.201.253.76', 7709):  # 免费服务器，可替换为其他服务器
            raise ConnectionError("无法连接到通达信服务器")

        market_code = get_market_code(code)
        stock_code = get_stock_code(code)

        # 获取最近10天的5分钟K线数据
        data = []
        # 修改为获取10天数据，每天48个5分钟周期
        for i in range(10):  # 获取10天的数据
            kline_data = api.get_security_bars(
                TDXParams.KLINE_TYPE_5MIN, 
                market_code, 
                stock_code, 
                (9-i) * 48,  # 每天最多48个5分钟周期，现在获取10天数据
                48
            )
            if kline_data:
                data.extend(kline_data)
        
        if not data:
            return pd.DataFrame()
            
        # 转换为DataFrame
        df = pd.DataFrame(data)
        print( df)
        # 处理日期时间字段
        df['datetime'] = pd.to_datetime(df['datetime'])
        
        # 重命名列以匹配原代码
        df = df.rename(columns={
            'open': 'open',
            'high': 'high',
            'low': 'low',
            'close': 'close',
            'vol': 'volume'
        })
        
        # 只保留需要的列
        required_cols = ['datetime', 'open', 'high', 'low', 'close', 'volume']
        df = df[required_cols]

        print( df)
        return df
        
    except Exception as e:
        print(f"获取股票{code}数据时出错: {e}")
        return pd.DataFrame()
    finally:
        api.disconnect()

def qfq(df: pd.DataFrame, code: str) -> pd.DataFrame:
    """模拟前复权处理"""
    # 实际应调用真实的复权逻辑
    df['adj_close'] = df['close']
    return df

# 缓存相关配置
CACHE_DIR = "cache/minute_data"
os.makedirs(CACHE_DIR, exist_ok=True)

def _get_cache_path(code: str) -> str:
    # 确保code是字符串类型
    code = str(code)
    return os.path.join(CACHE_DIR, f"{code}.pkl")

def _is_trading_time() -> bool:
    """判断当前是否为交易时间（交易日的9:30-15:00）"""
    now = datetime.now()
    # 获取当前时间的时和分
    current_time = now.time()
    # 交易时间范围
    start_time = time.strptime("09:30", "%H:%M").tm_hour * 3600 + time.strptime("09:30", "%H:%M").tm_min * 60
    end_time = time.strptime("15:00", "%H:%M").tm_hour * 3600 + time.strptime("15:00", "%H:%M").tm_min * 60
    # 当前时间（秒数）
    now_seconds = current_time.hour * 3600 + current_time.minute * 60 + current_time.second
    
    # 判断是否为工作日（周一到周五）
    if now.weekday() < 5:  # 0-4 代表周一到周五
        # 判断是否在交易时间范围内
        if start_time <= now_seconds <= end_time:
            return True
    return False

def _save_to_cache(code: str, df: pd.DataFrame):
    # 如果是交易时间，则不保存缓存
    if _is_trading_time():
        return
    cache_path = _get_cache_path(code)
    with open(cache_path, 'wb') as f:
        pickle.dump(df, f)

def _load_from_cache(code: str) -> Optional[pd.DataFrame]:
    cache_path = _get_cache_path(code)
    if os.path.exists(cache_path):
        with open(cache_path, 'rb') as f:
            return pickle.load(f)
    return None

def calculate_macd(df: pd.DataFrame, short=12, long=26, signal=9) -> pd.DataFrame:
    """计算MACD指标"""
    df = df.copy()
    df['ema_short'] = df['close'].ewm(span=short, adjust=False).mean()
    df['ema_long'] = df['close'].ewm(span=long, adjust=False).mean()
    df['dif'] = df['ema_short'] - df['ema_long']
    df['dea'] = df['dif'].ewm(span=signal, adjust=False).mean()
    df['bar'] = (df['dif'] - df['dea']) * 2
    return df

def calculate_boll(df: pd.DataFrame, window=20, num_std=2) -> pd.DataFrame:
    """计算布林带指标"""
    # df = df.copy()
    df['ma'] = df['close'].rolling(window=window).mean()
    df['std'] = df['close'].rolling(window=window).std()
    df['upper'] = df['ma'] + num_std * df['std']
    df['lower'] = df['ma'] - num_std * df['std']
    df['band_width'] = 100*(df['upper'] - df['lower']) / df['ma']
    return df

def process_single_code(code: str) -> Dict:
    """处理单个股票代码"""
    # 确保code是字符串类型
    code = str(code)
    # 尝试从缓存加载
    df = _load_from_cache(code)
    if df is None:
        # 获取原始分钟数据
        df = get_minute_data(code)
        # print( df)
        if df.empty:
            return {'code': code, 'error': 'No data'}
        # 前复权处理
        df = qfq(df, code)
        # 保存到缓存
        _save_to_cache(code, df)
    
    # 确保按时间排序
    df = df.sort_values('datetime').reset_index(drop=True)

    # 只保留需要的列
    required_cols = ['datetime', 'close']
    df = df[required_cols]

    # 计算MACD（直接使用5分钟K线数据）
    # df_macd = calculate_macd(df)
    # latest_macd = df_macd[['dif', 'dea', 'bar']].iloc[-1].to_dict()

    # 计算BOLL线
    df_boll = calculate_boll(df)
    print( df_boll)
    # latest_boll = df_boll['band_width'].iloc[-1]
    # 得到band_width最小值 和最后的比较
    latest_boll = df_boll['band_width'].iloc[-1]
    # 修复：使用df_boll而不是df来获取band_width列
    min_value = df_boll['band_width'].min()
    max_value = df_boll['band_width'].max()

    # 修复除零错误：检查min_value是否为0
    is_boll_low = latest_boll / min_value if min_value != 0 else 0

    return {
        'code': code,
        # 'latest_macd': latest_macd,
        'band_width': latest_boll,
        'min_value': min_value,
        'max_value': max_value,
        'is_boll_low': is_boll_low
    }

def process_multiple_codes(codes: List[str]) -> List[Dict]:
    """批量处理多个股票代码"""
    results = []
    for code in codes:
        # 确保每个code都是字符串类型
        code = str(code)
        result = process_single_code(code)
        results.append(result)
    return results



from pytdx.reader import TdxMinBarReader
import pandas as pd

def read_local_5m_data(stock_code, tdx_path=config['tdxdir']+'vipdoc'):
    """
    读取通达信本地5分钟数据
    :param stock_code: 股票代码，如'sh600000'或'sz000001'
    :param tdx_path: 通达信安装目录下的vipdoc路径
    :return: DataFrame
    """
    # 确定市场前缀
    market = 1 if stock_code.startswith('sh') else 0

    # 构建文件路径 D:\zd_haitong\vipdoc\sh\fzline
    file_path = f"{tdx_path}/{stock_code[:2]}/fzline/{stock_code}.lc5"

    # 使用PyTDX读取5分钟数据
    reader = TdxMinBarReader()
    df = reader.get_df(file_path)

    # print(df.head())
    # 打印数量
    print(len(df))
    # 打印最后10条
    print(df.tail(10))
    exit()

    # 转换时间格式
    df['datetime'] = pd.to_datetime(df['datetime'])
    df.set_index('datetime', inplace=True)

    return df

# # 示例使用
# df_5m = read_local_5m_data('sh600000')
# print(df_5m.head())


from pytdx.reader import TdxExHqDailyBarReader

def read_financial_data(stock_code, tdx_path=config['tdxdir']+'vipdoc'):
    """
    读取财务数据（含除权信息）
    :param stock_code: 股票代码
    :param tdx_path: 通达信安装目录
    :return: 包含除权信息的DataFrame
    """
    # 读取除权文件  D:\zd_haitong\vipdoc\cw
    file_path = f"{tdx_path}/cw/gp{stock_code}.dat"

    try:
        reader = TdxExHqDailyBarReader()
        # 添加异常处理，防止文件格式不匹配导致程序崩溃
        try:
            df_fin = reader.get_df(file_path)
        except struct.error as e:
            print(f"读取除权文件 {file_path} 时出错: {e}")
            print("可能是文件格式不匹配或文件损坏，返回空DataFrame")
            return pd.DataFrame()
        except Exception as e:
            print(f"读取除权文件 {file_path} 时发生未知错误: {e}")
            return pd.DataFrame()

        # 筛选出有除权信息的记录
        df_xr = df_fin[df_fin['amount'] == 0].copy()
        df_xr['date'] = pd.to_datetime(df_xr['datetime'])

        return df_xr[['date', 'open', 'close']].rename(
            columns={'open': 'xr_factor', 'close': 'dividend'})

    except FileNotFoundError:
        print(f"未找到除权文件: {file_path}")
        return pd.DataFrame()

# # 示例使用
# df_xr = read_financial_data('sh600000')
# print(df_xr.head())

def adjust_price(df_5m, df_xr):
    """
    对5分钟数据进行除权处理
    :param df_5m: 原始5分钟数据
    :param df_xr: 除权信息
    :return: 除权后的DataFrame
    """
    if df_xr.empty:
        return df_5m

    # 按除权日期降序排序
    df_xr = df_xr.sort_values('date', ascending=False)

    # 复制原始数据
    df_adjusted = df_5m.copy()

    # 初始化前复权因子
    adj_factor = 1.0

    # 对每个除权日进行处理
    for _, row in df_xr.iterrows():
        xr_date = row['date']
        xr_factor = row['xr_factor']
        dividend = row['dividend']

        # 计算复权因子
        # 除权因子 = (前收盘价 - 现金红利) / (前收盘价 * (1 + 送转比例))
        # 通达信中xr_factor已经是计算好的复权因子
        adj_factor *= xr_factor

        # 对除权日之前的数据进行复权
        mask = df_adjusted.index < xr_date
        df_adjusted.loc[mask, ['open', 'high', 'low', 'close']] *= adj_factor
        df_adjusted.loc[mask, 'volume'] /= adj_factor  # 成交量需要反向调整

    return df_adjusted

# 示例使用
# df_5m_adjusted = adjust_price(df_5m, df_xr)
# print(df_5m_adjusted.head())

# 示例调用
if __name__ == "__main__":
    codes = [
        "300436",
        "300224",
             ]
    results = process_multiple_codes(codes)
    # 将results 转成pd，并打印出来
    df = pd.DataFrame(results)
    print(df)
    exit()

    # todo 5分钟数据读取时间点以前的数据，结合最小的值和净量来判断是否可以买入，第二天爆发
    # 读取目录中的文件 dataanalysis/data/predictions/1600/07281517_1518.xlsx  获取代码
    # file_path = "../data/predictions/1600/07281517_1518.xlsx"
    # df = pd.read_excel(file_path)
    # print(df)
    # codes = df['代码'].to_list()
    # results = process_multiple_codes(codes)
    # print( results)
    # # 将result 转成pd，和df 合并，按照code 和 代码 相等的方法进行
    # df_bw = pd.DataFrame(results)
    # df = pd.merge(df,df_bw, left_index=True, right_index=True)
    # file_path = "../data/predictions/1600/07281517_151800.xlsx"
    # df.to_excel(file_path)


    # 1. 读取5分钟数据
    df_5m = read_local_5m_data('sh000689')

    # 2. 读取除权信息
    df_xr = read_financial_data('sh000689')

    exit()
    # 3. 进行除权处理
    df_5m_adjusted = adjust_price(df_5m, df_xr)
    # 4. 保存处理后的数据
    df_5m_adjusted.to_csv('sh600000_5m_adjusted.csv')

    # 5. 可视化对比
    import matplotlib.pyplot as plt

    plt.figure(figsize=(12, 6))
    plt.plot(df_5m['close'], label='原始数据')
    plt.plot(df_5m_adjusted['close'], label='复权后数据')
    plt.legend()
    plt.title('5分钟数据复权前后对比')
    plt.show()