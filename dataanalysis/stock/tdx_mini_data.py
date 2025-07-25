import os
import pandas as pd
import numpy as np
import pickle
from datetime import datetime
import time
from typing import List, Dict, Optional
from pytdx.hq import TdxHq_API
from pytdx.params import TDXParams

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
            raise ConnectionError("无法连接到通达信服务器")

        market_code = get_market_code(code)
        stock_code = get_stock_code(code)

        # 获取最近5天的5分钟K线数据
        data = []
        for i in range(5):  # 最多获取5天的数据
            kline_data = api.get_security_bars(
                TDXParams.KLINE_TYPE_5MIN, 
                market_code, 
                stock_code, 
                (4-i) * 48,  # 每天最多48个5分钟周期
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
    df = df.copy()
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

    return {
        'code': code,
        # 'latest_macd': latest_macd,
        'band_width': df_boll['band_width'].iloc[-1]
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

# 示例调用
if __name__ == "__main__":
    codes = [
        "300922",
        "300224",
             ]
    results = process_multiple_codes(codes)
    # 将results 转成pd，并打印出来
    df = pd.DataFrame(results)
    print(df)