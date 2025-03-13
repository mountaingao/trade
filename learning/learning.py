import akshare as ak
import pandas as pd

# 获取分时数据
symbol = "sz300718"  # 股票代码
minute_data = ak.stock_zh_a_minute(symbol=symbol, period="1", adjust="")
# 将索引转换为 DatetimeIndex
minute_data.index = pd.to_datetime(minute_data.index)
print(minute_data.head())

def extract_features(data):
    """
    从分时数据中提取特征
    """
    # 上午和下午的分段时间
    morning_data = data.between_time("09:30", "11:30")
    afternoon_data = data.between_time("13:00", "15:00")

    # 成交量特征
    morning_volume_ratio = morning_data["volume"].sum() / data["volume"].sum()
    afternoon_volume_ratio = afternoon_data["volume"].sum() / data["volume"].sum()

    # 价格特征
    morning_price_range = morning_data["high"].max() - morning_data["low"].min()
    afternoon_price_range = afternoon_data["high"].max() - afternoon_data["low"].min()

    # 时间特征
    time_features = {
        "morning_volume_ratio": morning_volume_ratio,
        "afternoon_volume_ratio": afternoon_volume_ratio,
        "morning_price_range": morning_price_range,
        "afternoon_price_range": afternoon_price_range,
    }
    return time_features

# 提取特征
features = extract_features(minute_data)
print(features)