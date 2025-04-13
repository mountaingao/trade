import akshare as ak
import pandas as pd
import numpy as np
import json
import os
import hashlib
import time, datetime
import pywencai
import mysql
import mysql.connector
from datetime import datetime

from mootdx.reader import Reader
from stockrating.read_local_info_tdx import read_tdx_block_data

# 本程序主要针对历史数据进行回溯分析，重新计算积分，以测试按照积分标准来重写会有什么影响和结果
# 1、读取通达信的某个板块的数据（如0327yu），得到股票列表
# 2、计算每个股票的积分，前四项数据可以根据日线来计算，后几项数据读取表stock_rating中按照代码和alert_date查询出结果，重新计算以后得到新的积分,同时计算以当日收盘价购买后，第二日、第三日的最高价/购买价的比例
# 3、新的积分和code写入库中的分析表stock_rating_history中，包含字段symbol,stockname, alert_date, score, alert_time,second_day,thirty_day,生成创建sql，并入库
# 4、获取50分以上的数据，比较second_day,thirty_day的大小，并按照倒序排列，保存为通达信的一个板块，新的板块名：板块名+001
# 请参考这个程序的内容，完成上述要求，测试结果


# 新增代码：读取配置文件
config_path = os.path.join(os.path.dirname(__file__), '..', 'config', 'config.json')
with open(config_path, 'r', encoding='utf-8') as config_file:  # 修改编码为utf-8
    config = json.load(config_file)

db_config = config['db_config']

reader = Reader.factory(market='std', tdxdir=config['tdxdir'])

# 定义缓存目录
CACHE_DIR = "cache"

# 确保缓存目录存在
os.makedirs(CACHE_DIR, exist_ok=True)

def get_cache_filename(code):
    """
    根据代码生成缓存文件名
    """
    # 使用哈希函数生成唯一的文件名
    hashed_code = hashlib.md5(code.encode()).hexdigest()
    return os.path.join(CACHE_DIR, f"{hashed_code}.json")

def load_cache(code):
    """
    从本地文件加载缓存数据
    """
    cache_file = get_cache_filename(code)
    if os.path.exists(cache_file):
        try:
            with open(cache_file, "r") as file:
                cache_data = json.load(file)
                if cache_data.get("timestamp") == time.strftime("%Y-%m-%d"):
                    return cache_data.get("data")
                else:
                    print(f"缓存文件 {cache_file} 已过期，将重新抓取数据")
                    return None
        except json.JSONDecodeError:
            print(f"警告：缓存文件 {cache_file} 格式错误，将忽略此缓存文件")
            return None
    return None

def save_cache(code, data):
    """
    将数据保存到本地文件
    """
    cache_file = get_cache_filename(code)
    cache_data = {
        "timestamp": time.strftime("%Y-%m-%d"),
        "data": data
    }
    try:
        with open(cache_file, "w") as file:
            json.dump(cache_data, file, ensure_ascii=False, indent=4)
            print(f"成功保存缓存文件 {cache_file}")
    except Exception as e:
        print(f"警告：无法保存缓存文件 {cache_file}，错误信息：{e}")

def get_stock_rating_data(symbol,date):
    """
    从 stock_rating 表中读取机构参与度、历史评分、用户关注指数和日度市场参与意愿
    :param symbol: 股票代码
    :return: 包含机构参与度、历史评分、用户关注指数和日度市场参与意愿的字典
    """
    with mysql.connector.connect(**db_config) as conn:
        cursor = conn.cursor()
        cursor.execute('''
            SELECT avg_jgcyd, avg_lspf, avg_focus, last_desire_daily,free_float_value
            FROM stock_rating
            WHERE symbol = %s and rating_date = %s
            ORDER BY rating_date DESC
            LIMIT 1
        ''', (symbol,date))
        result = cursor.fetchone()
        if result:
            return {
                "avg_jgcyd": result[0],
                "avg_lspf": result[1],
                "avg_focus": result[2],
                "last_desire_daily": result[3],
                "free_float_value": result[4]
            }
        else:
            return {
                "avg_jgcyd": None,
                "avg_lspf": None,
                "avg_focus": None,
                "last_desire_daily": None,
                "free_float_value": None
            }

def is_new_high(stock_data):
    """
    判断收盘价是否创1年新高
    :param stock_data: 包含历史收盘价数据的DataFrame
    :return: 如果创1年新高返回True，否则返回False
    """
    if len(stock_data) < 252:
        return False
    latest_close = stock_data['Close'].iloc[-1]
    max_close_past_year = stock_data['Close'].iloc[-252:].max()

    if latest_close >= max_close_past_year:
        high_rating = 10  # 假设创1年新高增加10分
    else:
        high_rating = 0
    return high_rating
#
# 1、近期成交额（15%），分成3个级别：5日内平均成交额10亿以上100分，5-10亿 60分，5亿以下0分
# 2、近期涨幅（10%），分成3个级别：20日内涨幅30%以上 100分，20%以内 50分 10%以内 0分
# 3、流通市值（10%），分成3个级别：50亿-300亿 100分，300-500亿 50分，50亿以下 0分
# 4、振幅（10）分成3个级别：50% 100分，30%-50% 50分，30%以内 0分
# 5、机构占比（5%）：机构占20%以上，0分，机构10-20% 50分，小于10%  100分
# 6、龙虎榜分析（5%），如果在里面，主力流入 100分，流入-1000万以内 50分，流出0
# 7、增加新闻报道分析（20%），如果多，则认为活跃度够，和热门板块重合
# 8、预估今日成交量和金额，10亿以上100分，5-10亿 60分，5亿以下0分
#
# 上述是对个股的评价系统，请用python 和akshare实现 ，后续需要进行微调，对各项分数比例进行评估和调整

# 新增下面几个维度的参考值，东方财富评级
# 9、机构调研（5%）：机构调研，如果多，则认为活跃度够，和热门板块重合
# 10、创1年新高（5%）：创1年新高
# 11、当日预估成交额（5%）：预估今日成交量和金额，10亿以上100分，5-10亿 60分，5亿以下0分


# 机构参与度
# 历史评分
# 用户关注指数
# 日度市场参与意愿

# 配置评分规则和权重
SCORE_RULES = {
    "recent_turnover": {"weight": 0.20, "levels": [(15, 100),(10, 80), (5, 60), (0, 0)]},  # 近期成交额
    "recent_increase": {"weight": 0.10, "levels": [(60, 100), (40, 50), (10, 0)]},  # 近期涨幅
    "market_cap": {"weight": 0.15, "levels": [(500, 0),(200, 50), (30, 100), (20, 0)]},  # 流通市值
    "amplitude": {"weight": 0.10, "levels": [(50, 100), (30, 50), (0, 0)]},  # 振幅 +5
    "jgcyd": {"weight": 0.10, "levels": [(50, 0), (42, 100), (30, 80), (0, 0)]},  # 机构参与度
    "lspf": {"weight": 0.10, "levels": [(67, 100), (60, 50), (0, 0)]},  # 历史评分
    "focus": {"weight": 0.10, "levels": [(87, 100), (80, 80), (0, 0)]},  # 用户关注指数
    "desire_daily": {"weight": 0.10, "levels": [(5, 100), (3, 50), (0, 0)]},  # 日度市场参与意愿  意义不大，可替换
    "dragon_tiger": {"weight": 0.00, "levels": [("inflow", 100), ("small_inflow", 50), ("outflow", 0)]},  # 龙虎榜
    "news_analysis": {"weight": 0.00, "levels": [(True, 100), (False, 0)]},  # 新闻报道分析
    "estimated_turnover": {"weight": 0.00, "levels": [(10, 100), (5, 60), (0, 0)]},  # 预估成交额
}

# 缓存字典
data_cache = {}
from stockrating.read_local_info import read_stock_market_value_from_db,get_stock_info
result_data = read_stock_market_value_from_db()

# 获取股票数据
def get_stock_data(symbol,date):
    """
    获取股票的基本数据、历史数据和龙虎榜数据
    """
    # 检查缓存中是否有数据,测试不用cache数据
    # cached_data = load_cache(symbol)
    # if cached_data:
    #     print(f"从缓存中获取 {symbol} 的数据")
    #     return cached_data

    # 定义返回数据字典
    return_data = {}
    return_data["code"] = symbol

    # 从 stock_rating 表中读取机构参与度、历史评分、用户关注指数和日度市场参与意愿
    rating_data = get_stock_rating_data(symbol,date)
    print(rating_data)
    return_data.update(rating_data)

    info = get_stock_info(result_data,symbol)
    print( info)
    # print(info)
    return_data["stockname"] = info["name"].values[0]
    # return_data["free_float_value"] = info["circulating_market_value"].values[0] / 1e8
    # 尝试将流通市值转换为数值类型
    try:
        free_float_value = pd.to_numeric(info["circulating_market_value"].values[0], errors='coerce')
        if pd.isna(free_float_value):
            raise ValueError("无法将流通市值转换为数值类型")
        return_data["free_float_value"] = free_float_value / 1e8
    except Exception as e:
        print(f"警告：获取或转换流通市值时出错：{e}")
        return_data["free_float_value"] = None



    # 先尝试从本地读取历史数据
    try:
        daily_data = reader.daily(symbol=symbol)
        # 打印 DataFrame 的列名
        # print("DataFrame 列名:", daily_data.columns)
        # 打印 DataFrame 的索引
        # print("DataFrame 索引:", daily_data.index)
        # 将日期索引转换为字符串
        daily_data.index = daily_data.index.strftime('%Y-%m-%d')
        # 将 DataFrame 转换为 JSON 格式，并带上索引
        return_data["stock_history"] = json.dumps(daily_data.reset_index().to_dict('records'), ensure_ascii=False)
        # print("日线数据：", return_data["stock_history"])
    except Exception as e:
        print(f"警告：从本地读取历史数据时出错：{e}")
        # 如果本地读取失败，再从网络获取历史行情数据
        try:
            stock_history = ak.stock_zh_a_hist(symbol=symbol, period="daily", adjust="qfq")
            if stock_history is not None:
                # 将日期字段转换为字符串
                stock_history_dict = stock_history.to_dict('records')
                for record in stock_history_dict:
                    record['日期'] = record['日期'].strftime('%Y-%m-%d')  # 将日期转换为字符串
                return_data["stock_history"] = json.dumps(stock_history_dict, ensure_ascii=False)
            else:
                print(f"警告：获取股票 {symbol} 的历史行情数据失败")
        except Exception as e:
            print(f"警告：获取股票 {symbol} 的历史行情数据时出错：{e}")

    # 将数据存入缓存
    data_cache = return_data
    save_cache(symbol, data_cache)
    return data_cache

# 清除缓存
def clear_cache():
    global data_cache
    data_cache = {}
    print("缓存已清除")

# 评分函数
def calculate_score(value, rule):
    """
    根据规则计算分数
    """
    for threshold, score in rule["levels"]:
        if isinstance(threshold, str):  # 处理龙虎榜等非数值规则
            if value == threshold:
                return score
        else:
            if value >= threshold:
                return score
    return 0
# //计算macd返回值
def calculate_macd(stock_history, date):
    # 将 JSON 字符串反序列化为 DataFrame
    stock_history_df = pd.DataFrame(json.loads(stock_history))
    # 将日期列设置为索引
    stock_history_df.set_index('date', inplace=True)
    macd = get_macd(stock_history_df)
    # 将 alert_date 转换为 datetime 对象
    cur_date = date.strftime("%Y-%m-%d")
    print(f"cur_date {cur_date}")

    # 从 DataFrame 中获取指定日期的 MACD 值
    macd_value = macd.loc[macd.index == cur_date, 'MACD'].values[0] if cur_date in macd.index else None
    print(f"macd_value {macd_value} ")

    # 获取第二日
    # 将日期转换为 datetime 对象以便比较
    cur_date_dt = datetime.strptime(cur_date, "%Y-%m-%d")
    # 获取 cur_date_dt 之前的两条数据
    prev_days = macd[macd.index < cur_date].tail(1)
    # 得到第一条数据
    prev_macd_value = prev_days['MACD'].values[0] if not prev_days.empty else None
    print(f"prev_macd_value {prev_macd_value}")

    if macd_value is not None and prev_macd_value is not None and macd_value > prev_macd_value:
        return 1
    else:
        return 0

# 计算各项得分
def calculate_symbol_score(symbol,date):
    """
    对股票进行评分
    """
    # 将date转换为时间
    date = datetime.strptime(date, "%Y%m%d")
    print( date)

    stock_data = get_stock_data(symbol,date)
    # print( stock_data)
    # exit()
    stock_history = json.loads(stock_data["stock_history"])  # 将 JSON 字符串反序列化为字典列表

    # 获取股票名称
    stockname = stock_data["stockname"]

    # 截取date日期以前的数据
    stock_history = [entry for entry in stock_history if datetime.strptime(entry["date"], "%Y-%m-%d") <= date]

    # 1. 近期成交额（15%） 成交量*均价 = 成交额  3天 有效
    turnover_array = np.array([entry["amount"] for entry in stock_history[-3:]])
    recent_turnover = turnover_array.mean() / 1e8  # 转换为亿
    turnover_score = calculate_score(recent_turnover, SCORE_RULES["recent_turnover"])

    # 2. 近期涨幅（10%）
    close_prices = np.array([entry["close"] for entry in stock_history])
    recent_increase = (close_prices[-1] - close_prices[-20]) / close_prices[-20] * 100
    increase_score = calculate_score(recent_increase, SCORE_RULES["recent_increase"])

    # 3. 流通市值（10%） f117
    # print("流通市值：", stock_data["free_float_value"])
    # market_cap = stock_data["free_float_value"] / 1e8  # 转换为亿
    market_cap = float(stock_data["free_float_value"])
    market_cap_score = calculate_score(market_cap, SCORE_RULES["market_cap"])

    # 4. 振幅（10%）
    high_prices = np.array([entry["high"] for entry in stock_history[-20:]])  # 修改为最近20天
    low_prices = np.array([entry["low"] for entry in stock_history[-20:]])  # 修改为最近20天
    amplitude = (high_prices.max() - low_prices.min()) / low_prices.min() * 100
    amplitude_score = calculate_score(amplitude, SCORE_RULES["amplitude"])

    # 机构参与度
    # 历史评分
    # 用户关注指数
    # 日度市场参与意愿

    # 这4个数据从表stock_rating里读出来
    # 5. 机构参与度（5%）
    jgcyd_score = calculate_score(stock_data['avg_jgcyd'], SCORE_RULES["jgcyd"])

    # 6. 历史评分（5%）
    lspf_score = calculate_score(stock_data['avg_lspf'], SCORE_RULES["lspf"])

    # 7. 用户关注指数（5%）
    focus_score = calculate_score(stock_data['avg_focus'], SCORE_RULES["focus"])

    # 8. 日度市场参与意愿（5%）
    desire_daily_score = calculate_score(stock_data['last_desire_daily'], SCORE_RULES["desire_daily"])

    # 9. 龙虎榜分析（5%）
    # if not dragon_tiger_data.empty:
    #     net_inflow = dragon_tiger_data["净买入额"].sum() / 1e4  # 转换为万元
    #     if net_inflow > 0:
    #         dragon_tiger_score = 100
    #     elif net_inflow > -1000:
    #         dragon_tiger_score = 50
    #     else:
    #         dragon_tiger_score = 0
    # else:
    #     dragon_tiger_score = 0
    dragon_tiger_score = 0

    # 10. 新闻报道分析（20%）
    # 假设通过某种方式获取新闻报道数量（这里用随机值模拟）
    # news_count = np.random.randint(0, 10)  # 模拟新闻报道数量
    news_score = 0

    #11 增加当日收盘价创1年新高的评分（盘中数据，和预测相关）
    high_rating = is_new_high(stock_data)

    # 13. 预估成交量和金额（25%）
    estimated_turnover = recent_turnover/10  # 假设预估成交额等于近期成交额
    estimated_score = calculate_score(estimated_turnover, SCORE_RULES["estimated_turnover"])

    # 计算各项得分
    total_score = (
            turnover_score * SCORE_RULES["recent_turnover"]["weight"]
            + increase_score * SCORE_RULES["recent_increase"]["weight"]
            + market_cap_score * SCORE_RULES["market_cap"]["weight"]
            + amplitude_score * SCORE_RULES["amplitude"]["weight"]
            + stock_data["avg_jgcyd"] * SCORE_RULES["jgcyd"]["weight"]
            + stock_data["avg_lspf"] * SCORE_RULES["lspf"]["weight"]
            + stock_data["avg_focus"] * SCORE_RULES["focus"]["weight"]
            # + stock_data["last_desire_daily"] * SCORE_RULES["desire_daily"]["weight"]
            # + dragon_tiger_score * SCORE_RULES["dragon_tiger"]["weight"]
            + high_rating
            # + dragon_tiger_score * SCORE_RULES["dragon_tiger"]["weight"]
            + news_score * SCORE_RULES["news_analysis"]["weight"]
            + estimated_score * SCORE_RULES["estimated_turnover"]["weight"]
    )

    # 将 numpy 数值类型转换为 Python 原生类型
    recent_turnover = float(turnover_array.mean() / 1e8)  # 转换为亿
    recent_increase = float((close_prices[-1] - close_prices[-20]) / close_prices[-20] * 100)
    market_cap = float(stock_data["free_float_value"])  # 转换为亿
    amplitude = float((high_prices.max() - low_prices.min()) / low_prices.min() * 100)
    avg_jgcyd = float(stock_data['avg_jgcyd']) if stock_data['avg_jgcyd'] is not None else None
    avg_lspf = float(stock_data['avg_lspf']) if stock_data['avg_lspf'] is not None else None
    avg_focus = float(stock_data['avg_focus']) if stock_data['avg_focus'] is not None else None
    last_desire_daily = float(stock_data['last_desire_daily']) if stock_data['last_desire_daily'] is not None else None
    free_float_value = float(stock_data['free_float_value']) if stock_data['free_float_value'] is not None else None

    # rating_date = datetime.date.strftime(datetime.date.today(),"%Y%m%d")
    rating_date = date

    second_day, third_day = calculate_second_and_third_day_ratio(stock_data, date)
    print(second_day,third_day)

    macd = calculate_macd(stock_data["stock_history"],date)
    print(macd)
    # 比较前一天macd，为正则是1，为负则是0
    # 插入数据到 stock_rating_history 表
    conn = mysql.connector.connect(**db_config)
    cursor = conn.cursor()
    cursor.execute('''
        INSERT INTO stock_rating_history (
            symbol,stockname, rating_date, recent_turnover, recent_increase, market_cap, amplitude,
            jgcyd, lspf, focus, desire_daily, dragon_tiger, news_analysis,
            estimated_turnover, total_score,second_day,third_day, avg_jgcyd, avg_lspf, avg_focus,
            last_desire_daily,free_float_value,macd
        ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
    ''', (
        symbol, stockname, rating_date, turnover_score, increase_score, market_cap_score, amplitude_score,
        jgcyd_score, lspf_score, focus_score, desire_daily_score,dragon_tiger_score,
        news_score, estimated_score, total_score,second_day,third_day,
        stock_data["avg_jgcyd"], stock_data["avg_lspf"], stock_data["avg_focus"], stock_data["last_desire_daily"],stock_data['free_float_value'],macd
    ))

    conn.commit()
    conn.close()

    print(f" {symbol} 的评分：{total_score}")
    return total_score






def get_stock_codes():
    """
    获取所有股票代码，并缓存结果
    """
    cache_file = os.path.join(CACHE_DIR, "stock_codes.json")
    
    # 检查缓存是否存在且未过期
    if os.path.exists(cache_file):
        try:
            with open(cache_file, "r") as file:
                cache_data = json.load(file)
                if cache_data.get("timestamp") == time.strftime("%Y-%m-%d"):
                    print("从缓存中获取股票代码")
                    return cache_data.get("data")
        except json.JSONDecodeError:
            print("警告：缓存文件格式错误，将重新抓取数据")
    
    # 从网络获取数据
    print("从网络获取股票代码")
    stock_info = ak.stock_info_a_code_name()
    stock_codes = stock_info['code'].tolist()
    
    # 保存到缓存
    cache_data = {
        "timestamp": time.strftime("%Y-%m-%d"),
        "data": stock_codes
    }
    try:
        with open(cache_file, "w") as file:
            json.dump(cache_data, file, ensure_ascii=False, indent=4)
            print("成功保存股票代码缓存")
    except Exception as e:
        print(f"警告：无法保存股票代码缓存，错误信息：{e}")
    
    return stock_codes

def calculate_all_stock_score():
    # # 修改执行部分
    symbols = get_stock_codes()
    print(f"获取到 {len(symbols)} 个股票代码")

    # 执行给定数组中的所有股票代码
    for symbol in symbols:
        print(f"正在处理股票代码：{symbol}")
        try:
            result =  calculate_symbol_score(symbol, result_data)
            print(f"股票评分结果：{symbol}", result)
            time.sleep(1)  # 添加1秒延迟
        except Exception as e:
            print(f"处理股票代码 {symbol} 时出错：{e}")
def get_macd(data):

    # 编写计算函数
    # 上市首日，DIFF、DEA、MACD = 0
    # 用来装变量的list
    EMA12 = []
    EMA26 = []
    DIFF = []
    DEA = []
    BAR = []
    # 如果是上市首日
    if len(data) == 1:
        # 则DIFF、DEA、MACD均为0
        DIFF = [0]
        DEA = [0]
        BAR = [0]

    # 如果不是首日
    else:
        # 第一日的EMA要用收盘价代替
        EMA12.append(data['close'].iloc[0])
        EMA26.append(data['close'].iloc[0])
        DIFF.append(0)
        DEA.append(0)
        BAR.append(0)

        # 计算接下来的EMA
        # 搜集收盘价
        close = list(data['close'].iloc[1:])    # 从第二天开始计算，去掉第一天
        for i in close:
            ema12 = EMA12[-1] * (11/13) + i * (2/13)
            ema26 = EMA26[-1] * (25/27) + i * (2/27)
            diff = ema12 - ema26
            dea = DEA[-1] * (8/10) + diff * (2/10)
            bar = 2 * (diff - dea)

            # 将计算结果写进list中
            EMA12.append(ema12)
            EMA26.append(ema26)
            DIFF.append(diff)
            DEA.append(dea)
            BAR.append(bar)

    # 返回全部的macd
    MACD = pd.DataFrame({'DIFF':DIFF,'DEA':DEA,'MACD':BAR})
    # 将计算出的 MACD 值直接添加到 data 中
    data['DIFF'] = DIFF
    data['DEA'] = DEA
    data['MACD'] = BAR

    return data

def calculate_second_and_third_day_ratio(stock_data, alert_date):
    """
    计算以当日收盘价购买后，第二日、第三日的最高价/购买价的比例
    :param stock_data: 股票历史数据
    :param alert_date: 提醒日期
    :return: (second_day_ratio, third_day_ratio)
    """
    stock_history = json.loads(stock_data["stock_history"])
    # 将 alert_date 转换为字符串（如果它是 datetime 对象）
    if isinstance(alert_date, datetime):
        alert_date = alert_date.strftime("%Y%m%d")
    # 将 alert_date 转换为 datetime 对象
    cur_date = datetime.strptime(alert_date, "%Y%m%d").strftime("%Y-%m-%d")
    print(f"cur_date {cur_date}")
    buy_price = next((entry["close"] for entry in stock_history if entry["date"] == cur_date), None)
    print(f"buy_price {buy_price} ")

    if not buy_price:
        return 0, 0

    # 获取第二日和第三日的最高价
    # 将日期转换为 datetime 对象以便比较
    cur_date_dt = datetime.strptime(cur_date, "%Y-%m-%d")
    # 获取 cur_date_dt 之后的两条数据
    next_two_days = [entry for entry in stock_history if datetime.strptime(entry["date"], "%Y-%m-%d") > cur_date_dt][:2]
    # 得到第一条数据
    first_data = next_two_days[0] if next_two_days else None
    # 得到第二条数据
    second_data = next_two_days[1] if len(next_two_days) > 1 else None
    # 计算这两条数据的最高价
    # second_day_high = max(entry["high"] for entry in next_two_days) if next_two_days else 0
    # second_day_high = next((entry["high"] for entry in stock_history if datetime.strptime(entry["date"], "%Y-%m-%d") > cur_date_dt), 0)
    second_day_high = first_data["high"]
    print(f"正在计算 {second_day_high} 第二日")
    third_day_high = second_data["high"]  # 由于只取了两条数据，第三日与第二日相同
    print(f"正在计算 {third_day_high} 第三日")

    second_day_ratio = ((second_day_high-buy_price) / buy_price) * 100 if second_day_high else 0
    third_day_ratio = ((third_day_high-buy_price) / buy_price) * 100 if third_day_high else 0

    return second_day_ratio, third_day_ratio

def save_to_stock_rating_history(symbol, stockname, alert_date, score, second_day, third_day):
    """
    将评分结果保存到 stock_rating_history 表
    :param symbol: 股票代码
    :param stockname: 股票名称
    :param alert_date: 提醒日期
    :param score: 评分
    :param second_day: 第二日收益比例
    :param third_day: 第三日收益比例
    """
    with mysql.connector.connect(**db_config) as conn:
        cursor = conn.cursor()
        cursor.execute('''
            INSERT INTO stock_rating_history (
                symbol, stockname, alert_date, score, second_day, third_day
            ) VALUES (%s, %s, %s, %s, %s, %s)
        ''', (symbol, stockname, alert_date, score, second_day, third_day))
        conn.commit()

def save_to_tdx_block(block_name, stock_list):
    """
    将股票列表保存为通达信板块文件
    :param block_name: 板块名称
    :param stock_list: 股票代码列表
    """
    new_block_name = f"{block_name}001"
    block_path = os.path.join(config['tdxdir'], 'T0002', 'blocknew', f"{new_block_name}.blk")

    with open(block_path, 'w', encoding='gbk') as f:
        for symbol in stock_list:
            f.write(f"{symbol}\n")

    print(f"成功保存板块 {new_block_name} 到 {block_path}")
def process_block(block_name):
    """
    处理指定板块的股票
    :param block_name: 板块名称
    """
    # 读取板块数据
    stock_list = read_tdx_block_data(block_name)
    if not stock_list:
        print(f"板块 {block_name} 数据为空，无法计算积分")
        return

    high_score_stocks = []
    
    for symbol in stock_list:
        # stock_data = get_stock_data(symbol,"20250408")
        year = datetime.today().strftime("%Y")
        # second_day, third_day = calculate_second_and_third_day_ratio(stock_data, alert_date)
        score =  calculate_symbol_score(symbol, f"{year}{block_name}")
        
        if score >= 40:
            high_score_stocks.append((symbol))
        #     save_to_stock_rating_history(symbol, stock_data["stockname"], alert_date, score, second_day, third_day)
        #
    # 按照 second_day 和 third_day 的比例倒序排列
    high_score_stocks.sort(key=lambda x: max(x[1], x[2]), reverse=True)
    print(high_score_stocks)
    save_to_tdx_block(block_name, [stock[0] for stock in high_score_stocks])

if __name__ == '__main__':


    # 单个股票评分
    # calculate_symbol_score("002570","20250408")

    block_name = "0401"
    process_block(block_name)
