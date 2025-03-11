import akshare as ak
import pandas as pd
import numpy as np
import json
import os
import hashlib
import time
import pywencai

from mootdx.reader import Reader

# 创建 Reader 对象
# reader = Reader.factory(market='std', tdxdir='D:/new_haitong/')
reader = Reader.factory(market='std', tdxdir='D:/zd_haitong/')



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
                if cache_data.get("timestamp") <= time.strftime("%Y-%m-%d"):
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
    "market_cap": {"weight": 0.15, "levels": [(500, 0),(200, 50), (35, 100), (20, 0)]},  # 流通市值
    "amplitude": {"weight": 0.10, "levels": [(50, 100), (30, 50), (0, 0)]},  # 振幅 +5
    "jgcyd": {"weight": 0.10, "levels": [(50, 0), (42, 50), (32, 60), (0, 100)]},  # 机构参与度
    "lspf": {"weight": 0.10, "levels": [(67, 100), (60, 50), (0, 0)]},  # 历史评分
    "focus": {"weight": 0.10, "levels": [(87, 100), (80, 50), (0, 0)]},  # 用户关注指数
    "desire_daily": {"weight": 0.10, "levels": [(5, 100), (3, 50), (0, 0)]},  # 日度市场参与意愿  意义不大，可替换
    "dragon_tiger": {"weight": 0.00, "levels": [("inflow", 100), ("small_inflow", 50), ("outflow", 0)]},  # 龙虎榜
    "news_analysis": {"weight": 0.00, "levels": [(True, 100), (False, 0)]},  # 新闻报道分析
    "estimated_turnover": {"weight": 0.00, "levels": [(10, 100), (5, 60), (0, 0)]},  # 预估成交额
}

# 缓存字典
data_cache = {}

# 获取股票数据
def get_stock_data(symbol):
    """
    获取股票的基本数据、历史数据和龙虎榜数据
    """
    # 检查缓存中是否有数据
    cached_data = load_cache(symbol)
    if cached_data:
        print(f"从缓存中获取 {symbol} 的数据")
        return cached_data

    # 定义返回数据字典
    return_data = {}
    return_data["code"] = symbol
    jgcyd= ak.stock_comment_detail_zlkp_jgcyd_em(symbol=symbol)
    # print("机构参与度：", jgcyd)
    # return_data["jgcyd"]=jgcyd
    # print(ak.__version__)

    # 计算机构参与度的平均值
    if not jgcyd.empty and "机构参与度" in jgcyd.columns:
        avg_jgcyd = jgcyd["机构参与度"].tail(5).mean()
    else:
        avg_jgcyd = None

    return_data["avg_jgcyd"]=avg_jgcyd

    lspf= ak.stock_comment_detail_zhpj_lspf_em(symbol=symbol)
    # print("综合评价-历史评分：", lspf)
    if not lspf.empty and "评分" in lspf.columns:
        avg_lspf = lspf["评分"].tail(1).mean()
    else:
        avg_lspf = None
    return_data["avg_lspf"]=avg_lspf

    focus= ak.stock_comment_detail_scrd_focus_em(symbol=symbol)
    # print("市场热度-用户关注指数：", focus)
    if not focus.empty and "用户关注指数" in focus.columns:
        avg_focus = focus["用户关注指数"].tail(3).mean()
    else:
        avg_focus = None
    return_data["avg_focus"]=avg_focus

    desire_daily= ak.stock_comment_detail_scrd_desire_daily_em(symbol=symbol)
    print("市场热度-日度市场参与意愿：", desire_daily)
    if not desire_daily.empty and "5日平均参与意愿变化" in desire_daily.columns:
        last_desire_daily = desire_daily["5日平均参与意愿变化"].iloc[-1]
    else:
        last_desire_daily = None
    return_data["last_desire_daily"]=last_desire_daily

    info = get_stock_info(result_data,symbol)
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

    # # 先尝试使用pywencai获取流通市值和股票名称
    # try:
    #     df = pywencai.get(query= f"{symbol} 流通市值")
    #     text = df['txt1']  # 获取第一个匹配结果的文本
    #     # 使用字符串查找和切片提取流通市值
    #     start_index = text.find("流通市值为") + len("流通市值为")
    #     end_index = text.find("亿元", start_index)
    #     circulating_market_value = text[start_index:end_index]
    #
    #     # 提取股票名称
    #     start_index_1 = text.find("日") + len("日")
    #     end_index_1 = text.find("的", start_index_1)
    #     name = text[start_index_1:end_index_1]
    #
    #     print(f"流通市值为: {circulating_market_value} 亿元")
    #     print(f"股票名称为: {name}")
    #     result_data【symbol】 = result_data
    #
    #
    #
    #     return_data["stockname"] = name
    #     return_data["free_float_value"] = circulating_market_value
    #     print(f"流通市值为: {circulating_market_value} 亿元")
    # except Exception as e:
    #     print(f"警告：使用pywencai获取流通市值和股票名称时出错：{e}")
    #     # 如果pywencai失败，再尝试使用akshare获取股票基本信息
    #     try:
    #         stock_info = ak.stock_individual_info_em(symbol=symbol)
    #         if stock_info is None:
    #             print(f"警告：股票代码 {symbol} 未找到相关信息，请检查代码格式是否正确。")
    #             return None, None, None
    #     except KeyError:
    #         print(f"警告：股票代码 {symbol} 未找到相关信息，请检查代码格式是否正确。")
    #         stock_info = None
    #
    #     if stock_info is not None:
    #         print("股票基本信息：", stock_info)
    #         print("流通市值：", stock_info.iloc[5]["value"])
    #         return_data["stockname"] = stock_info.iloc[1]["value"]
    #         return_data["free_float_value"] = stock_info.iloc[5]["value"]

    # 先尝试从本地读取历史数据
    try:
        daily_data = reader.daily(symbol=symbol)
        # print("日线数据：", daily_data)
        return_data["stock_history"] = json.dumps(daily_data.to_dict('records'), ensure_ascii=False)
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

# 计算各项得分
def evaluate_stock(symbol):
    """
    对股票进行评分
    """
    stock_data = get_stock_data(symbol)
    # print("stock_data:", stock_data)
    stock_history = json.loads(stock_data["stock_history"])  # 将 JSON 字符串反序列化为字典列表

    # 获取股票名称
    stockname = stock_data["stockname"]

    # 1. 近期成交额（15%） 成交量*均价 = 成交额
    turnover_array = np.array([entry["amount"] for entry in stock_history[-5:]])
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

    # 5. 机构参与度（5%）
    jgcyd_score = calculate_score(stock_data['avg_jgcyd'], SCORE_RULES["jgcyd"])

    # 6. 历史评分（5%）
    lspf_score = calculate_score(stock_data['avg_lspf'], SCORE_RULES["lspf"])

    # 7. 用户关注指数（5%）
    focus_score = calculate_score(stock_data['avg_focus'], SCORE_RULES["focus"])

    # 8. 日度市场参与意愿（5%）
    desire_daily_score = calculate_score(stock_data['last_desire_daily'], SCORE_RULES["desire_daily"])

    # 6. 龙虎榜分析（5%）
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

    # 7. 新闻报道分析（20%）
    # 假设通过某种方式获取新闻报道数量（这里用随机值模拟）
    # news_count = np.random.randint(0, 10)  # 模拟新闻报道数量
    news_score = 0

    # 增加当日收盘价创1年新高的评分（盘中数据，和预测相关）
    high_rating = is_new_high(stock_data)

    # 8. 预估成交量和金额（25%）
    estimated_turnover = recent_turnover/10  # 假设预估成交额等于近期成交额
    estimated_score = calculate_score(estimated_turnover, SCORE_RULES["estimated_turnover"])

    # 计算各项得分
    total_score = (
            turnover_score * SCORE_RULES["recent_turnover"]["weight"]
            + increase_score * SCORE_RULES["recent_increase"]["weight"]
            + market_cap_score * SCORE_RULES["market_cap"]["weight"]
            + amplitude_score * SCORE_RULES["amplitude"]["weight"]
            + jgcyd_score * SCORE_RULES["jgcyd"]["weight"]
            + lspf_score * SCORE_RULES["lspf"]["weight"]
            + focus_score * SCORE_RULES["focus"]["weight"]
            # + desire_daily_score * SCORE_RULES["desire_daily"]["weight"]
            # + dragon_tiger_score * SCORE_RULES["dragon_tiger"]["weight"]
            + high_rating
            # + dragon_tiger_score * SCORE_RULES["dragon_tiger"]["weight"]
            + news_score * SCORE_RULES["news_analysis"]["weight"]
            + estimated_score * SCORE_RULES["estimated_turnover"]["weight"]
    )

    # 构建返回结果
    result = {
        "symbol": symbol,
        "total_score": total_score,
        "scores": {
            "recent_turnover": turnover_score,
            "recent_increase": increase_score,
            "market_cap": market_cap_score,
            "amplitude": amplitude_score,
            "jgcyd": jgcyd_score,
            "lspf": lspf_score,
            "focus": focus_score,
            "high_rating": high_rating,
            "desire_daily": desire_daily_score,
            "dragon_tiger": dragon_tiger_score,
            "news_analysis": news_score,
            "estimated_turnover": estimated_score,
        },
        "source_data": {
            "avg_jgcyd": stock_data['avg_jgcyd'],
            "avg_lspf": stock_data['avg_lspf'],
            "avg_focus": stock_data['avg_focus'],
            "last_desire_daily": stock_data['last_desire_daily'],
            "free_float_value": stock_data['free_float_value']
        }
    }

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

    # 将评分结果和基础数据保存到 MySQL 数据库
    import mysql.connector
    # 数据库连接配置
    db_config = {
        "host": "localhost",  # 数据库主机地址
        "user": "root",  # 数据库用户名
        "password": "111111",  # 数据库密码
        "database": "trade"  # 数据库名称
    }
    conn = mysql.connector.connect(**db_config)
    cursor = conn.cursor()

    # 插入数据到 stock_rating 表
    cursor.execute('''
        INSERT INTO stock_rating (
            symbol,stockname, recent_turnover, recent_increase, market_cap, amplitude,
            jgcyd, lspf, focus, desire_daily, dragon_tiger, news_analysis,
            estimated_turnover, total_score, avg_jgcyd, avg_lspf, avg_focus,
            last_desire_daily, free_float_value
        ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
    ''', (
        symbol, stockname, turnover_score, increase_score, market_cap_score, amplitude_score,
        jgcyd_score, lspf_score, focus_score, desire_daily_score, dragon_tiger_score,
        news_score, estimated_score, total_score, avg_jgcyd,
        avg_lspf, avg_focus, last_desire_daily, free_float_value
    ))

    conn.commit()
    conn.close()

    print(f"股票 {symbol} 的评分：{total_score}")
    return total_score

# 测试
# symbol = "300718"  # 长盛轴承
# symbol = "300153"
# symbol = "300173"
# symbol = "300814"
# symbol = "300687"

# "recent_turnover": {"weight": 0.15, "levels": [(10, 100), (5, 60), (0, 0)]},  # 近期成交额
# "recent_increase": {"weight": 0.10, "levels": [(30, 100), (20, 50), (10, 0)]},  # 近期涨幅
# "market_cap": {"weight": 0.10, "levels": [(300, 100), (500, 50), (50, 0)]},  # 流通市值
# "amplitude": {"weight": 0.10, "levels": [(50, 100), (30, 50), (0, 0)]},  # 振幅
# "institution_ratio": {"weight": 0.05, "levels": [(20, 0), (10, 50), (0, 100)]},  # 机构占比
# "dragon_tiger": {"weight": 0.05, "levels": [("inflow", 100), ("small_inflow", 50), ("outflow", 0)]},  # 龙虎榜
# "news_analysis": {"weight": 0.20, "levels": [(True, 100), (False, 0)]},  # 新闻报道分析
# "estimated_turnover": {"weight": 0.25, "levels": [(10, 100), (5, 60), (0, 0)]},  # 预估成交额

# 股票评分结果： {'symbol': '300083', 'total_score': 70.0, 'scores': {'recent_turnover': 100, 'recent_increase': 100, 'market_cap': 100, 'amplitude': 100, 'institution_ratio': 0, 'dragon_tiger': 0, 'news_analysis': 0, 'estimated_turnover': 100}}
# symbol = "300083"
# 3月4日 失败 股票评分结果： {'symbol': '688393', 'total_score': 40.0, 'scores': {'recent_turnover': 0, 'recent_increase': 100, 'market_cap': 0, 'amplitude': 100, 'institution_ratio': 0, 'dragon_tiger': 0, 'news_analysis': 100, 'estimated_turnover': 0}}
# symbol = "688393"
# 3月4日 失败 股票评分结果： {'symbol': '301382', 'total_score': 60.0, 'scores': {'recent_turnover': 100, 'recent_increase': 100, 'market_cap': 0, 'amplitude': 100, 'institution_ratio': 0, 'dragon_tiger': 0, 'news_analysis': 0, 'estimated_turnover': 100}}
# symbol = "301382"
# 3月4日  股票评分结果： {'symbol': '688234', 'total_score': 55.0, 'scores': {'recent_turnover': 100, 'recent_increase': 50, 'market_cap': 0, 'amplitude': 100, 'institution_ratio': 0, 'dragon_tiger': 0, 'news_analysis': 0, 'estimated_turnover': 100}}
# symbol = "688234"
# 3月4日 股票评分结果： {'symbol': '301366', 'total_score': 80.0, 'scores': {'recent_turnover': 100, 'recent_increase': 100, 'market_cap': 0, 'amplitude': 100, 'institution_ratio': 0, 'dragon_tiger': 0, 'news_analysis': 100, 'estimated_turnover': 100}}
# symbol = "301366"
# 3月4日 17cm 70分 股票评分结果： {'symbol': '688521', 'total_score': 70.0, 'scores': {'recent_turnover': 100, 'recent_increase': 100, 'market_cap': 100, 'amplitude': 100, 'institution_ratio': 0, 'dragon_tiger': 0, 'news_analysis': 0, 'estimated_turnover': 100}}
# symbol = "688521"
# 3月4日 17cm 85分 股票评分结果： {'symbol': '300458', 'total_score': 85.0, 'scores': {'recent_turnover': 100, 'recent_increase': 50, 'market_cap': 100, 'amplitude': 100, 'institution_ratio': 0, 'dragon_tiger': 0, 'news_analysis': 100, 'estimated_turnover': 100}}
# symbol = "300458"
# 3月4日 涨停20cm  39分
# symbol = "300183"

# 3月4日 及格 65分
# symbol = "300223"

# 3月4日 及格 60分
# symbol = "300302"


# 3月7日 及格 60分
# symbol = "300953"

# # 3月7日 及格 60分
# symbol = "300475"


from stockrating.read_local_info import read_stock_market_value_from_db,get_stock_info
result_data = read_stock_market_value_from_db()


def calculate_symbol_score(symbol):

    # symbol = "301396"
    # # symbol = "301368"
    # # symbol = "688521"
    # # symbol = "300083"
    # # symbol = "002276"
    symbol = "300451"
    result = evaluate_stock(symbol)
#
def calculate_symbols_score(symbols):
    # 执行给定数组中的所有股票代码
    symbols = [
        "000665", "001339", "002196", "002760", "300148", "300258", "300475",
        "300515", "300657", "300840", "300953", "301128", "301368", "301325",
        "600367", "600588", "603039", "605069", "688306", "688685", "831832", "836208"
    ]
    # 0310
    symbols = [
        "300007",
        "300083",
        "301525",
        "300580",
        "301382",
        "688022",
        "688010",
        "688003",
        "688097",
        "688166",
        "688160",
        "300404",
        "300857",
        "300986",
        "301021",
        "688246",
        "688393",
        "688502",
        "300253",
        "300451",
        "300432",
        "300503",
        "300676",
        "300244",
        "300433"
    ];
    # 0311
    symbols = [
        "300657",
        "300738",
        "300042",
        "300895",
        "300296",
        "301377",
        "300100",
        "300153",
        "300441",
        "300503",
        "300083",
        "688037",
        "300840",
        "301392",
        "301389",
        "688306",
        "836263",
        "873726"
    ];
    #
    # # 提取并去重股票代码
    # stock_codes = [
    #     "002050", "003021", "002993", "002965", "002929", "002896", "002765", "002760",
    #     "002757", "002725", "002599", "002582", "002580", "002575", "002527", "002522",
    #     "002501", "002398", "002369", "002364", "002358", "002335", "002326", "002276",
    #     "002261", "002245", "002196", "002195", "002139", "002126", "002123", "002105",
    #     "002048", "002044", "002042", "002031", "001368", "001339", "001319", "001309",
    #     "001298", "000997", "000892", "000887", "000880", "000868", "000856", "000837",
    #     "000818", "000785", "000710", "000665", "000570", "000034", "000032", "688685",
    #     "688629", "688615", "688591", "688590", "688561", "688521", "688400", "688393",
    #     "688369", "688365", "688347", "688343", "688333", "688327", "688322", "688316",
    #     "688306", "688256", "688228", "688220", "688205", "688159", "688158", "688118",
    #     "688114", "688041", "688031", "688017", "688003", "605488", "605100", "605069",
    #     "605066", "603986", "603918", "603887", "603881", "603855", "603700", "603629",
    #     "603618", "603583", "603501", "603496", "603360", "603315", "603300", "603296",
    #     "603270", "603220", "603219", "603200", "603166", "603119", "603118", "603039",
    #     "603012", "601789", "601616", "601177", "600986", "600845", "600797", "600633",
    #     "600602", "600592", "600590", "600589", "600588", "600580", "600498", "600367",
    #     "600203"
    # ]
    #
    # # 去重
    # unique_stock_codes = list(set(stock_codes))
    # # 替换 symbols 列表
    # symbols = unique_stock_codes

    # 执行给定数组中的所有股票代码
    for symbol in symbols:
        print(symbol)
        result = evaluate_stock(symbol)
        print(f"股票评分结果：{symbol}", result)
        time.sleep(1)  # 添加3秒延迟

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
            result = evaluate_stock(symbol)
            print(f"股票评分结果：{symbol}", result)
            time.sleep(1)  # 添加1秒延迟
        except Exception as e:
            print(f"处理股票代码 {symbol} 时出错：{e}")

if __name__ == '__main__':
    # 所有股票评分
    calculate_all_stock_score()

    # 单个股票评分
    calculate_symbol_score("300541")

    # 多个股票评分
    calculate_symbols_score()