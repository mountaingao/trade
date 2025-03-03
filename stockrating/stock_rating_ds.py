import akshare as ak
import pandas as pd
import numpy as np
import json
import os
import hashlib
import time

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
        with open(cache_file, "r") as file:
            return json.load(file)
    return None

def save_cache(code, data):
    """
    将数据保存到本地文件
    """
    cache_file = get_cache_filename(code)
    with open(cache_file, "w") as file:
        json.dump(data, file)

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

# 配置评分规则和权重
SCORE_RULES = {
    "recent_turnover": {"weight": 0.15, "levels": [(10, 100), (5, 60), (0, 0)]},  # 近期成交额
    "recent_increase": {"weight": 0.10, "levels": [(60, 100), (40, 50), (10, 0)]},  # 近期涨幅
    "market_cap": {"weight": 0.10, "levels": [(500, 0),(300, 50), (50, 100), (20, 0)]},  # 流通市值
    "amplitude": {"weight": 0.10, "levels": [(50, 100), (30, 50), (0, 0)]},  # 振幅
    "institution_ratio": {"weight": 0.05, "levels": [(20, 0), (10, 50), (0, 100)]},  # 机构占比
    "dragon_tiger": {"weight": 0.05, "levels": [("inflow", 100), ("small_inflow", 50), ("outflow", 0)]},  # 龙虎榜
    "news_analysis": {"weight": 0.10, "levels": [(True, 100), (False, 0)]},  # 新闻报道分析
    "estimated_turnover": {"weight": 0.25, "levels": [(10, 100), (5, 60), (0, 0)]},  # 预估成交额
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


    jgcyd= ak.stock_comment_detail_zlkp_jgcyd_em(symbol=symbol)
    print("机构参与度：", jgcyd)
    # print(ak.__version__)

    # 计算机构参与度的平均值
    if not jgcyd.empty and "机构参与度" in jgcyd.columns:
        avg_jgcyd = jgcyd["机构参与度"].tail(5).mean()
    else:
        avg_jgcyd = None

    lspf= ak.stock_comment_detail_zhpj_lspf_em(symbol=symbol)
    print("综合评价-历史评分：", lspf)

    focus= ak.stock_comment_detail_scrd_focus_em(symbol=symbol)
    print("市场热度-用户关注指数：", focus)

    # 分时数据，无用
    # desire= ak.stock_comment_detail_scrd_desire_em(symbol=symbol)
    # print("市场热度-市场参与意愿：", desire)

    desire_daily= ak.stock_comment_detail_scrd_desire_daily_em(symbol=symbol)
    print("市场热度-日度市场参与意愿：", desire_daily)

    # cost=历史数据，不准确
    # cost= ak.stock_comment_detail_scrd_cost_em(symbol=symbol)
    # print("市场热度-市场成本：", cost)

    # jgcyd= ak.stock_comment_detail_zlkp_jgcyd_em(symbol=symbol)
    # print("机构调研：", jgcyd)
    #
    # jgcyd= ak.stock_comment_detail_zlkp_jgcyd_em(symbol=symbol)
    # print("机构调研：", jgcyd)


    # exit()
    try:
        # 获取股票基本信息
        stock_info = ak.stock_individual_info_em(symbol=symbol)
        if stock_info is None:
            print(f"警告：股票代码 {symbol} 未找到相关信息，请检查代码格式是否正确。")
            return None, None, None
    except KeyError:
        print(f"警告：股票代码 {symbol} 未找到相关信息，请检查代码格式是否正确。")
        stock_info = None
    print("股票基本信息：", stock_info)
    print("流通市值：", stock_info.iloc[5]["value"])
    # exit()
    try:
        # 获取历史行情数据
        stock_history = ak.stock_zh_a_hist(symbol=symbol, period="daily", adjust="qfq")
    except Exception as e:
        print(f"警告：获取股票 {symbol} 的历史行情数据时出错：{e}")
        stock_history = None

    print("历史行情数据：", stock_history)
    # try:
    #     # 获取龙虎榜数据 stock_lhb_detail_em(
    #     #     start_date: str = "20230403", end_date: str = "20230417"
    #     # )
    #     dragon_tiger_data = ak.stock_lhb_detail_em("20250201","20250228")
    # except Exception as e:
    #     print(f"警告：获取股票 {symbol} 的龙虎榜数据时出错：{e}")
    dragon_tiger_data = None

    print("龙虎榜数据：", dragon_tiger_data)


    # return stock_info, stock_history, dragon_tiger_data, avg_jgcyd,desire_daily, lspf,focus
    # 将数据存入缓存
    data_cache[symbol] = (stock_info, stock_history, dragon_tiger_data, avg_jgcyd, desire_daily, lspf, focus)

    save_cache(symbol,  data_cache[symbol])
    return data_cache[symbol]

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
    stock_info, stock_history, dragon_tiger_data, avg_jgcyd,desire_daily, lspf,focus = get_stock_data(symbol)

    # 1. 近期成交额（15%） 成交量*均价 = 成交额
    recent_turnover = stock_history["成交额"].tail(5).mean() / 1e8  # 转换为亿
    turnover_score = calculate_score(recent_turnover, SCORE_RULES["recent_turnover"])

    # 2. 近期涨幅（10%）
    recent_increase = (stock_history["收盘"].iloc[-1] - stock_history["收盘"].iloc[-20]) / stock_history["收盘"].iloc[-20] * 100
    increase_score = calculate_score(recent_increase, SCORE_RULES["recent_increase"])

    # 3. 流通市值（10%） f117
    print("流通市值：", stock_info.iloc[5]["value"])
    market_cap = stock_info.iloc[5]["value"] / 1e8  # 转换为亿
    market_cap_score = calculate_score(market_cap, SCORE_RULES["market_cap"])

    # 4. 振幅（10%）
    amplitude = (stock_history["最高"].max() - stock_history["最低"].min()) / stock_history["最低"].min() * 100
    amplitude_score = calculate_score(amplitude, SCORE_RULES["amplitude"])

    # 5. 机构占比（5%）
    institution_ratio = avg_jgcyd  # 假设机构占比在基本信息中
    institution_score = calculate_score(institution_ratio, SCORE_RULES["institution_ratio"])

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

    # 8. 预估成交量和金额（25%）
    estimated_turnover = recent_turnover  # 假设预估成交额等于近期成交额
    estimated_score = calculate_score(estimated_turnover, SCORE_RULES["estimated_turnover"])

    # 计算总分
    total_score = (
            turnover_score * SCORE_RULES["recent_turnover"]["weight"]
            + increase_score * SCORE_RULES["recent_increase"]["weight"]
            + market_cap_score * SCORE_RULES["market_cap"]["weight"]
            + amplitude_score * SCORE_RULES["amplitude"]["weight"]
            + institution_score * SCORE_RULES["institution_ratio"]["weight"]
            + dragon_tiger_score * SCORE_RULES["dragon_tiger"]["weight"]
            + news_score * SCORE_RULES["news_analysis"]["weight"]
            + estimated_score * SCORE_RULES["estimated_turnover"]["weight"]
    )

    return {
        "symbol": symbol,
        "total_score": total_score,
        "scores": {
            "recent_turnover": turnover_score,
            "recent_increase": increase_score,
            "market_cap": market_cap_score,
            "amplitude": amplitude_score,
            "institution_ratio": institution_score,
            "dragon_tiger": dragon_tiger_score,
            "news_analysis": news_score,
            "estimated_turnover": estimated_score,
        },
    }

# 测试
symbol = "300718"  # 长盛轴承
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
symbol = "688393"
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


result = evaluate_stock(symbol)
print("股票评分结果：", result)