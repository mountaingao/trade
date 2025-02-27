import akshare as ak
import pandas as pd
from datetime import datetime, timedelta
import os

# 配置
data_dir = "stock_data"
os.makedirs(data_dir, exist_ok=True)

# 获取股票历史行情数据
def get_stock_history(stock_code, days=30):
    end_date = datetime.now().strftime("%Y%m%d")
    start_date = (datetime.now() - timedelta(days=days)).strftime("%Y%m%d")
    try:
        stock_data = ak.stock_zh_a_hist(symbol=stock_code, period="daily", start_date=start_date, end_date=end_date, adjust="")
        return stock_data
    except Exception as e:
        print(f"获取 {stock_code} 的历史数据时出错：{e}")
        return None

# 获取股票基本信息
def get_stock_info(stock_code):
    try:
        stock_info = ak.stock_info_a_code_name()
        return stock_info[stock_info['code'] == stock_code]
    except Exception as e:
        print(f"获取 {stock_code} 的基本信息时出错：{e}")
        return None

# 获取股票所属板块
def get_stock_concept(stock_code):
    try:
        concept_data = ak.stock_board_industry_summary_ths()
        concept_data = concept_data[concept_data['板块代码'] == stock_code]
        return concept_data
    except Exception as e:
        print(f"获取 {stock_code} 的板块数据时出错：{e}")
        return None

# 获取股票流通市值
def get_stock_market_value(stock_code):
    try:
        market_value = ak.stock_info_a_code_name()
        return market_value[market_value['code'] == stock_code]['流通市值'].values[0]
    except Exception as e:
        print(f"获取 {stock_code} 的流通市值时出错：{e}")
        return None

# 获取股东结构数据
def get_stock_shareholder_structure(stock_code):
    try:
        shareholder_data = ak.stock_individual_shareholder_structure(symbol=stock_code)
        return shareholder_data
    except Exception as e:
        print(f"获取 {stock_code} 的股东结构数据时出错：{e}")
        return None

# 评级函数
def rate_stock(stock_code):
    stock_data = get_stock_history(stock_code)
    stock_info = get_stock_info(stock_code)
    concept_data = get_stock_concept(stock_code)
    market_value = get_stock_market_value(stock_code)
    shareholder_data = get_stock_shareholder_structure(stock_code)

    if stock_data is None or stock_info is None or concept_data is None or market_value is None or shareholder_data is None:
        return None

    # 1. 近期成交额
    avg_turnover = stock_data["成交额"].mean()
    turnover_score = 100 if avg_turnover > 1e9 else 80 if avg_turnover > 5e8 else 60 if avg_turnover > 1e8 else 40 if avg_turnover > 5e7 else 20

    # 2. 近期涨幅
    price_change = (stock_data["收盘"].iloc[-1] - stock_data["开盘"].iloc[0]) / stock_data["开盘"].iloc[0]
    price_change_score = 100 if price_change > 0.2 else 80 if price_change > 0.1 else 60 if price_change > 0.05 else 40 if price_change > 0 else 20

    # 3. 近期振幅
    amplitude = (stock_data["最高"].max() - stock_data["最低"].min()) / stock_data["开盘"].iloc[0]
    amplitude_score = 100 if amplitude > 0.3 else 80 if amplitude > 0.2 else 60 if amplitude > 0.1 else 40 if amplitude > 0.05 else 20

    # 4. 流通市值
    market_value_score = 100 if market_value > 1e12 else 80 if market_value > 5e11 else 60 if market_value > 1e11 else 40 if market_value > 5e10 else 20

    # 5. 是否热门板块
    hot_concept = concept_data["板块名称"].str.contains("热门").any()
    hot_concept_score = 100 if hot_concept else 80

    # 6. 机构占比
    institution_ratio = shareholder_data["机构持股比例"].values[0]
    institution_score = 100 if institution_ratio > 0.5 else 80 if institution_ratio > 0.3 else 60 if institution_ratio > 0.2 else 40 if institution_ratio > 0.1 else 20

    # 7. 散户占比
    retail_ratio = shareholder_data["散户持股比例"].values[0]
    retail_score = 100 if retail_ratio < 0.2 else 80 if retail_ratio < 0.3 else 60 if retail_ratio < 0.4 else 40 if retail_ratio < 0.5 else 20

    # 综合评级
    total_score = (turnover_score + price_change_score + amplitude_score + market_value_score + hot_concept_score + institution_score + retail_score) / 7
    return {
        "股票代码": stock_code,
        "综合评级": total_score,
        "成交额得分": turnover_score,
        "涨幅得分": price_change_score,
        "振幅得分": amplitude_score,
        "流通市值得分": market_value_score,
        "热门板块得分": hot_concept_score,
        "机构占比得分": institution_score,
        "散户占比得分": retail_score
    }

# 示例：对股票 "000001" 进行评级
if __name__ == "__main__":
    stock_code = "000001"
    rating = rate_stock(stock_code)
    if rating:
        print(f"股票代码：{rating['股票代码']}")
        print(f"综合评级：{rating['综合评级']:.2f}")
        print(f"成交额得分：{rating['成交额得分']}")
        print(f"涨幅得分：{rating['涨幅得分']}")
        print(f"振幅得分：{rating['振幅得分']}")
        print(f"流通市值得分：{rating['流通市值得分']}")
        print(f"热门板块得分：{rating['热门板块得分']}")
        print(f"机构占比得分：{rating['机构占比得分']}")
        print(f"散户占比得分：{rating['散户占比得分']}")