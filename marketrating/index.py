import tushare as ts
import pandas as pd
import datetime

# 初始化tushare
ts.set_token('296cdbdc3ad507506ec8785465785f5ea065c81e8ec5b38b8b4e35ba')  # 替换为你的tushare token  密码 mountain
pro = ts.pro_api()

# 定义评级函数
def rate_market(date):
    # 获取大盘指数数据
    index_data = pro.index_daily(ts_code='000001.SH', start_date=date, end_date=date)  # 上证指数
    index_data = index_data.iloc[0]  # 取当天数据

    # 获取市场情绪数据
    daily_data = pro.daily(trade_date=date)
    daily_data['change'] = daily_data['close'] - daily_data['pre_close']
    daily_data['rise'] = daily_data['change'] > 0
    daily_data['fall'] = daily_data['change'] < 0
    daily_data['limit_up'] = daily_data['close'] == daily_data['high']  # 涨停
    daily_data['limit_down'] = daily_data['close'] == daily_data['low']  # 跌停

    rise_count = daily_data['rise'].sum()
    fall_count = daily_data['fall'].sum()
    limit_up_count = daily_data['limit_up'].sum()
    limit_down_count = daily_data['limit_down'].sum()

    # 获取资金流向数据
    fund_flow = pro.moneyflow_hsgt(trade_date=date)  # 北向资金
    north_fund = fund_flow['north_money'].iloc[0]  # 北向资金流入量

    # 技术指标计算
    ma5 = index_data['ma5']
    ma10 = index_data['ma10']
    ma20 = index_data['ma20']
    macd = index_data['macd']
    kdj_k = index_data['kdj_k']
    kdj_d = index_data['kdj_d']

    # 评级逻辑
    score = 0

    # 技术指标评分
    if ma5 > ma10 > ma20:  # 均线多头排列
        score += 2
    if macd > 0:  # MACD金叉
        score += 1
    if kdj_k > kdj_d and kdj_k < 50:  # KDJ金叉且未超买
        score += 1

    # 市场情绪评分
    if rise_count > fall_count:
        score += 1
    if limit_up_count > limit_down_count:
        score += 1
    if limit_up_count > 100:  # 涨停板数量较多
        score += 1

    # 资金流向评分
    if north_fund > 0:  # 北向资金流入
        score += 1
    if north_fund > 100:  # 北向资金大幅流入
        score += 1

    # 确定评级
    if score <= 2:
        rating = 1
        strategy = "轻仓或空仓，只做超跌反弹"
    elif 3 <= score <= 4:
        rating = 3
        strategy = "中等仓位，高抛低吸"
    elif 5 <= score <= 6:
        rating = 5
        strategy = "重仓，追涨强势股"
    else:
        rating = 7
        strategy = "极强市场，积极进攻"

    return rating, strategy

# 主程序
if __name__ == "__main__":
    today = datetime.date.today().strftime('%Y%m%d')  # 获取当前日期
    rating, strategy = rate_market(today)
    print(f"日期：{today}")
    print(f"大盘评级：{rating}")
    print(f"操作策略：{strategy}")