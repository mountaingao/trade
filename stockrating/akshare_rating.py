import akshare as ak
import pandas as pd
import pickle
import os

# 获取股票数据
CACHE_DIR = './cache'  # 缓存文件夹路径
def get_stock_data():
    # 获取所有股票列表
    stock_list_cache_path = os.path.join(CACHE_DIR, 'stock_list.pkl')
    limit_up_data_cache_path = os.path.join(CACHE_DIR, 'limit_up_data.pkl')

    # 获取所有股票列表
    if os.path.exists(stock_list_cache_path):
        with open(stock_list_cache_path, 'rb') as f:
            stock_list = pickle.load(f)
    else:
        stock_info = ak.stock_info_a_code_name()
        stock_list = stock_info.rename(columns={'code': 'ts_code', 'name': 'name'})
        if not os.path.exists(CACHE_DIR):
            os.makedirs(CACHE_DIR)
        with open(stock_list_cache_path, 'wb') as f:
            pickle.dump(stock_list, f)

    # 获取涨停次数数据
    if os.path.exists(limit_up_data_cache_path):
        with open(limit_up_data_cache_path, 'rb') as f:
            limit_up_data = pickle.load(f)
    else:
        limit_up_data = ak.stock_zt_pool_em('20250227')  # 获取某一天的涨停数据
        with open(limit_up_data_cache_path, 'wb') as f:
            pickle.dump(limit_up_data, f)

# 计算板块地位评分
def calculate_sector_score(stock):
    # 假设板块内市值排名已计算
    if stock['market_rank'] == 1:
        return 100
    elif stock['market_rank'] <= 3:
        return 80
    elif stock['market_rank'] <= 5:
        return 60
    else:
        return 40

# 计算涨停次数评分
def calculate_limit_up_score(stock, limit_up_data):
    limit_up_count = limit_up_data[limit_up_data['代码'] == stock['ts_code']].shape[0]
    if limit_up_count >= 5:
        return 100
    elif limit_up_count >= 3:
        return 80
    elif limit_up_count >= 1:
        return 60
    else:
        return 40

# 计算综合评分
def calculate_total_score(stock, limit_up_data):
    sector_score = calculate_sector_score(stock)
    limit_up_score = calculate_limit_up_score(stock, limit_up_data)
    # 假设资金流向和技术形态评分已计算
    fund_score = 100  # 示例
    tech_score = 50   # 示例
    total_score = (sector_score * 0.3) + (limit_up_score * 0.25) + (fund_score * 0.25) + (tech_score * 0.2)
    return total_score

# 主函数
def main():
    stock_list, limit_up_data = get_stock_data()
    ratings = []
    for index, stock in stock_list.iterrows():
        total_score = calculate_total_score(stock, limit_up_data)
        if total_score >= 90:
            rating = 5
        elif total_score >= 80:
            rating = 4
        elif total_score >= 70:
            rating = 3
        elif total_score >= 60:
            rating = 2
        else:
            rating = 1
        ratings.append({'ts_code': stock['ts_code'], 'name': stock['name'], 'rating': rating})

    # 将评级结果保存到数据库或文件
    ratings_df = pd.DataFrame(ratings)
    ratings_df.to_csv('stock_ratings.csv', index=False)

if __name__ == "__main__":
    main()
