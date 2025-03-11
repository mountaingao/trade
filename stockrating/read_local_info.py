from mootdx.reader import Reader

# 创建 Reader 对象
# reader = Reader.factory(market='std', tdxdir='D:/new_haitong/')
reader = Reader.factory(market='std', tdxdir='D:/zd_haitong/')

# 获取历史日线数据
# daily_data = reader.daily(symbol='300264')
# print("日线数据：", daily_data)

# # 获取历史分时数据 无效 需下载
# minute_data = reader.minute(symbol='300264')
# print("分时数据：", minute_data)
#
# # 获取日线数据 无效 需下载
# fz = reader.fzline(symbol='300264')
# print("分钟数据：", fz)

# 不能用tushare
import pywencai

from selenium import webdriver
from bs4 import BeautifulSoup

# # 启动浏览器
# driver = webdriver.Chrome(executable_path='C:\Program Files\Google\Chrome\Application')
# driver.get("https://www.10jqka.com.cn/300083")
#
# # 解析网页
# soup = BeautifulSoup(driver.page_source, 'html.parser')
# circulating_market_value = soup.find("div", class_="流通市值").text
# print(f"流通市值: {circulating_market_value}")
#
# driver.quit()
# exit()

def get_stock_market_value_pywencai():
    df = pywencai.get(query="A股流通市值", loop=True)
    # df = pywencai.get(query="A股流通市值")

    print(df)

    # 将数据插入mysql数据库中
    from sqlalchemy import create_engine
    # 创建数据库连接
    engine = create_engine('mysql+pymysql://root:111111@localhost:3306/trade')

    # 将DataFrame插入到MySQL表中
    df.to_sql('stock_market_value', con=engine, if_exists='replace', index=False)

# 从数据库中读取表 stock_market_value 的所有字段名，并将 code、股票简称、流通市值组成字典
def read_stock_market_value_from_db():
    import pandas as pd
    from sqlalchemy import create_engine

    # 创建数据库连接
    engine = create_engine('mysql+pymysql://root:111111@localhost:3306/trade')

    # 读取表 stock_market_value 的所有数据
    query = "SELECT * FROM stock_market_value"
    df = pd.read_sql(query, con=engine)

    # 获取字段名
    columns = df.columns.tolist()
    # 动态获取流通市值字段名
    market_value_column = next((col for col in columns if 'a股市值' in col), None)
    if not market_value_column:
        raise ValueError("表中没有包含'a股市值'的字段")

    # 提取所需的字段并组成字典
    stock_dict = df.set_index('code')[['股票简称', market_value_column]].to_dict(orient='index')

    # 将字典转换为 DataFrame
    result_df = pd.DataFrame.from_dict(stock_dict, orient='index').reset_index()
    result_df.rename(columns={'index': 'code', market_value_column: 'circulating_market_value'}, inplace=True)
    result_df.rename(columns={'index': 'code', '股票简称': 'name'}, inplace=True)
    
    return result_df

# 示例调用


def get_stock_info(result_data,code):
    # print(result_data)
    return result_data[result_data['code'] == code]

if __name__ == '__main__':
    # 新抓取当日的市值数据
    # get_stock_market_value_pywencai()

    #使用方法
    result_data = read_stock_market_value_from_db()
    code = '300264'
    info = get_stock_info(result_data,code)
    print(info)
    print(info['name'].values[0])
    print(info['circulating_market_value'].values[0])
