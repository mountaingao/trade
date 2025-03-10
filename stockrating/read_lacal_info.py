from mootdx.reader import Reader

# 创建 Reader 对象
# reader = Reader.factory(market='std', tdxdir='D:/new_haitong/')
reader = Reader.factory(market='std', tdxdir='D:/zd_haitong/')

# 获取历史日线数据
daily_data = reader.daily(symbol='300264')
print("日线数据：", daily_data)

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

df = pywencai.get(query="A股流通市值", loop=True)
# df = pywencai.get(query="A股流通市值")

print(df)

# 将数据插入mysql数据库中
from sqlalchemy import create_engine
import mysql.connector
# 数据库连接配置
db_config = {
    "host": "localhost",  # 数据库主机地址
    "user": "root",  # 数据库用户名
    "password": "111111",  # 数据库密码
    "database": "trade"  # 数据库名称
}
conn = mysql.connector.connect(**db_config)

# 创建数据库连接
engine = create_engine('mysql+pymysql://root:111111@localhost:3306/trade')

# 将DataFrame插入到MySQL表中
df.to_sql('stock_market_value', con=engine, if_exists='replace', index=False)
