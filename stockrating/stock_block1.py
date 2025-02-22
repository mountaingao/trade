import requests
from bs4 import BeautifulSoup
import pandas as pd

def get_stock_concept_from_web(stock_code):
    url = f"http://q.10jqka.com.cn/gn/detail/code/{stock_code}/"  # 同花顺概念板块页面
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36"
    }
    response = requests.get(url, headers=headers)

    if response.status_code == 200:
        soup = BeautifulSoup(response.content, "html.parser")
        # 假设板块信息在某个特定的标签中
        concept_data = soup.find_all("div", class_="target-class")  # 修改为实际的类名
        concepts = [item.text for item in concept_data]
        return concepts
    else:
        print("请求失败，状态码：", response.status_code)
        return []

# 示例：获取特定股票的概念板块
stock_code = "600519"  # 示例股票代码
concepts = get_stock_concept_from_web(stock_code)
print("所属概念板块：", concepts)