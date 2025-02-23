import pywencai

def get_stock_concept(stock_code):
    # 构造查询语句
    query = f"{stock_code} 概念板块"
    # 调用问财接口
    result = pywencai.get(query=query, sort_key="股票代码", sort_order="desc")
    return result

if __name__ == "__main__":
    # 示例：获取特定股票的概念板块
    stock_code = "300718和300513、300870"  # 示例股票代码
    concept_data = get_stock_concept(stock_code)
    print(concept_data)