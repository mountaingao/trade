import pywencai
# 导入 a.py 和 b.py 中的函数
from stockrating.get_date_status_stockcode import query_stock_codes_by_date_and_status
from stockrating.ak_stock_block_ths import stock_profit_forecast_ths
import mysql.connector
from datetime import date
from datetime import datetime
import pandas as pd
import time


# 数据库连接配置
db_config = {
    "host": "localhost",
    "user": "root",
    "password": "111111",
    "database": "trade"
}



# 获取指定股票的现有板块数据
def get_existing_sectors_for_stock(cursor, stock_code):
    cursor.execute("SELECT sector FROM stockblock WHERE stock_code = %s AND status = 'active'", (stock_code,))
    return {row[0] for row in cursor.fetchall()}

# 插入新数据（批量）
def insert_new_sectors(cursor, new_sectors, join_time):
    """
    批量插入新的板块数据，并根据插入顺序为 rank 赋值。

    参数:
    cursor (mysql.connector.cursor): 数据库游标对象
    new_sectors (list of tuples): 包含股票代码和板块名称的列表，格式为 [(stock_code, sector), ...]
    join_time (datetime): 加入时间

    返回:
    None
    """
    insert_query = """
    INSERT INTO stockblock (stock_code, sector, status, ranking, join_time, delete_time, remark)
    VALUES (%s, %s, 'active', %s, %s, NULL, 'Initial addition')
    """

    # 使用 enumerate 为每个新板块分配 rank
    values = [
        (stock_code, sector, rank + 1, join_time)
        for rank, (stock_code, sector) in enumerate(new_sectors)
    ]

    try:
        cursor.executemany(insert_query, values)
        print(f"成功插入 {len(values)} 条新板块数据")
    except Exception as e:
        print(f"插入新板块数据时出错: {e}")
        raise

# 标记已删除的数据（批量）
def mark_deleted_sectors(cursor, deleted_sectors, delete_time):
    update_query = """
    UPDATE stockblock
    SET status = 'inactive', delete_time = %s, remark = 'Deleted in latest update'
    WHERE stock_code = %s AND sector = %s
    """
    cursor.executemany(update_query, [(delete_time, stock_code, sector) for stock_code, sector in deleted_sectors])

# 处理单个股票的板块数据
def process_stock_sectors(cursor, stock_code, sectors, current_time):
    # 获取该股票现有的板块数据
    existing_sectors = get_existing_sectors_for_stock(cursor, stock_code)

    # 比较现有数据和查询结果
    new_sectors = [(stock_code, sector) for sector in sectors if sector not in existing_sectors]
    deleted_sectors = [(stock_code, sector) for sector in existing_sectors if sector not in sectors]

    # 批量插入新数据
    if new_sectors:
        insert_new_sectors(cursor, new_sectors, current_time)

    # 批量标记已删除的数据
    if deleted_sectors:
        mark_deleted_sectors(cursor, deleted_sectors, current_time)

def process_stock_concept_data(cursor, stock_code):
    """
    处理单个股票的概念板块数据，并更新数据库。

    参数:
    cursor (mysql.connector.cursor): 数据库游标对象
    stock_code (string): 包含股票代码和其他相关信息的元组

    返回:
    bool: 是否成功处理了该股票的数据
    """
    try:
        concept_data = stock_profit_forecast_ths(symbol=stock_code)
        print(f"获取到股票 {stock_code} 的概念板块数据: {concept_data}")

        if not concept_data:
            print(f"股票 {stock_code} 没有概念板块数据")
            return False

        # 当前时间
        current_time = datetime.now()

        # 处理单个股票的板块数据
        process_stock_sectors(cursor, stock_code, concept_data, current_time)

        return concept_data
    except Exception as e:
        print(f"处理股票 {stock_code} 的概念板块数据时出错: {e}")
        return False
    



if __name__ == "__main__":
    # 示例：获取特定股票的概念板块
    # 查询条件
    target_date = date(2025, 3, 12)  # 目标日期
    target_status = "盘中"  # 目标状态
    target_status = "开盘"  # 目标状态
    # target_status = "开盘-自"  # 目标状态
    result = query_stock_codes_by_date_and_status(db_config, target_date, target_status)
    print(result)
    # 安全地提取每个元组中的第一个元素，并用逗号分隔
    stock_codes = ",".join(item[0] for item in result if item)

    print(stock_codes)  # 输出：300256,300383,300450
    # stock_codes = "300718和300513、300870"  # 示例股票代码
    # df = get_stock_concept(stock_codes)


    # 连接数据库
    conn = mysql.connector.connect(**db_config)
    cursor = conn.cursor()

    # 遍历查询结果
    for item in result:
        print(item)
        # stock_code = "300513"
        stock_code = item[0]
        process_stock_concept_data(cursor, stock_code)

        time.sleep(10)


    # 提交事务
    conn.commit()

    # 关闭连接
    cursor.close()
    conn.close()