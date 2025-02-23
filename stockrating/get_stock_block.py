import pywencai
# 导入 a.py 和 b.py 中的函数
from get_date_status_stockcode import query_stock_codes_by_date_and_status
from stock_block import get_stock_concept
import mysql.connector
from datetime import date
from datetime import datetime
import pandas as pd


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
    insert_query = """
    INSERT INTO stockblock (stock_code, sector, status, join_time, delete_time, remark)
    VALUES (%s, %s, 'active', %s, NULL, 'Initial addition')
    """
    cursor.executemany(insert_query, [(stock_code, sector, join_time) for stock_code, sector in new_sectors])

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

# 查询条件
target_date = date(2025, 2, 21)  # 目标日期
target_status = "盘中"  # 目标状态
target_status = "开盘"  # 目标状态
# target_status = "开盘-自"  # 目标状态

if __name__ == "__main__":
    # 示例：获取特定股票的概念板块
    result = query_stock_codes_by_date_and_status(db_config, target_date, target_status)
    print(result)
    # 安全地提取每个元组中的第一个元素，并用逗号分隔
    stock_codes = ",".join(item[0] for item in result if item)

    print(stock_codes)  # 输出：300256,300383,300450
    stock_codes = "300718和300513、300870"  # 示例股票代码
    df = get_stock_concept(stock_codes)
    stock_df = df[['代码', '名称', '所属概念']]
    print(stock_df)

    # 循环遍历字典列表
    for item in stock_df:
        print(f"股票代码: {item[0]}, 概念: {item[1]}")
    exit

    # 将字符串数据转换为 DataFrame
    from io import StringIO
    data = concept_data
    # 获取字符串数据
    data_str = data[list(data.keys())[0]]

    # 使用 StringIO 将字符串转换为类似文件对象
    data_io = StringIO(data_str.to_string())

    # 读取为 DataFrame
    df = pd.read_csv(data_io, sep=";", header=None, names=["所属概念", "代码", "名称"])

    # 转换为数组（列表）
    result_array = df.values.tolist()

    # 打印结果
    for item in result_array:
        print(item)
    exit
    if isinstance(concept_data, dict):
        # 将字典转换为 DataFrame
        concept_data = pd.DataFrame.from_dict(concept_data)

    # 连接数据库
    conn = mysql.connector.connect(**db_config)
    cursor = conn.cursor()

    # 当前时间
    current_time = datetime.now()

    # 遍历查询结果
    for index, row in concept_data.iterrows():
        stock_code = row['股票代码'].split('.')[0]  # 提取股票代码
        sectors = set(row['概念板块'].split(';'))  # 提取板块数据

        # 处理单个股票的板块数据
        process_stock_sectors(cursor, stock_code, sectors, current_time)

    # 提交事务
    conn.commit()

    # 关闭连接
    cursor.close()
    conn.close()