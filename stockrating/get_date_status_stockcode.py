import mysql.connector
from datetime import date

def query_stock_codes_by_date_and_status(db_config, target_date, target_status):
    try:
        # 连接数据库
        conn = mysql.connector.connect(**db_config)
        cursor = conn.cursor()

        # 构造 SQL 查询语句
        query = """
        SELECT stock_code
        FROM AlertData
        WHERE date = %s AND status = %s
        """
        # 执行查询
        cursor.execute(query, (target_date, target_status))

        # 获取查询结果
        results = cursor.fetchall()

        # 打印结果
        if results:
            print(f"在 {target_date} 日期下，状态为 '{target_status}' 的股票代码如下：")
            for row in results:
                print(row[0])  # row[0] 是股票代码
        else:
            print(f"未找到符合条件的股票代码。")

        return results

    except Exception as e:
        print(f"查询数据时出错：{e}")
    finally:
        # 关闭连接
        cursor.close()
        conn.close()

# 数据库连接配置
db_config = {
    "host": "localhost",
    "user": "root",
    "password": "111111",
    "database": "trade"
}

# 查询条件
target_date = date(2025, 2, 21)  # 目标日期
target_status = "盘中"  # 目标状态
target_status = "开盘"  # 目标状态
# target_status = "开盘-自"  # 目标状态

# 调用查询函数
query_stock_codes_by_date_and_status(db_config, target_date, target_status)