import sqlite3
import pandas as pd
import json
import os

class StockDataStorage:
    def __init__(self, db_path="../data/db/stock_block_data.db"):
        self.db_path = db_path
        self.init_database()

    def init_database(self):
        """初始化数据库表"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # 创建预测数据表
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS stockblock (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                code TEXT,                    -- 股票代码
                name TEXT,                    -- 股票名称
                date TEXT,                    -- 日期
                industry TEXT,                -- 细分行业
                blockname TEXT,               -- 板块名称
                status INTEGER,               -- 状态 (0: 默认, 1: 删除, 2: 其他)
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP  -- 创建时间
            )
        ''')

        conn.commit()
        conn.close()

    def save_stock_block(self, blockname, df):
        """保存预测数据"""
        conn = sqlite3.connect(self.db_path)

        # 添加blockname列
        df_to_save = df.copy()
        df_to_save['blockname'] = blockname

        # 保存到数据库
        df_to_save.to_sql('stockblock', conn, if_exists='append', index=False)
        conn.close()

    def insert_stock_record(self, code, name, date, industry, blockname, status=0):
        """插入单条股票记录"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO stockblock (code, name, date, industry, blockname, status)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', (code, name, date, industry, blockname, status))
        
        conn.commit()
        conn.close()

    def query_stock_block(self, where_clause="", params=None):
        """查询预测数据"""
        conn = sqlite3.connect(self.db_path)
        query = "SELECT * FROM stockblock"
        if where_clause:
            query += f" WHERE {where_clause}"

        df = pd.read_sql_query(query, conn, params=params)
        conn.close()
        return df

    def get_latest_stock_block(self, limit=100):
        """获取最新预测数据"""
        return self.query_stock_block(
            "id IN (SELECT id FROM stockblock ORDER BY created_at DESC LIMIT ?)",
            (limit,)
        )

    def query_by_code(self, code):
        """根据股票代码查询数据"""
        return self.query_stock_block("code = ?", (code,))

    def update_status(self, code, new_status):
        """更新指定代码的状态"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            UPDATE stockblock
            SET status = ?
            WHERE code = ?
        ''', (new_status, code))
        
        conn.commit()
        conn.close()

    def batch_import_from_dataframe(self, df):
        """批量导入数据，如有重复则更新"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        for _, row in df.iterrows():
            # 检查是否已存在该代码的记录
            cursor.execute("SELECT id FROM stockblock WHERE code = ?", (row['code'],))
            result = cursor.fetchone()
            
            if result:
                # 如果存在，更新记录
                cursor.execute('''
                    UPDATE stockblock
                    SET name = ?, date = ?, industry = ?, blockname = ?, status = ?
                    WHERE code = ?
                ''', (row['name'], row['date'], row['industry'], row['blockname'], row['status'], row['code']))
            else:
                # 如果不存在，插入新记录
                cursor.execute('''
                    INSERT INTO stockblock (code, name, date, industry, blockname, status)
                    VALUES (?, ?, ?, ?, ?, ?)
                ''', (row['code'], row['name'], row['date'], row['industry'], row['blockname'], row['status']))
        
        conn.commit()
        conn.close()

# 测试代码
if __name__ == "__main__":
    # 创建存储实例
    storage = StockDataStorage()
    
    # 1. 增加一条记录
    storage.insert_stock_record("000001", "平安银行", "2023-01-01", "银行", "金融板块", 0)
    
    # 2. 查询刚增加的记录
    print("查询刚添加的记录:")
    new_record = storage.query_by_code("000001")
    print(new_record)
    
    # 3. 单独查询某个代码的数据
    print("\n查询特定代码的数据:")
    specific_record = storage.query_by_code("000001")
    print(specific_record)
    
    # 4. 修改某个代码的状态
    print("\n将状态从0修改为1:")
    storage.update_status("000001", 1)
    updated_record = storage.query_by_code("000001")
    print(updated_record)
    
    # 5. 批量导入表格数据
    print("\n批量导入数据:")
    test_data = pd.DataFrame({
        'code': ['000002', '000003', '000001'],
        'name': ['万科A', '中粮可乐', '平安银行'],
        'date': ['2023-01-02', '2023-01-03', '2023-01-01'],
        'industry': ['房地产', '食品饮料', '银行'],
        'blockname': ['地产板块', '消费板块', '金融板块'],
        'status': [0, 0, 1]
    })
    
    storage.batch_import_from_dataframe(test_data)
    
    # 查询所有记录
    all_records = storage.query_stock_block()
    print("所有记录:")
    print(all_records)