import sqlite3
import pandas as pd
import json
import os
from data_prepare import prepare_ths_data
import datetime
import pytz

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

    def query_by_codes(self, codes):
        """根据多个股票代码查询数据"""
        # 修改: 处理numpy数组和pandas Series的情况
        if codes is None or len(codes) == 0:
            return pd.DataFrame()  # 返回空的DataFrame
        
        # 构造占位符
        placeholders = ','.join(['?'] * len(codes))
        query = f"SELECT * FROM stockblock WHERE code IN ({placeholders})"
        
        conn = sqlite3.connect(self.db_path)
        df = pd.read_sql_query(query, conn, params=codes)
        conn.close()
        return df

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
        # df_data = df[['代码', '名称', '日期', '细分行业', '概念', '状态']]

        print(df.columns)
        # 转换字段名
        df.rename(columns={'代码': 'code', '名称': 'name', '日期': 'date', '细分行业': 'industry', '概念': 'blockname', '状态': 'status'}, inplace=True)

        # blockname 字段 如果有+号，+号之前的数据 如果有空格分开，也取第一个
        df['blockname'] = df['blockname'].apply(lambda x: x.split(' ')[0])
        df['blockname'] = df['blockname'].apply(lambda x: x.split('+')[0])
        # 替换 -- 为空
        df['blockname'] = df['blockname'].str.replace('--', '')
        # 确保 code 字段是字符串类型
        df['code'] = df['code'].astype(str)
        for _, row in df.iterrows():
            print(row)
            print(row['code'])
            code = str(row['code'])
            print(code)

        # 检查是否已存在该代码的记录
            cursor.execute("SELECT id FROM stockblock WHERE code = ?", (code,))
            result = cursor.fetchone()
            
            if result:
                # 如果存在，更新记录
                cursor.execute('''
                    UPDATE stockblock
                    SET name = ?, date = ?, industry = ?, blockname = ?, status = ?
                    WHERE code = ?
                ''', (row['name'], row['date'], row['industry'], row['blockname'], row['status'], code))
            else:
                # 如果不存在，插入新记录
                cursor.execute('''
                    INSERT INTO stockblock (code, name, date, industry, blockname, status)
                    VALUES (?, ?, ?, ?, ?, ?)
                ''', (code, row['name'], row['date'], row['industry'], row['blockname'], row['status']))
        
        conn.commit()
        conn.close()

    def clear_all_data(self):
        """清除所有数据"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("DELETE FROM stockblock")
        
        conn.commit()
        conn.close()

    def clear_data_by_codes(self, codes):
        """根据股票代码列表清除指定数据"""
        if not codes:
            return
            
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # 构造占位符
        placeholders = ','.join(['?'] * len(codes))
        query = f"DELETE FROM stockblock WHERE code IN ({placeholders})"
        
        cursor.execute(query, codes)
        
        conn.commit()
        conn.close()

def insert_stock_block(code, name, date, industry, blockname, status=0):
    # 创建存储实例
    storage = StockDataStorage()

    # 1. 增加一条记录
    storage.insert_stock_record("000001", "平安银行", "2023-01-01", "银行", "金融板块", 0)

def main():
    # 测试代码
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
    
    # 6. 测试清除指定数据
    print("\n清除指定数据 (000001, 000002):")
    storage.clear_data_by_codes(['000001', '000002'])
    remaining_records = storage.query_stock_block()
    print("剩余记录:")
    print(remaining_records)
    
    # 7. 测试清除所有数据
    print("\n清除所有数据:")
    storage.clear_all_data()
    all_records_after_clear = storage.query_stock_block()
    print("清除所有数据后:")
    print(all_records_after_clear)


def update_stock_block_status():
    # 创建存储实例
    storage = StockDataStorage()
    # 清空数据
    # StockDataStorage().clear_all_data()

    # 读取同花顺目录下的所有的历史数据
    df = prepare_ths_data("0717","0920")
    # df_data = df[['代码', '名称', '日期', '细分行业', '概念', '状态']]

    print(df.columns)
    # 转换字段名
    df.rename(columns={'代码': 'code', '    名称': 'name',  '细分行业': 'industry', '备注': 'blockname'}, inplace=True)
    df['status'] = 0
    def get_beijing_time():
        """获取北京时间"""
    beijing_tz = pytz.timezone('Asia/Shanghai')

    df['date'] = datetime.datetime.now(beijing_tz).strftime('%Y-%m-%d')
    df['code'] = df['code'].str.replace('SH', '').str.replace('SZ', '')

    print(df.columns)
    df = df[['code', 'name', 'date', 'industry', 'blockname', 'status']]
    print(df.head(100))
    # exit()
    storage.batch_import_from_dataframe(df)
    # print(storage.query_stock_block())

def list_stock_block():
    # 创建存储实例
    storage = StockDataStorage()
    print(storage.query_stock_block())

def get_stocks_block(codes):
    # 创建存储实例
    storage = StockDataStorage()
    # print(codes)
    return storage.query_by_codes(codes)


# 把数据补充完整，增加概念字段
def add_blockname_data(df):
    if df.empty:
        return df
    codes = df['代码'].unique()
    # 查询得到板块数据
    block_df = get_stocks_block(codes.tolist())
    # print(block_df.tail(20))
    # print(block_df[['code', 'blockname']])

    # 修改: 统一代码列的数据类型为字符串,    NaN 转换为空字符串
    df['代码'] = df['代码'].astype(str)
    block_df['code'] = block_df['code'].fillna('').astype(str)

    # 合并df和block_df，根据code和代码进行合并, blockname列 赋给 概念 列
    df = pd.merge(df, block_df[['code', 'blockname']], left_on='代码', right_on='code', how='left')
    # 填充缺失的概念值为空字符串
    # 将blockname列 赋给 概念 列
    df['概念'] = df['blockname']
    # 过滤概念为空的数据
    # df['概念'] = df['概念'].fillna('')
    # df[group_by] = df[group_by].fillna('')
    # df = df.dropna(subset=[group_by])
    # 过滤概念为空的数据 或 ‘’ 字符串
    df = df[(df['概念'].notna()) & (df['概念'] != '')]
    print(f'增加概念字段，概念数据量：{len(df)}') # 查询得到板块数据
    return df

if __name__ == "__main__":
    # main()

    # 清空所有数据

    # update_stock_block_status()

    list_stock_block()

    value = get_stocks_block([688499, 688498])
    print(value)
