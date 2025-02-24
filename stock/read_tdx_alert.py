import os
import time
import tkinter as tk
from tkinter import messagebox
from playsound import playsound
import mysql.connector
import datetime

# 文件路径
file_path = r"alert1.txt"

# 记录文件的最后修改时间和内容
last_modified_time = os.path.getmtime(file_path)
with open(file_path, 'r', encoding='GBK') as file:
    last_content = file.read()  # 读取初始文件内容

# 数据库连接配置
db_config = {
    "host": "localhost",  # 数据库主机地址
    "user": "root",  # 数据库用户名
    "password": "111111",  # 数据库密码
    "database": "trade"  # 数据库名称
}

def show_alert(new_content, mp3_path):
    root = tk.Tk()
    root.title("提醒")  # 设置窗口标题
    root.attributes('-topmost', True)  # 确保窗口始终在最前面
    # 显示消息内容
    message = new_content
    label = tk.Label(root, text=message, wraplength=280, justify="center")
    label.pack(expand=True)
    playsound(mp3_path)
    # messagebox.showinfo("提醒", f"文件内容已更新！\n\n新增内容：\n{new_content}")

    # 设置定时器，5秒后关闭窗口
    root.after(5000, root.destroy)

    # 阻止窗口关闭按钮关闭窗口
    # root.protocol('WM_DELETE_WINDOW', lambda: None)

    # 运行主循环
    root.mainloop()

def import_to_database(df, file_date, db_config):
    try:
        # 连接数据库
        conn = mysql.connector.connect(**db_config)
        cursor = conn.cursor()

        # 遍历 DataFrame 并插入数据
        for _, row in df.iterrows():
            print(row)
            stock_name = row.iloc[0].strip()  # 股票名称
            stock_code = str(row.iloc[1]).strip()  # 股票代码
            print(stock_code)

            alert_time_str = row.iloc[2].strip()  # 预警时间（仅时间部分）
            current_price = float(str(row.iloc[3]).strip())  # 当前价格，确保转换为字符串
            price_change = float(str(row.iloc[4]).strip().rstrip('%'))  # 涨跌幅（去掉百分号）
            status = row.iloc[5].strip()  # 状态
            # 解析预警时间（仅时间部分）
            alert_time = datetime.strptime(alert_time_str, "%H:%M").time()
            print(alert_time)
            # 构造 SQL 插入语句
            insert_query = """
            INSERT INTO AlertData (stock_code, stock_name, alert_time, current_price, price_change, status, date)
            VALUES (%s, %s, %s, %s, %s, %s, %s)
            """
            values = (stock_code, stock_name, alert_time, current_price, price_change, status, file_date)
            print(insert_query, values)
            # 执行插入操作
            cursor.execute(insert_query, values)

        # 提交事务
        conn.commit()
        print("数据导入成功！")
    except Exception as e:
        print(f"导入数据时出错：{e}")
    finally:
        # 关闭连接
        cursor.close()
        conn.close()


def monitor_file(mp3_path,db_config):
    global last_modified_time, last_content
    while True:
        # 获取文件的当前修改时间和内容

        current_modified_time = os.path.getmtime(file_path)
        print(current_modified_time)
        print(last_modified_time)
        # print(last_content)
        with open(file_path, 'r', encoding='GBK') as file:
            current_content = file.read()

        # 如果文件被修改
        if current_modified_time != last_modified_time or current_content != last_content:
            last_modified_time = current_modified_time
            # 计算新增的内容
            added_content = current_content[len(last_content):].strip()
            last_content = current_content  # 更新记录的内容
            print(added_content)
            # 如果有新增内容，显示提醒
            if added_content:
                # 插入到数据库中
                df = ""
                file_date = datetime.now().strftime("%Y-%m-%d")
                import_to_database(df, file_date, db_config)

                # 弹出提示信息
                show_alert(added_content,mp3_path)

        # 每隔1秒检查一次
        time.sleep(2)

if __name__ == "__main__":
    # 获取脚本所在目录的上一级目录
    script_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    # 构造音频文件的完整路径
    mp3_path = os.path.join(script_dir, "mp3", "alarm.mp3")
    print(mp3_path)

    monitor_file(mp3_path,db_config)