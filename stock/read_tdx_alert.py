import os
import time
import tkinter as tk
from tkinter import messagebox
from playsound import playsound
from pydub import AudioSegment
from pydub.playback import play
import mysql.connector
import datetime
import pandas as pd
from datetime import datetime
from stockrating.stock_rating_ds import evaluate_stock
from stockrating.get_stock_block import process_stock_concept_data
from stockrating.read_local_info_tdx import expected_calculate_total_amount
import tempfile
import json

# 新增代码：读取配置文件
config_path = os.path.join(os.path.dirname(__file__), '..', 'config', 'config.json')
with open(config_path, 'r', encoding='utf-8') as config_file:  # 修改编码为utf-8
    config = json.load(config_file)

# 设置自定义临时目录
tempfile.tempdir = config['tempdir']

# 确保临时目录存在并且具有写权限
if not os.path.exists(tempfile.tempdir):
    os.makedirs(tempfile.tempdir, exist_ok=True)

# 文件路径
file_path_test = config['file_path_test']
file_path = config['file_path']
if config['istest']:
    file_path = file_path_test

print("正在读取文件..."+file_path)
# 检查文件是否存在，如果不存在则创建文件
if not os.path.exists(file_path):
    with open(file_path, 'w', encoding='GBK') as file:
        file.write("")  # 创建空文件

# 记录文件的最后修改时间
last_modified_time = os.path.getmtime(file_path)
with open(file_path, 'r', encoding='GBK') as file:
    last_content = file.read()  # 读取初始文件内容

# 数据库连接配置
db_config = config['db_config']

def show_alert(new_content, mp3_path):
    root = tk.Tk()
    root.title("提醒")  # 设置窗口标题
    root.attributes('-topmost', True)  # 确保窗口始终在最前面
    # 显示消息内容
    message = new_content
    label = tk.Label(root, text=message, wraplength=420, justify="left", padx=20,  # 内部水平填充
                     pady=20,  # 内部垂直填充
                     borderwidth=2,  # 边框宽度
                     relief="groove",  # 边框样式
                     font=("Arial", 10))  # 设置字体大小
                     # fg="red",  # 设置字体颜色
                     # bg="white")  # 设置背景颜色
    label.pack(expand=True, padx=20, pady=20)
    # messagebox.showinfo("提醒", f"文件内容已更新！\n\n新增内容：\n{new_content}")
    # playsound("alarm.mp3")

    # 设置定时器，10秒后关闭窗口
    root.after(10000, root.destroy)

    # 播放音频
    sound = AudioSegment.from_mp3(mp3_path)
    play(sound)
    # playsound(mp3_path)

    # 阻止窗口关闭按钮关闭窗口
    # root.protocol('WM_DELETE_WINDOW', lambda: None)

    # 运行主循环
    root.mainloop()

def import_to_database(data, conn):
    try:
        cursor = conn.cursor()
        insert_query = """
        INSERT INTO AlertData (stock_code, stock_name, alert_time, current_price, price_change, status, date, score, popup_status)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
        """
        for row in data:
            try:
                stock_code = row[0].strip()
                stock_name = str(row[1]).strip()
                alert_datetime_str = row[2].strip()  # 包含日期和时间的字符串
                current_price = float(str(row[3]).strip())
                price_change = float(str(row[4]).strip().rstrip('%'))
                status = row[6].strip()

                # 解析包含日期和时间的字符串
                alert_datetime = datetime.strptime(alert_datetime_str, "%Y-%m-%d %H:%M")
                alert_time = alert_datetime.time()  # 提取时间部分
                alert_date = alert_datetime.date()  # 提取日期部分

                # 计算分数和弹出状态
                score = evaluate_stock(stock_code)
                popup_status = 1 if score >= 50 else 0

                values = (stock_code, stock_name, alert_time, current_price, price_change, status, alert_date, score, popup_status)
                cursor.execute(insert_query, values)
                print(f"成功插入数据: {values}")
            except Exception as e:
                print(f"处理行数据时出错：{e}, 数据行: {row}")
        conn.commit()
        print("数据导入成功！")
    except Exception as e:
        print(f"导入数据时出错：{e}")
    finally:
        cursor.close()
        # conn.close()

def get_number_of_timer(time):
    # 解析时间字符串为 datetime 对象
    alert_time = datetime.strptime(time, "%Y-%m-%d %H:%M").time()

    # 计算时间对应的数字
    if 9 <= alert_time.hour < 11 or (alert_time.hour == 11 and alert_time.minute <= 30):
        time_number = (alert_time.hour - 9) * 60 + alert_time.minute-30
    elif 13 <= alert_time.hour < 15 or (alert_time.hour == 15 and alert_time.minute <= 0):
        time_number = 120 + (alert_time.hour - 13) * 60 + alert_time.minute
    else:
        time_number = -1  # 如果时间不在指定范围内，设置为 None
    print(time_number)
    return time_number

def get_alert_info(lines, conn):
    alert_info = []
    result = []
    for line in lines:
        fields = line.split("\t")  # 按制表符分割字段
        print(fields)
        if len(fields) >= 2:  # 确保每行至少有两个字段
            # 获取股票代码
            stock_code = fields[0].strip()
            cursor = conn.cursor()

            # 获取板块数据
            block = process_stock_concept_data(cursor, stock_code)
            block_str = ', '.join(block[:3])
            print(block_str)

            # 调用 evaluate_stock 方法获取评分
            score = evaluate_stock(stock_code)
            #计算当日成交量，是否能过10亿，如果可以，则弹出提示，很可能是涨停标的
            total_amount,current_amount = expected_calculate_total_amount(stock_code,get_number_of_timer( fields[2].strip()))

            # 将板块数据和评分加入到 fields 中
            fields.append(current_amount)
            fields.append(total_amount)
            fields.append(block_str)

            fields.append(str(score))

            # 将处理后的 fields 加入到 result 中
            result.append(fields)
            alert_info = format_alert_info(fields)
            # 将股票代码、板块数据和评分加入到 alert_info 中
            alert_info.append(alert_info)
    print(result)
    print(alert_info)
    return result, alert_info

def format_alert_info(fields):
    print(fields)
    formatted_lines = []
    formatted_lines.append("-------------------------------------")
    formatted_lines.append(f"|【{fields[1].strip()}】 {fields[0].strip()}  【{item[6].strip()}】")
    formatted_lines.append("-------------------------------------")
    formatted_lines.append(f"| 预警时间: {fields[2].strip()}           ")
    formatted_lines.append(f"| 当前价格: {fields[3].strip()} ({fields[4].strip()})          ")
    formatted_lines.append("-------------------------------------")
    formatted_lines.append("| 相关概念:                        ")
    for concept in block[:3]:
        formatted_lines.append(f"| - {concept}                         ")
    formatted_lines.append("-------------------------------------")
    formatted_lines.append(f"| 注: 上轨有效！                   ")

    formatted_lines.append("-------------------------------------")

    formatted_lines.append(f"|【{fields[1].strip()}】 {fields[0].strip()}  【{item[6].strip()}】")
    formatted_lines.append("-------------------------------------")
    formatted_lines.append(f"|【评分】: {fields[7]} 【{fields[6].strip()}】  ")
    formatted_lines.append(f"| 预警时间: {fields[2].strip()}           ")
    formatted_lines.append(f"| 当前价格: {fields[3].strip()} ({item[4].strip()})          ")
    formatted_lines.append("-------------------------------------")
    formatted_lines.append(f"| 当前成交额: {current_amount:.2f}亿           ")
    formatted_lines.append(f"| 预计成交额: {total_amount:.2f}亿              ")
    formatted_lines.append("-------------------------------------")
    formatted_lines.append("| 相关概念:                        ")
    for concept in block[:3]:
        formatted_lines.append(f"| - {concept}                         ")
    formatted_lines.append("-------------------------------------")
    formatted_lines.append(f"| 注: 上轨有效！                   ")

    return formatted_lines

def format_result(result,conn):
    """格式化 result 列表，提取每行的第一个和第二个字段，并用空格分隔，多条记录用换行符分隔
    过滤代码信息：通过结果集合 result 获取代码，并通过 evaluate_stock 方法得到积分，当大于50分时，返回给弹窗提示
    """
    formatted_lines = []
    for item in result:
        if len(item) >= 2:  # 确保每行至少有两个字段
            # 获取股票代码
            stock_code = item[0].strip()
            cursor = conn.cursor()

            # 获取板块数据
            block_str = ""
            block = process_stock_concept_data(cursor, stock_code)
            # print(block)
            # 数据库返回的板块数据和网络请求获取的值是否一致
            if len(block) > 3:
                # block = block[:3]
                block_str = ', '.join(block[:3])
            else:
                block_str = ', '.join(block)
            print(block_str)

            cursor.close()
            # 调用 evaluate_stock 方法获取评分
            if stock_code.startswith('8') or stock_code.startswith('4') or stock_code.startswith('9'):
                # 将 block 列表转换为字符串
                # formatted_line = f"{stock_code} {item[1].strip()} {item[2].strip()} {item[3].strip()} {item[4].strip()} "
                # formatted_lines.append(formatted_line)
                # formatted_lines.append(block_str)
                # formatted_lines.append(f"注: 上轨有效")
                formatted_lines.append("-------------------------------------")
                formatted_lines.append(f"|【{item[1].strip()}】 {stock_code}  【{item[6].strip()}】")
                formatted_lines.append("-------------------------------------")
                formatted_lines.append(f"| 预警时间: {item[2].strip()}           ")
                formatted_lines.append(f"| 当前价格: {item[3].strip()} ({item[4].strip()})          ")
                formatted_lines.append("-------------------------------------")
                formatted_lines.append("| 相关概念:                        ")
                for concept in block[:3]:
                    formatted_lines.append(f"| - {concept}                         ")
                formatted_lines.append("-------------------------------------")
                formatted_lines.append(f"| 注: 上轨有效！                   ")

            else:
                score = evaluate_stock(stock_code)
                #计算当日成交量，是否能过10亿，如果可以，则弹出提示，很可能是涨停标的
                total_amount,current_amount = expected_calculate_total_amount(stock_code,get_number_of_timer( item[2].strip()))
                # 如果评分大于50，添加到格式化结果中
                if score >= 50:
                    # 将 block 列表转换为字符串
                    formatted_lines.append("-------------------------------------")
                    formatted_lines.append(f"|【{item[1].strip()}】 {stock_code}   【{item[6].strip()}】")
                    formatted_lines.append("-------------------------------------")
                    formatted_lines.append(f"|【评分】: {score}   ")
                    formatted_lines.append(f"| 预警时间: {item[2].strip()}           ")
                    formatted_lines.append(f"| 当前价格: {item[3].strip()} ({item[4].strip()})          ")
                    formatted_lines.append("-------------------------------------")
                    formatted_lines.append(f"| 当前成交额: {current_amount:.2f}亿           ")
                    formatted_lines.append(f"| 预计成交额: {total_amount:.2f}亿              ")
                    formatted_lines.append("-------------------------------------")
                    formatted_lines.append("| 相关概念:                        ")
                    for concept in block[:3]:
                        formatted_lines.append(f"| - {concept}                         ")
                    formatted_lines.append("-------------------------------------")
                    formatted_lines.append(f"| 注: 上轨有效！                   ")
                else:
                    print(f"预计成交额{total_amount:.2f}亿")
                    if total_amount >= 10:
                        formatted_lines.append("-------------------------------------")
                        formatted_lines.append(f"|【{item[1].strip()}】 {stock_code}  【{item[6].strip()}】")
                        formatted_lines.append("-------------------------------------")
                        formatted_lines.append(f"|【评分】: {score} 【{item[6].strip()}】  ")
                        formatted_lines.append(f"| 预警时间: {item[2].strip()}           ")
                        formatted_lines.append(f"| 当前价格: {item[3].strip()} ({item[4].strip()})          ")
                        formatted_lines.append("-------------------------------------")
                        formatted_lines.append(f"| 当前成交额: {current_amount:.2f}亿           ")
                        formatted_lines.append(f"| 预计成交额: {total_amount:.2f}亿              ")
                        formatted_lines.append("-------------------------------------")
                        formatted_lines.append("| 相关概念:                        ")
                        for concept in block[:3]:
                            formatted_lines.append(f"| - {concept}                         ")
                        formatted_lines.append("-------------------------------------")
                        formatted_lines.append(f"| 注: 上轨有效！                   ")

    return "\n".join(formatted_lines)

def monitor_file(mp3_path,db_config):
    global last_modified_time, last_content

    # 连接数据库
    conn = mysql.connector.connect(**db_config)
    while True:
        # 获取文件的当前修改时间和内容

        current_modified_time = os.path.getmtime(file_path)
        formatted_time = datetime.fromtimestamp(current_modified_time).strftime('%Y-%m-%d %H:%M:%S')
        print(formatted_time)
        # print(last_modified_time)
        # print(last_content)
        with open(file_path, 'r', encoding='GBK') as file:  # 修改编码为utf-8
            current_content = file.read()

        # 如果文件被修改
        if current_modified_time != last_modified_time or current_content != last_content:
            last_modified_time = current_modified_time
            # 计算新增的内容
            added_content = current_content[len(last_content):].strip()
            last_content = current_content  # 更新记录的内容
            # 301396	宏景科技	2025-02-21 09:20	55.20	 0.00%	    0	开盘
            print(added_content)

            # 如果有新增内容，显示提醒
            if added_content:
                # 插入到数据库中
                # 将数据按行分割
                lines = added_content.strip().split("\n")

                # 解析每行数据
                result = []
                # get_alert_info(lines, conn)
                # exit()
                for line in lines:
                    fields = line.split("\t")  # 按制表符分割字段
                    # print(fields)
                    result.append(fields)
                # 过滤掉积分低于50的信号
                alertInfo = format_result(result,conn)
                if len(alertInfo)>0:
                    print(alertInfo)
                    # 弹出提示信息
                    popup_status = show_alert(alertInfo,mp3_path)

                # print(df)
                import_to_database(result,  conn)

        # 每隔1秒检查一次
        time.sleep(2)

if __name__ == "__main__":
    # 获取脚本所在目录的上一级目录
    script_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    # 构造音频文件的完整路径
    mp3_path = os.path.join(script_dir, "mp3", "alarm.mp3")
    print(mp3_path)

    mp3_path = os.path.join(script_dir, "mp3", "alarm.mp3")

    # sound = AudioSegment.from_mp3(mp3_path)
    # play(sound)
    # exit()
    monitor_file(mp3_path,db_config)

