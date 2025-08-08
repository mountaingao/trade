import struct
import os
import pandas as pd
import json

def read_lc5_file(file_path):
    """
    读取LC5文件并解析数据
    
    :param file_path: LC5文件的路径
    :return: 解析后的数据列表
    """
    data_list = []

    try:
        with open(file_path, 'rb') as f:
            while True:
                # 读取每条数据记录（32字节）
                chunk = f.read(32)
                if len(chunk) < 32:
                    break  # 读取到文件末尾

                # 解析数据
                date, minutes, open_price, high_price, low_price, close_price, turnover, volume = struct.unpack('<HHffffII', chunk[:28])

                # 解析日期
                year = (date // 2048) + 2004
                month = (date % 2048) // 100
                day = (date % 2048) % 100

                # 将从0点起的分钟数转换为具体的时间（小时和分钟）
                hour = minutes // 60
                minute = minutes % 60

                # 调整时间为5分钟前
                hour, minute = adjust_time(hour, minute)

                # 将每条数据保存为字典形式，并格式化价格数据到小数点后3位
                data_list.append({
                    '时间': f'{year}-{month:02}-{day:02} {hour:02}:{minute:02}:00',
                    '开盘价': round(open_price, 3),
                    '最高价': round(high_price, 3),
                    '最低价': round(low_price, 3),
                    '收盘价': round(close_price, 3),
                    '成交额': turnover,
                    '成交量': volume
                })

    except Exception as e:
        print(f"读取文件 {file_path} 时出错: {e}")

    return data_list


def save_to_excel(data_list, output_file):
    """
    将数据保存到Excel文件
    
    :param data_list: 解析后的数据列表
    :param output_file: 输出的Excel文件名
    """
    try:
        df = pd.DataFrame(data_list)
        df.to_excel(output_file, index=False)
        print(f"数据已保存至 {output_file}")

    except Exception as e:
        print(f"保存文件 {output_file} 时出错: {e}")


def process_lc5_files(directory):
    """
    处理指定目录下的所有LC5文件，将结果保存到对应的Excel文件中
    
    :param directory: LC5文件目录路径
    """
    for file_name in os.listdir(directory):
        if file_name.endswith('.lc5'):
            file_path = os.path.join(directory, file_name)
            print(f"正在处理文件: {file_path}")

            # 读取LC5文件
            data_list = read_lc5_file(file_path)

            # 获取基础文件名（不包含扩展名）
            base_name = os.path.splitext(file_name)[0]

            # 创建输出目录
            output_directory = os.path.join('./data', base_name)
            os.makedirs(output_directory, exist_ok=True)

            # 构建输出的Excel文件名
            output_file = os.path.join(output_directory, f'{base_name}_5min.xlsx')

            # 保存到Excel
            save_to_excel(data_list, output_file)


def adjust_time(hour, minute):
    """
    调整时间的辅助函数（需要实现具体逻辑）
    
    :param hour: 小时
    :param minute: 分钟
    :return: 调整后的小时和分钟
    """
    # TODO: 实现具体的时间调整逻辑
    return hour, minute

# 新增代码：读取配置文件
config_path = os.path.join(os.path.dirname(__file__), '../../', 'config', 'config.json')
with open(config_path, 'r', encoding='utf-8') as config_file:  # 修改编码为utf-8
    config = json.load(config_file)

def get_market_from_code(code: str) -> int:
    """根据股票代码判断市场代码"""
    # 确保code是字符串类型
    code = str(code)
    if code.startswith('sh') or code.startswith('6'):
        return 'sh'  # 上海
    elif code.startswith('sz') or code.startswith(('0', '3', '15', '16', '20', '30')):
        return 'sz'  # 深圳
    else:
        raise ValueError(f"Unknown market for code: {code}")

def get_m5_from_code(code: str) -> str:
    file_path = get_market_from_code(code)
    print(file_path)
    lc5_directory = f'{config["tdxdir"]}vipdoc/{file_path}/fzline'
    print(lc5_directory)


    file_path = os.path.join(lc5_directory, file_path+code+".lc5")
    print(f"正在处理文件: {file_path}")
    # 读取LC5文件
    data_list = read_lc5_file(file_path)
    print(data_list)

if __name__ == "__main__":
    # 指定LC5文件所在目录  D:\zd_haitong\vipdoc\sh\fzline
    # lc5_directory = config['tdxdir']+'vipdoc'+'/sh/fzline'

    # 转换文件
    # process_lc5_files(lc5_directory)

    # 根据代码获得文件的路径 如 上海 为 sh  深圳为sz 北京为bj
    code = "688048"
    get_m5_from_code(code)
