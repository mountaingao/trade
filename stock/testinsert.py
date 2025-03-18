import datetime
import time


file_path = r"ALERT.txt"
# file_path = r"ALERT-0311.txt"
# file_path = r"ALERT-0310.txt"
file_path = r"ALERT-0317.txt"
# file_path = r"D:/BaiduSyncdisk/个人/通达信/ALERT/ALERT.txt"

# 读取文件并解析时间
rows = []
with open(file_path, 'r', encoding='GB2312') as f:
    for line in f:
        line = line.rstrip('\n')  # 去除行尾换行符
        if not line.strip():
            continue
        parts = line.split('\t')
        if len(parts) < 7:
            continue  # 跳过列数不足的行
        time_str = parts[2].strip()
        try:
            time_obj = datetime.datetime.strptime(time_str, "%Y-%m-%d %H:%M")
        except ValueError:
            continue  # 时间格式错误，跳过
        rows.append((time_obj, line))

# 按时间排序
rows.sort(key=lambda x: x[0])

print(rows)
# 筛选符合条件的行（每隔1分钟）
selected = []
if rows:
    current_time = rows[0][0]
    selected.append(rows[0][1])
    for time_obj, line in rows[0:]:
        with open('alert1.txt', 'a', encoding='GB2312') as target_file:  # 以追加模式打开目标文件
            target_file.write(line)  # 写入当前行
            target_file.write('\n')  # 写入换行符（如果需要）
            print(f"写入一行：{line.strip()}")
        time.sleep(15)  # 暂停10S
