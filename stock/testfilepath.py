import os

file_path = r"D:\temp\tmpbtqbjz67.wav"

# 检查文件是否存在
if os.path.exists(file_path):
    print("文件存在，尝试访问...")
    try:
        with open(file_path, 'rb') as file:
            print("文件已成功打开")
    except PermissionError as e:
        print(f"权限错误: {e}")
else:
    print("文件不存在，请检查路径是否正确")