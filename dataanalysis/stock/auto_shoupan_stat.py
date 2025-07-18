import os
from datetime import datetime, timedelta
import json  # 假设文件是JSON格式，如果不是请替换相应的读取方式
from datetime import datetime, timedelta
from chinese_calendar import is_workday, is_holiday
import pandas as pd
import auto_shoupan

# /*
# 统计历史数据，得到最终的统计报告，对预测的成功率等进行计算

# ?
def get_previous_trading_day(date):
    previous_date = date - timedelta(days=1)
    while not is_workday(previous_date) or is_holiday(previous_date):
        previous_date -= timedelta(days=1)
    return previous_date

# 示例Python代码
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
# 相关性分析
def corr_matrix( df):
    # 计算相关系数
    corr_matrix = df.corr()

    # 查看与目标变量的相关性
    target_corr = corr_matrix['次日涨幅'].sort_values(ascending=False)

    # 可视化
    plt.figure(figsize=(12,8))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
    plt.show()

from sklearn.ensemble import RandomForestRegressor

# 随机森林来分析特征
def RandomForest_tezheng(df):
    # 准备数据
    X = df.drop(['次日涨幅', '序号', '名称'], axis=1)  # 移除目标变量和无关字段
    y = df['次日涨幅']

    # 训练模型
    model = RandomForestRegressor()
    model.fit(X, y)

    # 获取特征重要性
    feature_importances = pd.DataFrame({
        'feature': X.columns,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)

def process_prediction_files_stat(base_dir="../data/predictions/"):
    # 1. 获取当前月日（例如：今天是7月8日，则得到 "0708"）
    # 上一个交易日的月日
    md = datetime.now().date()
    print(md)
    previous_mmdd = get_previous_trading_day(md).strftime("%m%d")
    print(previous_mmdd)
    # current_mmdd = datetime.now().strftime("%m%d")
    # previous_mmdd = '0716'
    # 2. 遍历base_dir下的所有文件夹
    for folder_name in os.listdir(base_dir):
        folder_path = os.path.join(base_dir, folder_name)

        # 确保是文件夹
        if not os.path.isdir(folder_path):
            continue

        print(f"正在处理文件夹: {folder_name}")

        df = pd.DataFrame()
        # 3. 遍历文件夹中的所有文件，按目录输出一个统计数据文件，输出文件名为 "日期.xlsx"
        for filename in os.listdir(folder_path):
            print(f"正在处理文件: {filename}")
            print(f"filename: {filename[:4]}")

            # 检查文件名前4位是否匹配当前月日
            if len(filename) >= 4 and filename[:4] <= previous_mmdd:
                file_path = os.path.join(folder_path, filename)
                print(f"找到匹配文件: {file_path}")
                filename_without_extension = os.path.splitext(filename)[0]

                # try:
                # 4. 读取文件内容
                df_pred = pd.read_excel(file_path)
                print(df_pred.head(100))

                # 计算成功比例 AI预测成功比例
                # df里添加一行记录

                # 修复1：确保使用正确的列名
                if '是否成功' not in df_pred.columns:
                    print(f"警告: 文件 {filename} 中缺少 '是否成功' 列")
                    continue

                # 修复2：转换数据类型
                df_pred['是否成功'] = pd.to_numeric(df_pred['是否成功'], errors='coerce').fillna(0)
                df_pred['预测'] = df_pred['预测'].map({'是': 1, '否': 0}).fillna(0)
                df_pred['AI预测'] = pd.to_numeric(df_pred['AI预测'], errors='coerce').fillna(0)

                # 修复3：计算成功率
                ai_pred_sum = df_pred['AI预测'].sum()
                rule_pred_sum = df_pred['预测'].sum()
                success_sum = df_pred['是否成功'].sum()

                # 修复4：避免除零错误
                ai_success_ratio = success_sum / ai_pred_sum if ai_pred_sum > 0 else 0
                rule_success_ratio = success_sum / rule_pred_sum if rule_pred_sum > 0 else 0

                # 修复5：计算重合成功率
                overlap_mask = (df_pred['预测'] == 1) & (df_pred['AI预测'] == 1)
                overlap_success = df_pred.loc[overlap_mask & (df_pred['是否成功'] == 1)].shape[0]
                overlap_total = df_pred.loc[overlap_mask].shape[0]
                overlap_success_ratio = overlap_success / overlap_total if overlap_total > 0 else 0

                # 创建结果DataFrame ,值都使用百分比

                result_data = {
                    "日期": [filename_without_extension],
                    "AI预测成功比例": [ai_success_ratio*100],
                    "预测成功比例": [rule_success_ratio*100],
                    "重合成功比例": [overlap_success_ratio*100]
                }
                # 将result_data 增加到 df 中
                df = pd.concat([df, pd.DataFrame(result_data)])
                # df = pd.concat([df, pd.DataFrame(result_data)])
                    # except Exception as e:
                    #     print(f"处理文件 {filename} 时出错: {e}")
        print( df)
        output_file = os.path.join("../data/stat", f"{md.strftime('%y-%m-%d')}-{folder_name}.xlsx")
        print( output_file)
        df.to_excel(output_file, index=False)

if __name__ == "__main__":
    process_prediction_files_stat()