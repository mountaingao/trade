"""
模型调用和预测结果
"""

import pandas as pd
from model_xunlian_alert_1 import predictions_model_data_file,predictions_model_data


# 示例调用修改
if __name__ == "__main__":
    # 使用多个数据集训练并生成模型 ,
    model = {
        'reg_weights': '../models/250709_feature_weights_reg.csv',
        'clf_weights': '../models/250709_feature_weights_clf.csv',
        'reg_model': '../models/250709_model_reg.json',
        'clf_model': '../models/250709_model_clf.json'}

    #预测文件中的数据
    predictions_model_data_file("../alert/0711.xlsx",model)

    exit()

    # 比较0709 和070903 两个文件中重合的代码后做数据分析
    # 预测 准备数据
    df_1 = pd.read_excel("../alert/0709.xlsx")
    df_2 = pd.read_excel("../alert/070903.xlsx")

    print(df_1.head(10))
    print(df_2.head(5))
    df_3 = df_1[df_1['代码'].isin(df_2['代码'])]
    print(df_3.head(50))
    df = df_3[['代码','当日涨幅', '信号天数', '净额', '净流入', '当日资金流入', '是否领涨','最高价']]

    result = predictions_model_data(df, model)


# # 预测一行数据的结果
#     # 新增：单行数据预测示例
#     print("\n===== 开始单行数据测试 =====")
#     # 示例数据（包含所有必需特征）
#     example_data = {
#         '最高价': '是',      # 将被映射为1
#         '是否领涨': '是',    # 将被映射为1
#         '当日涨幅': 0.05,    # 5%涨幅
#         '信号天数': 3,       # 连续3天出现信号
#         '净额': 1000000,    # 净额100万
#         '净流入': 500000,    # 净流入50万
#         '当日资金流入': 2000000  # 当日资金流入200万
#     }
#
#     print("测试数据:", example_data)
#     result = predictions_model_data(example_data, model)
#     print("\n预测结果:")
#     print(f"分类预测: {result['分类预测']}")
#     print(f"回归预测(次日最高涨幅): {result['回归预测']:.4f}")
#     print("===== 测试结束 =====")