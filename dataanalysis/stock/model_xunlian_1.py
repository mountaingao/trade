"""
提取重要特征参数
一、特征重要性分析与关键参数识别
1. 特征重要性评估方法
2. 关键参数识别标准
稳定性：在不同时间窗口下重要性排名一致

一致性：在回归和分类任务中均表现重要

解释性：符合金融逻辑（如量价关系）

独立性：与其他特征相关性低（相关系数<0.6）
"""

import xgboost as xgb
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.metrics import classification_report, r2_score, mean_squared_error
from sklearn.metrics import accuracy_score, precision_score



def generate_model_data(input_file):
    """
    根据输入文件生成模型数据并保存特征权重
    
    参数:
        input_file (str): 输入Excel文件路径
    """
    df = pd.read_excel(input_file)
    
    # 新增数据清洗步骤：处理标签列中的无效值
    # ======= 新增开始 =======
    # 检查并处理NaN值
    nan_mask = df['1_day_close'].isna()
    if nan_mask.any():
        print(f"警告：发现{nan_mask.sum()}个NaN值，已删除对应行")
        df = df[~nan_mask]
    
    # 检查并处理无穷大值
    inf_mask = np.isinf(df['1_day_close'])
    if inf_mask.any():
        print(f"警告：发现{inf_mask.sum()}个无穷大值，已删除对应行")
        df = df[~inf_mask]
    
    # 检查过大值（假设大于1e10为无效）
    large_mask = np.abs(df['1_day_close']) > 1e10
    if large_mask.any():
        print(f"警告：发现{large_mask.sum()}个过大值，已删除对应行")
        df = df[~large_mask]
    # ======= 新增结束 =======

    # 特征选择
    features = [
        'sma_up', 'sma_down', 'macd', 'is_up',
        'upper_days_counts', 'ma_amount_days_ratio_3', 'ma_amount_days_ratio_5','ma_amount_days_ratio_8',"ma_amount_days_ratio_11",
        'total_score','amount', 'free_amount', 'increase', 'amplitude', 'jgcyd', 'lspf', 'focus',
        'last_desire_daily'
    ]
    
    X_train = df[features]
    y_reg = df['1_day_close']
    y_clf = (df['1_day_close'] > 0).astype(int)
    
    # 训练模型
    model_reg = xgb.XGBRegressor()
    model_reg.fit(X_train, y_reg)
    
    model_clf = xgb.XGBClassifier()
    model_clf.fit(X_train, y_clf)
    
    # 获取特征重要性
    feat_importance_reg = model_reg.feature_importances_
    feat_importance_clf = model_clf.feature_importances_
    
    # 标准化权重
    feature_weights_reg = feat_importance_reg / feat_importance_reg.sum()
    feature_weights_clf = feat_importance_clf / feat_importance_clf.sum()
    
    # 生成文件名前缀（从输入路径提取）
    file_prefix = input_file.split('/')[-1].split('.')[0]
    
    # 保存回归模型特征权重
    weights_reg = pd.DataFrame({
        'Feature': features,
        'Weight': feature_weights_reg
    })
    reg_filename = f'../data/{file_prefix}_feature_weights_reg.csv'
    weights_reg.to_csv(reg_filename, index=False)
    print(f"回归模型特征权重已保存至: {reg_filename}")
    
    # 保存分类模型特征权重
    weights_clf = pd.DataFrame({
        'Feature': features,
        'Weight': feature_weights_clf
    })
    clf_filename = f'../data/{file_prefix}_feature_weights_clf.csv'
    weights_clf.to_csv(clf_filename, index=False)
    print(f"分类模型特征权重已保存至: {clf_filename}")

# 模型效果评估
def metrics_sklearn(y_valid, y_pred_):
    """模型效果评估"""
    r2 = r2_score(y_valid, y_pred_)
    print('r2_score:{0}'.format(r2))

    mse = mean_squared_error(y_valid, y_pred_)
    print('mse:{0}'.format(mse))

def calculate_model_data(input_file):
    # 预测
    df2 = pd.read_excel(input_file)
    y_test = df2[features]
    # y_pred_proba = model_clf.predict_proba(X_test)[:, 1]  # 获取正类的概率

    model_reg = xgb.XGBRegressor()

    model_clf = xgb.XGBClassifier()

    # 预测是
    y_pred = model_clf.predict(y_test)
    # print(y_pred)


    y2_clf = (df2['1_day_close'] > 0).astype(int)
    # y2_clf = df2['1_day_close']
    # print('1_day_close：', y2_clf)

    # 比较 y_pred 和 y2_clf 中的值，统计他们之中当预测正确的比例；
    print('整体预测正确的比例：', sum(y_pred == y2_clf) / len(y2_clf))

    # 统计当预测值为1时的预测结果正确的比例（精确率）
    true_positives = sum((y_pred == 1) & (y2_clf == 1))
    predicted_positives = sum(y_pred == 1)
    precision = true_positives / predicted_positives if predicted_positives > 0 else 0
    print('当预测值为1时预测正确的比例（精确率）:', precision)

    # 列出所有的预测结果，逐行遍历打印出来
    for m, n in zip(y_pred, y2_clf):
        print('预测值为{0}, 真实结果为{1}'.format(m, n))
        # if m / n - 1 > 0.2:
        #     print('预测值为{0}, 真是结果为{1}, 预测结果偏差大于20%'.format(m, n))


    """模型效果评估"""
    metrics_sklearn(y2_clf, y_pred)
    print("准确率:", accuracy_score(y2_clf, y_pred))
    print("精确率:", precision_score(y2_clf, y_pred))
    print(classification_report(y2_clf, y_pred))

    # 预测涨幅
    y_pred = model_reg.predict(y_test)
    # print(y_pred)

    y2_reg = df2['1_day_close']
    # print('1_day_close：', y2_reg)
    # 列出所有的预测结果，逐行遍历打印出来
    i = 0
    for m, n in zip(y_pred, y2_reg):
        print('预测值为{0}, 真实结果为{1}'.format(m, n))
        # 添加防除零保护
        if n == 0:
            print('真实结果为0，跳过偏差计算')
            continue

        # 检查预测偏差
        if m / n - 1 > 0.2:
            print('预测值为{0}, 真实结果为{1}, 预测结果偏差大于20%'.format(m, n))
            i = i + 1

    # i/len(y2_clf)
    print('预测结果偏差大于20%的比例：', i/len(y2_reg))
    metrics_sklearn(y2_reg, y_pred)
    exit()

# 示例调用
if __name__ == "__main__":
    # 使用示例文件生成模型数据
    # generate_model_data("../data/0401-0531.xlsx")
    generate_model_data("../data/0601-0619.xlsx")

    # 预测示例
    # calculate_model_data("../data/0610.xlsx")


