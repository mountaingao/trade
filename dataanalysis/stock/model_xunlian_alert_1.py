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
import os


def generate_model_data(input_files):
    """
    根据输入文件列表生成模型数据并保存特征权重
    
    参数:
        input_files (list): 输入Excel文件路径列表
    """
    # 读取并合并所有文件
    dfs = []
    for file in input_files:
        df_part = pd.read_excel(file)
        dfs.append(df_part)
    df = pd.concat(dfs, ignore_index=True)

    # df['value'] = (df['value'] > 0).astype(int)
    df['value'] = df['最高价'].map({'是': 1, '否': 0})
    df['是否领涨'] = df['是否领涨'].map({'是': 1, '否': 0})


    print(df.head(10))
    # 新增数据清洗步骤：处理标签列中的无效值
    # ======= 新增开始 =======
    # 检查并处理NaN值
    nan_mask = df['value'].isna()
    if nan_mask.any():
        print(f"警告：发现{nan_mask.sum()}个NaN值，已删除对应行")
        df = df[~nan_mask]
    
    # 检查并处理无穷大值
    inf_mask = np.isinf(df['value'])
    if inf_mask.any():
        print(f"警告：发现{inf_mask.sum()}个无穷大值，已删除对应行")
        df = df[~inf_mask]
    
    # 检查过大值（假设大于1e10为无效）
    large_mask = np.abs(df['value']) > 1e10
    if large_mask.any():
        print(f"警告：发现{large_mask.sum()}个过大值，已删除对应行")
        df = df[~large_mask]
    # ======= 新增结束 =======

    # 特征选择
    features = [
        '当日涨幅', '信号天数', '净额', '净流入', '当日资金流入', '是否领涨'
    ]

    
    X_train = df[features]
    y_reg = df['次日最高涨幅']
    # y_clf = (df['value'] > 0).astype(int)
    y_clf = df['value']

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
    
    # 生成文件名前缀（使用组合文件名）,每日一个文件
    # file_prefix = "_".join([f.split('/')[-1].split('.')[0] for f in input_files])
    file_prefix = pd.Timestamp.now().strftime("%y%m%d")
    # 保存回归模型特征权重
    weights_reg = pd.DataFrame({
        'Feature': features,
        'Weight': feature_weights_reg
    })
    reg_filename = f'../models/{file_prefix}_feature_weights_reg.csv'
    weights_reg.to_csv(reg_filename, index=False)
    print(f"回归模型特征权重已保存至: {reg_filename}")
    
    # 保存分类模型特征权重
    weights_clf = pd.DataFrame({
        'Feature': features,
        'Weight': feature_weights_clf
    })
    clf_filename = f'../models/{file_prefix}_feature_weights_clf.csv'
    weights_clf.to_csv(clf_filename, index=False)
    print(f"分类模型特征权重已保存至: {clf_filename}")

    # 保存模型
    reg_model_path = model_save_type(model_reg, file_prefix,type='reg')
    clf_model_path = model_save_type(model_clf, file_prefix,type='clf')

    # 返回权重文件 + 模型文件路径
    return {
        'reg_weights': reg_filename,
        'clf_weights': clf_filename,
        'reg_model': reg_model_path,
        'clf_model': clf_model_path
    }

def model_save_type(base_model, file_prefix,type='reg'):
    # 模型训练完成后做持久化，模型保存为model模式，便于调用预测
    base_modelname = f'../models/{file_prefix}_model_{type}.json'
    base_model.save_model(base_modelname)

    # 模型保存为文本格式，便于分析、优化和提供可解释性
    base = base_model.get_booster()
    base.dump_model(f'../models/{file_prefix}_model_{type}_dump.txt')
    return base_modelname

# 模型效果评估
def metrics_sklearn(y_valid, y_pred_):
    """模型效果评估"""
    r2 = r2_score(y_valid, y_pred_)
    print('r2_score:{0}'.format(r2))

    mse = mean_squared_error(y_valid, y_pred_)
    print('mse:{0}'.format(mse))

def clf_model_save_load(model, x_transform):
    # 模型加载
    clf = xgb.XGBClassifier()
    booster = xgb.Booster()
    booster.load_model(model)
    clf._Booster = booster

    # 数据预测
    y_pred = [round(value) for value in clf.predict(x_transform)]
    y_pred_proba = clf.predict_proba(x_transform)
    print('y_pred：', y_pred)
    print('y_pred_proba：', y_pred_proba)

def reg_model_save_load(model, x_transform):
    # 模型加载
    reg = xgb.XGBRegressor()
    booster = xgb.Booster()
    booster.load_model(model)
    reg._Booster = booster

    # 数据预测
    y_pred = [round(value) for value in reg.predict(x_transform)]
    y_pred_proba = reg.predict_proba(x_transform)
    print('y_pred：', y_pred)
    print('y_pred_proba：', y_pred_proba)


def checking_model_data(input_file,model):
    # 预测 准备数据
    df2 = pd.read_excel(input_file)
    features = [
        '当日涨幅', '信号天数', '净额', '净流入', '当日资金流入', '是否领涨'
    ]

    df2['value'] = df2['最高价'].map({'是': 1, '否': 0})
    df2['是否领涨'] = df2['是否领涨'].map({'是': 1, '否': 0})
    y_test = df2[features]

    print(df2.head(10))
    # 加载模型
    model_reg = xgb.XGBRegressor()
    model_reg.load_model(model['reg_model'])

    model_clf = xgb.XGBClassifier()
    model_clf.load_model(model['clf_model'])

    # 预测
    y_pred_clf = model_clf.predict(y_test)
    y_pred_reg = model_reg.predict(y_test)



    # 预测是
    y_pred = model_clf.predict(y_test)
    # print(y_pred)


    # y2_clf = (df2['value'] > 0).astype(int)
    y2_clf = df2['value']
    print('value：', y2_clf)

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

    y2_reg = df2['次日最高涨幅']
    print('value：', y2_reg)
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

    # 构造结果 DataFrame
    result_df = df2.copy()
    result_df['实际标签_value'] = y2_clf
    result_df['分类预测_y_pred_clf'] = y_pred_clf
    result_df['回归预测_y_pred_reg'] = y_pred_reg

    # 保存为 Excel 文件  predictions_202507021753.xlsx
    # output_file = f'../data/checking_{pd.Timestamp.now().strftime("%Y%m%d%H%M")}.xlsx'
    file_name = os.path.basename(input_file)
    # output_file = f'../data/checking_{pd.Timestamp.now().strftime("%H%M")}_{file_name}'
    file_root, file_ext = os.path.splitext(file_name)  # 分离文件名和后缀
    output_file = f'../data/checking/{file_root}_{pd.Timestamp.now().strftime("%H%M")}{file_ext}'
    result_df.to_excel(output_file, index=False)

    print(f"验证结果已保存至: {output_file}")

def predictions_model_data_file(input_file,model,output_dir):
    print(input_file)
    # 预测 准备数据
    df_calculate = pd.read_excel(input_file)
    features = [
        '当日涨幅', '信号天数', '净额', '净流入', '当日资金流入', '是否领涨'
    ]

    df_calculate['value'] = df_calculate['最高价'].map({'是': 1, '否': 0})
    df_calculate['是否领涨'] = df_calculate['是否领涨'].map({'是': 1, '否': 0})
    y_test = df_calculate[features]

    # print(df_calculate.head(10))
    # 加载模型
    model_reg = xgb.XGBRegressor()
    model_reg.load_model(model['reg_model'])

    model_clf = xgb.XGBClassifier()
    model_clf.load_model(model['clf_model'])

    # 预测
    y_pred_clf = model_clf.predict(y_test)
    y_pred_reg = model_reg.predict(y_test)

    # print(y_pred_clf)
    # print(y_pred_reg)


    # 构造结果 DataFrame
    result_df = df_calculate.copy()
    result_df['分类预测_y_pred_clf'] = y_pred_clf
    result_df['回归预测_y_pred_reg'] = y_pred_reg

    # 保存为 Excel 文件
    file_name = os.path.basename(input_file)
    file_root, file_ext = os.path.splitext(file_name)  # 分离文件名和后缀
    if(output_dir == ''):
        output_dir = f'../data/predictions'

    output_file = f'{output_dir}/{file_root}_{pd.Timestamp.now().strftime("%H%M")}{file_ext}'
    result_df.to_excel(output_file, index=False)

    print(f"预测结果已保存至: {output_file}")
    return output_file

def predictions_model_data(data, model):
    # 将单行数据转换为DataFrame
    # df_calculate = pd.DataFrame([data])
    
    features = [
        '当日涨幅', '信号天数', '净额', '净流入', '当日资金流入', '是否领涨'
    ]

    # 判断输入类型并做相应处理
    if isinstance(data, dict):
        # 处理单个样本
        df_calculate = pd.DataFrame([data])
        single_sample = True
    elif isinstance(data, pd.DataFrame):
        # 处理DataFrame
        df_calculate = data.copy()
        single_sample = False
    else:
        raise ValueError("输入数据必须是字典或DataFrame")
    
    # 特征映射处理
    df_calculate['value'] = df_calculate['最高价'].map({'是': 1, '否': 0})
    df_calculate['是否领涨'] = df_calculate['是否领涨'].map({'是': 1, '否': 0})
    
    # 提取特征
    X_test = df_calculate[features]
    
    # 加载模型
    model_reg = xgb.XGBRegressor()
    model_reg.load_model(model['reg_model'])
    
    model_clf = xgb.XGBClassifier()
    model_clf.load_model(model['clf_model'])
    
    # 预测
    y_pred_clf = model_clf.predict(X_test)
    y_pred_reg = model_reg.predict(X_test)
    
    # 返回预测结果
    return {
        '分类预测': y_pred_clf[0],
        '回归预测': y_pred_reg[0]
    }

# 示例调用修改
if __name__ == "__main__":
    # 使用多个数据集训练并生成模型
    files= [
        "../alert/0630.xlsx",
        "../alert/0701.xlsx",
        "../alert/0702.xlsx",
        "../alert/0703.xlsx",
        "../alert/0704.xlsx",
        "../alert/0707.xlsx",
        "../alert/0708.xlsx",
     ]
    model = generate_model_data(files)
    print(model)
    # todo 模型调优还没有做
    
    # 预测文件数据
    predictions_model_data_file("../alert/0709.xlsx",model)
    
    # 新增：单行数据预测示例
    print("\n===== 开始单行数据测试 =====")
    # 示例数据（包含所有必需特征）
    example_data = {
        '最高价': '是',      # 将被映射为1
        '是否领涨': '是',    # 将被映射为1
        '当日涨幅': 0.05,    # 5%涨幅
        '信号天数': 3,       # 连续3天出现信号
        '净额': 1000000,    # 净额100万
        '净流入': 500000,    # 净流入50万
        '当日资金流入': 2000000  # 当日资金流入200万
    }
    
    print("测试数据:", example_data)
    result = predictions_model_data(example_data, model)
    print("\n预测结果:")
    print(f"分类预测: {result['分类预测']}")
    print(f"回归预测(次日最高涨幅): {result['回归预测']:.4f}")
    print("===== 测试结束 =====")
