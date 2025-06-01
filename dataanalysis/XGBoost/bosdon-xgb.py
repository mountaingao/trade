"""第三方库导入"""
from xgboost import XGBRegressor
from sklearn import datasets
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import r2_score, mean_squared_error
import xgboost as xgb
import pandas as pd

import csv
from os.path import dirname, expanduser, isdir, join, splitext
import numpy as np


"""波士顿房价数据集导入"""
# 修改前（错误的分隔符和列名处理）
# data = pd.read_csv('boston_house_prices.csv', sep=r"\s+", skiprows=1)

# 修改后（正确解析CSV格式）
data = pd.read_csv('boston_house_prices.csv', sep=',', skiprows=1,header=0)
data.rename(columns={'MEDV': 'MEDV'}, inplace=True)  # 确保列名一致性

# print(data.tail(5))
# print(data["MEDV"])
# - CRIM     per capita crime rate by town\n      # 按城镇划分的犯罪率
# - ZN       proportion of residential land zoned for lots over 25,000 sq.ft.\n  # 划分为25000平方英尺以上地块的住宅用地比例
# - INDUS    proportion of non-retail business acres per town\n     # 每每个城镇的非零售商业用地比例
# - CHAS     Charles River dummy variable (= 1 if tract bounds river; 0 otherwise)\n        # 靠近查尔斯河，则为1；否则为0
# - NOX      nitric oxides concentration (parts per 10 million)\n      # 一氧化氮浓度（百万分之一）
# - RM       average number of rooms per dwelling\n  # 每个住宅的平均房间数
# - AGE      proportion of owner-occupied units built prior to 1940\n     # 1940年之前建造的自住单位比例
# - DIS      weighted distances to five Boston employment centres\n     # 到波士顿五个就业中心的加权距离
# - RAD      index of accessibility to radial highways\n    # 辐射状公路可达性指数
# - TAX      full-value property-tax rate per $10,000\n   # 每10000美元的全额财产税税率
# - PTRATIO  pupil-teacher ratio by town\n    # 按城镇划分的师生比例
# - B        1000(Bk - 0.63)^2 where Bk is the proportion of blacks by town\n  # 1000（Bk-0.63）^2其中Bk是按城镇划分的黑人比例
# - LSTAT    % lower status of the population\n    # 人口密度
# - MEDV     Median value of owner-occupied homes in $1000's\n # 住房屋的中值（单位：1000美元）

"""训练集 验证集构建"""
X_train, X_test, y_train, y_test = train_test_split(
    data.drop('MEDV', axis=1),  # 特征矩阵
    data['MEDV'],  # 目标变量
    test_size=0.2,
    random_state=42
)

"""模型训练"""
model = XGBRegressor()

# model = XGBRegressor(booster='gbtree',  # gblinear
#                      n_estimators=150,  # 迭代次数
#                      learning_rate=0.01,  # 步长
#                      max_depth=10,  # 树的最大深度
#                      min_child_weight=0.5,  # 决定最小叶子节点样本权重和
#                      seed=123,  # 指定随机种子，为了复现结果
#                      )

model.fit(X_train, y_train, verbose=True)

y_pred = model.predict(X_test)
# print(y_pred)

for m, n in zip(y_pred, y_test):
    if m / n - 1 > 0.2:
        print('预测值为{0}, 真是结果为{1}, 预测结果偏差大于20%'.format(m, n))


def metrics_sklearn(y_valid, y_pred_):
    """模型效果评估"""
    r2 = r2_score(y_valid, y_pred_)
    print('r2_score:{0}'.format(r2))

    mse = mean_squared_error(y_valid, y_pred_)
    print('mse:{0}'.format(mse))


"""模型效果评估"""
metrics_sklearn(y_test, y_pred)

# 2.1调参过程
# 第1次调参，选择'booster': ['gbtree', 'gblinear']和'n_estimators': [75, 125, 200, 250, 300]，params如下：
# 第2次调参，选择'n_estimators': [75, 125, 200, 250, 300]和'learning_rate': [0.01, 0.03, 0.05, 0.1]，params如下：


def adj_params():
    """模型调参"""
    params = {
        # 'booster': ['gbtree', 'gblinear'],
        # 'n_estimators': [20, 50, 100, 150, 200],
        'n_estimators': [75, 125, 200, 250, 300],
        'learning_rate': [0.01, 0.03, 0.05, 0.1],
        # 'max_depth': [5, 8, 10, 12]
    }

    # model_adj = XGBRegressor()

    other_params = {'subsample': 0.8, 'colsample_bytree': 0.8, 'seed': 123}
    model_adj = XGBRegressor(**other_params)

    # sklearn提供的调参工具，训练集k折交叉验证(消除数据切分产生数据分布不均匀的影响)
    optimized_param = GridSearchCV(estimator=model_adj, param_grid=params, scoring='r2', cv=5, verbose=1)
    # 模型训练
    optimized_param.fit(X_train, y_train)

    # 对应参数的k折交叉验证平均得分
    means = optimized_param.cv_results_['mean_test_score']
    params = optimized_param.cv_results_['params']
    for mean, param in zip(means, params):
        print("mean_score: %f,  params: %r" % (mean, param))
    # 最佳模型参数
    print('参数的最佳取值：{0}'.format(optimized_param.best_params_))
    # 最佳参数模型得分
    print('最佳模型得分:{0}'.format(optimized_param.best_score_))


adj_params()

model = XGBRegressor(booster='gbtree',  # gblinear
                     n_estimators=300,  # 迭代次数
                     learning_rate=0.03,  # 步长
                     # max_depth=10,  # 树的最大深度
                     # min_child_weight=0.5,  # 决定最小叶子节点样本权重和
                     seed=123,  # 指定随机种子，为了复现结果
                     )

model.fit(X_train, y_train, verbose=True)

"""模型保存"""
model.save_model('xgb_regressor_boston.json')  # 使用.json扩展名显式指定格式

"""模型加载"""
clf = XGBRegressor()
clf.load_model('xgb_regressor_boston.json')  # 保持加载文件名一致

"""模型参数打印"""
bst = xgb.Booster(model_file='xgb_regressor_boston.json')  # 同步修改扩展名

# print(bst.attributes())
print('模型参数值-开始'.center(20, '='))
for attr_name, attr_value in bst.attributes().items():
    # scikit_learn 的参数逐一解析
    if attr_name == 'scikit_learn':
        import json

        dict_attr = json.loads(attr_value)
        # 打印 模型 scikit_learn 参数
        for sl_name, sl_value in dict_attr.items():
            if sl_value is not None:
                print(f"{sl_name}:{sl_value}")
    else:
        print(f"{attr_name}:{attr_value}")
print('模型参数值-结束'.center(20, '='))

"""预测验证数据"""
y_pred = clf.predict(X_test)

"""模型效果评估"""
metrics_sklearn(y_test, y_pred)

# 内容解释：
#
# 【注释1】R-squared（R2）分数是回归模型性能的一种常见评估指标。它测量模型对观测数据的拟合程度。该分数介于0和1之间，越接近1表示模型对数据的拟合越好。具体来说，R2分数是观测数据和回归模型之间差异的比率。这个比率由1减去误差平方和(SSE)和总偏差平方和(TSS)之比得到。
#
# 计算公式为：
#
# R2 = 1 - (SSE / TSS)
#
# 其中，SSE(sum of squared errors)是模型预测值与实际观测值之间差异的平方和，TSS(total sum of squares)是所有观测值与其均值差异的平方和。
#
# 【注释2】SSE和TSS是回归分析中常用的两个指标，分别代表回归模型的误差平方和和总偏差平方和。
# SSE（Sum of Squared Errors）是指在回归分析中，对于给定的自变量，在模型中计算出的因变量值与实际观察值之间的误差，即模型拟合的不准确程度。SSE等于所有误差的平方和，可以通过对每个数据点的误差（预测值与实际值之差）的平方求和得到。SSE越小，代表回归模型与实际观察值的拟合程度越好。
# TSS（Total Sum of Squares）是指将每个数据点的实际观察值和所有观察值的平均值之差的平方求和，这个值代表了数据的总方差，即数据中每个点偏离数据的平均值的程度。TSS用于评估模型的预测能力，因为它反映了实际观察值的变化范围。TSS越小，代表数据相对于它的平均值离散程度越小。
#
# 计算公式如下：
#
# SSE = sum((y_true - y_pred) ** 2)
#
# TSS = sum((y_true - np.mean(y_true)) ** 2)
#
# 其中，y_true为真实观察值，y_pred为模型预测值，np.mean(y_true)为真实观察值的均值。
#
# 原文链接：https://blog.csdn.net/LMTX069/article/details/131314600
