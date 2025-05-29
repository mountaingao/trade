
# Python自动化炒股：利用CatBoost和XGBoost进行股票市场预测的详细指南
# https://blog.csdn.net/yunce_touzi/article/details/146420257

# 安装：pip install pandas numpy matplotlib seaborn catboost xgboost
# pip install lightgbm
# LightGBM（Light Gradient Boosting Machine）是由微软（Microsoft）研发的一种基于梯度提升算法（Gradient Boosting）的机器学习框架，它在处理大规模数据时非常高效。作为一种集成学习方法，LightGBM通过构建多个弱学习器来提高模型的整体性能，尤其擅长处理高维稀疏数据、类别特征和大规模数据集。
# ————————————————
#
# 版权声明：本文为博主原创文章，遵循 CC 4.0 BY-SA 版权协议，转载请附上原文出处链接和本声明。
#
# 原文链接：https://blog.csdn.net/liu1983robin/article/details/144942761

# 这个例子使用了 xgboost 和 catboost 、lightgbm 三种方法来训练模型

import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import pandas as pd
import matplotlib
matplotlib.use('Agg')

# 1. 数据加载
df = pd.read_excel("../result/tdx_block_pre_data_401-403.xlsx")
# 数据信息
df.info()
print(df.head())
# 数据计算统计 ，需要列出所有字段
df.describe()
print(df.describe().round(2))
df.plot(figsize=(12, 6))
exit()

# 选择特征和标签（技术指标 + 评分指标） # 选择特征和标签
features = [
    'sma_up', 'sma_down', 'macd', 'is_up', 'consecutive_upper_days',
    'upper_days_counts', 'ma_amount_days_ratio_3', 'ma_amount_days_ratio_5',
    'ma_amount_days_ratio_8', 'ma_amount_days_ratio_11',
    'amount', 'free_amount', 'increase', 'amplitude',
    # 'jgcyd', 'lspf', 'focus',
    'last_desire_daily', 'high_rating'
]

X = df[features]

y = df['1_day_close']


# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 训练XGBoost模型
model_xgb = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=100)
model_xgb.fit(X_train, y_train)

# 预测和评估
y_pred_xgb = model_xgb.predict(X_test)
mse_xgb = mean_squared_error(y_test, y_pred_xgb)
print(f'XGBoost MSE: {mse_xgb}')

from catboost import CatBoostRegressor

# 训练CatBoost模型
model_cb = CatBoostRegressor(iterations=100, learning_rate=0.1, depth=5, verbose=200)
model_cb.fit(X_train, y_train)

# 预测和评估
y_pred_cb = model_cb.predict(X_test)
mse_cb = mean_squared_error(y_test, y_pred_cb)
print(f'CatBoost MSE: {mse_cb}')


#训练 lightgbm 模型
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
# 加载鸢尾花数据集
from sklearn.datasets import load_iris
data = load_iris()
X = data.data
y = data.target
# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
# 训练LightGBM模型
# 构建LightGBM数据格式
train_data = lgb.Dataset(X_train, label=y_train)
test_data = lgb.Dataset(X_test, label=y_test, reference=train_data)

# 设置LightGBM参数
params = {
    'objective': 'multiclass',     # 多分类任务
    'num_class': 3,                # 类别数
    'metric': 'multi_logloss',     # 损失函数
    'boosting_type': 'gbdt',       # 基于梯度提升树
    'num_leaves': 31,              # 最大叶子数
    'learning_rate': 0.05,         # 学习率
    'feature_fraction': 0.9,        # 特征选择比例
    'force_col_wise': True,         # 强制使用列式存储
    'early_stopping_rounds': 10     # 将 early_stopping_rounds 移到这里
}

# 训练模型
num_round = 100
bst = lgb.train(params, train_data, num_round, valid_sets=[test_data])

# 使用训练好的模型进行预测
y_pred = bst.predict(X_test, num_iteration=bst.best_iteration)

# 由于预测结果为每个类别的概率分布，我们选择概率最大的一类作为预测结果
y_pred_max = np.argmax(y_pred, axis=1)

# 计算准确率
accuracy = accuracy_score(y_test, y_pred_max)
print(f'准确率: {accuracy}')
