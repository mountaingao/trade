"""
    xgboost模型调参、训练、保存、预测
    官网信息辅助理解：
    xgboost官网参数(default) https://xgboost.readthedocs.io/en/latest/parameter.html#general-parameters
    sklearn评估指标(Metrics) https://scikit-learn.org/stable/modules/model_evaluation.html
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score,classification_report
import xgboost as xgb
from sklearn.metrics import mean_squared_error
from xgboost import XGBClassifier, plot_importance
import warnings

warnings.filterwarnings('ignore')
plt.rcParams['font.family'] = ['sans-serif']
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
# 1. 数据加载
df = pd.read_excel("result/tdx_block_pre_data_401-430.xlsx")



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

# XGBoost训练模型的核心步骤是使用XGBClassifier（用于分类任务）或者XGBRegressor（用于回归任务）进行模型训练。我们在这里使用分类任务作为示例。
# 创建XGBoost分类器
model = xgb.XGBClassifier(objective='binary:logistic', use_label_encoder=False)
# 训练模型
model.fit(X_train, y_train)

# 在测试集上进行预测
y_pred = model.predict(X_test)

# 输出准确率和分类报告
print('准确率:', accuracy_score(y_test, y_pred))
print('分类报告:\n', classification_report(y_test, y_pred))



# # 训练XGBoost模型-回归任务
# model_xgb = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=100)
# model_xgb.fit(X_train, y_train)

# # 预测和评估
# y_pred_xgb = model_xgb.predict(X_test)
# mse_xgb = mean_squared_error(y_test, y_pred_xgb)
# print(f'XGBoost MSE: {mse_xgb}')


# 三、XGBoost模型调优
# 3.1 常用调参策略
# XGBoost提供了丰富的参数来优化模型性能，常见的调参策略包括：
#
# 学习率（learning_rate 或 eta）：控制每棵树对最终模型的贡献。较小的学习率可以提高模型的泛化能力，但需要更多的树。
# 树的数量（n_estimators）：控制模型的复杂度，通常与学习率配合使用。较多的树可以提高准确率，但也可能导致过拟合。
# 最大深度（max_depth）：控制树的最大深度。较大的深度可以捕捉更多的特征交互关系，但容易导致过拟合。
# 最小样本分裂数（min_child_weight）：控制一个叶子节点的最小样本权重。较大的值会使得模型更保守，减少过拟合。
# 子样本比例（subsample）：控制每棵树训练时使用的数据比例。可以用于防止过拟合。
# 3.2 使用交叉验证调参
# XGBoost支持使用交叉验证来选择最佳超参数，常用的调参方法是通过GridSearchCV或RandomizedSearchCV进行参数搜索。
from sklearn.model_selection import GridSearchCV

# 参数网格
param_grid = {
    'max_depth': [3, 4, 5],
    'learning_rate': [0.1, 0.01],
    'n_estimators': [100, 200, 300],
    'subsample': [0.8, 0.9, 1.0]
}

# 网格搜索
grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=3, n_jobs=-1, verbose=2)
grid_search.fit(X_train, y_train)

# 输出最佳参数
print("最佳参数: ", grid_search.best_params_)


# 3.3 早停机制（Early Stopping）
# XGBoost还提供了早停机制，允许在验证集上的表现不再提升时提前终止训练，从而避免过拟合。
# 设置早停参数
model = xgb.XGBClassifier(objective='binary:logistic', use_label_encoder=False, n_estimators=1000)

# 训练模型时使用早停
model.fit(X_train, y_train, eval_metric="logloss", eval_set=[(X_test, y_test)], early_stopping_rounds=10)


#
# 四、XGBoost的应用实例
# 4.1 使用XGBoost进行房价预测
# 在这个实例中，我们使用XGBoost进行房价预测。假设我们有一个包含房屋特征（如面积、卧室数量、地理位置等）和房价标签的数据集。
import xgboost as xgb
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# 加载数据
data = pd.read_csv('house_prices.csv')

# 特征和标签
X = data.drop(columns=['price'])
y = data['price']

# 数据集划分
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 创建XGBoost回归器
model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=500)

# 训练模型


model.fit(X_train, y_train)

# 预测与评估
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print('均方误差:', mse)

#
# 五、总结
# XGBoost是一种强大且高效的梯度提升算法，它在许多机器学习竞赛中取得了显著的成功。通过引入正则化、二阶导数信息、并行计算等技术，XGBoost大大提高了模型的训练速度和预测精度。
