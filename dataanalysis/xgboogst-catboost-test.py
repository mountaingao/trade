"""
利用CatBoost和XGBoost进行股票市场预测的详细指南
    https://blog.csdn.net/yunce_touzi/article/details/146420257

    !pip install pandas numpy matplotlib seaborn catboost xgboost
    !pip install seaborn catboost

"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score,classification_report
import xgboost as xgb
from sklearn.metrics import mean_squared_error
import matplotlib
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
# 训练XGBoost模型 回归方法
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

# 预测和评估.
# 它的作用是计算均方误差，也就是MSE。  那参数部分呢，y_test应该是测试集的真实标签，而y_pred_cb是模型对测试集的预测结果。函数会计算这两个数组之间的均方误差。
# 均方误差的计算方法应该是取预测值和真实值差的平方的平均数。这样得到的值越小，说明模型预测得越准确。
# 这段代码使用scikit-learn的mean_squared_error函数计算回归模型的预测误差。具体功能：
# 输入y_test为测试集真实值
# y_pred_cb为模型预测值
# 函数通过计算两数组元素间平方差的均值，返回均方误差(MSE)指标 该值越小表示模型预测越接近真实值，常用于评估回归模型的准确性。
y_pred_cb = model_cb.predict(X_test)
mse_cb = mean_squared_error(y_test, y_pred_cb)
print(f'CatBoost MSE: {mse_cb}')

import matplotlib.pyplot as plt
# 添加JupyterLab内嵌显示魔法命令 在JupyterLab中显示
# %matplotlib inline

# 绘制实际值与预测值
plt.figure(figsize=(10, 6))
plt.plot(y_test.index, y_test, label='Actual')
plt.plot(y_test.index, y_pred_xgb, label='XGBoost')
plt.plot(y_test.index, y_pred_cb, label='CatBoost')
plt.legend()
plt.title('Stock Price Prediction')
plt.show()
