
import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, f1_score

# 加载数据
# df = pd.read_excel("result/tdx_block_pre_data_0508.xlsx")
# Best parameters: {'colsample_bytree': 0.6, 'gamma': 0.2, 'learning_rate': 0.3, 'max_depth': 7, 'n_estimators': 50, 'reg_alpha': 0, 'reg_lambda': 0, 'subsample': 0.6}
# Best F1 score: 0.22999999999999998
df = pd.read_excel("source/0508-1.xlsx")
# Best parameters: {'colsample_bytree': 0.6, 'gamma': 0, 'learning_rate': 0.1, 'max_depth': 3, 'n_estimators': 50, 'reg_alpha': 1, 'reg_lambda': 0, 'subsample': 1.0}
# Best F1 score: 0.5142857142857142

# 选择特征（技术指标 + 评分指标）
features = [
    'sma_up', 'sma_down', 'macd', 'is_up', 'consecutive_upper_days',
    'upper_count_in_days', 'ma_amount_3_days_ratio', 'ma_amount_5_days_ratio',
    'ma_amount_8_days_ratio', 'ma_amount_11_days_ratio', 'total_score',
    'amount', 'free_amount', 'increase', 'amplitude', 'jgcyd', 'lspf', 'focus',
    'last_desire_daily', 'high_rating'
]

# 目标变量：1_day_close是否为正（1=正，0=非正）
df['target'] = (df['1_day_close'] > 0).astype(int)

X = df[features]
y = df['target']

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 初始化模型
model = XGBClassifier()

# 参数网格
param_grid = {
    'max_depth': [3, 5, 7],          # 树深度
    'learning_rate': [0.01, 0.1, 0.3],  # 学习率
    'n_estimators': [50, 100, 200],   # 树数量
    'subsample': [0.6, 0.8, 1.0],     # 样本采样比例
    'colsample_bytree': [0.6, 0.8, 1.0],  # 特征采样比例
    'gamma': [0, 0.1, 0.2],           # 最小分裂损失
    'reg_alpha': [0, 0.1, 1],         # L1正则化
    'reg_lambda': [0, 0.1, 1]         # L2正则化
}

# 网格搜索（使用F1分数评估）
grid = GridSearchCV(
    estimator=model,
    param_grid=param_grid,
    scoring='f1',  # 关注正类的F1分数
    cv=5,
    n_jobs=-1
)

grid.fit(X_train, y_train)

# 输出最佳参数
print("Best parameters:", grid.best_params_)
print("Best F1 score:", grid.best_score_)

