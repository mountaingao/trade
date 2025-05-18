
import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, f1_score
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import xgboost as xgb  # 习惯缩写为 xgb



# 加载数据
# df = pd.read_excel("source/0508-1.xlsx")
df = pd.read_excel("source/0509-1.xlsx")

# 选择特征（技术指标 + 评分指标）
features = [
    'sma_up', 'sma_down', 'macd', 'is_up', 'consecutive_upper_days',
    'upper_days_counts', 'ma_amount_days_ratio_3', 'ma_amount_days_ratio_5',
    'ma_amount_days_ratio_8', 'ma_amount_days_ratio_11',
    'amount', 'free_amount', 'increase', 'amplitude', 'jgcyd', 'lspf', 'focus',
    'last_desire_daily', 'high_rating'
]
# date	code	close	amount	zhang	zhen	sma_up	sma_down	macd	is_up	consecutive_upper_days	upper_days_counts	ma_amount_days_ratio_3
# ma_amount_days_ratio_5	ma_amount_days_ratio_8	ma_amount_days_ratio_11	total_score	amount_score	free_amount	free_amount_score	increase	increase_score	amplitude	amplitude_score	jgcyd	jgcyd_score	lspf	lspf_score	focus	focus_score	last_desire_daily	desire_daily_score	high_rating	0_open	0_high	0_low	0_close	0_amount	0_day_open	0_day_max	0_day_min	0_day_close	1_open	1_high	1_low	1_close	1_amount	1_day_open	1_day_max	1_day_min	1_day_close	2_open	2_high	2_low	2_close	2_amount	2_day_open	2_day_max	2_day_min	2_day_close	3_open	3_high	3_low	3_close	3_amount	3_day_open	3_day_max	3_day_min	3_day_close

# 目标变量：1_day_close是否为正（1=正，0=非正）
df['target'] = (df['1_day_close'] > 0).astype(int)

X = df[features]
y = df['target']

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Best parameters: {'colsample_bytree': 0.6, 'gamma': 0, 'learning_rate': 0.1, 'max_depth': 3, 'n_estimators': 50, 'reg_alpha': 1, 'reg_lambda': 0, 'subsample': 1.0}

# 使用最佳参数训练模型
best_model = XGBClassifier(
    colsample_bytree=0.6,
    gamma=0,
    learning_rate=0.1,
    max_depth=3,
    n_estimators=50,
    reg_alpha=1,
    reg_lambda=0,
    subsample=1,
    objective='binary:logistic',
    eval_metric='logloss'
)

best_model.fit(X_train, y_train)

# 预测概率（正类的概率）
y_pred_proba = best_model.predict_proba(X_test)[:, 1]

# 调整阈值（默认0.5，可优化）
from sklearn.metrics import precision_recall_curve
precision, recall, thresholds = precision_recall_curve(y_test, y_pred_proba)
f1_scores = 2 * (precision * recall) / (precision + recall)
best_threshold = thresholds[np.argmax(f1_scores)]
print("最佳阈值:", best_threshold)  # 例如0.6

# 应用最优阈值
y_pred = (y_pred_proba >= best_threshold).astype(int)

# 评估
print("准确性:", accuracy_score(y_test, y_pred))
print("F1 Score:", f1_score(y_test, y_pred))


# 绘制特征重要性
xgb.plot_importance(best_model)
# plt.show()
plt.savefig('feature_importance.png')  # 添加保存图片功能替代显示
plt.close()  # 清理图形资源
