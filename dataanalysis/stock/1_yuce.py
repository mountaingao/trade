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
from sklearn.metrics import classification_report,r2_score,mean_squared_error
from sklearn.metrics import accuracy_score, precision_score



df = pd.read_excel("../data/0604.xlsx")


# 选择特征（技术指标 + 评分指标） consecutive_upper_days 连续天数
features = [
    'sma_up', 'sma_down', 'macd', 'is_up', 'consecutive_upper_days',
    'upper_days_counts', 'ma_amount_days_ratio_3', 'ma_amount_days_ratio_5','ma_amount_days_ratio_8',"ma_amount_days_ratio_11",
    'total_score','amount', 'free_amount', 'increase', 'amplitude', 'jgcyd', 'lspf', 'focus',
    'last_desire_daily', 'high_rating'
]

features = [
    'sma_up', 'sma_down', 'macd', 'is_up',
    'upper_days_counts', 'ma_amount_days_ratio_3', 'ma_amount_days_ratio_5','ma_amount_days_ratio_8',"ma_amount_days_ratio_11",
    'total_score','amount', 'free_amount', 'increase', 'amplitude', 'jgcyd', 'lspf', 'focus',
    'last_desire_daily'
]

X_train = df[features]
y_reg = df['1_day_close']

y_clf = (df['1_day_close'] > 0).astype(int)

# 训练回归模型预测涨跌幅
model_reg = xgb.XGBRegressor()
model_reg.fit(X_train, y_reg)

# 训练分类模型预测涨跌方向
model_clf = xgb.XGBClassifier()
model_clf.fit(X_train, y_clf)

# 获取特征重要性
feat_importance_reg = model_reg.feature_importances_
feat_importance_clf = model_clf.feature_importances_


# 3. 标准化为权重（总和为1）
feature_weights_clf = feat_importance_clf / feat_importance_clf.sum()

# 3. 标准化为权重（总和为1）
feature_weights_reg = feat_importance_reg / feat_importance_reg.sum()

print( '特征权重：', feature_weights_reg)
print( '特征权重：', feature_weights_clf)
# 构造 DataFrame
weights_reg = pd.DataFrame({
    'Feature': features,
    'Weight': feature_weights_reg
})

# 保存为 CSV 文件
weights_reg.to_csv('../data/feature_weights_reg.csv', index=False)
print("特征权重已保存为 CSV 文件")

# 构造 DataFrame
weights_clf = pd.DataFrame({
    'Feature': features,
    'Weight': feature_weights_clf
})

# 保存为 CSV 文件
weights_clf.to_csv('../data/feature_weights_clf.csv', index=False)
print("特征权重已保存为 CSV 文件")


# 预测
# df2 = pd.read_excel("../source/0516-1.xlsx")
df2 = pd.read_excel("../data/0610.xlsx")
y_test = df2[features]
# y_pred_proba = model_clf.predict_proba(X_test)[:, 1]  # 获取正类的概率

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


def metrics_sklearn(y_valid, y_pred_):
    """模型效果评估"""
    r2 = r2_score(y_valid, y_pred_)
    print('r2_score:{0}'.format(r2))

    mse = mean_squared_error(y_valid, y_pred_)
    print('mse:{0}'.format(mse))


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

# 默认阈值（0.5）
y_pred_default = model_clf.predict(X_test)

# 调整阈值（如0.6）
custom_threshold = 0.6
y_pred_custom = (y_pred_proba >= custom_threshold).astype(int)

# 评估
print("Default Threshold (0.5):")
print(classification_report(y_test, y_pred_default))

print(f"Custom Threshold ({custom_threshold}):")
print(classification_report(y_test, y_pred_custom))


# 可视化
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.barh(features, feat_importance_reg)
plt.title('Regression Feature Importance')
plt.subplot(1, 2, 2)
plt.barh(features, feat_importance_clf)
plt.title('Classification Feature Importance')
plt.tight_layout()
plt.show()

# 二、加权评分机制设计
# 1. 动态权重分配公式
#
# 特征得分 = 0.4×回归重要性 + 0.4×分类重要性 + 0.2×经济逻辑系数
#
# 经济逻辑系数：
# 1.0 - 核心量价指标（成交量、波动率）
# 0.8 - 技术指标（RSI、MACD）
# 0.6 - 市场情绪指标
# 0.4 - 基本面指标

# 将权重features 和 feature_weights_reg 保存为文件，作为预测模型的输入





def calculate_stock_score(features):
    # 特征加权
    weighted_features = features * feature_weights

    # 计算维度得分
    technical_score = np.sum(weighted_features[['RSI', 'MACD', 'Volatility']])
    volume_score = np.sum(weighted_features[['Volume', 'OBV']])
    sentiment_score = np.sum(weighted_features[['Sentiment', 'News_Impact']])

    # 综合评分
    total_score = (0.4 * technical_score +
                   0.3 * volume_score +
                   0.2 * sentiment_score +
                   0.1 * momentum_score)

    # 归一化到0-100分
    return min(max(total_score * 10, 0), 100)


# 集成预测模型
class StockPredictionSystem:
    def __init__(self):
        self.model_reg = xgb.XGBRegressor(objective='reg:squarederror')
        self.model_clf = xgb.XGBClassifier(objective='binary:logistic')
        self.model_interval = xgb.XGBRegressor(objective='reg:quantileerror')

    def train(self, X, y_reg, y_clf):
        # 训练核心模型
        self.model_reg.fit(X, y_reg)
        self.model_clf.fit(X, y_clf)

        # 训练区间预测模型（残差学习）
        preds = self.model_reg.predict(X)
        residuals = y_reg - preds
        self.model_interval.fit(X, residuals)

    def predict(self, X):
        # 基础预测
        prob_up = self.model_clf.predict_proba(X)[:, 1]
        pred_change = self.model_reg.predict(X)

        # 区间预测
        residual_pred = self.model_interval.predict(X)
        lower_bound = pred_change - 1.96 * np.abs(residual_pred)
        upper_bound = pred_change + 1.96 * np.abs(residual_pred)

        return prob_up, pred_change, lower_bound, upper_bound