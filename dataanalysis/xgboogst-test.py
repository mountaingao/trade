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
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score
import xgboost as xgb
from xgboost import XGBClassifier, plot_importance
import warnings

warnings.filterwarnings('ignore')
plt.rcParams['font.family'] = ['sans-serif']
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
# 1. 数据加载
df = pd.read_excel("result/tdx_block_pre_data_401-403.xlsx")

# 选择特征（技术指标 + 评分指标）
features = [
    'sma_up', 'sma_down', 'macd', 'is_up', 'consecutive_upper_days',
    'upper_days_counts', 'ma_amount_days_ratio_3', 'ma_amount_days_ratio_5',
    'ma_amount_days_ratio_8', 'ma_amount_days_ratio_11',
    'amount', 'free_amount', 'increase', 'amplitude',
    # 'jgcyd', 'lspf', 'focus',
    'last_desire_daily', 'high_rating'
]

# 目标变量：1_day_close是否为正（1=正，0=非正）
df['target'] = (df['1_day_close'] > 0).astype(int)