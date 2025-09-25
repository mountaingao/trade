import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
from data_prepare import get_dir_files_data


'''
# /*
#  * 逻辑回归和随机森林模型
我们有两个数据集：shuju.xlsx和shuju1.xlsx。为了构建一个更稳健的模型，我们可以将两个数据集合并，从而增加样本量。
但是请注意，两个数据集的字段不完全相同。shuju.xlsx有更多的技术指标（如band_width, min_value, max_value, is_boll_low等），而shuju1.xlsx则没有。
不过，两个数据集都包含以下共同字段：当日涨幅、量比、总金额、细分行业、信号天数、Q、Q_1、Q3、净额、净流入、当日资金流入、value。
由于我们重点分析的是这些共同字段，因此可以合并这两个数据集。

步骤：

读取两个文件，提取共同字段。

合并两个数据集。

数据预处理：处理缺失值，对分类变量（如细分行业）进行编码。

构建逻辑回归和随机森林模型。

评估模型并分析特征重要性。

由于细分行业类别较多，且可能对预测有帮助，我们将对其进行独热编码（One-Hot Encoding）。

注意：两个数据集的日期不同，但我们是基于当日的特征预测次日的涨幅（value），所以日期本身不需要作为特征。

开始操作：

我将使用逻辑回归和随机森林模型对数据进行建模，量化各特征的重要性，并进行组合预测。以下是完整的分析过程：
#  */
'''

# 选择共同的特征列
features = ['当日涨幅', '量比', '总金额', '信号天数', 'Q', 'Q_1', 'Q3', '净额', '净流入', '当日资金流入']
target = 'value'

# 假设df1和df2分别是两个数据集

dates= get_dir_files_data("../data/predictions/1000/",start_md="0801",end_mmdd="0916")
print(len(dates))
# print( dates)
# exit(0)
# 合并数据集
df_combined = dates.copy()

# 检查缺失值
print("缺失值情况:")
print(df_combined.isnull().sum())

# 数据基本信息
print(f"总样本数: {len(df_combined)}")
print(f"正样本数(value=1): {df_combined[target].sum()}")
print(f"负样本数(value=0): {len(df_combined) - df_combined[target].sum()}")

# 处理可能的异常值
df_clean = df_combined.copy()

# 对数值型特征进行标准化
scaler = StandardScaler()
X = df_clean[features]
y = df_clean[target]

# 分割数据集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

# 标准化特征
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 逻辑回归模型
lr_model = LogisticRegression(random_state=42, max_iter=1000)
lr_model.fit(X_train_scaled, y_train)

# 预测
y_pred_lr = lr_model.predict(X_test_scaled)
y_pred_proba_lr = lr_model.predict_proba(X_test_scaled)[:, 1]

# 评估
lr_accuracy = accuracy_score(y_test, y_pred_lr)
print("逻辑回归准确率:", lr_accuracy)
print("\n逻辑回归分类报告:")
print(classification_report(y_test, y_pred_lr))

# 特征重要性（系数绝对值）
lr_importance = pd.DataFrame({
    'feature': features,
    'importance': np.abs(lr_model.coef_[0])
}).sort_values('importance', ascending=False)

print("\n逻辑回归特征重要性:")
print(lr_importance)

# 随机森林模型
rf_model = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=5)
rf_model.fit(X_train, y_train)

# 预测
y_pred_rf = rf_model.predict(X_test)
y_pred_proba_rf = rf_model.predict_proba(X_test)[:, 1]

# 评估
rf_accuracy = accuracy_score(y_test, y_pred_rf)
print("随机森林准确率:", rf_accuracy)
print("\n随机森林分类报告:")
print(classification_report(y_test, y_pred_rf))

# 特征重要性
rf_importance = pd.DataFrame({
    'feature': features,
    'importance': rf_model.feature_importances_
}).sort_values('importance', ascending=False)

print("\n随机森林特征重要性:")
print(rf_importance)


# 绘制特征重要性对比图
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

# 逻辑回归特征重要性
sns.barplot(data=lr_importance, x='importance', y='feature', ax=ax1)
ax1.set_title('逻辑回归特征重要性（系数绝对值）')
ax1.set_xlabel('重要性')

# 随机森林特征重要性
sns.barplot(data=rf_importance, x='importance', y='feature', ax=ax2)
ax2.set_title('随机森林特征重要性')
ax2.set_xlabel('重要性')

plt.tight_layout()
plt.show()

# 组合预测（加权平均）
def ensemble_predict(lr_proba, rf_proba, lr_weight=0.4, rf_weight=0.6):
    """组合预测函数"""
    combined_proba = lr_weight * lr_proba + rf_weight * rf_proba
    return (combined_proba > 0.5).astype(int), combined_proba

# 组合预测
y_pred_ensemble, y_proba_ensemble = ensemble_predict(y_pred_proba_lr, y_pred_proba_rf)

# 评估组合模型
ensemble_accuracy = accuracy_score(y_test, y_pred_ensemble)
print("组合模型准确率:", ensemble_accuracy)
print("\n组合模型分类报告:")
print(classification_report(y_test, y_pred_ensemble))


# 简易决策规则
def trading_signal(row):
    """生成交易信号"""
    if row['量比'] > 5 and row['Q'] > 3 and row['总金额'] > 80000:
        if row['净流入'] > 0 or row['当日涨幅'] > 2:
            return "强烈看多"
        else:
            return "谨慎看多"
    elif row['量比'] < 2 or row['Q'] < 1:
        return "观望"
    else:
        return "中性"

df_clean['trading_signal'] = df_clean.apply(trading_signal, axis=1)