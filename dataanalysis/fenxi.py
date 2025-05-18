import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report


# 使用逻辑回归模型分析数据  pip install scikit-learn 机器学习
# 读取数据
data = pd.read_csv("result/result.txt", sep="\t")

# 筛选 is_up=1 的数据
data_filtered = data[data["is_up"] == 1]

# 定义特征和标签
features = [
    "sma_up", "sma_down", "macd", "consecutive_upper_days", "upper_count_in_days",
    "ma_amount_3_days_ratio", "ma_amount_5_days_ratio", "ma_amount_8_days_ratio", "ma_amount_11_days_ratio",
    "total_score", "amount", "free_amount", "increase", "amplitude",
    "jgcyd", "lspf", "focus", "last_desire_daily", "high_rating"
]
labels = ["1_day_close", "2_day_close", "3_day_close","1_day_max", "2_day_max", "3_day_max"]

# 训练模型并分析
for label in labels:
    # 定义标签（是否为正数）
    data_filtered[label + "_positive"] = (data_filtered[label] > 0).astype(int)

    # 划分训练集和测试集
    X = data_filtered[features]
    y = data_filtered[label + "_positive"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 训练逻辑回归模型
    model = LogisticRegression()
    model.fit(X_train, y_train)

    # 预测并评估模型
    y_pred = model.predict(X_test)
    print(f"Label: {label}")
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("Classification Report:\n", classification_report(y_test, y_pred))

    # 输出特征系数
    print("特征系数:")
    for feature, coef in zip(features, model.coef_[0]):
        print(f"{feature}: {coef}")
    print("\n")
#
# from sklearn.preprocessing import StandardScaler
#
# # 数据标准化
# scaler = StandardScaler()
# X_train_scaled = scaler.fit_transform(X_train)
# X_test_scaled = scaler.transform(X_test)
#
# # 使用不同的求解器并增加最大迭代次数
# model = LogisticRegression(solver='liblinear', max_iter=1000)
# model.fit(X_train_scaled, y_train)
#
# # 预测并评估模型
# y_pred = model.predict(X_test_scaled)
# print(f"Label: {label}")
# print("Accuracy:", accuracy_score(y_test, y_pred))
# print("Classification Report:\n", classification_report(y_test, y_pred))
#
# # 输出特征系数
# print("特征系数:")
# for feature, coef in zip(features, model.coef_[0]):
#     print(f"{feature}: {coef}")
# print("\n")
