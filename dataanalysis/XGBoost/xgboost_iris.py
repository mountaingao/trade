# 导入所需库
import xgboost as xgb
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from xgboost.callback import EarlyStopping
from joblib import dump, load
import numpy as np


print(xgb.__version__)


# 1. 数据准备
# 加载鸢尾花数据集
iris = load_iris()
X, y = iris.data, iris.target

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 转换为DMatrix格式（XGBoost的高效数据格式）
dtrain = xgb.DMatrix(X_train, label=y_train)
dtest = xgb.DMatrix(X_test, label=y_test)

# 2. 参数调优
# 定义参数网格
param_grid = {
    'max_depth': [3, 5, 7],
    'learning_rate': [0.1, 0.01, 0.05],
    'n_estimators': [50, 100, 200],
    'gamma': [0, 0.1, 0.2]
}

# 初始化基础模型
base_model = xgb.XGBClassifier(
    objective='multi:softprob',
    num_class=3,
    random_state=42,
    eval_metric='mlogloss'
)

# 网格搜索
grid_search = GridSearchCV(
    estimator=base_model,
    param_grid=param_grid,
    scoring='accuracy',
    cv=5,
    n_jobs=-1
)

grid_search.fit(X_train, y_train)

# 获取最佳参数
best_params = grid_search.best_params_
print(f"Best parameters: {best_params}")

# 3. 模型训练（使用最佳参数）
final_model = xgb.XGBClassifier(
    **best_params,
    objective='multi:softprob',
    num_class=3,
    random_state=42,
    use_label_encoder=False,
    eval_metric='mlogloss'
)

# 训练模型并添加早停法
early_stop = EarlyStopping(rounds=10, metric_name='mlogloss', data_name='validation_0')

# 训练模型并添加早停法（正确参数）
final_model.fit(
    X_train, y_train,
    eval_set=[(X_test, y_test)],
    callbacks=[early_stop],  # ✅ 替代方案
    verbose=False
)

# 4. 模型评估
# 预测测试集
y_pred = final_model.predict(X_test)

# 计算准确率
accuracy = accuracy_score(y_test, y_pred)
print(f"\nTest Accuracy: {accuracy:.4f}")

# 分类报告
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=iris.target_names))

# 混淆矩阵
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# 5. 模型保存
dump(final_model, 'xgboost_iris_model.joblib')
print("\nModel saved successfully!")

# 6. 模型加载与预测
# 加载保存的模型
loaded_model = load('xgboost_iris_model.joblib')

# 创建新样本进行预测
new_samples = np.array([
    [5.1, 3.5, 1.4, 0.2],  # 预期类别0
    [6.7, 3.1, 4.4, 1.4],  # 预期类别1
    [7.2, 3.0, 5.8, 1.6]   # 预期类别2
])

# 预测新样本
predictions = loaded_model.predict(new_samples)
probabilities = loaded_model.predict_proba(new_samples)

# 输出预测结果
print("\nNew samples predictions:")
for i, (pred, prob) in enumerate(zip(predictions, probabilities)):
    print(f"Sample {i+1}:")
    print(f"Predicted class: {iris.target_names[pred]}")
    print(f"Probabilities: {dict(zip(iris.target_names, prob.round(3)))}")
    print()