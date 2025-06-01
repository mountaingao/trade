"""第三方库导入"""
from xgboost import XGBRegressor
from sklearn import datasets
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import r2_score, mean_squared_error
import xgboost as xgb
import pandas as pd


data = datasets.load_wine()

print(data)

"""训练集 验证集构建"""
X_train, X_test, y_train, y_test = train_test_split(data.data, data.target, test_size=0.2,
                                                    random_state=42)


"""模型训练"""
model = XGBRegressor()

# model = XGBRegressor(booster='gbtree',  # gblinear
#                      n_estimators=150,  # 迭代次数
#                      learning_rate=0.01,  # 步长
#                      max_depth=10,  # 树的最大深度
#                      min_child_weight=0.5,  # 决定最小叶子节点样本权重和
#                      seed=123,  # 指定随机种子，为了复现结果
#                      )

model.fit(X_train, y_train, verbose=True)

y_pred = model.predict(X_test)
# print(y_pred)

for m, n in zip(y_pred, y_test):
    if m / n - 1 > 0.2:
        print('预测值为{0}, 真是结果为{1}, 预测结果偏差大于20%'.format(m, n))


def metrics_sklearn(y_valid, y_pred_):
    """模型效果评估"""
    r2 = r2_score(y_valid, y_pred_)
    print('r2_score:{0}'.format(r2))

    mse = mean_squared_error(y_valid, y_pred_)
    print('mse:{0}'.format(mse))


"""模型效果评估"""
metrics_sklearn(y_test, y_pred)
