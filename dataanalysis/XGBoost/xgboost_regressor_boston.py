"""第三方库导入"""
from xgboost import XGBRegressor
from sklearn import datasets
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import r2_score, mean_squared_error
import xgboost as xgb

"""波士顿房价数据集导入"""
data = datasets.load_boston()
# print(data)

"""训练集 验证集构建"""
X_train, X_test, y_train, y_test = train_test_split(data.data, data.target, test_size=0.2,
                                                    random_state=42)


# print(len(X_test))


def adj_params():
    """模型调参"""
    params = {
        # 'booster': ['gbtree', 'gblinear'],
        # 'n_estimators': [20, 50, 100, 150, 200],
        'n_estimators': [75, 125, 200, 250, 300],
        'learning_rate': [0.01, 0.03, 0.05, 0.1],
        # 'max_depth': [5, 8, 10, 12]
    }

    # model_adj = XGBRegressor()

    other_params = {'booster': 'gbtree', 'seed': 123}
    model_adj = XGBRegressor(**other_params)

    # sklearn提供的调参工具，训练集k折交叉验证(消除数据切分产生数据分布不均匀的影响)
    optimized_param = GridSearchCV(estimator=model_adj, param_grid=params, scoring='r2', cv=5, verbose=1)
    # 模型训练
    optimized_param.fit(X_train, y_train)

    # 对应参数的k折交叉验证平均得分
    means = optimized_param.cv_results_['mean_test_score']
    params = optimized_param.cv_results_['params']
    for mean, param in zip(means, params):
        print("mean_score: %f,  params: %r" % (mean, param))
    # 最佳模型参数
    print('参数的最佳取值：{0}'.format(optimized_param.best_params_))
    # 最佳参数模型得分
    print('最佳模型得分:{0}'.format(optimized_param.best_score_))


# adj_params()

"""模型训练"""
# model = XGBRegressor()

model = XGBRegressor(booster='gbtree',  # gblinear
                     n_estimators=300,  # 迭代次数
                     learning_rate=0.03,  # 步长
                     # max_depth=10,  # 树的最大深度
                     # min_child_weight=0.5,  # 决定最小叶子节点样本权重和
                     seed=123,  # 指定随机种子，为了复现结果
                     )

model.fit(X_train, y_train, verbose=True)

"""模型保存"""
model.save_model('xgb_regressor_boston.model')

"""模型加载"""
clf = XGBRegressor()
clf.load_model('xgb_regressor_boston.model')

"""模型参数打印"""
bst = xgb.Booster(model_file='xgb_regressor_boston.model')

print(bst.attributes())
print('模型参数值-开始'.center(20, '='))
for attr_name, attr_value in bst.attributes().items():
    # scikit_learn 的参数逐一解析
    if attr_name == 'scikit_learn':
        import json

        dict_attr = json.loads(attr_value)
        # 打印 模型 scikit_learn 参数
        for sl_name, sl_value in dict_attr.items():
            if sl_value is not None:
                print(f"{sl_name}:{sl_value}")
    else:
        print(f"{attr_name}:{attr_value}")
print('模型参数值-结束'.center(20, '='))

"""预测验证数据"""
y_pred = clf.predict(X_test)

# y_pred = model.predict(X_test)
# print(y_pred)

for m, n in zip(y_pred, y_test):
    if m / n - 1 > 0.2:
        print('预测值为{0}, 真是结果为{1}, 预测结果偏差{2}'.format(m, n, m / n - 1))


def metrics_sklearn(y_valid, y_pred_):
    """模型效果评估"""
    r2 = r2_score(y_valid, y_pred_)
    print('r2_score:{0}'.format(r2))

    mse = mean_squared_error(y_valid, y_pred_)
    print('mse:{0}'.format(mse))


"""模型效果评估"""
metrics_sklearn(y_test, y_pred)

# r2_score:0.9089456048929225
# mse:6.677363266576786
