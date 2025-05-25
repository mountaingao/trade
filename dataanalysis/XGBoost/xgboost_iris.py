# 导入所需库
import xgboost as xgb
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score, f1_score, \
    classification_report
from joblib import dump

# 1. 数据准备 E:\ai\.conda\trade\Lib\site-packages\sklearn\datasets\data\iris.csv
iris = load_iris()
X, y = iris.data, iris.target
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 2. 参数调优（简化示例）
param_grid = {
    'max_depth': [3, 5],
    'learning_rate': [0.1, 0.05],
    'n_estimators': [50, 100]
}

model = xgb.XGBClassifier(
    objective='multi:softprob',
    num_class=3,
    random_state=42,
    eval_metric='mlogloss'
)

grid_search = GridSearchCV(model, param_grid, cv=3)
grid_search.fit(X_train, y_train)

# 3. 模型训练（正确使用早停法）
best_model = grid_search.best_estimator_

# 正确设置早停参数的位置
best_model.set_params(
    early_stopping_rounds=10,
    eval_metric="mlogloss"
)

# 正确格式的eval_set
best_model.fit(
    X_train, y_train,
    eval_set=[(X_test, y_test)],
    verbose=False
)

# 4. 模型评估
print(f"Best params: {grid_search.best_params_}")
print(f"Accuracy: {accuracy_score(y_test, best_model.predict(X_test)):.3f}")

# 5. 保存与加载
dump(best_model, 'best_model.joblib')

y_pred = best_model.predict(X_test)  # 修改：将model改为训练后的best_model
y_proba = best_model.predict_proba(X_test)

# print(y_pred)

for m, n in zip(y_pred, y_test):
    # 添加除数非零判断，使用绝对值计算双向误差
    if n != 0 and abs(m/n - 1) > 0.2:
        # 修正错别字"真是"->"真实"
        print('预测值为{0}, 真实结果为{1}, 预测结果偏差大于20%'.format(m, n))
    else:
        print('预测值为{0}, 真实结果为{1}, 预测结果偏差小于20%'.format(m, n))


# 准确率
accuracy_ = accuracy_score(y_test, y_pred)
# print('Accuracy：%.2f%%' % (accuracy_ * 100))


def metrics_sklearn(class_num, y_valid, y_pred_, y_prob):
    """模型效果评估"""
    # 准确率
    # 准确度 accuracy_score:分类正确率分数，函数返回一个分数，这个分数或是正确的比例，或是正确的个数，不考虑正例负例的问题，区别于 precision_score
    # 一般不直接使用准确率，主要是因为类别不平衡问题，如果大部分是negative的 而且大部分模型都很容易判别出来，那准确率都很高， 没有区分度，也没有实际意义(因为negative不是我们感兴趣的)
    accuracy = accuracy_score(y_valid, y_pred_)
    print('Accuracy：%.2f%%' % (accuracy * 100))

    # 精准率
    if class_num == 2:
        precision = precision_score(y_valid, y_pred_)
    else:
        precision = precision_score(y_valid, y_pred_, average='macro')
    print('Precision：%.2f%%' % (precision * 100))

    # 召回率
    # 召回率/查全率R recall_score:预测正确的正样本占预测正样本的比例， TPR 真正率
    # 在二分类任务中，召回率表示被分为正例的个数占所有正例个数的比例；如果是多分类的话，就是每一类的平均召回率
    if class_num == 2:
        recall = recall_score(y_valid, y_pred_)
    else:
        recall = recall_score(y_valid, y_pred_, average='macro')
    print('Recall：%.2f%%' % (recall * 100))

    # F1值
    if class_num == 2:
        f1 = f1_score(y_valid, y_pred_)
    else:
        f1 = f1_score(y_valid, y_pred_, average='macro')
    print('F1：%.2f%%' % (f1 * 100))

    # auc曲线下面积
    # 曲线下面积 roc_auc_score 计算AUC的值，即输出的AUC（二分类任务中 auc 和 roc_auc_score 数值相等）
    # 计算auc，auc就是曲线roc下面积，这个数值越高，则分类器越优秀。这个曲线roc所在坐标轴的横轴是FPR，纵轴是TPR。
    # 真正率：TPR = TP/P = TP/(TP+FN)、假正率：FPR = FP/N = FP/(FP+TN)
    # auc:不支持多分类任务 计算ROC曲线下的面积
    # 二分类问题直接用预测值与标签值计算：auc = roc_auc_score(Y_test, Y_pred)
    # 多分类问题概率分数 y_prob：auc = roc_auc_score(Y_test, Y_pred_prob, multi_class='ovo') 其中multi_class必选
    if class_num == 2:
        auc = roc_auc_score(y_valid, y_pred_)
    else:
        auc = roc_auc_score(y_valid, y_prob, multi_class='ovo')
        # auc = roc_auc_score(y_valid, y_prob, multi_class='ovr')
    print('AUC：%.2f%%' % (auc * 100))

    # 评估效果报告
    if class_num == 3:
        print(classification_report(y_test, y_pred, target_names=['0:setosa', '1:versicolor', '2:virginica']))
    else:
        print(classification_report(y_test, y_pred))


"""模型效果评估"""
n_classes = len(set(y_train))
metrics_sklearn(n_classes, y_test, y_pred, y_proba)
