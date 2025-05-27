"""第三方库导入"""
from xgboost import XGBClassifier
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score, f1_score, \
    classification_report
import xgboost as xgb

"""鸢尾花卉数据集导入"""
data = datasets.load_iris()

"""训练集 验证集构建"""
X_train, X_test, y_train, y_test = train_test_split(data.data, data.target, test_size=0.3,
                                                    random_state=123)

print(len(X_test))
print(len(X_train))

# y_train = [1 if y > 0 else 0 for y in y_train]
#
# y_test = [1 if y > 0 else 0 for y in y_test]

"""模型训练"""
model = XGBClassifier()

# model = XGBClassifier(booster='gbtree',
#                       n_estimators=20,  # 迭代次数
#                       learning_rate=0.1,  # 步长
#                       max_depth=5,  # 树的最大深度
#                       min_child_weight=1,  # 决定最小叶子节点样本权重和
#                       subsample=0.8,  # 每个决策树所用的子样本占总样本的比例（作用于样本）
#                       colsample_bytree=0.8,  # 建立树时对特征随机采样的比例（作用于特征）典型值：0.5-1
#                       nthread=4,
#                       seed=27,  # 指定随机种子，为了复现结果
#                       # num_class=4,  # 标签类别数
#                       # objective='multi:softmax',  # 多分类
#                       )

model.fit(X_train, y_train, verbose=True)

"""模型保存"""
model.save_model('xgb_classifier_model.model')

"""模型加载"""
clf = XGBClassifier()
clf.load_model('xgb_classifier_model.model')

"""模型参数打印"""
bst = xgb.Booster(model_file='xgb_classifier_model.model')

# print(bst.attributes())
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
# y_pred = clf.predict(X_test)
# y_proba = clf.predict_proba(X_test)

y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)

for m, n, p in zip(y_proba, y_pred, y_test):
    if n == p:
        q = '预测正确'
    else:
        q = '预测错误'
    print('预测概率为{0}, 预测概率为{1}, 真是结果为{2}, {3}'.format(m, n, p, q))

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
