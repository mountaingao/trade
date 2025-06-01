# -*- coding: UTF-8 -*-
"""
    xgboost模型调参、训练、保存、预测
    官网信息辅助理解：
    xgboost官网参数(default) https://xgboost.readthedocs.io/en/latest/parameter.html#general-parameters
    sklearn评估指标(Metrics) https://scikit-learn.org/stable/modules/model_evaluation.html
"""

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, precision_score, recall_score, confusion_matrix, \
    roc_curve
import xgboost as xgb
from xgboost import XGBClassifier, plot_importance
import warnings
import matplotlib
matplotlib.use('Agg')

warnings.filterwarnings('ignore')
plt.rcParams['font.family'] = ['sans-serif']
plt.rcParams['font.sans-serif'] = ['SimHei']

dataset = pd.read_csv('Oil_well_parameters_train.csv', engine='python')

X_train, X_test, y_train, y_test = train_test_split(dataset.iloc[:, :-1], dataset.iloc[:, -1], test_size=0.2,
                                                    random_state=42)


def xgboost_parameters():
    # 第一步：确定迭代次数 n_estimators
    # 参数的最佳取值：{'n_estimators': 75}
    # 最佳模型得分: 0.9247619047619047
    # params = {'n_estimators': [5, 10, 50, 75, 100, 200]}

    # 第二步：min_child_weight[default=1],range: [0,∞] 和 max_depth[default=6],range: [0,∞]
    # min_child_weight:如果树分区步骤导致叶节点的实例权重之和小于min_child_weight,那么构建过程将放弃进一步的分区,最小子权重越大,算法就越保守
    # max_depth:树的最大深度,增加该值将使模型更复杂,更可能过度拟合,0表示深度没有限制
    # 参数的最佳取值：{'max_depth': 2, 'min_child_weight': 0.3}
    # 最佳模型得分: 0.9247619047619049
    # params = {'max_depth': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10], 'min_child_weight': [0.1, 0.2, 0.3, 0.5, 1, 2, 3]}

    # 第三步:gamma[default=0, alias: min_split_loss],range: [0,∞]
    # gamma:在树的叶子节点上进行进一步分区所需的最小损失下降,gamma越大,算法就越保守;后剪枝时，用于控制是否后剪枝的参数
    # 参数的最佳取值：{'gamma': 0.1}
    # 最佳模型得分:0.9314285714285715
    # params = {'gamma': [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6]}

    # 第四步：subsample[default=1],range: (0,1] 和 colsample_bytree[default=1],range: (0,1]
    # subsample:训练实例的子样本比率。将其设置为0.5意味着XGBoost将在种植树木之前随机抽样一半的训练数据。这将防止过度安装。每一次提升迭代中都会进行一次子采样。
    # colsample_bytree:用于列的子采样的参数,用来控制每颗树随机采样的列数的占比。有利于满足多样性要求,避免过拟合
    # 参数的最佳取值：{'colsample_bytree': 1, 'subsample': 1}
    # 最佳模型得分: 0.9314285714285715
    # params = {'subsample': [0.6, 0.7, 0.8, 0.9, 1], 'colsample_bytree': [0.6, 0.7, 0.8, 0.9, 1]}

    # 第五步：alpha[default=0, alias: reg_alpha], 和 lambda[default=1, alias: reg_lambda]
    # alpha:关于权重的L1正则化项。增加该值将使模型更加保守
    # lambda:关于权重的L2正则化项。增加该值将使模型更加保守
    # 参数的最佳取值：{'alpha': 0, 'lambda': 1}
    # 最佳模型得分: 0.9314285714285715
    # params = {'alpha': [0, 0.01, 0.02, 0.03, 0.04, 0.05, 0.1, 1, 2, 3], 'lambda': [0.05, 0.1, 1, 2, 3, 4]}

    # 第六步：learning_rate[default=0.3, alias: eta],range: [0,1]
    # learning_rate:一般这时候要调小学习率来测试,学习率越小训练速度越慢,模型可靠性越高,但并非越小越好
    # 参数的最佳取值：{'learning_rate': 0.3}
    # 最佳模型得分:0.9380952380952381, 无提高即默认值
    params = {'learning_rate': [0.01, 0.05, 0.07, 0.1, 0.2, 0.25, 0.3, 0.4]}

    # 其他参数设置，每次调参将确定的参数加入
    fine_params = {'n_estimators': 50, 'max_depth': 2, 'min_child_weight': 0.5}
    return params, fine_params


def model_adjust_parameters(cv_params, other_params):
    """模型调参：GridSearchCV"""
    # 模型基本参数
    model = XGBClassifier(**other_params)
    # sklearn提供的调参工具，训练集k折交叉验证(消除数据切分产生数据分布不均匀的影响)
    optimized_param = GridSearchCV(estimator=model, param_grid=cv_params, scoring='roc_auc', cv=5, verbose=1)
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

    # 模型参数调整得分变化曲线绘制
    parameters_score = pd.DataFrame(params, means)
    parameters_score['means_score'] = parameters_score.index
    parameters_score = parameters_score.reset_index(drop=True)
    parameters_score.to_excel('parameters_score.xlsx', index=False)
    # 画图
    plt.figure(figsize=(15, 12))
    plt.subplot(2, 1, 1)
    plt.plot(parameters_score.iloc[:, :-1], 'o-')
    plt.legend(parameters_score.columns.to_list()[:-1], loc='upper left')
    plt.title('Parameters_size', loc='left', fontsize='xx-large', fontweight='heavy')
    plt.subplot(2, 1, 2)
    plt.plot(parameters_score.iloc[:, -1], 'r+-')
    plt.legend(parameters_score.columns.to_list()[-1:], loc='upper left')
    plt.title('Score', loc='left', fontsize='xx-large', fontweight='heavy')
    plt.show()


def feature_importance_selected(clf_model):
    """模型特征重要性提取与保存"""
    # 模型特征重要性打印和保存
    feature_importance = clf_model.get_booster().get_fscore()
    feature_importance = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
    feature_ipt = pd.DataFrame(feature_importance, columns=['特征名称', '重要性'])
    feature_ipt.to_csv('feature_importance.csv', index=False)
    print('特征重要性:', feature_importance)

    # 模型特征重要性绘图
    plot_importance(clf_model)
    plt.show()


def metrics_sklearn(y_valid, y_pred_):
    """模型对验证集和测试集结果的评分"""
    # 准确率
    accuracy = accuracy_score(y_valid, y_pred_)
    print('Accuracy：%.2f%%' % (accuracy * 100))

    # 精准率
    precision = precision_score(y_valid, y_pred_)
    print('Precision：%.2f%%' % (precision * 100))

    # 召回率
    recall = recall_score(y_valid, y_pred_)
    print('Recall：%.2f%%' % (recall * 100))

    # F1值
    f1 = f1_score(y_valid, y_pred_)
    print('F1：%.2f%%' % (f1 * 100))

    # auc曲线下面积
    auc = roc_auc_score(y_valid, y_pred_)
    print('AUC：%.2f%%' % (auc * 100))

    # ks值
    fpr, tpr, thresholds = roc_curve(y_valid, y_pred_)
    ks = max(abs(fpr - tpr))
    print('KS：%.2f%%' % (ks * 100))


def confusion_matrix_df(y_valid, y_pred_):
    """验证集或测试集预测结果的混淆矩阵"""
    matrix = confusion_matrix(y_valid, y_pred_)
    plt.matshow(matrix, cmap=plt.cm.Greens)

    # 构建标签
    for s in range(len(matrix)):
        for t in range(len(matrix)):
            plt.annotate(matrix[s, t], xy=(s, t))

    # 打印结果
    plt.xlabel('True label')
    plt.ylabel('Predicted label')
    plt.show()


def model_plot_trees(clf, features, **n):
    """绘制树模型"""
    # fmap文件的结构，文件中每一行都是如下结构（注意：这里的尖括号是不需要的）：
    # [feature_id, feature_name, q or i or int]:
    # 第1列为特征的id，第2列为特征的名称；第3列为数据类型
    # 其中：
    # feature_id：从0开始直到特征的个数为止，从小到大排列。
    # i：表示是二分类特征q表示数值变量，如年龄，时间等。
    # q：可以缺省int表示特征为整数(int时，决策边界将为整数)
    with open('xgb.fmap', 'w', encoding="utf-8") as fmap:
        for k, ft in enumerate(features):
            fmap.write(''.join(str([k, ft, 'p']) + '\n'))

    c_node_params = {'shape': 'box',
                     'style': 'filled,rounded',
                     'fillcolor': '#78bceb'
                     }
    l_node_params = {'shape': 'box',
                     'style': 'filled',
                     'fillcolor': '#e48038'
                     }

    # 树模型显示
    # 绘制和保存第num_trees+1棵树
    digraph = xgb.to_graphviz(clf, num_trees=0, condition_node_params=c_node_params,
                              leaf_node_params=l_node_params, fmap='xgb.fmap')
    # digraph.format = 'png'
    # digraph.view('oil_xgb_trees')
    #
    # xgb.plot_tree(model, num_trees=0)
    # plt.show()

    # 分别绘制子图，不保存
    # for i in range(n.get('n')):
    #     xgb.plot_tree(clf, num_trees=i, condition_node_params=c_node_params,
    #                   leaf_node_params=l_node_params, fmap='xgb.fmap')
    #     plt.show()


def model_fit():
    """模型训练"""
    # XGBoost训练过程，下面的参数是调试出来的最佳参数组合
    model = XGBClassifier(learning_rate=0.3, n_estimators=50, max_depth=2, min_child_weight=1,
                          subsample=1, colsample_bytree=1, gamma=0.1, reg_alpha=0.01, reg_lambda=3)
    model.fit(X_train, y_train)

    # 对验证集进行预测——类别
    y_pred = model.predict(X_test)
    y_test_ = y_test.values
    print('y_test：', y_test_.tolist())
    print('y_pred：', y_pred.tolist())

    # 对验证集进行预测——概率
    y_pred_proba = model.predict_proba(X_test)
    # 结果类别是1的概率
    y_pred_proba_ = []
    for i in y_pred_proba.tolist():
        y_pred_proba_.append(i[1])
    print('y_pred_proba：', y_pred_proba_)

    # 模型对验证集预测结果评分
    metrics_sklearn(y_test_, y_pred)

    # 模型对验证集预测结果的混淆矩阵
    confusion_matrix_df(y_test_, y_pred)

    # 模型特征重要性提取、展示和保存
    feature_importance_selected(model)

    # 绘制树模型
    model_plot_trees(model, dataset.columns[:-1])

    return model


def model_save_type(clf_model):
    # 模型训练完成后做持久化，模型保存为model模式，便于调用预测
    clf_model.save_model('xgboost_classifier_model.model')

    # 模型保存为文本格式，便于分析、优化和提供可解释性
    clf = clf_model.get_booster()
    clf.dump_model('dump.txt')


def model_save_load(model, x_transform):
    # 模型加载
    clf = xgb.XGBClassifier()
    booster = xgb.Booster()
    booster.load_model(model)
    clf._Booster = booster

    # 数据预测
    y_pred = [round(value) for value in clf.predict(x_transform)]
    y_pred_proba = clf.predict_proba(x_transform)
    print('y_pred：', y_pred)
    print('y_pred_proba：', y_pred_proba)


if __name__ == '__main__':
    """
        模型调参
        调参策略：逐个或逐类参数调整，避免所有参数一起调整导致模型复杂度过高
    """
    # xgboost参数组合
    # adj_params, fixed_params = xgboost_parameters()
    # 模型调参
    # model_adjust_parameters(adj_params, fixed_params)
    """
        模型训练、评分与保存
        结论：训练集k折交叉验证带来的模型评分提升，未必会在测试集上得到提升
    """

    # 模型训练
    model_xgb_clf = model_fit()
    # 模型保存：model和txt两种格式
    model_save_type(model_xgb_clf)

    """
        模型加载与数据预测
        结论：持久化的模型用来预测数据结果，随着业务的变化模型也需要随之调整
    """
    # x_pred = np.array([[0.63, 0.72, 7.6, 85.4, 40, 38, 0.598787852, 0.474784735],
    #                    [0.39, 0.42, 6.2, 95.4, 39, 38, 0.71287283, 0.5838785491],
    #                    [0.29, 0.32, 20.43, 92.7, 41, 39, 0.498825525, 0.476575973]])
    # model_save_load('xgboost_classifier_model.model', x_pred)
