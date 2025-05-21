def model_plot_trees(clf, features, **n):
    """绘制树模型"""
    # 保存特征名称到fmap文件，用于图形绘制
    with open('xgb.fmap', 'w', encoding="utf-8") as fmap:
        for k, ft in enumerate(features):
            fmap.write(''.join(str([k, ft, 'p']) + '\n'))

    # 构建节点的形状、填充颜色
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
    for i in range(n.get('n')):
        digraph = xgb.to_graphviz(clf, num_trees=i, condition_node_params=c_node_params,
                                  leaf_node_params=l_node_params, fmap='xgb.fmap')
        # digraph.format = 'png'
        digraph.view('./oil_xgb_trees{}'.format(i))

    # 分别绘制子图，不保存
    # for i in range(n.get('n')):
    #     xgb.plot_tree(clf, num_trees=i, condition_node_params=c_node_params,
    #                   leaf_node_params=l_node_params, fmap='xgb.fmap')
    #     plt.show()


def model_fit():
    """模型训练"""
    # XGBoost训练过程，下面的参数是调试出来的最佳参数组合
    model = XGBClassifier(learning_rate=0.3, n_estimators=10, max_depth=5, min_child_weight=1,
                          subsample=1, colsample_bytree=1, gamma=0.1, reg_alpha=0.01, reg_lambda=3)
    model.fit(X_train, y_train)

    # 模型保存为txt格式，便于分析、优化和提供可解释性
    clf = model.get_booster()
    clf.dump_model('dump_10tree.txt')

    # 绘制树模型:n = n_estimators
    model_plot_trees(model, dataset.columns[:-1], n=10)

    return model


if __name__ == '__main__':
    import pandas as pd
    import matplotlib.pyplot as plt
    from sklearn.model_selection import train_test_split
    import xgboost as xgb
    from xgboost import XGBClassifier
    import warnings

    warnings.filterwarnings('ignore')
    plt.rcParams['font.family'] = ['sans-serif']
    plt.rcParams['font.sans-serif'] = ['SimHei']
    # 数据集
    dataset = pd.read_csv('Oil_well_parameters_train.csv', engine='python')
    # 划分训练集和验证集
    X_train, X_test, y_train, y_test = train_test_split(dataset.iloc[:, :-1], dataset.iloc[:, -1], test_size=0.2,
                                                        random_state=42)

    # 模型训练
    model_xgb_clf = model_fit()
