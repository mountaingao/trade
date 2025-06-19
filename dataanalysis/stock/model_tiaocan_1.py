def xgboost_parameters():
    """模型调参过程"""
    # 第一步：确定迭代次数 n_estimators
    # 参数的最佳取值：{'n_estimators': 50}
    # 最佳模型得分:0.9180952380952381
    # params = {'n_estimators': [5, 10, 50, 75, 100, 200]}

    # 第二步：min_child_weight[default=1],range: [0,∞] 和 max_depth[default=6],range: [0,∞]
    # min_child_weight:如果树分区步骤导致叶节点的实例权重之和小于min_child_weight,那么构建过程将放弃进一步的分区,最小子权重越大,算法就越保守
    # max_depth:树的最大深度,增加该值将使模型更复杂,更可能过度拟合,0表示深度没有限制
    # 参数的最佳取值：{'max_depth': 2, 'min_child_weight': 1}
    # 最佳模型得分:0.9180952380952381，模型分数未提高
    # params = {'max_depth': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10], 'min_child_weight': [1, 2, 3, 4, 5, 6]}

    # 第三步:gamma[default=0, alias: min_split_loss],range: [0,∞]
    # gamma:在树的叶子节点上进行进一步分区所需的最小损失下降,gamma越大,算法就越保守
    # 参数的最佳取值：{'gamma': 0.1}
    # 最佳模型得分:0.9247619047619049
    # params = {'gamma': [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6]}

    # 第四步：subsample[default=1],range: (0,1] 和 colsample_bytree[default=1],range: (0,1]
    # subsample:训练实例的子样本比率。将其设置为0.5意味着XGBoost将在种植树木之前随机抽样一半的训练数据。这将防止过度安装。每一次提升迭代中都会进行一次子采样。
    # colsample_bytree:用于列的子采样的参数,用来控制每颗树随机采样的列数的占比。有利于满足多样性要求,避免过拟合
    # 参数的最佳取值：{'colsample_bytree': 1, 'subsample': 1}
    # 最佳模型得分:0.9247619047619049, 无提高即默认值
    # params = {'subsample': [0.6, 0.7, 0.8, 0.9, 1], 'colsample_bytree': [0.6, 0.7, 0.8, 0.9, 1]}

    # 第五步：alpha[default=0, alias: reg_alpha], 和 lambda[default=1, alias: reg_lambda]
    # alpha:L1关于权重的正则化项。增加该值将使模型更加保守
    # lambda:关于权重的L2正则化项。增加该值将使模型更加保守
    # 参数的最佳取值：{'reg_alpha': 0.01, 'reg_lambda': 3}
    # 最佳模型得分:0.9380952380952381
    # params = {'alpha': [0.01, 0.02, 0.03, 0.04, 0.05, 0.1, 1, 2, 3], 'lambda': [0.05, 0.1, 1, 2, 3, 4]}

    # 第六步：learning_rate[default=0.3, alias: eta],range: [0,1]
    # learning_rate:一般这时候要调小学习率来测试,学习率越小训练速度越慢,模型可靠性越高,但并非越小越好
    # 参数的最佳取值：{'learning_rate': 0.3}
    # 最佳模型得分:0.9380952380952381, 无提高即默认值
    params = {'learning_rate': [0.01, 0.05, 0.07, 0.1, 0.2, 0.25, 0.3, 0.4]}

    # 其他参数设置，每次调参将确定的参数加入
    fine_params = {'n_estimators': 50, 'max_depth': 2, 'min_child_weight': 1, 'gamma': 0.1, 'colsample_bytree': 1,
                   'subsample': 1, 'reg_alpha': 0.01, 'reg_lambda': 3, 'learning_rate': 0.3}
    return params, fine_params
