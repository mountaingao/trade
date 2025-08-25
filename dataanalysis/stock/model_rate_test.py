

from model_rate import predict_with_models


# 训练一个阈值为14的模型，

import joblib
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb

def prepare_data():
    """
    准备数据集，包括历史数据和预测数据
    """
    # 假设你已经训练好了一个模型
    model = RandomForestClassifier()
    model.fit(X_train, y_train)

    # 将模型保存到文件 'my_model.joblib'
    joblib.dump(model, 'my_model.joblib')

    # ... 以后或者其他地方 ...

    # 从文件加载模型，无需重新训练
    loaded_model = joblib.load('my_model.joblib')
    prediction = loaded_model.predict(X_new)


if __name__ == "__main__":


    # input_file = "../data/predictions/1000/08220954_1003.xlsx"
    # input_file = "../data/predictions/1200/08221132_1134.xlsx"
    # input_file = "../data/predictions/1400/08221404_1406.xlsx"
    # input_file = "../data/predictions/1600/08221505_1506.xlsx"


    input_file = "../data/predictions/1600/08251518_1520.xlsx"


    output_dir = "../data/predictions"
    predict_with_models(input_file)