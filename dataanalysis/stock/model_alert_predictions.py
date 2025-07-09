"""
模型调用和预测结果
"""


from model_xunlian_alert_1 import predictions_model_data_file,predictions_model_data


# 示例调用修改
if __name__ == "__main__":
    # 使用多个数据集训练并生成模型 ,
    model = {
        'reg_weights': '../models/250709_feature_weights_reg.csv',
        'clf_weights': '../models/250709_feature_weights_clf.csv',
        'reg_model': '../models/250709_model_reg.json',
        'clf_model': '../models/250709_model_clf.json'}

    #预测文件中的数据
    predictions_model_data_file("../alert/0709.xlsx",model)

    # 预测一行数据的结果
    predictions_model_data("../alert/0709.xlsx",model)


