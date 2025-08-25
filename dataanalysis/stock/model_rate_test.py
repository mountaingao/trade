

from model_rate import predict_with_models


训练一个阈值为14的模型，

if __name__ == "__main__":


    # input_file = "../data/predictions/1000/08220954_1003.xlsx"
    # input_file = "../data/predictions/1200/08221132_1134.xlsx"
    # input_file = "../data/predictions/1400/08221404_1406.xlsx"
    # input_file = "../data/predictions/1600/08221505_1506.xlsx"


    input_file = "../data/predictions/1600/08251518_1520.xlsx"


    output_dir = "../data/predictions"
    predict_with_models(input_file)