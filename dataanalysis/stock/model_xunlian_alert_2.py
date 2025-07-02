import numpy as np
import pandas as pd

def calculate_stock_score(features, feature_weights, feature_names):
    """
    计算股票综合评分（0-100分）
    
    参数:
    features -- 包含所有特征值的数组或Series
    feature_weights -- 各特征的权重数组
    feature_names -- 特征名称列表
    
    返回:
    total_score -- 综合评分(0-100)
    """
    # 确保输入为NumPy数组
    features = np.array(features)
    feature_weights = np.array(feature_weights)

    # 特征加权
    weighted_features = features * feature_weights

    # 综合评分（所有特征加权和）
    total_score = np.sum(weighted_features)

    # 归一化到0-100分（假设原始评分范围在0-10之间）
    total_score = min(max(total_score * 10, 0), 100)

    return total_score

# 使用示例
if __name__ == "__main__":
    # 定义特征名称

    feature_names = [
        'sma_up', 'sma_down', 'macd', 'is_up',
        'upper_days_counts', 'ma_amount_days_ratio_3', 'ma_amount_days_ratio_5','ma_amount_days_ratio_8',"ma_amount_days_ratio_11",
        'total_score','amount', 'free_amount', 'increase', 'amplitude', 'jgcyd', 'lspf', 'focus',
        'last_desire_daily'
    ]

    df = pd.read_excel("../data/0610.xlsx")
    stock_features = df[feature_names]

    # 示例特征权重（根据特征重要性确定）
    # 读取data目录下的feature_weights_clf.csv文件，得到特征权重
    feature_weights_clf = pd.read_csv("../data/feature_weights_clf.csv")

    # 读取data目录下的feature_weights_reg.csv文件，得到特征权重
    feature_weights_reg = pd.read_csv("../data/feature_weights_reg.csv")





    feature_weights = np.array([
        0.15, 0.12, 0.10,   # 技术指标权重
        0.20, 0.08,         # 成交量指标权重
        0.12, 0.13,         # 情绪指标权重
        0.05, 0.03, 0.02   # 动量指标权重
    ])


    # 计算股票评分
    score = calculate_stock_score(
        features=stock_features,
        feature_weights=feature_weights_clf['Weight'],
        feature_names=feature_names
    )

    print(f"股票综合评分: {score:.1f}/100")
    print("评分解读:")

    if score > 80:
        print("→ 强烈买入信号：技术面强劲，市场情绪积极")
    elif score > 65:
        print("→ 买入信号：多个指标显示积极信号")
    elif score > 50:
        print("→ 中性：建议持有或观望")
    elif score > 35:
        print("→ 卖出信号：多个指标显示负面信号")
    else:
        print("→ 强烈卖出信号：技术面疲软，市场情绪消极")