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

    # 创建特征字典便于访问
    feature_dict = dict(zip(feature_names, features))

    # 特征加权
    weighted_features = features * feature_weights

    # 检查特征是否存在
    def get_feature_value(name, default=0):
        return feature_dict.get(name, default) if name in feature_names else default

    # 计算维度得分
    technical_score = (
            get_feature_value('RSI') * feature_weights[feature_names.index('RSI')] +
            get_feature_value('MACD') * feature_weights[feature_names.index('MACD')] +
            get_feature_value('Volatility') * feature_weights[feature_names.index('Volatility')]
    )

    volume_score = (
            get_feature_value('Volume') * feature_weights[feature_names.index('Volume')] +
            get_feature_value('OBV') * feature_weights[feature_names.index('OBV')]
    )

    sentiment_score = (
            get_feature_value('Sentiment') * feature_weights[feature_names.index('Sentiment')] +
            get_feature_value('News_Impact') * feature_weights[feature_names.index('News_Impact')]
    )

    # 动量得分 - 使用多个动量指标
    momentum_score = (
            get_feature_value('Momentum_1D') * feature_weights[feature_names.index('Momentum_1D')] +
            get_feature_value('Momentum_3D') * feature_weights[feature_names.index('Momentum_3D')] +
            get_feature_value('Momentum_5D') * feature_weights[feature_names.index('Momentum_5D')]
    )

    # 综合评分（加权求和）
    total_score = (
            0.4 * technical_score +
            0.3 * volume_score +
            0.2 * sentiment_score +
            0.1 * momentum_score
    )

    # 归一化到0-100分（假设原始评分范围在0-10之间）
    total_score = min(max(total_score * 10, 0), 100)

    return total_score

# 使用示例
if __name__ == "__main__":
    # 定义特征名称
    feature_names = [
        'RSI', 'MACD', 'Volatility',
        'Volume', 'OBV',
        'Sentiment', 'News_Impact',
        'Momentum_1D', 'Momentum_3D', 'Momentum_5D'
    ]

    # 示例特征权重（根据特征重要性确定）
    feature_weights = np.array([
        0.15, 0.12, 0.10,   # 技术指标权重
        0.20, 0.08,         # 成交量指标权重
        0.12, 0.13,         # 情绪指标权重
        0.05, 0.03, 0.02   # 动量指标权重
    ])

    # 示例股票特征数据
    stock_features = np.array([
        65,    # RSI
        0.5,   # MACD
        0.25,  # Volatility
        1500000, # Volume
        0.8,   # OBV
        0.75,  # Sentiment
        0.85,  # News_Impact
        0.03,  # Momentum_1D
        0.08,  # Momentum_3D
        0.12   # Momentum_5D
    ])

    # 计算股票评分
    score = calculate_stock_score(
        features=stock_features,
        feature_weights=feature_weights,
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