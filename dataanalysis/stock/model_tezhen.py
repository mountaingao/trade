import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
# 新增：显式设置matplotlib后端
import matplotlib
matplotlib.use('Agg')  # 使用非交互式后端避免GUI线程问题
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, accuracy_score
from sklearn.preprocessing import LabelEncoder
import model_xunlian


import xgboost as xgb
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt

def xgboost_feature_analysis(df, target_col, params=None, plot_top_n=15):
    """
    XGBoost特征重要性分析

    参数:
        df: 包含特征和目标值的数据框
        target_col: 目标列名
        params: XGBoost参数字典
        plot_top_n: 显示最重要的前N个特征
    """
    if params is None:
        params = {
            'objective': 'reg:squarederror',
            'eval_metric': 'mae',
            'learning_rate': 0.1,
            'max_depth': 6,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'seed': 42
        }

    # 数据准备
    # 新增：复制数据避免修改原数据
    data = df.copy()
    print("数据集大小:", data.shape)
    
    # 新增：检查目标列是否存在
    if target_col not in data.columns:
        raise ValueError(f"目标列 '{target_col}' 不存在于数据中")
    
    # 新增：处理目标列缺失值
    if data[target_col].isna().any():
        na_count = data[target_col].isna().sum()
        print(f"警告: 目标列 '{target_col}' 包含 {na_count} 个缺失值，已自动移除")
        data = data.dropna(subset=[target_col])
    
    # 新增：目标列数值类型转换
    try:
        data[target_col] = pd.to_numeric(data[target_col], errors='coerce')
        if data[target_col].isna().any():
            invalid_count = data[target_col].isna().sum()
            print(f"警告: 目标列 '{target_col}' 包含 {invalid_count} 个无效值(如'--')，已自动移除")
            data = data.dropna(subset=[target_col])
    except Exception as e:
        print(f"目标列转换错误: {e}")
        raise
    
    # 分离特征和目标
    X = data.drop(columns=[target_col])
    y = data[target_col]
    
    # 新增：处理分类特征
    categorical_cols = X.select_dtypes(include=['object', 'category']).columns
    if not categorical_cols.empty:
        print(f"正在对分类列进行编码: {list(categorical_cols)}")
        le = LabelEncoder()
        for col in categorical_cols:
            X[col] = le.fit_transform(X[col].astype(str))

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 转换数据格式
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dtest = xgb.DMatrix(X_test, label=y_test)

    # 训练模型
    model = xgb.train(
        params,
        dtrain,
        num_boost_round=100,
        evals=[(dtrain, 'train'), (dtest, 'test')],
        early_stopping_rounds=10,
        verbose_eval=False
    )

    # 评估
    y_pred = model.predict(dtest)
    mae = mean_absolute_error(y_test, y_pred)
    print(f"MAE: {mae:.4f}")

    # 获取特征重要性
    importance = model.get_score(importance_type='weight')  # 也可用'gain'或'cover'
    importance_df = pd.DataFrame({
        'feature': list(importance.keys()),
        'importance': list(importance.values())
    }).sort_values('importance', ascending=False)

    print("特征重要性:")
    print(importance_df.head(plot_top_n))

    # # 可视化
    # plt.figure(figsize=(12, 8))
    # importance_df.head(plot_top_n).plot.barh(x='feature', y='importance')
    # plt.title('XGBoost Feature Importance')
    # plt.tight_layout()
    # plt.show()

    return importance_df, model

# 使用示例
# xgb_importance, xgb_model = xgboost_feature_analysis(df, '次日涨幅')
import shap

def shap_analysis(model, X, sample_size=100):
    """
    SHAP值分析（解释模型决策）

    参数:
        model: 训练好的XGBoost模型
        X: 特征数据框
        sample_size: 分析样本量（大数据集时可减小）
    """
    # 创建解释器
    explainer = shap.TreeExplainer(model)

    # 计算SHAP值（抽样计算以提高速度）
    X_sample = X.sample(min(sample_size, len(X)))
    shap_values = explainer.shap_values(X_sample)

    # 全局重要性
    plt.figure(figsize=(12, 8))
    shap.summary_plot(shap_values, X_sample, plot_type="bar")
    plt.title('Global Feature Importance (SHAP)')
    plt.show()

    # 特征影响方向分析
    plt.figure(figsize=(12, 8))
    shap.summary_plot(shap_values, X_sample)
    plt.title('Feature Impact Direction')
    plt.show()

    return shap_values

# 使用示例
# shap_values = shap_analysis(xgb_model, X)

def random_forest_feature_analysis(df, target_col, problem_type='regression',
                                   test_size=0.2, random_state=42, n_estimators=100,
                                   plot_top_n=15, figsize=(12, 8)):
    """
    使用随机森林分析特征重要性

    参数:
    -----------
    df : DataFrame
        包含特征和目标变量的数据集
    target_col : str
        目标变量列名
    problem_type : str ('regression' 或 'classification')
        问题类型，默认为回归
    test_size : float
        测试集比例，默认为0.2
    random_state : int
        随机种子，默认为42
    n_estimators : int
        随机森林中树的数量，默认为100
    plot_top_n : int
        可视化显示最重要的前N个特征，默认为15
    figsize : tuple
        图形大小，默认为(12, 8)

    返回:
    -----------
    feature_importance_df : DataFrame
        包含特征重要性排序的DataFrame
    model : 训练好的随机森林模型
    """

    # 复制数据避免修改原数据
    data = df.copy()
    print("数据集大小:", data.shape)

    # 检查目标列是否存在
    if target_col not in data.columns:
        raise ValueError(f"目标列 '{target_col}' 不存在于数据中")
    
    # 新增：检查目标列缺失值并处理
    if data[target_col].isna().any():
        na_count = data[target_col].isna().sum()
        print(f"警告: 目标列 '{target_col}' 包含 {na_count} 个缺失值，已自动移除")
        data = data.dropna(subset=[target_col])
    
    # 新增：目标列数值类型转换
    try:
        # 尝试转换为数值类型，无法转换的值设为NaN
        data[target_col] = pd.to_numeric(data[target_col], errors='coerce')
        # 把不符合的数据行打印出来
        # print(data[data[target_col].apply(lambda x: not isinstance(x, (int, float)))])
        # 检查转换后是否有新的NaN
        if data[target_col].isna().any():
            invalid_count = data[target_col].isna().sum()
            print(f"警告: 目标列 '{target_col}' 包含 {invalid_count} 个无效值(如'--')，已自动移除")
            # print(data[target_col])
            data = data.dropna(subset=[target_col])
    except Exception as e:
        print(f"目标列转换错误: {e}")
        raise
    
    # 分离特征和目标
    X = data.drop(columns=[target_col])
    y = data[target_col]

    # 处理分类特征
    categorical_cols = X.select_dtypes(include=['object', 'category']).columns
    if not categorical_cols.empty:
        print(f"正在对分类列进行编码: {list(categorical_cols)}")
        le = LabelEncoder()
        for col in categorical_cols:
            X[col] = le.fit_transform(X[col].astype(str))

    # 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state)

    # 初始化模型
    if problem_type == 'regression':
        model = RandomForestRegressor(n_estimators=n_estimators,
                                      random_state=random_state)
        metric_name = 'MAE'
    elif problem_type == 'classification':
        model = RandomForestClassifier(n_estimators=n_estimators,
                                       random_state=random_state)
        metric_name = 'Accuracy'
    else:
        raise ValueError("problem_type 必须是 'regression' 或 'classification'")

    # 训练模型
    model.fit(X_train, y_train)

    # 预测并评估
    y_pred = model.predict(X_test)
    if problem_type == 'regression':
        metric_value = mean_absolute_error(y_test, y_pred)
    else:
        metric_value = accuracy_score(y_test, y_pred)

    print(f"\n模型评估 ({metric_name}): {metric_value:.4f}")

    # 获取特征重要性
    importances = model.feature_importances_
    feature_importance_df = pd.DataFrame({
        'feature': X.columns,
        'importance': importances
    }).sort_values('importance', ascending=False)

    # 打印最重要的特征
    print("\n特征重要性排名:")
    print(feature_importance_df.head(plot_top_n).to_string(index=False))

    # 可视化
    # plt.figure(figsize=figsize)
    # top_features = feature_importance_df.head(plot_top_n)
    # sns.barplot(x='importance', y='feature', data=top_features, palette='viridis')
    # plt.title(f'Top {plot_top_n} 最重要的特征 (随机森林)')
    # plt.xlabel('重要性分数')
    # plt.ylabel('特征')
    # plt.tight_layout()
    # plt.savefig('feature_importance.png')  # 保存为图片文件
    # print("特征重要性图已保存为 feature_importance.png")
    
    return feature_importance_df, model

# 比较模型特征
def compare_models(df, target_col):
    """对比随机森林和XGBoost的特征重要性"""
    # 随机森林
    rf_importance, rf_model = random_forest_feature_analysis(df, target_col)

    # XGBoost
    xgb_importance, xgb_model = xgboost_feature_analysis(df, target_col)

    # 标准化重要性分数
    rf_importance['importance'] = rf_importance['importance'] / rf_importance['importance'].max()
    xgb_importance['importance'] = xgb_importance['importance'] / xgb_importance['importance'].max()

    # 合并结果
    comparison = pd.merge(
        rf_importance.rename(columns={'importance': 'RF_importance'}),
        xgb_importance.rename(columns={'importance': 'XGB_importance'}),
        on='feature',
        how='outer'
    ).fillna(0)

    # 计算差异
    comparison['difference'] = comparison['XGB_importance'] - comparison['RF_importance']

    print("\n特征差异:")
    print(comparison.sort_values('difference', ascending=False).head(15).to_string(index=False))
    # # 可视化
    # plt.figure(figsize=(12, 8))
    # comparison.set_index('feature')[['RF_importance', 'XGB_importance']].head(15).plot.barh()
    # plt.title('Random Forest vs XGBoost Feature Importance')
    # plt.xlabel('Normalized Importance')
    # plt.tight_layout()
    # plt.show()
    print(stability_index(xgb_importance))
    print(stability_index(rf_importance))
    return comparison
# 使用示例
# feature_comparison = compare_models(df, '次日涨幅')

from sklearn.feature_selection import SelectFromModel

# 1. 特征选择流水线
def feature_selection_pipeline(df, target_col, threshold='median'):
    """
    基于XGBoost的特征选择流水线
    """
    X = df.drop(columns=[target_col])
    y = df[target_col]

    # 训练XGBoost
    # 使用示例 2. 动态特征权重调整  todo 不知道怎么用
    # model = DynamicFeatureWeighter(xgb.XGBRegressor())
    # model.fit(X, y)
    # adjusted_weights = model.adjusted_weights

    model = xgb.XGBRegressor()
    model.fit(X, y)

    # 特征选择
    selector = SelectFromModel(model, threshold=threshold, prefit=True)
    selected_features = X.columns[selector.get_support()]

    print(f"原始特征数: {X.shape[1]}, 选择后特征数: {len(selected_features)}")
    print("重要特征:", list(selected_features))

    return selected_features

# 使用示例
# selected_features = feature_selection_pipeline(df, '次日涨幅')

class DynamicFeatureWeighter:
    def __init__(self, base_model, volatility_window=5):
        self.base_model = base_model
        self.window = volatility_window

    def calculate_feature_volatility(self, df):
        """计算特征波动率"""
        returns = df.pct_change().dropna()
        volatility = returns.rolling(self.window).std()
        return volatility.mean()

    def fit(self, X, y):
        # 训练基础模型
        self.base_model.fit(X, y)

        # 计算特征波动率
        self.feature_volatility = self.calculate_feature_volatility(X)

        # 调整特征权重
        base_importance = self.base_model.feature_importances_
        self.adjusted_weights = base_importance / (self.feature_volatility + 1e-6)
        self.adjusted_weights = self.adjusted_weights / self.adjusted_weights.sum()

    def predict(self, X):
        return self.base_model.predict(X)

# 使用示例
# model = DynamicFeatureWeighter(xgb.XGBRegressor())
# model.fit(X_train, y_train)
# adjusted_weights = model.adjusted_weights
def stability_index(importance_list):
    """计算各特征排名变化的标准差"""
    # 修复：处理单个DataFrame输入的情况
    if isinstance(importance_list, pd.DataFrame):
        # 如果是DataFrame，提取importance列作为Series
        importance_list = [importance_list['importance']]
    # 确保输入是Series列表
    ranks = pd.DataFrame([s.rank(ascending=False) for s in importance_list])
    return ranks.std().mean()

# 使用3σ原则+分位数组合过滤
def winsorize_series(s, sigma=3, quantile_range=(0.05, 0.95)):
    # 新增：确保输入为数值类型
    s = pd.to_numeric(s, errors='coerce')
    mean, std = s.mean(), s.std()
    lower = max(s.quantile(quantile_range[0]), mean - sigma*std)
    upper = min(s.quantile(quantile_range[1]), mean + sigma*std)
    return s.clip(lower, upper)

# 多目标学习架构
class MultiTargetXGB:
    def __init__(self):
        self.models = {
            'direction': xgb.XGBClassifier(),  # 涨跌方向
            'magnitude': xgb.XGBRegressor()    # 涨幅幅度
        }

    def fit(self, X, y):
        # 方向分类目标
        y_direction = (y > 0).astype(int)
        self.models['direction'].fit(X, y_direction)

        # 幅度回归目标
        y_magnitude = y.abs()
        self.models['magnitude'].fit(X, y_magnitude)

    def predict(self, X):
        dir_pred = self.models['direction'].predict(X)
        mag_pred = self.models['magnitude'].predict(X)
        return dir_pred * mag_pred

# 使用示例
# mtxgb = MultiTargetXGB()
# mtxgb.fit(X_train, y_train)


from bayes_opt import BayesianOptimization
# 贝叶斯参数优化
def xgb_cv(max_depth, gamma, min_child_weight):
    params = {
        'max_depth': int(max_depth),
        'gamma': gamma,
        'min_child_weight': min_child_weight,
        'subsample': 0.8,
        'eta': 0.1
    }
    cv_results = xgb.cv(
        params,
        dtrain,
        num_boost_round=100,
        nfold=5,
        metrics={'mae'},
        early_stopping_rounds=10
    )
    return -cv_results['test-mae-mean'].min()

optimizer = BayesianOptimization(
    f=xgb_cv,
    pbounds={
        'max_depth': (3, 10),
        'gamma': (0, 5),
        'min_child_weight': (1, 10)
    },
    random_state=42
)
# optimizer.maximize(n_iter=10)
def trading_signal(row):
    if row['AI幅度'] > 5 and row['当日资金流入'] > 1.5:
        return '强买入'
    elif row['AI幅度'] > 3 and row['当日资金流入'] > 0:
        return '买入'
    elif row['AI幅度'] < -3:
        return '卖出'
    else:
        return '观望'

# 风险暴露控制：
def calculate_position_size(row, max_risk=0.02, atr_multiplier=2):
    atr = row['ATR_14']  # 需预先计算ATR指标
    risk_amount = portfolio_value * max_risk
    position_size = risk_amount / (atr * atr_multiplier)
    return min(position_size, max_position_size)

# 在线学习机制：
class OnlineXGB:
    def __init__(self):
        self.model = xgb.XGBRegressor()
        self.warm_start = False

    def update(self, new_X, new_y):
        if not self.warm_start:
            self.model.fit(new_X, new_y)
            self.warm_start = True
        else:
            self.model.fit(
                new_X, new_y,
                xgb_model=self.model.get_booster()
            )
# 预测一致性指数：
def consistency_index(y_true, y_pred, window=5):
    direction_match = (np.sign(y_true.rolling(window).mean()) ==
                       np.sign(y_pred.rolling(window).mean()))
    return direction_match.mean()


# 超额收益夏普比率：
def sharpe_ratio(returns, risk_free=0):
    excess_returns = returns - risk_free
    return excess_returns.mean() / excess_returns.std()

if __name__ == "__main__":
    # 使用示例
    # 假设df是您的DataFrame，'次日涨幅'是目标变量
    # 使用多个数据集训练并生成模型
    files= [
        # "../alert/0630.xlsx",
        "../alert/0701.xlsx",
        "../alert/0702.xlsx",
        "../alert/0703.xlsx",
        "../alert/0704.xlsx",
        "../alert/0707.xlsx",
        "../alert/0708.xlsx",
        "../alert/0709.xlsx",
        "../alert/0710.xlsx",
        "../alert/0711.xlsx",
        "../alert/0714.xlsx",
        "../alert/0715.xlsx",
        "../alert/0716.xlsx",
        "../alert/0717.xlsx",
    ]
    df = model_xunlian.generate_model_data_from_files(files)
    features = [
        '当日涨幅', '信号天数', '净额', '净流入', '当日资金流入','次日涨幅'
    ]
    
    # 异常值鲁棒处理
    # 修改：移除dropna()调用，在winsorize_series内部处理类型转换
    df['次日涨幅'] = winsorize_series(df['次日涨幅'])

    df['交易信号'] = df.apply(trading_signal, axis=1)
    # 比较模型
    feature_comparison = compare_models(df[features], '次日涨幅')

    # 使用示例 1. 特征选择流水线
    selected_features = feature_selection_pipeline(df, '次日涨幅')

    # 参数调优
    critical_params = {
        'max_depth': (3, 10),
        'gamma': (0, 5),
        'subsample': (0.6, 1.0),
        'colsample_bytree': (0.6, 1.0)
    }





