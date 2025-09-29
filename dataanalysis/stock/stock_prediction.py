import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, roc_auc_score, precision_recall_curve, auc
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline
import xgboost as xgb
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
import warnings
warnings.filterwarnings('ignore')
from re_train_history_datat import get_dir_files_data_value,select_stock_with_block_and_date
from data_prepare import get_prediction_files_data
from sklearn.feature_selection import SelectKBest, f_classif

import os
os.environ['MPLBACKEND'] = 'Agg'
import matplotlib
matplotlib.use('Agg')
import joblib
import logging
import datetime

# 设置日志
logger = logging.getLogger(__name__)
# logger.setLevel(logging.DEBUG)
logger.setLevel(logging.INFO)

# 创建控制台处理器
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)

# 创建格式器
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)

# 添加处理器到日志记录器
logger.addHandler(console_handler)

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

class StockPredictor:
    def __init__(self):
        self.models = {}
        self.scaler = StandardScaler()
        self.feature_names = []
        self.results = {}

    def load_and_preprocess(self, df):
        """加载和预处理数据"""
        logger.debug("=== 数据预处理 ===")

        # 选择特征列
        # features = ['当日涨幅', '量比', '总金额', '信号天数', 'Q', 'Q_1', 'Q3', '净额', '净流入', '当日资金流入']
        features = ['当日涨幅', '量比', '总金额', '信号天数', 'Q', '净额', '净流入', '当日资金流入','time']
        target = 'value'

        # 检查缺失值
        logger.debug(f"数据形状: {df.shape}")
        logger.debug(f"缺失值情况:\n{df[features + [target]].isnull().sum()}")


        # 检查目标变量分布
        logger.debug(f"目标变量分布:\n{df[target].value_counts()}")

        # 处理无限值和异常值
        df_clean = df[features + [target]].copy()
        # 特别处理time列 - 转换为数值类型
        if 'time' in df_clean.columns:
            # 将time列转换为数值类型（例如：'1000' -> 1000）
            df_clean['time'] = pd.to_numeric(df_clean['time'], errors='coerce').fillna(0)

        for col in features:
            if df_clean[col].dtype in ['float64', 'int64']:
                # 替换无限值
                df_clean[col] = df_clean[col].replace([np.inf, -np.inf], np.nan)
                # 用中位数填充缺失值
                df_clean[col].fillna(df_clean[col].median(), inplace=True)
                # 缩尾处理异常值（保留99%的数据）
                lower = df_clean[col].quantile(0.01)  # 扩大范围
                upper = df_clean[col].quantile(0.99)
                df_clean[col] = np.clip(df_clean[col], lower, upper)

        # 关键：处理目标变量中的缺失值
        df_clean[target] = df_clean[target].replace([np.inf, -np.inf], np.nan)
        # 删除目标变量为NaN的行
        df_clean = df_clean.dropna(subset=[target])
        # 确保目标变量为整数类型
        df_clean[target] = df_clean[target].astype(int)

        # 检查是否有足够的正负样本
        positive_samples = df_clean[target].sum()
        negative_samples = len(df_clean) - positive_samples
        logger.debug(f"正样本: {positive_samples}, 负样本: {negative_samples}")

        if positive_samples < 10 or negative_samples < 10:
            logger.warning("警告: 正负样本数量过少，可能影响模型训练")

        # 创建新特征
        df_clean = self._create_features(df_clean)

        logger.debug(f"正样本比例: {df_clean[target].mean():.3f} ({df_clean[target].sum()}/{len(df_clean)})")

        return df_clean

    def _create_features(self, df):
        """创建新特征"""
        # 量价关系特征
        df['量价比'] = df['量比'] / (abs(df['当日涨幅']) + 1e-6)  # 更小的epsilon值
        df['资金强度'] = df['净流入'] / (df['总金额'] + 1e-6)
        df['Q动量'] = df['Q'] * df['当日涨幅']

        # 技术指标组合
        # df['Q系列均值'] = (df['Q'] + df['Q_1'] + df['Q3']) / 3
        # df['Q系列稳定性'] = df[['Q', 'Q_1', 'Q3']].std(axis=1)
        # df['Q系列趋势'] = (df['Q'] - df['Q3']) / (df['Q系列均值'] + 1e-6)

        # 相对强度特征
        df['涨幅强度'] = df['当日涨幅'] / (df['量比'] + 1e-6)
        df['金额强度'] = df['总金额'] / (df['总金额'].median() + 1e-6)

        # 信号持续性
        df['信号强度'] = df['信号天数'] * df['Q']

        # 新增特征
        # df['Q_变化率'] = (df['Q'] - df['Q_1']) / (df['Q_1'] + 1e-6)
        df['资金流入比例'] = df['净流入'] / (df['总金额'] + 1e-6)
        # 增加更多技术指标
        df['涨跌动能'] = df['当日涨幅'] * df['量比']
        df['价格趋势'] = df['当日涨幅'].rolling(3).mean()  # 3日价格趋势
        df['量能趋势'] = df['量比'].rolling(3).mean()      # 3日量能趋势
        df['波动率'] = df['当日涨幅'].rolling(5).std()  # 5日波动率
        # 填充NaN值
        df[['价格趋势', '量能趋势', '波动率']] = df[['价格趋势', '量能趋势', '波动率']].fillna(method='bfill')
        return df


    def prepare_features(self, df, target='value'):
        """准备特征矩阵和目标向量"""
        # 选择最终特征集
        # base_features = ['当日涨幅', '量比', '总金额', '信号天数', 'Q', 'Q_1', 'Q3', '净额', '净流入', '当日资金流入']
        base_features = ['当日涨幅', '量比', '总金额', '信号天数', 'Q', '净额', '净流入', '当日资金流入','time']
        # new_features = ['量价比', '资金强度', 'Q动量', 'Q系列均值', 'Q系列稳定性', 'Q系列趋势','涨幅强度', '金额强度', '信号强度', 'Q_变化率', '资金流入比例']
        # new_features = ['量价比', '资金强度','涨幅强度', '金额强度', '信号强度', '资金流入比例']
        # new_features = [ '资金强度', '信号强度', '资金流入比例']
        new_features = ['资金流入比例','涨幅强度','价格趋势','量能趋势','波动率']

        all_features = base_features + new_features
        self.feature_names = all_features

        X = df[all_features]
        y = df[target]

        # 检查特征是否有常数列
        constant_features = []
        for col in X.columns:
            if X[col].nunique() <= 1:
                constant_features.append(col)

        if constant_features:
            logger.debug(f"发现常数特征: {constant_features}")
            # 移除常数特征
            X = X.drop(columns=constant_features)
            self.feature_names = [f for f in self.feature_names if f not in constant_features]

        # 特征选择：选择最重要的15个特征
        selector = SelectKBest(score_func=f_classif, k=min(15, X.shape[1]))
        X_selected = selector.fit_transform(X, y)

        # 更新特征名称
        selected_features_idx = selector.get_support(indices=True)
        self.feature_names = [self.feature_names[i] for i in selected_features_idx]

        logger.debug(f"特征选择后矩阵形状: {X_selected.shape}")
        logger.debug(f"选择的特征: {self.feature_names}")

        # 检查是否有NaN或inf值
        logger.debug(f"NaN值数量: {np.isnan(X_selected).sum()}")
        logger.debug(f"Inf值数量: {np.isinf(X_selected).sum()}")

        return X_selected, y

    def train_models(self, X, y, test_size=0.3):
        """训练多个模型"""
        logger.debug("\n=== 模型训练 ===")
        # 检查并重新编码目标变量，确保标签是连续的
        unique_labels = np.unique(y)
        logger.debug(f"原始目标变量唯一值: {unique_labels}")

        if len(unique_labels) > 2 and not np.array_equal(unique_labels, np.arange(len(unique_labels))):
            # 如果标签不是连续的，需要重新编码
            from sklearn.preprocessing import LabelEncoder
            label_encoder = LabelEncoder()
            y_encoded = label_encoder.fit_transform(y)
            logger.debug(f"重新编码后的目标变量唯一值: {np.unique(y_encoded)}")
            y = y_encoded
        elif len(unique_labels) == 2 and not np.array_equal(unique_labels, [0, 1]):
            # 如果是二分类但标签不是0和1，也需要重新编码
            from sklearn.preprocessing import LabelEncoder
            label_encoder = LabelEncoder()
            y_encoded = label_encoder.fit_transform(y)
            logger.debug(f"重新编码后的二分类目标变量唯一值: {np.unique(y_encoded)}")
            y = y_encoded


        # 分割数据 移除分层抽样参数 stratify=y
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42
        )

        # 标准化特征
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        # 对于需要标准化的模型，使用采样技术处理不平衡
        from imblearn.over_sampling import SMOTE
        from imblearn.under_sampling import RandomUnderSampler
        from imblearn.pipeline import Pipeline as ImbPipeline

        # 创建采样管道
        smote = SMOTE(random_state=42, sampling_strategy=0.6)  # 增加正样本比例到60%

        # 对于RandomForest使用采样
        # 修改rf_pipeline部分
        rf_pipeline = ImbPipeline([
            ('smote', SMOTE(random_state=42, sampling_strategy=0.3)),  # 减少正样本比例
            ('classifier', RandomForestClassifier(
                n_estimators=150,
                max_depth=5,
                random_state=42,
                min_samples_split=25,
                min_samples_leaf=12,
                max_features='sqrt',
                ccp_alpha=0.005
            ))
        ])

        models = {
            # 如果您追求的是高质量、高可信度的交易信号
            # 'XGBoost': xgb.XGBClassifier(
            #     scale_pos_weight=len(y_train[y_train==0]) / len(y_train[y_train==1]),
            #     max_depth=5,
            #     learning_rate=0.02,       # 进一步降低学习率
            #     n_estimators=250,         # 增加树的数量
            #     min_child_weight=8,       # 略微降低
            #     subsample=0.9,            # 增加样本采样比例
            #     colsample_bytree=0.9,     # 增加特征采样比例
            #     reg_alpha=0.2,            # 略微降低正则化
            #     reg_lambda=0.2,           # 略微降低正则化
            #     random_state=42,
            #     eval_metric='logloss'
            # ),
            # 望最大化机会捕捉
            # 'XGBoost': xgb.XGBClassifier(
            #     scale_pos_weight=len(y_train[y_train==0]) / len(y_train[y_train==1]),
            #     max_depth=5,
            #     learning_rate=0.015,      # 进一步降低学习率
            #     n_estimators=300,         # 增加树的数量
            #     min_child_weight=6,       # 略微降低
            #     subsample=0.95,           # 略微增加
            #     colsample_bytree=0.95,    # 略微增加
            #     reg_alpha=0.1,            # 进一步降低正则化
            #     reg_lambda=0.1,           # 进一步降低正则化
            #     random_state=42,
            #     eval_metric='logloss'
            # ),
            'XGBoost': xgb.XGBClassifier(
                scale_pos_weight=len(y_train[y_train==0]) / len(y_train[y_train==1]),
                max_depth=5,              # 略微降低深度防止过拟合
                learning_rate=0.015,       # 进一步降低学习率
                n_estimators=400,         # 增加树的数量
                min_child_weight=6,      # 增加最小叶子节点样本数
                subsample=0.8,            # 适度采样
                colsample_bytree=0.8,     # 适度特征采样
                reg_alpha=0.1,            # 增加L1正则化
                reg_lambda=0.1,           # 增加L2正则化
                random_state=42,
                eval_metric='logloss'
            ),
            # 'LightGBM': LGBMClassifier(
            #     class_weight='balanced',
            #     random_state=42,
            #     n_estimators=50,
            #     max_depth=3,          # 减小深度
            #     learning_rate=0.1,
            #     min_child_samples=30, # 增加最小叶子节点样本数
            #     subsample=0.7,        # 减少随机采样比例
            #     colsample_bytree=0.7, # 减少特征采样比例
            #     reg_alpha=1,          # 增加L1正则化
            #     reg_lambda=1,         # 增加L2正则化
            #     verbose=-1
            # ),
            # 'LightGBM': LGBMClassifier(  #LightGBM      0.650    0.679  0.500  0.653  0.566  0.462
            #     class_weight='balanced',
            #     random_state=42,
            #     n_estimators=100,         # 增加树的数量
            #     max_depth=6,              # 适当增加深度
            #     learning_rate=0.05,       # 调整学习率
            #     num_leaves=31,            # 控制叶子节点数
            #     min_child_samples=20,     # 调整子节点最小样本数
            #     subsample=0.8,            # 增加样本采样比例
            #     colsample_bytree=0.8,     # 增加特征采样比例
            #     reg_alpha=0.1,            # L1正则化
            #     reg_lambda=0.1,           # L2正则化
            #     min_split_gain=0.01,      # 最小分割增益
            #     verbose=-1
            # ),
            'LightGBM': LGBMClassifier(
                class_weight='balanced',
                random_state=42,
                n_estimators=200,         # 增加树的数量
                max_depth=6,              # 适当增加深度
                learning_rate=0.05,       # 降低学习率
                num_leaves=31,            # 增加叶子节点数
                min_child_samples=20,     # 减少子节点最小样本数
                subsample=0.8,            # 增加样本采样比例
                colsample_bytree=0.8,     # 增加特征采样比例
                reg_alpha=0.1,            # 减少L1正则化
                reg_lambda=0.1,           # 减少L2正则化
                min_split_gain=0.01,      # 最小分割增益
                verbose=-1
            ),
            'RandomForest': RandomForestClassifier(
                class_weight='balanced',
                n_estimators=200,
                max_depth=5,              # 降低深度防止过拟合
                random_state=42,
                min_samples_split=50,     # 增加分割所需最小样本数
                min_samples_leaf=25,      # 增加叶节点最小样本数
                max_features='sqrt',
                bootstrap=True,
                ccp_alpha=0.01           # 增加剪枝强度
            )
        }

        # 存储结果
        self.results = {
            'X_train': X_train, 'X_test': X_test, 'y_train': y_train, 'y_test': y_test,
            'X_train_scaled': X_train_scaled, 'X_test_scaled': X_test_scaled
        }

        # 训练每个模型
        for name, model in models.items():
            logger.debug(f"\n训练 {name}...")

            if name in ['XGBoost', 'LightGBM']:
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                y_proba = model.predict_proba(X_test)[:, 1]
            else:
                model.fit(X_train_scaled, y_train)
                y_pred = model.predict(X_test_scaled)
                y_proba = model.predict_proba(X_test_scaled)[:, 1]

            # 计算评估指标
            accuracy = accuracy_score(y_test, y_pred)
            auc_roc = roc_auc_score(y_test, y_proba)

            # 找到最优阈值
            precision, recall, thresholds = precision_recall_curve(y_test, y_proba)
            f1_scores = 2 * precision * recall / (precision + recall + 1e-8)
            optimal_idx = np.argmax(f1_scores)
            optimal_threshold = thresholds[optimal_idx]

            # 使用最优阈值重新预测
            y_pred_optimal = (y_proba >= optimal_threshold).astype(int)

            # 存储模型和结果
            self.models[name] = {
                'model': model,
                'y_pred': y_pred,
                'y_pred_optimal': y_pred_optimal,
                'y_proba': y_proba,
                'accuracy': accuracy,
                'auc_roc': auc_roc,
                'optimal_threshold': optimal_threshold,
                'classification_report': classification_report(y_test, y_pred_optimal, output_dict=True)
            }

            logger.debug(f"{name} - 准确率: {accuracy:.3f}, AUC-ROC: {auc_roc:.3f}, 最优阈值: {optimal_threshold:.3f}")

    def evaluate_models(self):
        """评估所有模型"""
        logger.debug("\n=== 模型评估 ===")

        y_test = self.results['y_test']

        # 创建对比表格
        evaluation_df = pd.DataFrame()

        for name, result in self.models.items():
            report = result['classification_report']

            evaluation_df.loc[name, '准确率'] = result['accuracy']
            evaluation_df.loc[name, 'AUC-ROC'] = result['auc_roc']
            evaluation_df.loc[name, '正类精确率'] = report['1']['precision']
            evaluation_df.loc[name, '正类召回率'] = report['1']['recall']
            evaluation_df.loc[name, '正类F1'] = report['1']['f1-score']
            evaluation_df.loc[name, '最优阈值'] = result['optimal_threshold']

        logger.debug("模型性能对比:")
        logger.debug(evaluation_df.round(3))

        return evaluation_df

    def plot_feature_importance(self, top_n=15):
        """绘制特征重要性图"""
        logger.debug("\n=== 特征重要性分析 ===")

        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        axes = axes.ravel()

        best_model_name = None
        best_score = 0

        for idx, (name, result) in enumerate(self.models.items()):
            if idx >= 4:  # 最多显示4个图
                break

            if name == 'XGBoost':
                importance = result['model'].feature_importances_
            elif name == 'LightGBM':
                importance = result['model'].feature_importances_
            else:  # RandomForest
                importance = result['model'].feature_importances_

            # 创建特征重要性DataFrame
            importance_df = pd.DataFrame({
                'feature': self.feature_names,
                'importance': importance
            }).sort_values('importance', ascending=False).head(top_n)

            # 绘制条形图
            axes[idx].barh(range(len(importance_df)), importance_df['importance'])
            axes[idx].set_yticks(range(len(importance_df)))
            axes[idx].set_yticklabels(importance_df['feature'])
            axes[idx].set_title(f'{name} - 特征重要性 Top {top_n}')
            axes[idx].set_xlabel('重要性')

            # 记录最佳模型
            if result['auc_roc'] > best_score:
                best_score = result['auc_roc']
                best_model_name = name

        # 显示最佳模型的特征重要性表格
        if best_model_name:
            best_importance = self.models[best_model_name]['model'].feature_importances_
            best_importance_df = pd.DataFrame({
                '特征': self.feature_names,
                '重要性': best_importance
            }).sort_values('重要性', ascending=False).head(10)

            logger.debug(f"\n最佳模型 ({best_model_name}) 的特征重要性 Top 10:")
            logger.debug(best_importance_df.round(4))

        plt.tight_layout()
        plt.show()

        return best_importance_df if best_model_name else None

    def plot_confusion_matrices(self):
        """绘制混淆矩阵"""
        logger.debug("\n=== 混淆矩阵 ===")

        fig, axes = plt.subplots(1, 3, figsize=(15, 4))

        for idx, (name, result) in enumerate(list(self.models.items())[:3]):
            cm = confusion_matrix(self.results['y_test'], result['y_pred_optimal'])
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[idx])
            axes[idx].set_title(f'{name} - 混淆矩阵')
            axes[idx].set_xlabel('预测标签')
            axes[idx].set_ylabel('真实标签')

        plt.tight_layout()
        plt.show()

    def create_trading_strategy(self, top_features=5):
        """创建交易策略规则"""
        logger.debug("\n=== 交易策略建议 ===")

        # 获取最佳模型的特征重要性
        best_importance_df = self.plot_feature_importance(top_n=top_features)

        if best_importance_df is not None:
            top_features_list = best_importance_df['特征'].head(top_features).tolist()
            top_importances = best_importance_df['重要性'].head(top_features).tolist()

            logger.debug(f"\n基于 Top {top_features} 特征的交易策略规则:")

            rules = []
            for feature, importance in zip(top_features_list, top_importances):
                if '量比' in feature:
                    rules.append(f"量比 > 3 (重要性: {importance:.3f})")
                elif 'Q' in feature and '系列' not in feature:
                    rules.append(f"Q > 2.5 (重要性: {importance:.3f})")
                elif '当日涨幅' in feature:
                    rules.append(f"当日涨幅 > 2% (重要性: {importance:.3f})")
                elif '总金额' in feature:
                    rules.append(f"总金额 > 中位数 (重要性: {importance:.3f})")
                elif '净流入' in feature:
                    rules.append(f"净流入 > 0 (重要性: {importance:.3f})")
                elif '信号天数' in feature:
                    rules.append(f"信号天数 > 5 (重要性: {importance:.3f})")
                elif '资金强度' in feature:
                    rules.append(f"资金强度 > 0.1 (重要性: {importance:.3f})")
                elif 'Q动量' in feature:
                    rules.append(f"Q动量 > 0 (重要性: {importance:.3f})")
                else:
                    rules.append(f"{feature} > 阈值 (重要性: {importance:.3f})")

            for i, rule in enumerate(rules, 1):
                logger.debug(f"{i}. {rule}")

            return rules

        return None

    def predict_new_data(self, new_df, model_name='XGBoost'):
        """预测新数据"""
        if model_name not in self.models:
            logger.debug(f"模型 {model_name} 不存在")
            return None

        # 预处理新数据
        new_df_processed = self._create_features(new_df)
        X_new = new_df_processed[self.feature_names]

        # 预测
        model = self.models[model_name]['model']
        threshold = self.models[model_name]['optimal_threshold']

        if model_name in ['XGBoost', 'LightGBM']:
            y_proba = model.predict_proba(X_new)[:, 1]
        else:
            X_new_scaled = self.scaler.transform(X_new)
            y_proba = model.predict_proba(X_new_scaled)[:, 1]

        y_pred = (y_proba >= threshold).astype(int)

        # 创建结果DataFrame
        result_df = new_df.copy()
        result_df['预测概率'] = y_proba
        result_df['预测结果'] = y_pred
        result_df['交易信号'] = result_df['预测结果'].map({1: '看多', 0: '观望'})

        return result_df


    # 添加到StockPredictor类中
    def save_trained_model(self, model_name, save_path):
        """
        保存训练好的模型

        Parameters:
        model_name (str): 要保存的模型名称 ('XGBoost', 'LightGBM', 'RandomForest')
        save_path (str): 保存路径
        """
        if model_name not in self.models:
            logger.debug(f"错误: 模型 {model_name} 未训练或不存在")
            return False

        try:
            # 确保保存目录存在
            os.makedirs(os.path.dirname(save_path), exist_ok=True)

            # 保存模型和相关信息
            model_package = {
                'model': self.models[model_name]['model'],
                'feature_names': self.feature_names,
                'scaler': self.scaler,
                'optimal_threshold': self.models[model_name]['optimal_threshold'],
                'model_type': model_name
            }

            joblib.dump(model_package, save_path)
            logger.debug(f"模型 {model_name} 已成功保存到: {save_path}")
            return True
        except Exception as e:
            logger.debug(f"保存模型时出错: {e}")
            return False

    def load_trained_model(self, model_path):
        """
        加载已保存的模型

        Parameters:
        model_path (str): 模型文件路径

        Returns:
        str: 加载的模型名称
        """
        try:
            if not os.path.exists(model_path):
                logger.debug(f"错误: 模型文件 {model_path} 不存在")
                return None

            model_package = joblib.load(model_path)

            model_name = model_package['model_type']
            self.models[model_name] = {
                'model': model_package['model'],
                'optimal_threshold': model_package['optimal_threshold']
            }
            self.feature_names = model_package['feature_names']
            self.scaler = model_package['scaler']

            logger.debug(f"模型 {model_name} 已成功加载")
            return model_name
        except Exception as e:
            logger.debug(f"加载模型时出错: {e}")
            return None

    def predict_dataframe(self, df, model_name='XGBoost'):
        """
        使用训练好的模型对DataFrame进行预测

        Parameters:
        df (pd.DataFrame): 需要预测的数据框
        model_name (str): 使用的模型名称，默认为'XGBoost'

        Returns:
        pd.DataFrame: 包含预测结果的DataFrame
        """
        if model_name not in self.models:
            logger.debug(f"错误: 模型 {model_name} 未训练或不存在")
            return None

        logger.debug(f"使用 {model_name} 模型进行预测...")
        # print(df.columns)
        # print(df[['代码', '名称','当日涨幅', '量比', '总金额', '信号天数', 'Q', '净额', '净流入', '当日资金流入','time']])
    # try:
        # 对数据进行预处理和特征工程
        df_processed = self._create_features(df)

        # 选择训练时使用的特征
        X_new = df_processed[self.feature_names]

        # 获取模型和阈值
        model = self.models[model_name]['model']
        threshold = self.models[model_name]['optimal_threshold']

        # 进行预测
        if model_name in ['XGBoost', 'LightGBM']:
            y_proba = model.predict_proba(X_new)[:, 1]
        else:
            # 对于需要标准化的模型
            X_new_scaled = self.scaler.transform(X_new)
            y_proba = model.predict_proba(X_new_scaled)[:, 1]

        # 根据最优阈值进行分类
        y_pred = (y_proba >= threshold).astype(int)

        # 创建结果DataFrame
        result_df = df.copy()
        result_df['预测概率'] = y_proba
        result_df['预测结果'] = y_pred
        result_df['交易信号'] = result_df['预测结果'].map({1: '看多', 0: '观望'})

        logger.debug(f"预测完成，共预测 {len(result_df)} 条记录")
        logger.debug(f"看多信号: {result_df['预测结果'].sum()} 条")
        logger.debug(f"观望信号: {len(result_df) - result_df['预测结果'].sum()} 条")
        logger.debug(result_df[['日期', '代码', '名称', 'blockname','次日涨幅','次日最高涨幅','预测概率', '预测结果', '交易信号']])

        return result_df

    # except Exception as e:
    #     logger.debug(f"预测过程中出错: {e}")
    #     return None

    def predict_ensemble(self, df, method='average'):
        """
        使用集成模型进行预测
        
        Parameters:
        df (pd.DataFrame): 需要预测的数据框
        method (str): 集成方法，'average'(平均概率), 'weighted'(加权平均), 'voting'(投票)
        
        Returns:
        pd.DataFrame: 包含集成预测结果的DataFrame
        """
        logger.debug(f"使用{method}集成方法进行预测...")
        
        # 检查是否有训练好的模型
        if not self.models:
            logger.debug("错误: 没有训练好的模型可供预测")
            return None
            
        # 检查是否所有需要的模型都存在
        available_models = list(self.models.keys())
        logger.debug(f"可用模型: {available_models}")
        
        # 确保至少有一个模型
        if len(available_models) == 0:
            logger.debug("错误: 没有可用的模型进行集成预测")
            return None
            
        try:
            # 对数据进行预处理和特征工程
            df_processed = self._create_features(df)
            X_new = df_processed[self.feature_names]
            
            # 存储各模型的预测概率
            predictions_proba = {}
            predictions_pred = {}
            
            # 获取各模型的预测概率和预测结果
            for name, model_info in self.models.items():
                try:
                    model = model_info['model']
                    threshold = model_info['optimal_threshold']
                    
                    if name in ['XGBoost', 'LightGBM']:
                        y_proba = model.predict_proba(X_new)[:, 1]
                    else:
                        X_new_scaled = self.scaler.transform(X_new)
                        y_proba = model.predict_proba(X_new_scaled)[:, 1]
                    
                    predictions_proba[name] = y_proba
                    predictions_pred[name] = (y_proba >= threshold).astype(int)
                except Exception as e:
                    logger.debug(f"模型 {name} 预测时出错: {e}")
                    continue
            
            # 检查是否有成功预测的模型
            if not predictions_proba:
                logger.debug("错误: 没有模型成功生成预测结果")
                return None
                
            logger.debug(f"成功预测的模型: {list(predictions_proba.keys())}")
            
            # 根据不同方法进行集成
            if method == 'average':
                # 平均概率
                ensemble_proba = np.mean(list(predictions_proba.values()), axis=0)
                # 使用第一个模型的最优阈值作为默认阈值
                first_model = list(predictions_proba.keys())[0]
                optimal_threshold = self.models[first_model]['optimal_threshold']
                ensemble_pred = (ensemble_proba >= optimal_threshold).astype(int)
            elif method == 'weighted':
                # 加权平均 (根据模型数量平均分配权重)
                num_models = len(predictions_proba)
                weights = {name: 1/num_models for name in predictions_proba.keys()}
                ensemble_proba = np.zeros(len(df))
                for name, weight in weights.items():
                    ensemble_proba += predictions_proba[name] * weight
                # 使用第一个模型的最优阈值作为默认阈值
                first_model = list(predictions_proba.keys())[0]
                optimal_threshold = self.models[first_model]['optimal_threshold']
                ensemble_pred = (ensemble_proba >= optimal_threshold).astype(int)
            elif method == 'voting':
                # 投票 (多数模型预测为1则为1)
                predictions_array = np.array(list(predictions_pred.values()))
                ensemble_pred = np.mean(predictions_array, axis=0) >= 0.5
                ensemble_pred = ensemble_pred.astype(int)
                # 对于投票方法，使用平均概率作为参考
                ensemble_proba = np.mean(list(predictions_proba.values()), axis=0)
            else:
                raise ValueError("method参数必须是'average', 'weighted'或'voting'之一")
            
            # 创建结果DataFrame
            result_df = df.copy()
            result_df['预测概率'] = ensemble_proba
            result_df['预测结果'] = ensemble_pred
            result_df['交易信号'] = result_df['预测结果'].map({1: '看多', 0: '观望'})
            
            # 添加各模型的预测结果和概率
            for name, pred in predictions_pred.items():
                result_df[f'{name}_预测结果'] = pred
            for name, proba in predictions_proba.items():
                result_df[f'{name}_预测概率'] = proba
            
            logger.debug(f"集成预测完成，共预测 {len(result_df)} 条记录")
            logger.debug(f"看多信号: {result_df['预测结果'].sum()} 条")
            logger.debug(f"观望信号: {len(result_df) - result_df['预测结果'].sum()} 条")
            
            return result_df
            
        except Exception as e:
            logger.debug(f"集成预测过程中出错: {e}")
            import traceback
            traceback.print_exc()
            return None

    def predict_all_models_and_ensemble(self, df):
        """
        并行使用所有模型进行预测，包括单独模型和集成模型
        
        Parameters:
        df (pd.DataFrame): 需要预测的数据框
        
        Returns:
        dict: 包含所有预测结果的字典
        """
        logger.debug("并行使用所有模型进行预测...")
        
        results = {}
        
        # 单独模型预测
        for model_name in self.models.keys():
            logger.debug(f"\n使用 {model_name} 模型进行预测...")
            predictions = self.predict_dataframe(df, model_name)
            if predictions is not None:
                results[model_name] = predictions
        
        # 集成模型预测
        logger.debug("\n使用集成模型进行预测...")
        ensemble_methods = ['average', 'weighted', 'voting']
        for method in ensemble_methods:
            ensemble_predictions = self.predict_ensemble(df, method)
            if ensemble_predictions is not None:
                results[f'ensemble_{method}'] = ensemble_predictions
        
        return results

    def save_ensemble_model(self, save_path):
        """
        保存集成模型
        
        Parameters:
        save_path (str): 保存路径
        """
        try:
            # 确保保存目录存在
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            
            # 保存所有模型和相关信息
            ensemble_package = {
                'models': self.models,
                'feature_names': self.feature_names,
                'scaler': self.scaler,
                'model_types': list(self.models.keys())
            }
            
            joblib.dump(ensemble_package, save_path)
            logger.debug(f"集成模型已成功保存到: {save_path}")
            return True
        except Exception as e:
            logger.debug(f"保存集成模型时出错: {e}")
            return False

    def load_ensemble_model(self, model_path):
        """
        加载集成模型
        
        Parameters:
        model_path (str): 模型文件路径
        
        Returns:
        bool: 是否加载成功
        """
        try:
            if not os.path.exists(model_path):
                logger.debug(f"错误: 模型文件 {model_path} 不存在")
                return False
            
            ensemble_package = joblib.load(model_path)
            
            self.models = ensemble_package['models']
            self.feature_names = ensemble_package['feature_names']
            self.scaler = ensemble_package['scaler']
            
            logger.debug(f"集成模型已成功加载，包含模型: {ensemble_package['model_types']}")
            return True
        except Exception as e:
            logger.debug(f"加载集成模型时出错: {e}")
            return False

    def save_predictions_to_excel(self, df, filename, model_name='XGBoost'):
        """
        对DataFrame进行预测并将结果保存到Excel文件

        Parameters:
        df (pd.DataFrame): 需要预测的数据框
        filename (str): 保存的文件名
        model_name (str): 使用的模型名称，默认为'XGBoost'
        """
        # 进行预测
        predictions = self.predict_dataframe(df, model_name)

        if predictions is not None:
            try:
                # 保存到Excel文件
                predictions.to_excel(filename, index=False)
                logger.debug(f"预测结果已保存到: {filename}")
                return predictions
            except Exception as e:
                logger.debug(f"保存文件时出错: {e}")
                return predictions
        else:
            logger.debug("预测失败，无法保存结果")
            return None


def model_train():
    # 初始化预测器
    predictor = StockPredictor()

    # 加载数据
    df = pd.read_excel("temp/0801-0923.xlsx")
    logger.debug(len(df))

    # 数据预处理
    df_processed = predictor.load_and_preprocess(df)

    # 准备特征
    X, y = predictor.prepare_features(df_processed)

    # 训练模型
    predictor.train_models(X, y)

    # 保存所有模型
    for model_name in ['XGBoost', 'LightGBM', 'RandomForest']:
        predictor.save_trained_model(model_name, f'saved_models/{model_name.lower()}_stock_model.pkl')

    # 评估模型
    evaluation_df = predictor.evaluate_models()

    return predictor


# 使用示例
def main():
    # 初始化预测器
    predictor = StockPredictor()

    # 假设df是您的数据框，包含1980条记录
    # df = pd.read_excel('your_data.xlsx')  # 请替换为您的数据加载代码
    # df= get_dir_files_data("../data/predictions/1000/",start_md="0801",end_mmdd="0916")
    start = "0801"
    end = "0923"
    df0 =get_dir_files_data_value("1000",start_md=start,end_mmdd=end)
    df1 =get_dir_files_data_value("1200",start_md=start,end_mmdd=end)
    df2 =get_dir_files_data_value("1400",start_md=start,end_mmdd=end)
    df3 =get_dir_files_data_value("1600",start_md=start,end_mmdd=end)
    df = pd.concat([df0,df1,df2,df3])
    # 将df写入到临时文件中 temp/0801-0923.csv
    df.to_excel(f"temp/{start}-{end}.xlsx", index=False)
    # 读取tmp/0801-0923.xlsx
    # df = pd.read_excel(f"temp/{start}-{end}.xlsx")

    # df.to_excel(df, "")
    logger.debug(len(df))

    # 数据预处理
    df_processed = predictor.load_and_preprocess(df)

    # 准备特征
    X, y = predictor.prepare_features(df_processed)

    # 训练模型
    predictor.train_models(X, y)

    # 评估模型
    evaluation_df = predictor.evaluate_models()

    # 绘制混淆矩阵
    predictor.plot_confusion_matrices()

    # 创建交易策略
    rules = predictor.create_trading_strategy(top_features=5)

    # 显示最佳模型详细报告
    best_model_name = evaluation_df['AUC-ROC'].idxmax()
    logger.debug(f"\n=== 最佳模型详细报告 ({best_model_name}) ===")
    best_report = predictor.models[best_model_name]['classification_report']
    logger.debug(classification_report(predictor.results['y_test'],
                                predictor.models[best_model_name]['y_pred_optimal']))

    return predictor


def load_model_and_predict():
    # 初始化预测器
    predictor = StockPredictor()

    # 加载已保存的模型
    logger.debug("加载XGBoost模型...")
    xgboost_loaded = predictor.load_trained_model('saved_models/xgboost_stock_model.pkl')
    
    logger.debug("加载LightGBM模型...")
    lightgbm_loaded = predictor.load_trained_model('saved_models/lightgbm_stock_model.pkl')
    
    logger.debug("加载RandomForest模型...")
    rf_loaded = predictor.load_trained_model('saved_models/randomforest_stock_model.pkl')

    loaded_models = []
    if xgboost_loaded:
        loaded_models.append('XGBoost')
    if lightgbm_loaded:
        loaded_models.append('LightGBM')
    if rf_loaded:
        loaded_models.append('RandomForest')
    
    logger.debug(f"成功加载的模型: {loaded_models}")

    if loaded_models:
        start = "0801"
        end = "0923"
        # 加载需要预测的数据
        new_df = get_dir_files_data_value("1000", start_md=start, end_mmdd=end)

        # 1. 单独模型预测
        logger.debug("\n=== 单独模型预测 ===")
        single_predictions = {}
        for model_name in loaded_models:
            logger.debug(f"\n使用 {model_name} 模型进行预测...")
            prediction = predictor.predict_dataframe(new_df, model_name)
            if prediction is not None:
                # 只保留 预测结果为1的行
                prediction = prediction[prediction['预测结果'] == 1]
                single_predictions[model_name] = prediction

        # 2. 集成模型预测（仅在有多个模型时）
        if len(loaded_models) > 1:
            logger.debug("\n=== 集成模型预测 ===")
            ensemble_predictions = {}
            
            # 平均集成
            avg_pred = predictor.predict_ensemble(new_df, 'average')
            if avg_pred is not None:
                ensemble_predictions['average'] = avg_pred
            
            # 加权集成
            weighted_pred = predictor.predict_ensemble(new_df, 'weighted')
            if weighted_pred is not None:
                ensemble_predictions['weighted'] = weighted_pred
            
            # 投票集成
            voting_pred = predictor.predict_ensemble(new_df, 'voting')
            if voting_pred is not None:
                ensemble_predictions['voting'] = voting_pred
            
            # 显示集成预测结果
            for method, pred in ensemble_predictions.items():
                logger.debug(f"\n{method} 集成方法预测结果:")
                if pred is not None:
                    logger.debug(pred[pred['预测结果'] == 1][
                        ['日期', '代码', '名称', 'blockname', '次日涨幅', '次日最高涨幅', '预测概率', '预测结果', '交易信号']
                    ].head(10))
        else:
            logger.debug("\n=== 注意: 仅加载了一个模型，跳过集成预测 ===")

        # 保存所有预测结果
        logger.debug("\n=== 保存预测结果 ===")
        # 保存单独模型预测结果
        for model_name, pred in single_predictions.items():
            filename = f"temp/{start}-{end}{model_name.lower()}_predictions.xlsx"
            try:
                pred.to_excel(filename, index=False)
                logger.debug(f"{model_name} 预测结果已保存到: {filename}")
            except Exception as e:
                logger.debug(f"保存 {model_name} 预测结果时出错: {e}")
        
        # 如果有集成预测结果，保存集成结果
        if len(loaded_models) > 1:
            # 这里可以保存集成预测结果
            for method, pred in ensemble_predictions.items():
                filename = f"temp/{start}-{end}{method}_predictions.xlsx"
                try:
                    pred.to_excel(filename, index=False)
                    logger.debug(f"{method} 集成方法预测结果已保存到: {filename}")
                except Exception as e:
                    logger.debug(f"保存 {method} 集成方法预测结果时出错: {e}")
        else:
            logger.debug("\n=== 注意: 仅加载了一个模型，跳过集成预测 ===")
            pass

        return single_predictions
    else:
        logger.debug("模型加载失败")
        return None

# 所有数据建立一个新模型，来比较效果
def all_data_model():
    predictor = StockPredictor()

    # 假设df是您的数据框，包含1980条记录
    df = get_prediction_files_data("../data/predictions/","0801","0923")
    # df.to_excel(df, "")
    logger.debug(len(df))

    # 数据预处理
    df_processed = predictor.load_and_preprocess(df)

    # 准备特征
    X, y = predictor.prepare_features(df_processed)

    # 训练模型
    predictor.train_models(X, y)

    # 评估模型
    evaluation_df = predictor.evaluate_models()

    # 绘制混淆矩阵
    predictor.plot_confusion_matrices()

    # 创建交易策略
    rules = predictor.create_trading_strategy(top_features=5)

    # 显示最佳模型详细报告
    best_model_name = evaluation_df['AUC-ROC'].idxmax()
    logger.debug(f"\n=== 最佳模型详细报告 ({best_model_name}) ===")
    best_report = predictor.models[best_model_name]['classification_report']
    logger.debug(classification_report(predictor.results['y_test'],
                                predictor.models[best_model_name]['y_pred_optimal']))

    return predictor


def load_model_and_predict_from_dataframe(new_df):
    # 初始化预测器
    predictor = StockPredictor()

    # 加载已保存的模型
    model_name = predictor.load_trained_model('saved_models/xgboost_stock_model.pkl')

    if model_name and len(new_df) > 0:
        # 加载需要预测的数据
        # new_df =get_dir_files_data_value("1000",start_md="0925",end_mmdd="0926")
        df_result = select_stock_with_block_and_date(new_df)
        logger.debug(len(df_result))
        logger.debug(df_result)
        if isinstance(df_result, dict) and 'df_max_up' in df_result:
            strong_leaders_df = df_result['df_max_up']
            if not strong_leaders_df.empty:
                # 进行预测
                predictions = predictor.predict_dataframe(strong_leaders_df, 'XGBoost')

                if predictions is not None:
                    # 保存预测结果
                    predictions.to_excel(f"temp/new_predictions_{model_name.lower()}_{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}.xlsx", index=False)
                    logger.debug("新数据预测完成并已保存")

                    # 显示部分预测结果
                    logger.debug("\n前10条预测结果:")
                    # 打印 预测结果为1的记录 ,'次日最高涨幅'
                    logger.debug(predictions[predictions['预测结果'] == 1][['日期', '代码', '名称', '当日涨幅','blockname','次日涨幅','次日最高涨幅','预测概率', '预测结果', '交易信号']])
                    # 打印 次日涨幅 的和  次日最高涨幅的和
                    logger.debug(predictions[predictions['预测结果'] == 1]['次日涨幅'].sum())
                    logger.debug(predictions[predictions['预测结果'] == 1]['次日最高涨幅'].sum())

                return predictions
    else:
        logger.debug("模型加载失败")
        return None


if __name__ == "__main__":
    # 分析数据
    # predictor = main()

    # 训练模型
    # predictor = model_train()
    # logger.debug("训练完成")
    # predictor.run_optimized_predictor(df, predictor.results)
    # 加载模型并进行预测
    # load_model_and_predict()

    # 所有数据的分析
    # all_data_model()

# 单独预测一个文件
#     predictions_file = "../data/predictions/1600/09241501_1515.xlsx"
#     predictions_file = "../data/predictions/1600/09251526_1528.xlsx"
    predictions_file = "../data/predictions/1000/09290941_0942.xlsx"
    df = pd.read_excel(predictions_file)

    df['time'] = 1000
    df['blockname'] = df['概念']

    result= load_model_and_predict_from_dataframe(df)
    print( result)