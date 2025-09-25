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
from re_train_history_datat import get_dir_files_data_value

from sklearn.feature_selection import SelectKBest, f_classif

import os
os.environ['MPLBACKEND'] = 'Agg'
import matplotlib
matplotlib.use('Agg')

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
        print("=== 数据预处理 ===")

        # 选择特征列
        # features = ['当日涨幅', '量比', '总金额', '信号天数', 'Q', 'Q_1', 'Q3', '净额', '净流入', '当日资金流入']
        features = ['当日涨幅', '量比', '总金额', '信号天数', 'Q', '净额', '净流入', '当日资金流入']
        target = 'value'

        # 检查缺失值
        print(f"数据形状: {df.shape}")
        print(f"缺失值情况:\n{df[features + [target]].isnull().sum()}")

        # 检查目标变量分布
        print(f"目标变量分布:\n{df[target].value_counts()}")

        # 处理无限值和异常值
        df_clean = df[features + [target]].copy()
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

        # 检查是否有足够的正负样本
        positive_samples = df_clean[target].sum()
        negative_samples = len(df_clean) - positive_samples
        print(f"正样本: {positive_samples}, 负样本: {negative_samples}")

        if positive_samples < 10 or negative_samples < 10:
            print("警告: 正负样本数量过少，可能影响模型训练")

        # 创建新特征
        df_clean = self._create_features(df_clean)

        print(f"正样本比例: {df_clean[target].mean():.3f} ({df_clean[target].sum()}/{len(df_clean)})")

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

        return df


    def prepare_features(self, df, target='value'):
        """准备特征矩阵和目标向量"""
        # 选择最终特征集
        # base_features = ['当日涨幅', '量比', '总金额', '信号天数', 'Q', 'Q_1', 'Q3', '净额', '净流入', '当日资金流入']
        base_features = ['当日涨幅', '量比', '总金额', '信号天数', 'Q', '净额', '净流入', '当日资金流入']
        # new_features = ['量价比', '资金强度', 'Q动量', 'Q系列均值', 'Q系列稳定性', 'Q系列趋势','涨幅强度', '金额强度', '信号强度', 'Q_变化率', '资金流入比例']
        new_features = ['量价比', '资金强度','涨幅强度', '金额强度', '信号强度', '资金流入比例']

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
            print(f"发现常数特征: {constant_features}")
            # 移除常数特征
            X = X.drop(columns=constant_features)
            self.feature_names = [f for f in self.feature_names if f not in constant_features]

        # 特征选择：选择最重要的15个特征
        selector = SelectKBest(score_func=f_classif, k=min(15, X.shape[1]))
        X_selected = selector.fit_transform(X, y)

        # 更新特征名称
        selected_features_idx = selector.get_support(indices=True)
        self.feature_names = [self.feature_names[i] for i in selected_features_idx]

        print(f"特征选择后矩阵形状: {X_selected.shape}")
        print(f"选择的特征: {self.feature_names}")

        # 检查是否有NaN或inf值
        print(f"NaN值数量: {np.isnan(X_selected).sum()}")
        print(f"Inf值数量: {np.isinf(X_selected).sum()}")

        return X_selected, y

    def train_models(self, X, y, test_size=0.3):
        """训练多个模型"""
        print("\n=== 模型训练 ===")

        # 分割数据
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
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
        rf_pipeline = ImbPipeline([
            ('smote', smote),
            ('classifier', RandomForestClassifier(
                n_estimators=50,
                max_depth=4,
                random_state=42,
                min_samples_split=20,
                min_samples_leaf=10,
                max_features='sqrt'
            ))
        ])

        # 定义模型（进一步优化参数）
        models = {
            'XGBoost': xgb.XGBClassifier(
                scale_pos_weight=len(y_train[y_train==0]) / len(y_train[y_train==1]),
                max_depth=3,  # 进一步减小深度
                learning_rate=0.1,
                n_estimators=50,  # 减少树的数量
                random_state=42,
                eval_metric='logloss',
                min_child_weight=20,  # 增加最小叶子节点样本数
                subsample=0.7,        # 减少随机采样比例
                colsample_bytree=0.7, # 减少特征采样比例
                reg_alpha=1,          # 增加L1正则化
                reg_lambda=1          # 增加L2正则化
            ),
            'LightGBM': LGBMClassifier(
                class_weight='balanced',
                random_state=42,
                n_estimators=50,
                max_depth=3,          # 减小深度
                learning_rate=0.1,
                min_child_samples=30, # 增加最小叶子节点样本数
                subsample=0.7,        # 减少随机采样比例
                colsample_bytree=0.7, # 减少特征采样比例
                reg_alpha=1,          # 增加L1正则化
                reg_lambda=1,         # 增加L2正则化
                verbose=-1
            ),
            'RandomForest': RandomForestClassifier(
                class_weight='balanced',
                n_estimators=50,
                max_depth=4,
                random_state=42,
                min_samples_split=20,  # 增加分割所需的最小样本数
                min_samples_leaf=10,   # 增加叶子节点最小样本数
                max_features='sqrt'    # 减少每次分割考虑的特征数
            )
        }

        # 存储结果
        self.results = {
            'X_train': X_train, 'X_test': X_test, 'y_train': y_train, 'y_test': y_test,
            'X_train_scaled': X_train_scaled, 'X_test_scaled': X_test_scaled
        }

        # 训练每个模型
        for name, model in models.items():
            print(f"\n训练 {name}...")

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

            print(f"{name} - 准确率: {accuracy:.3f}, AUC-ROC: {auc_roc:.3f}, 最优阈值: {optimal_threshold:.3f}")

    def evaluate_models(self):
        """评估所有模型"""
        print("\n=== 模型评估 ===")

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

        print("模型性能对比:")
        print(evaluation_df.round(3))

        return evaluation_df

    def plot_feature_importance(self, top_n=15):
        """绘制特征重要性图"""
        print("\n=== 特征重要性分析 ===")

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

            print(f"\n最佳模型 ({best_model_name}) 的特征重要性 Top 10:")
            print(best_importance_df.round(4))

        plt.tight_layout()
        plt.show()

        return best_importance_df if best_model_name else None

    def plot_confusion_matrices(self):
        """绘制混淆矩阵"""
        print("\n=== 混淆矩阵 ===")

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
        print("\n=== 交易策略建议 ===")

        # 获取最佳模型的特征重要性
        best_importance_df = self.plot_feature_importance(top_n=top_features)

        if best_importance_df is not None:
            top_features_list = best_importance_df['特征'].head(top_features).tolist()
            top_importances = best_importance_df['重要性'].head(top_features).tolist()

            print(f"\n基于 Top {top_features} 特征的交易策略规则:")

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
                print(f"{i}. {rule}")

            return rules

        return None

    def predict_new_data(self, new_df, model_name='XGBoost'):
        """预测新数据"""
        if model_name not in self.models:
            print(f"模型 {model_name} 不存在")
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

# 使用示例
def main():
    # 初始化预测器
    predictor = StockPredictor()

    # 假设df是您的数据框，包含1980条记录
    # df = pd.read_excel('your_data.xlsx')  # 请替换为您的数据加载代码
    # df= get_dir_files_data("../data/predictions/1000/",start_md="0801",end_mmdd="0916")
    df =get_dir_files_data_value("1000",start_md="0801",end_mmdd="0923")
    # 将df写入到临时文件中 temp/0801-0923.csv
    df.to_excel("temp/0801-0923.xlsx", index=False)
    # df.to_excel(df, "")
    print(len(df))

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
    print(f"\n=== 最佳模型详细报告 ({best_model_name}) ===")
    best_report = predictor.models[best_model_name]['classification_report']
    print(classification_report(predictor.results['y_test'],
                                predictor.models[best_model_name]['y_pred_optimal']))

    return predictor

if __name__ == "__main__":
    predictor = main()