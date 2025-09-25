import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler, QuantileTransformer
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, roc_auc_score, precision_recall_curve, auc
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
import xgboost as xgb
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
import warnings
warnings.filterwarnings('ignore')
from data_prepare import get_dir_files_data

class AdvancedStockPredictor:
    def __init__(self):
        self.models = {}
        self.scaler = StandardScaler()
        self.feature_names = []
        self.results = {}
        self.best_model = None

    def load_and_preprocess(self, df):
        """加载和预处理数据 - 优化版本"""
        print("=== 高级数据预处理 ===")

        features = ['当日涨幅', '量比', '总金额', '信号天数', 'Q', 'Q_1', 'Q3', '净额', '净流入', '当日资金流入']
        target = 'value'

        df_clean = df[features + [target]].copy()

        # 更严格的异常值处理
        for col in features:
            if df_clean[col].dtype in ['float64', 'int64']:
                df_clean[col] = df_clean[col].replace([np.inf, -np.inf], np.nan)
                df_clean[col].fillna(df_clean[col].median(), inplace=True)

                # 使用IQR方法处理异常值
                Q1 = df_clean[col].quantile(0.05)
                Q3 = df_clean[col].quantile(0.95)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                df_clean[col] = np.clip(df_clean[col], lower_bound, upper_bound)

        # 创建更有区分度的特征
        df_clean = self._create_advanced_features(df_clean)

        print(f"正样本比例: {df_clean[target].mean():.3f}")
        print(f"处理后的数据形状: {df_clean.shape}")

        return df_clean

    def _create_advanced_features(self, df):
        """创建更高级的特征工程"""
        # 价格动量特征
        df['价格动量'] = df['当日涨幅'] * df['量比']
        df['强势指标'] = (df['当日涨幅'] > 0).astype(int) * df['Q']

        # 资金流特征
        df['主力净流入率'] = df['净流入'] / (df['总金额'] + 1e-8)
        df['资金效率'] = df['当日涨幅'] / (abs(df['主力净流入率']) + 0.01)

        # 波动性特征
        q_features = ['Q', 'Q_1', 'Q3']
        df['Q波动率'] = df[q_features].std(axis=1)
        df['Q趋势'] = (df['Q'] - df['Q_1']) / (abs(df['Q_1']) + 0.01)

        # 量价关系特征
        df['量价背离'] = abs(df['量比'] - abs(df['当日涨幅']) / 10)
        df['放量上涨'] = ((df['当日涨幅'] > 2) & (df['量比'] > 3)).astype(int)

        # 技术指标组合
        df['综合强度'] = (df['Q'] * 0.3 + df['量比'] * 0.3 + df['当日涨幅'] * 0.2 +
                          df['主力净流入率'] * 0.2)

        # 信号持续性特征
        df['有效信号'] = ((df['信号天数'] > 3) & (df['Q'] > 2)).astype(int)

        return df

    def prepare_features(self, df, target='value'):
        """准备特征矩阵和目标向量 - 新增方法"""
        # 基础特征
        base_features = ['当日涨幅', '量比', '总金额', '信号天数', 'Q', 'Q_1', 'Q3', '净额', '净流入', '当日资金流入']

        # 新创建的特征
        new_features = ['价格动量', '强势指标', '主力净流入率', '资金效率', 'Q波动率',
                        'Q趋势', '量价背离', '放量上涨', '综合强度', '有效信号']

        # 确保所有特征都存在
        available_features = []
        for feature in base_features + new_features:
            if feature in df.columns:
                available_features.append(feature)

        self.feature_names = available_features

        X = df[available_features]
        y = df[target]

        print(f"使用的特征数量: {len(available_features)}")
        print(f"特征列表: {available_features}")

        return X, y

    def feature_selection(self, X, y, k=15):
        """特征选择优化"""
        print("\n=== 特征选择 ===")

        # 使用多种方法进行特征选择
        selector_anova = SelectKBest(score_func=f_classif, k=min(k, X.shape[1]))
        selector_mutual = SelectKBest(score_func=mutual_info_classif, k=min(k, X.shape[1]))

        X_anova = selector_anova.fit_transform(X, y)
        X_mutual = selector_mutual.fit_transform(X, y)

        # 获取特征得分
        anova_scores = pd.DataFrame({
            'feature': X.columns,
            'anova_score': selector_anova.scores_
        })

        mutual_scores = pd.DataFrame({
            'feature': X.columns,
            'mutual_score': selector_mutual.scores_
        })

        # 合并得分
        feature_scores = pd.merge(anova_scores, mutual_scores, on='feature')
        feature_scores['combined_score'] = (feature_scores['anova_score'] +
                                            feature_scores['mutual_score'] * 100)  # 调整权重

        selected_features = feature_scores.nlargest(min(k, X.shape[1]), 'combined_score')['feature'].tolist()

        print(f"选择的特征数量: {len(selected_features)}")
        print("重要特征:", selected_features[:10])

        return selected_features

    def train_optimized_models(self, X, y, test_size=0.3):
        """训练优化后的模型"""
        print("\n=== 优化模型训练 ===")

        # 特征选择
        selected_features = self.feature_selection(X, y, k=15)
        X_selected = X[selected_features]
        self.feature_names = selected_features

        # 数据分割
        X_train, X_test, y_train, y_test = train_test_split(
            X_selected, y, test_size=test_size, random_state=42, stratify=y
        )

        # 使用SMOTE处理不平衡数据，但控制过采样比例
        smote = SMOTE(sampling_strategy=0.8, random_state=42)  # 正样本增加到80%的负样本数量
        X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

        # 标准化
        self.scaler.fit(X_train_resampled)
        X_train_scaled = self.scaler.transform(X_train_resampled)
        X_test_scaled = self.scaler.transform(X_test)

        # 计算更精确的类别权重
        scale_pos_weight = len(y_train_resampled[y_train_resampled==0]) / len(y_train_resampled[y_train_resampled==1])

        # 定义优化后的模型
        models = {
            'XGBoost_Opt': xgb.XGBClassifier(
                scale_pos_weight=scale_pos_weight,
                max_depth=4,  # 减少深度防止过拟合
                learning_rate=0.05,  # 降低学习率
                n_estimators=200,
                subsample=0.8,
                colsample_bytree=0.8,
                reg_alpha=0.1,
                reg_lambda=0.1,
                random_state=42
            ),
            'LightGBM_Opt': LGBMClassifier(
                class_weight='balanced',
                n_estimators=200,
                max_depth=4,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                reg_alpha=0.1,
                reg_lambda=0.1,
                random_state=42
            ),
            'RandomForest_Opt': RandomForestClassifier(
                class_weight='balanced',
                n_estimators=200,
                max_depth=8,
                min_samples_split=20,
                min_samples_leaf=10,
                random_state=42
            ),
            'GradientBoosting': GradientBoostingClassifier(
                n_estimators=200,
                max_depth=4,
                learning_rate=0.05,
                subsample=0.8,
                random_state=42
            )
        }

        # 存储数据
        self.results = {
            'X_train': X_train_resampled, 'X_test': X_test,
            'y_train': y_train_resampled, 'y_test': y_test,
            'X_train_scaled': X_train_scaled, 'X_test_scaled': X_test_scaled
        }

        # 训练模型
        for name, model in models.items():
            print(f"\n训练 {name}...")

            if 'XGBoost' in name or 'LightGBM' in name:
                model.fit(X_train_resampled, y_train_resampled)
                y_proba = model.predict_proba(X_test)[:, 1]
            else:
                model.fit(X_train_scaled, y_train_resampled)
                y_proba = model.predict_proba(X_test_scaled)[:, 1]

            # 使用精确率-召回率平衡的阈值
            precision, recall, thresholds = precision_recall_curve(y_test, y_proba)
            f1_scores = 2 * precision * recall / (precision + recall + 1e-8)

            # 寻找平衡精确率和召回率的阈值
            balanced_idx = np.argmin(np.abs(precision - recall))
            balanced_threshold = thresholds[balanced_idx] if balanced_idx < len(thresholds) else 0.5

            y_pred_balanced = (y_proba >= balanced_threshold).astype(int)

            # 计算评估指标
            accuracy = accuracy_score(y_test, y_pred_balanced)
            auc_roc = roc_auc_score(y_test, y_proba)

            report = classification_report(y_test, y_pred_balanced, output_dict=True)

            self.models[name] = {
                'model': model,
                'y_pred': y_pred_balanced,
                'y_proba': y_proba,
                'accuracy': accuracy,
                'auc_roc': auc_roc,
                'threshold': balanced_threshold,
                'classification_report': report,
                'precision_1': report['1']['precision'] if '1' in report else 0,
                'recall_1': report['1']['recall'] if '1' in report else 0,
                'f1_1': report['1']['f1-score'] if '1' in report else 0
            }

            print(f"{name} - 准确率: {accuracy:.3f}, 精确率: {report['1']['precision']:.3f}, "
                  f"召回率: {report['1']['recall']:.3f}, F1: {report['1']['f1-score']:.3f}")

    def optimize_threshold_for_precision(self, model_name, min_precision=0.5):
        """为特定模型优化阈值以达到最小精确率要求"""
        if model_name not in self.models:
            return None

        y_test = self.results['y_test']
        y_proba = self.models[model_name]['y_proba']

        precision, recall, thresholds = precision_recall_curve(y_test, y_proba)

        # 找到满足最小精确率要求的阈值
        valid_thresholds = []
        for i, p in enumerate(precision[:-1]):  # 最后一个precision是1，recall是0
            if p >= min_precision:
                valid_thresholds.append((thresholds[i], p, recall[i]))

        if valid_thresholds:
            # 选择召回率最高的阈值
            best_threshold, best_precision, best_recall = max(valid_thresholds, key=lambda x: x[2])

            y_pred_optimized = (y_proba >= best_threshold).astype(int)
            new_report = classification_report(y_test, y_pred_optimized, output_dict=True)

            print(f"\n{model_name} 阈值优化结果:")
            print(f"新阈值: {best_threshold:.3f}")
            print(f"精确率: {best_precision:.3f}, 召回率: {best_recall:.3f}")
            if '1' in new_report:
                print(f"F1-score: {new_report['1']['f1-score']:.3f}")

            # 更新模型结果
            self.models[model_name].update({
                'y_pred_optimized': y_pred_optimized,
                'optimized_threshold': best_threshold,
                'optimized_report': new_report
            })

            return best_threshold
        else:
            print(f"无法达到最小精确率 {min_precision}")
            return None

    def ensemble_prediction(self):
        """集成学习预测"""
        print("\n=== 集成学习 ===")

        # 获取各模型的预测概率
        proba_list = []
        model_weights = {}

        for name, result in self.models.items():
            # 根据F1-score分配权重
            weight = result['f1_1']
            proba_list.append(result['y_proba'] * weight)
            model_weights[name] = weight

        # 加权平均
        ensemble_proba = np.sum(proba_list, axis=0) / sum(model_weights.values())

        # 找到集成模型的最优阈值
        precision, recall, thresholds = precision_recall_curve(self.results['y_test'], ensemble_proba)
        f1_scores = 2 * precision * recall / (precision + recall + 1e-8)
        optimal_idx = np.argmax(f1_scores)
        optimal_threshold = thresholds[optimal_idx]

        y_pred_ensemble = (ensemble_proba >= optimal_threshold).astype(int)

        # 评估集成模型
        accuracy = accuracy_score(self.results['y_test'], y_pred_ensemble)
        auc_roc = roc_auc_score(self.results['y_test'], ensemble_proba)
        report = classification_report(self.results['y_test'], y_pred_ensemble, output_dict=True)

        self.models['Ensemble'] = {
            'y_pred': y_pred_ensemble,
            'y_proba': ensemble_proba,
            'accuracy': accuracy,
            'auc_roc': auc_roc,
            'threshold': optimal_threshold,
            'classification_report': report,
            'precision_1': report['1']['precision'] if '1' in report else 0,
            'recall_1': report['1']['recall'] if '1' in report else 0,
            'f1_1': report['1']['f1-score'] if '1' in report else 0
        }

        print(f"集成模型 - 准确率: {accuracy:.3f}, 精确率: {report['1']['precision']:.3f}, "
              f"召回率: {report['1']['recall']:.3f}, F1: {report['1']['f1-score']:.3f}")

        return ensemble_proba

    def evaluate_and_compare(self):
        """综合评估和比较"""
        print("\n=== 模型综合评估 ===")

        evaluation_df = pd.DataFrame()

        for name, result in self.models.items():
            evaluation_df.loc[name, '准确率'] = result['accuracy']
            evaluation_df.loc[name, 'AUC-ROC'] = result['auc_roc']
            evaluation_df.loc[name, '正类精确率'] = result['precision_1']
            evaluation_df.loc[name, '正类召回率'] = result['recall_1']
            evaluation_df.loc[name, '正类F1'] = result['f1_1']
            evaluation_df.loc[name, '阈值'] = result['threshold']

        # 按F1-score排序
        evaluation_df = evaluation_df.sort_values('正类F1', ascending=False)

        print("模型性能对比 (按F1-score排序):")
        print(evaluation_df.round(3))

        # 确定最佳模型
        self.best_model = evaluation_df.index[0]
        print(f"\n最佳模型: {self.best_model}")

        return evaluation_df

    def plot_improvement_comparison(self, old_results, new_results):
        """绘制改进对比图"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))

        # 精确率对比
        models = ['XGBoost', 'LightGBM', 'RandomForest']
        old_precision = [old_results.loc[model, '正类精确率'] for model in models]

        # 确保新结果中有对应的模型
        new_models = [model for model in models if model+'_Opt' in new_results.index]
        new_precision = [new_results.loc[model+'_Opt', '正类精确率'] for model in models if model+'_Opt' in new_results.index]

        x = np.arange(len(models))
        width = 0.35

        axes[0,0].bar(x - width/2, old_precision, width, label='优化前', alpha=0.7)
        if new_precision:
            axes[0,0].bar(x[:len(new_precision)] + width/2, new_precision, width, label='优化后', alpha=0.7)
        axes[0,0].set_title('正类精确率对比')
        axes[0,0].set_xticks(x)
        axes[0,0].set_xticklabels(models)
        axes[0,0].legend()

        # F1-score对比
        old_f1 = [old_results.loc[model, '正类F1'] for model in models]
        new_f1 = [new_results.loc[model+'_Opt', '正类F1'] for model in models if model+'_Opt' in new_results.index]

        axes[0,1].bar(x - width/2, old_f1, width, label='优化前', alpha=0.7)
        if new_f1:
            axes[0,1].bar(x[:len(new_f1)] + width/2, new_f1, width, label='优化后', alpha=0.7)
        axes[0,1].set_title('正类F1-score对比')
        axes[0,1].set_xticks(x)
        axes[0,1].set_xticklabels(models)
        axes[0,1].legend()

        # 阈值对比
        old_threshold = [old_results.loc[model, '最优阈值'] for model in models]
        new_threshold = [new_results.loc[model+'_Opt', '阈值'] for model in models if model+'_Opt' in new_results.index]

        axes[1,0].bar(x - width/2, old_threshold, width, label='优化前', alpha=0.7)
        if new_threshold:
            axes[1,0].bar(x[:len(new_threshold)] + width/2, new_threshold, width, label='优化后', alpha=0.7)
        axes[1,0].set_title('预测阈值对比')
        axes[1,0].set_xticks(x)
        axes[1,0].set_xticklabels(models)
        axes[1,0].legend()

        # AUC-ROC对比
        old_auc = [old_results.loc[model, 'AUC-ROC'] for model in models]
        new_auc = [new_results.loc[model+'_Opt', 'AUC-ROC'] for model in models if model+'_Opt' in new_results.index]

        axes[1,1].bar(x - width/2, old_auc, width, label='优化前', alpha=0.7)
        if new_auc:
            axes[1,1].bar(x[:len(new_auc)] + width/2, new_auc, width, label='优化后', alpha=0.7)
        axes[1,1].set_title('AUC-ROC对比')
        axes[1,1].set_xticks(x)
        axes[1,1].set_xticklabels(models)
        axes[1,1].legend()

        plt.tight_layout()
        plt.show()

# 使用优化版本
def run_optimized_predictor(df, old_results):
    """运行优化后的预测器"""
    predictor = AdvancedStockPredictor()

    # 数据预处理
    df_processed = predictor.load_and_preprocess(df)

    # 准备特征
    X, y = predictor.prepare_features(df_processed)

    # 训练优化模型
    predictor.train_optimized_models(X, y)

    # 集成学习
    predictor.ensemble_prediction()

    # 评估结果
    new_results = predictor.evaluate_and_compare()

    # 绘制改进对比
    predictor.plot_improvement_comparison(old_results, new_results)

    # 对最佳模型进行精确率优化
    best_model_name = new_results.index[0]
    predictor.optimize_threshold_for_precision(best_model_name, min_precision=0.5)

    return predictor, new_results

# 主函数
def main():
    # 您的原始结果数据
    old_results = pd.DataFrame({
        '准确率': [0.613, 0.598, 0.613],
        'AUC-ROC': [0.567, 0.566, 0.586],
        '正类精确率': [0.310, 0.307, 0.340],
        '正类召回率': [1.000, 0.995, 0.819],
        '正类F1': [0.473, 0.470, 0.481],
        '最优阈值': [0.107, 0.092, 0.373]
    }, index=['XGBoost', 'LightGBM', 'RandomForest'])

    df= get_dir_files_data("../data/predictions/1000/",start_md="0801",end_mmdd="0916")
    print(len(df))


    # 运行优化版本
    predictor, new_results = run_optimized_predictor(df, old_results)

    return predictor, new_results

if __name__ == "__main__":
    predictor, new_results = main()