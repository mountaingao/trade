import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error, accuracy_score, r2_score, mean_squared_error
# 添加新的评估指标导入
from sklearn.metrics import precision_score, recall_score, f1_score, classification_report
import xgboost as xgb
# 添加集成学习相关导入
from sklearn.ensemble import VotingRegressor, VotingClassifier, StackingRegressor, StackingClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import cross_val_score
import os
from datetime import datetime
import warnings
# 添加模型保存和加载所需库
import joblib
warnings.filterwarnings('ignore')

# 导入原有数据读取函数
from model_xunlian import generate_model_data_from_files, get_prediction_files_data

'''
参考本程序，生成一个新的程序model_rate，主要实现以下功能：
1、读取所有数据，进行整理，可以保留这个读取数据部分；
2、对数据集提取不同字段，分别采用 随机森林 和 xgb 两种模型进行机器学习；
3、比较两种模型的预测的准确率，并对关键字段进行分析；
4、比较同一模型的不同字段的训练和预测结果，得到主要相关字段；
5、最后的结果生成报表文件，保存为excel ，并同时打印展示出来

进一步优化建议
重点优化分类模型：分类任务表现明显优于回归任务，应重点关注分类模型
增加更多技术指标：考虑添加更多量化因子和市场情绪指标
调整分类阈值：可以尝试不同的分类阈值（当前为7%）来优化精确率和召回率平衡
特征工程优化：进一步分析Q因子等新特征的作用机制
模型集成：考虑使用模型集成方法进一步提升性能

进一步优化建议
继续微调分类阈值：可以尝试10.5%或11.5%的阈值看是否能进一步提升性能
优化堆叠模型：使用更复杂的元学习器替代逻辑回归
增加更多技术指标：继续寻找有效的量化因子和市场情绪指标
模型调参：对RandomForest和XGBoost进行超参数优化
交叉验证：使用更稳健的交叉验证方法评估模型性能


'''

def prepare_data():
    """
    准备数据集，包括历史数据和预测数据
    """
    # 检查临时文件是否存在
    temp_file_path = "../data/bak/model_data.xlsx"
    if os.path.exists(temp_file_path):
        print("检测到临时文件，直接读取...")
        df = pd.read_excel(temp_file_path)
        print(f'历史数据量：{len(df)}')
        return df
    
    # 读取历史数据文件
    files = [
        # "../alert/0630.xlsx",
        # "../alert/0701.xlsx",
        # "../alert/0702.xlsx",
        # "../alert/0703.xlsx",
        # "../alert/0704.xlsx",
        # "../alert/0707.xlsx",
        # "../alert/0708.xlsx",
        # "../alert/0709.xlsx",
        # "../alert/0710.xlsx",
        # "../alert/0711.xlsx",
        # "../alert/0714.xlsx",
        # "../alert/0715.xlsx",
        "../alert/0716.xlsx",
    ]
    
    # 读取文件中的数据
    df = generate_model_data_from_files(files)
    print(f'历史数据量：{len(df)}')

    # 提取有效字段
    # df = df[['日期', '代码', '当日涨幅', '信号天数', '净额', '净流入', '当日资金流入', '次日涨幅', '次日最高涨幅']]

    # 读取其他数据 每日整理的数据集
    df_other = get_prediction_files_data("../data/predictions/",'0730')
    
    if df_other is not None and not df_other.empty:
        print(f'预测数据量：{len(df_other)}')
        # 提取有效字段
        # df_other = df_other[['日期', '代码', '当日涨幅', '量比','总金额', '信号天数','Q', 'band_width','min_value','max_value','净额', '净流入', '当日资金流入', '次日涨幅', '次日最高涨幅']]
        # 合并数据
        df = pd.concat([df, df_other], ignore_index=True)
    
    print(f'总数据量：{len(df)}')
    # 将df写入临时文件，供下次使用
    df.to_excel("../data/bak/model_data.xlsx", index=False)
    return df


def prepare_all_data():
    """
    准备数据集，包括历史数据和预测数据
    """
    # 检查临时文件是否存在
    temp_file_path = "../data/bak/model_data_all.xlsx"
    if os.path.exists(temp_file_path):
        print("检测到临时文件，直接读取...")
        df = pd.read_excel(temp_file_path)
        print(f'历史数据量：{len(df)}')
        return df

    # 读取历史数据文件
    files = [
        "../alert/0630.xlsx",
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
    ]

    # 读取文件中的数据
    df = generate_model_data_from_files(files)
    print(f'历史数据量：{len(df)}')

    # 提取有效字段
    # df = df[['日期', '代码', '当日涨幅', '信号天数', '净额', '净流入', '当日资金流入', '次日涨幅', '次日最高涨幅']]

    # 读取其他数据 每日整理的数据集
    df_other = get_prediction_files_data("../data/predictions/",'0717')

    if df_other is not None and not df_other.empty:
        print(f'预测数据量：{len(df_other)}')
        # 提取有效字段
        # df_other = df_other[['日期', '代码', '当日涨幅', '量比','总金额', '信号天数','Q', 'band_width','min_value','max_value','净额', '净流入', '当日资金流入', '次日涨幅', '次日最高涨幅']]
        # 合并数据
        df = pd.concat([df, df_other], ignore_index=True)

    print(f'总数据量：{len(df)}')
    # 将df写入临时文件，供下次使用
    df.to_excel("../data/bak/model_data_all.xlsx", index=False)
    return df
def evaluate_model_performance(y_true, y_pred, model_type='regression'):
    """
    评估模型性能
    """
    if model_type == 'regression':
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        return {'MAE': mae, 'R2': r2, 'RMSE': rmse}
    else:  # classification
        accuracy = accuracy_score(y_true, y_pred)
        # 添加精确率、召回率和F1分数计算
        # 处理多类分类时的平均方法
        precision = precision_score(y_true, y_pred, average='weighted', zero_division=0)
        recall = recall_score(y_true, y_pred, average='weighted', zero_division=0)
        f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
        return {'Accuracy': accuracy, 'Precision': precision, 'Recall': recall, 'F1_Score': f1}

def train_and_evaluate_models(df, target_col, feature_cols, model_type='regression', threshold=20, return_models=False):
    """
    使用随机森林和XGBoost训练模型并评估性能
    """
    # 数据预处理
    data = df.copy()
    
    # 移除目标列中的缺失值
    data = data.dropna(subset=[target_col])
    
    # 分离特征和目标
    X = data[feature_cols]
    y = data[target_col]
    
    # 处理分类特征
    categorical_cols = X.select_dtypes(include=['object', 'category']).columns
    if not categorical_cols.empty:
        le = LabelEncoder()
        for col in categorical_cols:
            X[col] = le.fit_transform(X[col].astype(str))
    
    # 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)
    
    # 初始化模型
    if model_type == 'regression':
        rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
        xgb_model = xgb.XGBRegressor(random_state=42)
        # 创建集成模型
        voting_model = VotingRegressor([
            ('rf', rf_model),
            ('xgb', xgb_model)
        ])
        # 创建堆叠模型
        stacking_model = StackingRegressor([
            ('rf', rf_model),
            ('xgb', xgb_model)
        ], final_estimator=LinearRegression())
    else:  # classification
        # 创建分类目标变量
        if target_col == '次日最高涨幅':
            # 根据传入的阈值来创建分类标签
            y_train = (y_train > threshold).astype(int)
            y_test = (y_test > threshold).astype(int)
        rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
        xgb_model = xgb.XGBClassifier(random_state=42)
        # 创建集成模型
        voting_model = VotingClassifier([
            ('rf', rf_model),
            ('xgb', xgb_model)
        ], voting='soft')
        # 创建堆叠模型
        stacking_model = StackingClassifier([
            ('rf', rf_model),
            ('xgb', xgb_model)
        ], final_estimator=LogisticRegression())
    
    # 训练模型
    rf_model.fit(X_train, y_train)
    xgb_model.fit(X_train, y_train)
    voting_model.fit(X_train, y_train)
    stacking_model.fit(X_train, y_train)
    
    # 预测
    rf_pred = rf_model.predict(X_test)
    xgb_pred = xgb_model.predict(X_test)
    voting_pred = voting_model.predict(X_test)
    stacking_pred = stacking_model.predict(X_test)
    
    # 评估模型
    rf_metrics = evaluate_model_performance(y_test, rf_pred, model_type)
    xgb_metrics = evaluate_model_performance(y_test, xgb_pred, model_type)
    voting_metrics = evaluate_model_performance(y_test, voting_pred, model_type)
    stacking_metrics = evaluate_model_performance(y_test, stacking_pred, model_type)
    
    # 获取特征重要性
    rf_importance = rf_model.feature_importances_
    xgb_importance = xgb_model.feature_importances_
    
    # 整理结果
    results = {
        'RandomForest': {
            # 'model': rf_model,
            'predictions': rf_pred,
            'metrics': rf_metrics,
            'feature_importance': dict(zip(feature_cols, rf_importance))
        },
        'XGBoost': {
            # 'model': xgb_model,
            'predictions': xgb_pred,
            'metrics': xgb_metrics,
            'feature_importance': dict(zip(feature_cols, xgb_importance))
        },
        'VotingEnsemble': {
            # 'model': voting_model,
            'predictions': voting_pred,
            'metrics': voting_metrics,
            'feature_importance': dict(zip(feature_cols, rf_importance))  # 使用RF的重要性作为代表
        },
        'StackingEnsemble': {
            # 'model': stacking_model,
            'predictions': stacking_pred,
            'metrics': stacking_metrics,
            'feature_importance': dict(zip(feature_cols, rf_importance))  # 使用RF的重要性作为代表
        },
        'test_data': {
            'X_test': len(X_test),
            'y_test': len(y_test)
        }
    }
    
    # 如果需要返回模型，则添加模型到结果中
    if return_models:
        results['RandomForest']['model'] = rf_model
        results['XGBoost']['model'] = xgb_model
        results['VotingEnsemble']['model'] = voting_model
        results['StackingEnsemble']['model'] = stacking_model
        results['feature_cols'] = feature_cols
        results['label_encoder'] = le if not categorical_cols.empty else None
        results['categorical_cols'] = categorical_cols
        
    print(results)
    return results

def compare_models_performance(results_dict):
    """
    比较不同模型的性能
    """
    comparison_data = []
    
    for task, task_results in results_dict.items():
        for feature_name, results in task_results.items():
            for model_name, model_results in results.items():
                if model_name in ['RandomForest', 'XGBoost', 'VotingEnsemble', 'StackingEnsemble']:
                    row = {'Task': task, 'Feature_Combination': feature_name, 'Model': model_name}
                    row.update(model_results['metrics'])
                    comparison_data.append(row)
    
    df = pd.DataFrame(comparison_data)
    # 确保列的顺序和存在性
    cols = ['Task', 'Feature_Combination', 'Model'] + [col for col in df.columns if col not in ['Task', 'Feature_Combination', 'Model']]
    df = df[cols]
    return df

def analyze_feature_importance(results_dict):
    """
    分析特征重要性
    """
    importance_data = []
    
    for task, task_results in results_dict.items():
        for feature_name, results in task_results.items():
            for model_name, model_results in results.items():
                if model_name in ['RandomForest', 'XGBoost', 'VotingEnsemble', 'StackingEnsemble']:
                    for feature, importance in model_results['feature_importance'].items():
                        importance_data.append({
                            'Task': task,
                            'Feature_Combination': feature_name,
                            'Model': model_name,
                            'Feature': feature,
                            'Importance': importance
                        })
    
    return pd.DataFrame(importance_data)

def generate_report(df_results, df_importance, output_file):
    """
    生成报表并保存为Excel文件
    """
    with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
        # 保存模型性能比较
        df_results.to_excel(writer, sheet_name='Model_Performance', index=False)
        
        # 保存特征重要性
        df_importance.to_excel(writer, sheet_name='Feature_Importance', index=False)
        
        # 生成汇总统计
        if not df_results.empty and 'Task' in df_results.columns and 'Model' in df_results.columns:
            summary_stats = df_results.groupby(['Task', 'Model']).mean(numeric_only=True).round(4)
            summary_stats.to_excel(writer, sheet_name='Summary_Stats')
        else:
            print("警告：无法生成汇总统计，缺少必要的列或数据为空")
    
    print(f"报表已保存至: {output_file}")
    return output_file

def analyze_feature_impact_on_accuracy(df_performance):
    """
    分析不同特征组合对模型准确率的影响
    """
    print("\n=== 特征组合对准确率影响分析 ===")
    
    # 只分析分类任务
    classification_data = df_performance[df_performance['Task'].str.contains('Classification')]
    
    if classification_data.empty:
        print("无分类任务数据用于分析")
        return
    
    for task in classification_data['Task'].unique():
        print(f"\n任务: {task}")
        task_data = classification_data[classification_data['Task'] == task]
        
        # 获取每个特征组合的最佳模型准确率
        feature_performance = task_data.groupby('Feature_Combination')['Accuracy'].max().sort_values(ascending=False)
        
        print("特征组合按准确率排序:")
        for i, (feature_comb, accuracy) in enumerate(feature_performance.items(), 1):
            print(f"  {i}. {feature_comb}: {accuracy:.4f}")
        
        # 计算准确率差异
        max_accuracy = feature_performance.max()
        min_accuracy = feature_performance.min()
        accuracy_range = max_accuracy - min_accuracy
        
        print(f"\n准确率范围: {min_accuracy:.4f} - {max_accuracy:.4f}")
        print(f"准确率差异: {accuracy_range:.4f}")
        
        # 判断差异是否显著（阈值设为1%）
        if accuracy_range < 0.01:
            print("准确率差异较小，可以考虑使用基本特征组合")
        else:
            print("准确率差异较大，建议使用最佳特征组合")

def save_selected_models(df):
    """
    保存指定的模型到model/目录下
    """
    # 创建model目录（如果不存在）
    os.makedirs("../models", exist_ok=True)
    
    # 定义特征组合
    feature_combinations = {
        'Basic_Features': ['当日涨幅', '信号天数', '净额', '净流入', '当日资金流入'],
        'All_Features': ['当日涨幅', '量比','总金额', '信号天数','Q','band_width','min_value','max_value','净额', '净流入', '当日资金流入']
    }
    
    # 定义目标任务和阈值
    tasks = {
        'Next_Day_High_Increase_Classification_14': ('次日最高涨幅', 'classification', 14),
        'Next_Day_High_Increase_Classification_20': ('次日最高涨幅', 'classification', 20)
    }
    
    # 保存阈值=14 和 All_Features 的 StackingEnsemble 模型
    target_col, task_type, threshold = tasks['Next_Day_High_Increase_Classification_14']
    feature_cols = feature_combinations['All_Features']
    # 移除目标列中的缺失值
    results = train_and_evaluate_models(df, target_col, feature_cols, task_type, threshold, return_models=True)
    
    # 保存模型
    model_data_14 = {
        'model': results['StackingEnsemble']['model'],
        'feature_cols': results['feature_cols'],
        'label_encoder': results['label_encoder'],
        'categorical_cols': results['categorical_cols'],
        'threshold': threshold,
        'task_type': task_type
    }
    joblib.dump(model_data_14, "../models/stacking_ensemble_threshold_14_all_features.pkl")
    print("已保存阈值=14 和 All_Features 的 StackingEnsemble 模型")
    
    # 保存阈值=20 和 Basic_Features 的 RandomForest 模型
    target_col, task_type, threshold = tasks['Next_Day_High_Increase_Classification_20']
    feature_cols = feature_combinations['Basic_Features']
    results = train_and_evaluate_models(df, target_col, feature_cols, task_type, threshold, return_models=True)
    
    # 保存模型
    model_data_20 = {
        'model': results['RandomForest']['model'],
        'feature_cols': results['feature_cols'],
        'label_encoder': results['label_encoder'],
        'categorical_cols': results['categorical_cols'],
        'threshold': threshold,
        'task_type': task_type
    }
    joblib.dump(model_data_20, "../models/random_forest_threshold_20_basic_features.pkl")
    print("已保存阈值=20 和 Basic_Features 的 RandomForest 模型")

def predict_with_models(file_path, output_path=None):
    """
    读取文件并使用保存的模型进行预测，将结果添加到原始文件中并保存
    """
    # 读取输入文件
    df = pd.read_excel(file_path)
    
    # 加载模型
    model_14 = joblib.load("../models/stacking_ensemble_threshold_14_all_features.pkl")
    model_20 = joblib.load("../models/random_forest_threshold_20_basic_features.pkl")
    
    # 准备特征数据
    # 对模型14 (All_Features)
    X_14 = df[model_14['feature_cols']].copy()
    if not model_14['categorical_cols'].empty:
        for col in model_14['categorical_cols']:
            if col in X_14.columns:
                X_14[col] = model_14['label_encoder'].fit_transform(X_14[col].astype(str))
    
    # 对模型20 (Basic_Features)
    X_20 = df[model_20['feature_cols']].copy()
    if not model_20['categorical_cols'].empty:
        for col in model_20['categorical_cols']:
            if col in X_20.columns:
                X_20[col] = model_20['label_encoder'].fit_transform(X_20[col].astype(str))
    
    # 进行预测
    predictions_14 = model_14['model'].predict(X_14)
    predictions_20 = model_20['model'].predict(X_20)


    # 添加预测结果到数据框
    df['Prediction_Threshold_14'] = predictions_14
    df['Prediction_Threshold_20'] = predictions_20
    # 打印出df['Prediction_Threshold_14'] 预测结果大于0的值
    print(df[df['Prediction_Threshold_14'] > 0])
    print(predictions_20[predictions_20 > 0])

    # 根据阈值创建分类结果
    df['Classification_Threshold_14'] = (predictions_14 > 0.5).astype(int)  # 二分类结果
    df['Classification_Threshold_20'] = (predictions_20 > 0.5).astype(int)  # 二分类结果
    
    # 保存结果
    if output_path is None:
        output_path = file_path.replace('.xlsx', '_with_predictions.xlsx')
    print(f"保存结果到 {output_path}")
    print( df)
    df.to_excel(output_path, index=False)
    print(f"预测结果已保存至: {output_path}")
    
    return df

def backtest_threshold_14(df):
    """
    回测模型，测试阈值14的模型，并将符合条件的值写入临时目录
    """
    # 创建临时目录（如果不存在）
    os.makedirs("temp", exist_ok=True)
    
    # 定义特征组合
    feature_combinations = {
        'All_Features': ['当日涨幅', '量比','总金额','信号天数','Q','band_width','min_value','max_value','净额', '净流入', '当日资金流入']
    }
    
    print("\n=== 回测任务: 阈值14 (Threshold_14) ===")
    
    # 选择特征组合和目标
    feature_cols_14 = feature_combinations['All_Features']
    target_col_14 = '次日最高涨幅'
    task_type_14 = 'classification'
    threshold_14 = 14

    # 加载模型
    # 训练模型
    # results_14 = train_and_evaluate_models(df, target_col_14, feature_cols_14, task_type_14, threshold_14, return_models=True)
    
    # 获取测试数据
    X_test_14 = df[feature_cols_14].copy()
    
    # 处理分类特征
    categorical_cols_14 = X_test_14.select_dtypes(include=['object', 'category']).columns
    if not categorical_cols_14.empty:
        le_14 = LabelEncoder()
        for col in categorical_cols_14:
            X_test_14[col] = le_14.fit_transform(X_test_14[col].astype(str))
    
    # 使用StackingEnsemble模型进行预测
    # model_14 = results_14['StackingEnsemble']['model']
    model_14 = joblib.load("../models/stacking_ensemble_threshold_14_all_features.pkl")

    # 进行预测
    predictions_14 = model_14['model'].predict(X_test_14)
    
    # 根据阈值创建分类结果
    classifications_14 = (predictions_14 > 0.5).astype(int)
    
    # 将预测结果添加到原始数据
    df_result_14 = df.copy()
    df_result_14['Prediction_14'] = predictions_14
    df_result_14['Classification_14'] = classifications_14
    
    # 筛选出分类结果为1的记录（即符合条件的记录）
    filtered_df_14 = df_result_14[df_result_14['Classification_14'] == 1]
    
    # 保存符合条件的记录到临时目录
    output_file_14 = "temp/backtest_results_threshold_14.xlsx"
    if not filtered_df_14.empty:
        filtered_df_14.to_excel(output_file_14, index=False)
        print(f"阈值 14 的回测结果已保存到: {output_file_14}")
        print(f"符合条件的记录数: {len(filtered_df_14)}")
        print("符合条件的前几条记录:")
        print(filtered_df_14[['日期', '代码', '当日涨幅', 'Prediction_14', 'Classification_14']].head())
    else:
        print(f"阈值 14 没有符合条件的记录")
        # 创建空文件
        filtered_df_14.to_excel(output_file_14, index=False)
    
    backtest_results_14 = {
        'predictions': predictions_14,
        'classifications': classifications_14,
        'filtered_data': filtered_df_14,
        'count': len(filtered_df_14)
    }
    
    return backtest_results_14

def backtest_threshold_20(df):
    """
    回测模型，测试阈值20的模型，并将符合条件的值写入临时目录
    """
    # 创建临时目录（如果不存在）
    os.makedirs("temp", exist_ok=True)
    
    # 定义特征组合
    feature_combinations = {
        'Basic_Features': ['当日涨幅', '信号天数', '净额', '净流入', '当日资金流入']
    }
    
    print("\n=== 回测任务: 阈值20 (Threshold_20) ===")
    
    # 选择特征组合和目标
    feature_cols_20 = feature_combinations['Basic_Features']
    target_col_20 = '次日最高涨幅'
    task_type_20 = 'classification'
    threshold_20 = 20
    
    # 训练模型
    # results_20 = train_and_evaluate_models(df, target_col_20, feature_cols_20, task_type_20, threshold_20, return_models=True)

    # 获取测试数据
    X_test_20 = df[feature_cols_20].copy()
    
    # 处理分类特征
    categorical_cols_20 = X_test_20.select_dtypes(include=['object', 'category']).columns
    if not categorical_cols_20.empty:
        le_20 = LabelEncoder()
        for col in categorical_cols_20:
            X_test_20[col] = le_20.fit_transform(X_test_20[col].astype(str))
    
    # 使用RandomForest模型进行预测
    # model_20 = results_20['RandomForest']['model']
    model_20 = joblib.load("../models/random_forest_threshold_20_basic_features.pkl")

    print(len(X_test_20))
    # 进行预测
    predictions_20 = model_20['model'].predict(X_test_20)
    
    # 根据阈值创建分类结果
    classifications_20 = (predictions_20 > 0.5).astype(int)
    
    # 将预测结果添加到原始数据
    df_result_20 = df.copy()
    df_result_20['Prediction_20'] = predictions_20
    df_result_20['Classification_20'] = classifications_20
    
    # 筛选出分类结果为1的记录（即符合条件的记录）
    filtered_df_20 = df_result_20[df_result_20['Classification_20'] == 1]
    
    # 保存符合条件的记录到临时目录
    output_file_20 = "temp/backtest_results_threshold_20.xlsx"
    if not filtered_df_20.empty:
        filtered_df_20.to_excel(output_file_20, index=False)
        print(f"阈值 20 的回测结果已保存到: {output_file_20}")
        print(f"符合条件的记录数: {len(filtered_df_20)}")
        print("符合条件的前几条记录:")
        print(filtered_df_20[['日期', '代码', '当日涨幅', 'Prediction_20', 'Classification_20']].head())
    else:
        print(f"阈值 20 没有符合条件的记录")
        # 创建空文件
        filtered_df_20.to_excel(output_file_20, index=False)
    
    backtest_results_20 = {
        'predictions': predictions_20,
        'classifications': classifications_20,
        'filtered_data': filtered_df_20,
        'count': len(filtered_df_20)
    }
    
    return backtest_results_20

def backtest_models(df):
    """
    回测模型，分别测试阈值14和20的模型，并将符合条件的值写入临时目录
    """
    # 调用阈值14的回测
    results_14 = backtest_threshold_14(df)
    
    # 调用阈值20的回测

    results_20 = backtest_threshold_20(df)
    
    backtest_results = {
        'Threshold_14': results_14,
        'Threshold_20': results_20
    }
    
    return backtest_results

def main():
    """
    主函数
    """
    print("开始数据准备...")
    df = prepare_data()
    
    # 保存指定模型
    save_selected_models(df)
    
    # 定义多种特征列组合进行测试
    feature_combinations = {
        'Basic_Features': ['当日涨幅', '信号天数', '净额', '净流入', '当日资金流入'],
        'Price_and_Flow_Features': ['当日涨幅', '总金额', '净额', '净流入', '当日资金流入'],
        'Timing_Features':  ['当日涨幅', '量比', '总金额','信号天数', '净额', '净流入', '当日资金流入'],
        'Flow_Features': ['当日涨幅', '量比','总金额','信号天数','Q','净额', '净流入', '当日资金流入'],
        'last_Features': ['当日涨幅', '量比','总金额','信号天数','Q','band_width','净额', '净流入', '当日资金流入'],
        'All_Features': ['当日涨幅', '量比','总金额','信号天数','Q','band_width','min_value','max_value','净额', '净流入', '当日资金流入']
    }
    
    # 定义目标任务和阈值
    tasks = {
        'Next_Day_High_Increase_Classification_14': ('次日最高涨幅', 'classification', 14),
        'Next_Day_High_Increase_Classification_20': ('次日最高涨幅', 'classification', 20)
    }
    
    # 存储所有结果
    all_results = {}
    
    print("开始模型训练和评估...")
    for task_name, (target_col, task_type, threshold) in tasks.items():
        print(f"处理任务: {task_name}")
        task_results = {}
        for feature_name, feature_cols in feature_combinations.items():
            print(f"  测试特征组合: {feature_name}")
            # 为每个阈值训练模型
            results = train_and_evaluate_models(df, target_col, feature_cols, task_type, threshold)
            task_results[feature_name] = results

        all_results[task_name] = task_results
    
    print("比较模型性能...")
    # 比较模型性能
    df_performance = compare_models_performance(all_results)
    
    print("分析特征重要性...")
    # 分析特征重要性
    # 在main函数中显示特征重要性之前添加去重逻辑

    df_importance = analyze_feature_importance(all_results)
    # print(df_importance)
    # 生成报表文件名
    output_file = f"reports/model_comparison_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx"
    
    print("生成报表...")
    # 生成报表
    generate_report(df_performance, df_importance, output_file)
    
    # 打印结果摘要
    print("\n=== 模型性能比较 ===")
    print(df_performance.to_string(index=False))
    
    # 添加特征组合对准确率影响分析
    analyze_feature_impact_on_accuracy(df_performance)
    
    print("\n=== 特征重要性分析 ===")
    # 显示每个任务和模型中最重要的3个特征

    if not df_importance.empty:
        # 在main函数中显示特征重要性之前添加去重逻辑
        df_importance = df_importance.drop_duplicates(subset=['Task', 'Model', 'Feature'], keep='first')

        for task in df_importance['Task'].unique():
            print(f"\n任务: {task}")
            task_data = df_importance[df_importance['Task'] == task]
            for model in task_data['Model'].unique():
                print(f"  模型: {model}")
                model_data = task_data[task_data['Model'] == model].sort_values('Importance', ascending=False)
                for i, row in model_data.head(10).iterrows():
                    print(f"    {row['Feature']}: {row['Importance']:.4f}")
    else:
        print("无特征重要性数据")
    
    # 添加集成模型性能提升分析
    print("\n=== 集成模型性能提升分析 ===")
    for task in df_performance['Task'].unique():
        task_performance = df_performance[df_performance['Task'] == task]
        print(f"\n任务: {task}")
        for feature_comb in task_performance['Feature_Combination'].unique():
            feature_data = task_performance[task_performance['Feature_Combination'] == feature_comb]
            print(f"  特征组合: {feature_comb}")
            
            # 找到最佳基础模型
            if 'classification' in task.lower():
                best_base = feature_data.loc[feature_data['Accuracy'].idxmax()]
            else:
                best_base = feature_data.loc[feature_data['MAE'].idxmin()]
            
            # 检查集成模型性能
            voting_model = feature_data[feature_data['Model'] == 'VotingEnsemble']
            stacking_model = feature_data[feature_data['Model'] == 'StackingEnsemble']
            
            if not voting_model.empty:
                voting_model = voting_model.iloc[0]
                print(f"    投票集成模型性能:")
                if 'classification' in task.lower():
                    improvement = voting_model['Accuracy'] - best_base['Accuracy']
                    print(f"      Accuracy提升: {improvement:.4f}")
                else:
                    improvement = best_base['MAE'] - voting_model['MAE']
                    print(f"      MAE改善: {improvement:.4f}")
            
            if not stacking_model.empty:
                stacking_model = stacking_model.iloc[0]
                print(f"    堆叠集成模型性能:")
                if 'classification' in task.lower():
                    improvement = stacking_model['Accuracy'] - best_base['Accuracy']
                    print(f"      Accuracy提升: {improvement:.4f}")
                else:
                    improvement = best_base['MAE'] - stacking_model['MAE']
                    print(f"      MAE改善: {improvement:.4f}")
    
    return df_performance, df_importance



if __name__ == "__main__":
    # performance_df, importance_df = main()

    # df = prepare_data()
    # backtest_models(df)


    df = prepare_all_data()
    backtest_models(df)
    results_20 = backtest_threshold_20(df)


    # input_file = "../data/predictions/1000/08220954_1003.xlsx"
    # input_file = "../data/predictions/1200/08221132_1134.xlsx"
    # input_file = "../data/predictions/1400/08221404_1406.xlsx"
    # input_file = "../data/predictions/1600/08221505_1506.xlsx"
    # output_dir = "../data/predictions"
    # predict_with_models(input_file)