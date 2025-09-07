import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error, accuracy_score, r2_score, mean_squared_error
from sklearn.metrics import precision_score, recall_score, f1_score
import os
from datetime import datetime
import warnings
import joblib
# 添加XGBoost导入
import xgboost as xgb
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
warnings.filterwarnings('ignore')


# 请参考上面的程序新建一个程序 model_random_forest.py,实现一个随机森林的机器学习程序，要求如下： 1、读取某个目录下的所有文件，获取数据，参考已有方法； 2、用这些数据进行训练，生成一个模型进行保存，需要两种模型； 3、模型生成好了以后，可以指定读取文件，来调用这个模型进行推理，结果保存到temp目录下； 4、也可以读取指定目录下的文件，来调用这个模型； 5、要求提供模型的优化和检查的依据，可以进行调参； 6、写出测试方法，并可以运行；

# 导入原有数据读取函数
from model_xunlian import generate_model_data_from_files, get_prediction_files_data
from data_prepare import prepare_all_data,get_dir_files_date,get_dir_files,prepare_prediction_data


def prepare_data_from_directory(directory_path):
    """
    从指定目录读取所有数据文件并准备数据集
    """
    # 使用绝对路径
    base_dir = os.path.dirname(os.path.abspath(__file__))
    
    # 检查临时文件是否存在
    temp_file_path = os.path.join(base_dir, "..", "data", "bak", "model_data_rf.xlsx")
    if os.path.exists(temp_file_path):
        print("检测到临时文件，直接读取...")
        try:
            df = pd.read_excel(temp_file_path, engine='openpyxl')
            print(f'历史数据量：{len(df)}')
            return df
        except Exception as e:
            print(f"读取临时文件失败: {e}，重新生成数据...")
    
    # 读取目录下所有xlsx文件
    files = []
    for file in os.listdir(directory_path):
        if file.endswith('.xlsx'):
            files.append(os.path.join(directory_path, file))
    
    if not files:
        raise ValueError(f"目录 {directory_path} 中未找到xlsx文件")
    
    print(f"找到 {len(files)} 个数据文件")
    
    # 读取文件中的数据
    df = generate_model_data_from_files(files)
    print(f'历史数据量：{len(df)}')

    # 读取其他数据 每日整理的数据集
    predictions_dir = os.path.join(base_dir, "..", "data", "predictions")
    df_other = get_prediction_files_data(predictions_dir, '0730')
    
    if df_other is not None and not df_other.empty:
        print(f'预测数据量：{len(df_other)}')
        # 合并数据
        df = pd.concat([df, df_other], ignore_index=True)
    
    print(f'总数据量：{len(df)}')
    # 将df写入临时文件，供下次使用
    os.makedirs(os.path.dirname(temp_file_path), exist_ok=True)
    df.to_excel("../data/bak/model_data_rf.xlsx", index=False)
    return df

def optimize_model_parameters(X_train, y_train, model_type='regression', algorithm='random_forest'):
    """
    使用网格搜索优化模型参数
    """
    print("开始模型参数优化...")
    
    if algorithm == 'random_forest':
        if model_type == 'regression':
            model = RandomForestRegressor(random_state=42)
            param_grid = {
                'n_estimators': [50, 100, 200],
                'max_depth': [None, 10, 20, 30],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4]
            }
        else:
            model = RandomForestClassifier(random_state=42)
            param_grid = {
                'n_estimators': [50, 100, 200],
                'max_depth': [None, 10, 20, 30],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4]
            }
    elif algorithm == 'xgboost':
        if model_type == 'regression':
            model = xgb.XGBRegressor(random_state=42)
            param_grid = {
                'n_estimators': [50, 100, 200],
                'max_depth': [3, 6, 9],
                'learning_rate': [0.01, 0.1, 0.2],
                'subsample': [0.8, 1.0]
            }
        else:
            model = xgb.XGBClassifier(random_state=42)
            param_grid = {
                'n_estimators': [50, 100, 200],
                'max_depth': [3, 6, 9],
                'learning_rate': [0.01, 0.1, 0.2],
                'subsample': [0.8, 1.0]
            }
    
    # 网格搜索
    grid_search = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        cv=3,
        n_jobs=-1,
        verbose=1,
        scoring='accuracy' if model_type == 'classification' else 'neg_mean_absolute_error'
    )
    
    grid_search.fit(X_train, y_train)
    
    print(f"最佳参数: {grid_search.best_params_}")
    print(f"最佳得分: {grid_search.best_score_}")
    
    return grid_search.best_estimator_

def find_optimal_threshold(df, feature_cols, algorithm='random_forest', model_type='classification'):
    """
    寻找最优的threshold值
    """
    print("开始寻找最优threshold值...")
    
    # 数据预处理
    data = df.copy()
    
    # 移除目标列中的缺失值
    target_col = '次日最高涨幅'
    data = data.dropna(subset=[target_col])
    
    # 分离特征和目标
    X = data[feature_cols]
    y = data[target_col]
    
    # 清理数据中的无效值
    # 将 '--' 等无效字符串替换为 NaN
    X = X.replace(['--', 'None', 'null', ''], np.nan)
    y = y.replace(['--', 'None', 'null', ''], np.nan)
    
    # 移除包含 NaN 的行
    valid_indices = X.dropna().index.intersection(y.dropna().index)

    X = X.loc[valid_indices]
    y = y.loc[valid_indices]

    print(f"数据...{len(X)}，“列”数据量：{len(y)}")
    # 处理分类特征
    categorical_cols = X.select_dtypes(include=['object', 'category']).columns
    if not categorical_cols.empty:
        le = LabelEncoder()
        for col in categorical_cols:
            X[col] = le.fit_transform(X[col].astype(str))
    
    # 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)
    
    if model_type == 'classification':
        # 测试不同的threshold值
        thresholds = range(5, 15, 1)  # 从0到30，步长为1
        best_threshold = 0
        best_score = -1
        best_metrics = {}
        
        print(f"测试threshold值范围: {min(thresholds)} - {max(thresholds)}")
        
        results = []
        
        for threshold in thresholds:
            # 创建分类目标变量
            y_train_cls = (y_train > threshold).astype(int)
            y_test_cls = (y_test > threshold).astype(int)
            
            # 初始化模型
            if algorithm == 'random_forest':
                model = RandomForestClassifier(n_estimators=100, random_state=42)
            elif algorithm == 'xgboost':
                model = xgb.XGBClassifier(n_estimators=100, random_state=42)
            
            # 对模型进行超参数优化
            print(f"对threshold={threshold}进行模型调优...")
            optimized_model = optimize_model_parameters(X_train, y_train_cls, 'classification', algorithm)
            
            # 训练优化后的模型
            optimized_model.fit(X_train, y_train_cls)
            
            # 预测
            y_pred = optimized_model.predict(X_test)
            
            # 计算评估指标
            accuracy = accuracy_score(y_test_cls, y_pred)
            precision = precision_score(y_test_cls, y_pred, average='weighted', zero_division=0)
            recall = recall_score(y_test_cls, y_pred, average='weighted', zero_division=0)
            f1 = f1_score(y_test_cls, y_pred, average='weighted', zero_division=0)
            
            # 使用F1分数作为主要评估指标
            score = f1
            
            results.append({
                'threshold': threshold,
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1_score': f1
            })
            
            if score > best_score:
                best_score = score
                best_threshold = threshold
                best_metrics = {
                    'threshold': threshold,
                    'accuracy': accuracy,
                    'precision': precision,
                    'recall': recall,
                    'f1_score': f1
                }
            
            print(f"Threshold: {threshold:2d}, Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")
        
        print(f"\n最优threshold值: {best_threshold}")
        print(f"对应的评估指标: {best_metrics}")
        
        # 将结果保存到文件
        results_df = pd.DataFrame(results)
        os.makedirs("temp", exist_ok=True)
        results_df.to_excel(f"temp/threshold_optimization_results_{algorithm}_{model_type}.xlsx", index=False)
        print(f"threshold优化结果已保存至: temp/threshold_optimization_results_{algorithm}_{model_type}.xlsx")
        
        return best_threshold, best_metrics, results_df
    
    elif model_type == 'regression':
        # 对于回归模型，我们使用不同的评估方法
        # 我们将预测值与实际值的差异作为评估标准
        
        # 初始化模型
        if algorithm == 'random_forest':
            model = RandomForestRegressor(n_estimators=100, random_state=42)
        elif algorithm == 'xgboost':
            model = xgb.XGBRegressor(n_estimators=100, random_state=42)
        
        # 对模型进行超参数优化
        print("对回归模型进行调优...")
        optimized_model = optimize_model_parameters(X_train, y_train, 'regression', algorithm)
        
        # 训练优化后的模型
        optimized_model.fit(X_train, y_train)
        
        # 预测
        y_pred = optimized_model.predict(X_test)
        
        # 计算回归评估指标
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        
        metrics = {
            'MAE': mae,
            'R2': r2,
            'RMSE': rmse
        }
        
        print(f"回归模型评估结果: MAE={mae:.4f}, R2={r2:.4f}, RMSE={rmse:.4f}")
        
        # 创建结果数据框
        results_df = pd.DataFrame([{
            'MAE': mae,
            'R2': r2,
            'RMSE': rmse
        }])
        
        # 保存结果
        os.makedirs("temp", exist_ok=True)
        results_df.to_excel(f"temp/threshold_optimization_results_{algorithm}_{model_type}.xlsx", index=False)
        print(f"回归模型评估结果已保存至: temp/threshold_optimization_results_{algorithm}_{model_type}.xlsx")
        
        # 回归模型不需要threshold，返回默认值
        return None, metrics, results_df

def train_and_save_models(df, target_col, feature_cols, model_type='regression', threshold=7, algorithm='random_forest', date_suffix='', model_name=''):
    """
    训练随机森林模型并保存
    """
    # 数据预处理
    data = df.copy()
    
    # 移除目标列中的缺失值
    data = data.dropna(subset=[target_col])
    
    # 分离特征和目标
    X = data[feature_cols]
    y = data[target_col]
    
    # 清理数据中的无效值
    # 将 '--' 等无效字符串替换为 NaN
    X = X.replace(['--', 'None', 'null', ''], np.nan)
    y = y.replace(['--', 'None', 'null', ''], np.nan)
    
    # 移除包含 NaN 的行
    valid_indices = X.dropna().index.intersection(y.dropna().index)
    X = X.loc[valid_indices]
    y = y.loc[valid_indices]
    print(f"数据...{len(X)}，“列”数据量：{len(y)}")

    # 处理分类特征
    categorical_cols = X.select_dtypes(include=['object', 'category']).columns
    if not categorical_cols.empty:
        le = LabelEncoder()
        for col in categorical_cols:
            X[col] = le.fit_transform(X[col].astype(str))
    
    # 划分训练集和测试集
    # X_train, X_test, y_train, y_test = train_test_split(
    #     X, y, test_size=0.2, random_state=42)

    X_train = X
    y_train = y

    # 初始化模型
    if algorithm == 'random_forest':
        if model_type == 'regression':
            base_model = RandomForestRegressor(n_estimators=100, random_state=42)
        else:  # classification
            # 创建分类目标变量
            if target_col == '次日最高涨幅':
                y_train = (y_train > threshold).astype(int)
                # y_test = (y_test > threshold).astype(int)
            base_model = RandomForestClassifier(n_estimators=100, random_state=42)
    elif algorithm == 'xgboost':
        if model_type == 'regression':
            base_model = xgb.XGBRegressor(n_estimators=100, random_state=42)
        else:  # classification
            # 创建分类目标变量
            if target_col == '次日最高涨幅':
                y_train = (y_train > threshold).astype(int)
                # y_test = (y_test > threshold).astype(int)
            base_model = xgb.XGBClassifier(n_estimators=100, random_state=42)
    
    # 训练基础模型
    base_model.fit(X_train, y_train)

    # 创建模型目录
    os.makedirs("../models", exist_ok=True)
    y = (y > threshold).astype(int)
    base_model.fit(X, y)

    # 生成模型文件名
    model_identifier = model_name if model_name else f"{algorithm}_{model_type}_{threshold}"
    date_identifier = f"_{date_suffix}" if date_suffix else ""


    # 保存基础模型，添加日期后缀
    model_data_basic = {
        'model': base_model,
        'feature_cols': feature_cols,
        'label_encoder': le if not categorical_cols.empty else None,
        'categorical_cols': categorical_cols,
        'model_type': model_type,
        'threshold': threshold if model_type == 'classification' else None,
        # 'metrics': basic_metrics,
        'algorithm': algorithm,
        'date_suffix': date_suffix,
        'model_name': model_name
    }
    joblib.dump(model_data_basic, f"../models/{model_identifier}_basic_model{date_identifier}.pkl")
    print(f"基础{model_type}模型已保存: ../models/{model_identifier}_basic_model{date_identifier}.pkl")

    return model_data_basic
    # # 先不用优化模型，计算基础的结果
    # optimized_model, basic_metrics,optimized_metrics = optimize_and_predict(X_train,X_test, y_train, y_test,base_model,algorithm=['random_forest'], model_type='basic', )
    # optimized_model.fit(X, y)
    #
    # # 保存优化模型，添加日期后缀
    # model_data_optimized = {
    #     'model': optimized_model,
    #     'feature_cols': feature_cols,
    #     'label_encoder': le if not categorical_cols.empty else None,
    #     'categorical_cols': categorical_cols,
    #     'model_type': model_type,
    #     'threshold': threshold if model_type == 'classification' else None,
    #     'metrics': optimized_metrics,
    #     'algorithm': algorithm,
    #     'date_suffix': date_suffix,
    #     'model_name': model_name
    # }
    # joblib.dump(model_data_optimized, f"../models/{model_identifier}_optimized_model{date_identifier}.pkl")
    # print(f"优化{model_type}模型已保存")
    #
    # return model_data_basic, model_data_optimized

# 优化和预测、评估模型

def optimize_and_predict(X_train,X_test, y_train, y_test,base_model,algorithm=['random_forest'], model_type='basic', ):
    # 优化参数的模型
    optimized_model = optimize_model_parameters(X_train, y_train, model_type, algorithm)
    optimized_model.fit(X_train, y_train)

    # 预测
    basic_pred = base_model.predict(X_test)
    optimized_pred = optimized_model.predict(X_test)

    # 评估模型
    if model_type == 'regression':
        basic_metrics = {
            'MAE': mean_absolute_error(y_test, basic_pred),
            'R2': r2_score(y_test, basic_pred),
            'RMSE': np.sqrt(mean_squared_error(y_test, basic_pred))
        }

        optimized_metrics = {
            'MAE': mean_absolute_error(y_test, optimized_pred),
            'R2': r2_score(y_test, optimized_pred),
            'RMSE': np.sqrt(mean_squared_error(y_test, optimized_pred))
        }
    else:
        basic_metrics = {
            'Accuracy': accuracy_score(y_test, basic_pred),
            'Precision': precision_score(y_test, basic_pred, average='weighted', zero_division=0),
            'Recall': recall_score(y_test, basic_pred, average='weighted', zero_division=0),
            'F1_Score': f1_score(y_test, basic_pred, average='weighted', zero_division=0)
        }

        optimized_metrics = {
            'Accuracy': accuracy_score(y_test, optimized_pred),
            'Precision': precision_score(y_test, optimized_pred, average='weighted', zero_division=0),
            'Recall': recall_score(y_test, optimized_pred, average='weighted', zero_division=0),
            'F1_Score': f1_score(y_test, optimized_pred, average='weighted', zero_division=0)
        }

    print(f"基础模型评估结果: {basic_metrics}")
    print(f"优化模型评估结果: {optimized_metrics}")

    return optimized_model, basic_metrics,optimized_metrics

def predict_with_saved_models(file_path, output_path=None, algorithms=['random_forest'], model='basic', date_suffix=''):
    """
    使用保存的模型进行预测 model_type='basic' or 'optimized'
    """
    # 判断文件是否存在
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"文件 {file_path} 不存在")
    # 读取输入文件
    df = pd.read_excel(file_path, engine='openpyxl')
    
    # 创建temp目录，使用绝对路径
    base_dir = os.path.dirname(os.path.abspath(__file__))
    temp_dir = os.path.join(base_dir, "temp")
    os.makedirs(temp_dir, exist_ok=True)
    
    # 存储所有预测结果
    results = df.copy()
    
    # 遍历所有算法
    for algorithm in algorithms:
        # 加载模型，使用日期后缀
        model_dir = os.path.join(base_dir, "..", "models")
        if algorithm == 'random_forest':
            regression_model = joblib.load(os.path.join(model_dir, f"{algorithm}_regression_{model}_model_{date_suffix}.pkl"))
            classification_model = joblib.load(os.path.join(model_dir, f"{algorithm}_classification_{model}_model_{date_suffix}.pkl"))
        elif algorithm == 'xgboost':
            regression_model = joblib.load(os.path.join(model_dir, f"xgboost_regression_{model}_model_{date_suffix}.pkl"))
            classification_model = joblib.load(os.path.join(model_dir, f"xgboost_classification_{model}_model_{date_suffix}.pkl"))
        
        # 准备回归模型特征数据
        X_reg = df[regression_model['feature_cols']].copy()
        if regression_model['label_encoder'] is not None and not regression_model['categorical_cols'].empty:
            for col in regression_model['categorical_cols']:
                if col in X_reg.columns:
                    X_reg[col] = regression_model['label_encoder'].fit_transform(X_reg[col].astype(str))
        
        # 准备分类模型特征数据
        X_cls = df[classification_model['feature_cols']].copy()
        if classification_model['label_encoder'] is not None and not classification_model['categorical_cols'].empty:
            for col in classification_model['categorical_cols']:
                if col in X_cls.columns:
                    X_cls[col] = classification_model['label_encoder'].fit_transform(X_cls[col].astype(str))
        
        # 进行预测
        reg_predictions = regression_model['model'].predict(X_reg)
        cls_predictions = classification_model['model'].predict(X_cls)
        
        # 添加预测结果到数据框，使用算法名称作为前缀
        results[f'{algorithm}_Reg'] = reg_predictions
        results[f'{algorithm}_Cf'] = cls_predictions

        # 打印出预测结果为1的行
        print(f"预测结果为1的行：{model} {algorithm}")
        print(results[results[f'{algorithm}_Cf'] == 1])
        
        # 打印出预测结果大于14的行
        print(f"预测结果大于14的行：{model} {algorithm}")
        print(results[results[f'{algorithm}_Reg'] > 14])
    
    # 保存结果
    if output_path is None:
        # 文件名改为输入文件名加上_with_results.xlsx，加上目录 temp,只取文件名，增加传入模型的参数
        file_name = os.path.basename(file_path)
        # 文件名改为输入文件名加上_with_results.xlsx
        # 修改文件名以包含所有算法名称
        algorithms_str = '_'.join(algorithms)
        output_path = os.path.join(temp_dir, f'{file_name}_{algorithms_str}_{model}_{date_suffix}.xlsx')

    results.to_excel(output_path, index=False)
    print(f"预测结果已保存至: {output_path}")
    
    return results, output_path

def predict_from_directory(directory_path, algorithms=['random_forest'], date_suffix=''):
    """
    读取指定目录下的所有文件并进行预测
    """
    # 创建结果目录，使用绝对路径
    base_dir = os.path.dirname(os.path.abspath(__file__))
    results_dir = os.path.join(base_dir, "temp", "directory_predictions")
    os.makedirs(results_dir, exist_ok=True)
    
    # 读取目录下所有xlsx文件
    files = []
    for file in os.listdir(directory_path):
        if file.endswith('.xlsx'):
            files.append(os.path.join(directory_path, file))
    
    if not files:
        raise ValueError(f"目录 {directory_path} 中未找到xlsx文件")
    
    print(f"找到 {len(files)} 个数据文件进行预测")
    
    results = []
    for file_path in files:
        try:
            print(f"处理文件: {file_path}")
            result = predict_with_saved_models(file_path, 
                                             os.path.join(results_dir, os.path.basename(file_path)),
                                             algorithms, date_suffix=date_suffix)
            results.append(result)
        except Exception as e:
            print(f"处理文件 {file_path} 时出错: {e}")
    
    print(f"已完成 {len(results)} 个文件的预测")
    return results

def model_evaluation_report(model_data, X_test, y_test, model_name):
    """
    生成模型评估报告
    """
    model = model_data['model']
    predictions = model.predict(X_test)
    
    # 创建评估报告
    report = {
        'Model_Name': model_name,
        'Number_of_Features': len(model_data['feature_cols']),
        'Number_of_Samples': len(y_test)
    }
    
    if model_data['model_type'] == 'regression':
        report.update({
            'MAE': mean_absolute_error(y_test, predictions),
            'R2': r2_score(y_test, predictions),
            'RMSE': np.sqrt(mean_squared_error(y_test, predictions))
        })
    else:
        report.update({
            'Accuracy': accuracy_score(y_test, predictions),
            'Precision': precision_score(y_test, predictions, average='weighted', zero_division=0),
            'Recall': recall_score(y_test, predictions, average='weighted', zero_division=0),
            'F1_Score': f1_score(y_test, predictions, average='weighted', zero_division=0)
        })
    
    return report

def validate_model_accuracy(df, target_col, feature_cols, model_type='regression', threshold=7, algorithm='random_forest', date_suffix=''):
    """
    使用后续数据验证模型准确率
    """
    # 数据预处理
    data = df.copy()
    
    # 移除目标列中的缺失值
    data = data.dropna(subset=[target_col])
    
    # 分离特征和目标
    X = data[feature_cols]
    y = data[target_col]
    
    # 清理数据中的无效值
    X = X.replace(['--', 'None', 'null', ''], np.nan)
    y = y.replace(['--', 'None', 'null', ''], np.nan)
    
    # 移除包含 NaN 的行
    valid_indices = X.dropna().index.intersection(y.dropna().index)
    X = X.loc[valid_indices]
    y = y.loc[valid_indices]
    
    # 处理分类特征
    categorical_cols = X.select_dtypes(include=['object', 'category']).columns
    if not categorical_cols.empty:
        le = LabelEncoder()
        for col in categorical_cols:
            X[col] = le.fit_transform(X[col].astype(str))
    
    # 加载模型
    base_dir = os.path.dirname(os.path.abspath(__file__))
    model_dir = os.path.join(base_dir, "..", "models")
    
    if model_type == 'regression':
        model_path = os.path.join(model_dir, f"{algorithm}_{model_type}_basic_model_{date_suffix}.pkl")
    else:
        model_path = os.path.join(model_dir, f"{algorithm}_{model_type}_basic_model_{date_suffix}.pkl")
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"模型文件 {model_path} 不存在")
    
    model_data = joblib.load(model_path)
    model = model_data['model']
    
    # 预测
    predictions = model.predict(X)
    
    # 评估模型
    if model_type == 'regression':
        metrics = {
            'MAE': mean_absolute_error(y, predictions),
            'R2': r2_score(y, predictions),
            'RMSE': np.sqrt(mean_squared_error(y, predictions))
        }
    else:
        # 创建分类目标变量
        y_cls = (y > threshold).astype(int)
        predictions_cls = (predictions > 0.5).astype(int)  # 假设分类模型输出概率
        metrics = {
            'Accuracy': accuracy_score(y_cls, predictions_cls),
            'Precision': precision_score(y_cls, predictions_cls, average='weighted', zero_division=0),
            'Recall': recall_score(y_cls, predictions_cls, average='weighted', zero_division=0),
            'F1_Score': f1_score(y_cls, predictions_cls, average='weighted', zero_division=0)
        }
    
    return metrics

def create_accuracy_charts(accuracy_data):
    """
    创建准确率图表
    """
    # 创建子图
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('准确率对比', '精确率对比', '召回率对比', 'F1分数对比'),
        specs=[[{"type": "bar"}, {"type": "bar"}],
               [{"type": "bar"}, {"type": "bar"}]]
    )
    
    # 提取数据
    algorithms = list(accuracy_data.keys())
    accuracy_values = [accuracy_data[alg]['Accuracy'] for alg in algorithms]
    precision_values = [accuracy_data[alg]['Precision'] for alg in algorithms]
    recall_values = [accuracy_data[alg]['Recall'] for alg in algorithms]
    f1_values = [accuracy_data[alg]['F1_Score'] for alg in algorithms]
    
    # 添加图表
    fig.add_trace(
        go.Bar(x=algorithms, y=accuracy_values, name='准确率'),
        row=1, col=1
    )
    
    fig.add_trace(
        go.Bar(x=algorithms, y=precision_values, name='精确率'),
        row=1, col=2
    )
    
    fig.add_trace(
        go.Bar(x=algorithms, y=recall_values, name='召回率'),
        row=2, col=1
    )
    
    fig.add_trace(
        go.Bar(x=algorithms, y=f1_values, name='F1分数'),
        row=2, col=2
    )
    
    fig.update_layout(
        title_text="模型准确率对比",
        showlegend=False,
        height=600
    )
    
    return fig

def create_date_comparison_chart(accuracy_history):
    """
    创建不同日期模型准确率对比图表
    """
    # 创建子图
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('准确率趋势', '精确率趋势', '召回率趋势', 'F1分数趋势'),
        specs=[[{"type": "scatter"}, {"type": "scatter"}],
               [{"type": "scatter"}, {"type": "scatter"}]]
    )
    
    # 为每个算法绘制趋势线
    for algorithm in accuracy_history:
        dates = list(accuracy_history[algorithm].keys())
        accuracy_values = [accuracy_history[algorithm][date]['Accuracy'] for date in dates]
        precision_values = [accuracy_history[algorithm][date]['Precision'] for date in dates]
        recall_values = [accuracy_history[algorithm][date]['Recall'] for date in dates]
        f1_values = [accuracy_history[algorithm][date]['F1_Score'] for date in dates]
        
        # 转换日期格式
        date_labels = [str(date) for date in dates]
        
        fig.add_trace(
            go.Scatter(x=date_labels, y=accuracy_values, mode='lines+markers', name=f'{algorithm}-准确率'),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Scatter(x=date_labels, y=precision_values, mode='lines+markers', name=f'{algorithm}-精确率'),
            row=1, col=2
        )
        
        fig.add_trace(
            go.Scatter(x=date_labels, y=recall_values, mode='lines+markers', name=f'{algorithm}-召回率'),
            row=2, col=1
        )
        
        fig.add_trace(
            go.Scatter(x=date_labels, y=f1_values, mode='lines+markers', name=f'{algorithm}-F1分数'),
            row=2, col=2
        )
    
    fig.update_layout(
        title_text="不同日期模型准确率趋势对比",
        showlegend=True,
        height=600
    )
    
    return fig

# def test_model_functionality():
#     """
#     测试模型功能
#     """
#     print("=== 开始测试模型功能 ===")
#
#     # 1. 准备数据
#     print("1. 准备数据...")
#     # 确保测试目录存在
#     test_data_dir = "../alert/"
#     os.makedirs(test_data_dir, exist_ok=True)
#
#     # 检查目录中是否有数据文件，如果没有则创建示例数据
#     data_files = [f for f in os.listdir(test_data_dir) if f.endswith('.xlsx')]
#     if not data_files:
#         print("未找到测试数据文件，创建示例数据...")
#         # 创建示例数据
#         sample_data = pd.DataFrame({
#             '当日涨幅': np.random.randn(200),
#             '量比': np.random.randn(200),
#             '总金额': np.random.randn(200),
#             '信号天数': np.random.randint(1, 10, 200),
#             'Q': np.random.randn(200),
#             'band_width': np.random.randn(200),
#             'min_value': np.random.randn(200),
#             'max_value': np.random.randn(200),
#             '净额': np.random.randn(200),
#             '净流入': np.random.randn(200),
#             '当日资金流入': np.random.randn(200),
#             '次日涨幅': np.random.randn(200),
#             '次日最高涨幅': np.random.randint(0, 2, 200) * 30
#         })
#         sample_data.to_excel(os.path.join(test_data_dir, "sample_data.xlsx"), index=False)
#
#     df = prepare_data_from_directory("../alert/")
#
#     # 2. 定义特征和目标
#     feature_combinations = {
#         'Basic_Features': ['当日涨幅', '信号天数', '净额', '净流入', '当日资金流入'],
#         'All_Features': ['当日涨幅', '量比','总金额', '信号天数','Q','band_width','min_value','max_value','净额', '净流入', '当日资金流入']
#     }
#
#     target_col_reg = '次日最高涨幅'
#     target_col_cls = '次日最高涨幅'
#
#     # 3. 训练回归模型
#     print("2. 训练回归模型...")
#     reg_basic, reg_opt = train_and_save_models(
#         df, target_col_reg, feature_combinations['All_Features'], 'regression'
#     )
#
#     # 4. 训练分类模型
#     print("3. 训练分类模型...")
#     cls_basic, cls_opt = train_and_save_models(
#         df, target_col_cls, feature_combinations['Basic_Features'], 'classification', threshold=14
#     )
#
#     # 5. 测试单文件预测
#     print("4. 测试单文件预测...")
#     # 创建测试数据文件
#     test_df = df.head(100)  # 取前100行作为测试数据
#     test_file = "temp/test_data.xlsx"
#     test_df.to_excel(test_file, index=False)
#
#     result = predict_with_saved_models(test_file)
#     print(f"单文件预测完成，结果行数: {len(result)}")
#
#     # 6. 测试目录预测
#     print("5. 测试目录预测...")
#     os.makedirs("temp/test_directory", exist_ok=True)
#     test_df.head(50).to_excel("temp/test_directory/test1.xlsx", index=False)
#     test_df.tail(50).to_excel("temp/test_directory/test2.xlsx", index=False)
#
#     dir_results = predict_from_directory("temp/test_directory")
#     print(f"目录预测完成，处理文件数: {len(dir_results)}")
#
#     print("=== 模型功能测试完成 ===")

def save_model(threshold,feature_cols,date_suffix='0827'):
    """
    主函数
    """
    # 运行测试

    # 如果需要实际使用，可以取消下面的注释
    df = prepare_all_data(date_suffix)
    if df is None:
        return

    # 训练并保存模型 - 随机森林
    if feature_cols is not None:
        feature_cols = feature_cols
    else:
        # feature_cols = None
        # feature_cols = ['当日涨幅', '信号天数', '净额', '净流入', '当日资金流入']
        feature_cols =  ['当日涨幅', '量比','总金额','信号天数','Q','band_width','净额', '净流入', '当日资金流入']


    # 定义多种特征列组合进行测试
    feature_combinations = {
        'Basic_Features': ['当日涨幅', '信号天数', '净额', '净流入', '当日资金流入'],
        'Price_and_Flow_Features': ['当日涨幅', '总金额', '净额', '净流入', '当日资金流入'],
        'Timing_Features':  ['当日涨幅', '量比', '总金额','信号天数', '净额', '净流入', '当日资金流入'],
        'Flow_Features': ['当日涨幅', '量比','总金额','信号天数','Q','净额', '净流入', '当日资金流入'],
        'last_Features': ['当日涨幅', '量比','总金额','信号天数','Q','band_width','净额', '净流入', '当日资金流入'],
        'All_Features': ['当日涨幅', '量比','总金额','信号天数','Q','band_width','min_value','max_value','净额', '净流入', '当日资金流入']
    }

    # 模型调优和参数调优
    # find_optimal_threshold(df, feature_cols, algorithm='random_forest', model_type='classification')
    # find_optimal_threshold(df, feature_cols, algorithm='xgboost', model_type='classification')

    # find_optimal_threshold(df, feature_cols, algorithm='xgboost', model_type='regression')
    # find_optimal_threshold(df, feature_cols, algorithm='random_forest', model_type='regression')


# def train_and_save_models(df, target_col, feature_cols, model_type='regression', threshold=7, algorithm='random_forest', date_suffix='', model_name=''):

    train_and_save_models(df, '次日最高涨幅', feature_cols, 'regression', threshold, algorithm='random_forest', date_suffix=date_suffix)
    train_and_save_models(df, '次日最高涨幅', feature_cols, 'classification', threshold,algorithm='random_forest', date_suffix=date_suffix)
    #
    # # 训练并保存模型 - XGBoost
    train_and_save_models(df, '次日最高涨幅', feature_cols, 'regression',threshold, algorithm='xgboost', date_suffix=date_suffix)
    train_and_save_models(df, '次日最高涨幅', feature_cols, 'classification',threshold, algorithm='xgboost', date_suffix=date_suffix)
    #
    # # 使用模型进行预测
    # predict_with_saved_models("../data/predictions/1000/08250950_0952.xlsx", algorithms=['random_forest','xgboost'])


def main():

    dates= get_dir_files_date("../data/predictions/1600/",start_md="0717",end_mmdd="0905")
    print(dates)
    if len(dates) > 0:

        # feature_cols=['当日涨幅', '量比','总金额','信号天数','Q','band_width','min_value','max_value','净额', '净流入', '当日资金流入']
        feature_cols=['当日涨幅', '信号天数', '净额', '净流入', '当日资金流入']

        for date in dates:
            # threshold 从7到14循环
            for threshold in range(7,14):
                save_model(threshold,feature_cols,date)
                # save_model(threshold=20,feature_cols=feature_cols,date_suffix=date)

            # exit()

# 得到文件列表，使用所有符合调价你的模型进行推测
#     files = get_dir_files("../data/predictions/1600/")
#     for date in dates:
#         for file in files:
#             if date in file:
#                 # predict_with_saved_models(file)
#                 predict_from_directory(file,date)

    # save_model(threshold=20,feature_cols=['当日涨幅', '量比','总金额','信号天数','Q','band_width','min_value','max_value','净额', '净流入', '当日资金流入'],date_suffix='0827')


def streamlit_app():
    """
    Streamlit应用界面
     streamlit run F:/project/trade/dataanalysis/stock/model_train_random_forest.py
    """
    st.set_page_config(page_title="股票预测模型", layout="wide")
    st.title("📈 股票预测模型系统")
    
    # 侧边栏
    st.sidebar.header("模型操作")
    operation = st.sidebar.radio("选择操作", ["模型训练", "模型预测", "模型验证", "查看模型信息"])
    
    if operation == "模型训练":
        st.header("模型训练")
        
        # 数据准备
        if st.button("准备训练数据"):
            with st.spinner("正在准备数据..."):
                try:
                    df = prepare_all_data("0827")
                    st.success(f"数据准备完成，共 {len(df)} 条记录")
                    st.session_state['train_data'] = df
                except Exception as e:
                    st.error(f"数据准备失败: {e}")
        
        # 特征选择
        feature_cols = ['当日涨幅', '量比','总金额','信号天数','Q','band_width','净额', '净流入', '当日资金流入']
        
        # 模型训练参数
        date_suffix = st.text_input("模型日期后缀", "0827")
        threshold = st.number_input("分类阈值", min_value=0, max_value=30, value=7)
        
        # 模型训练
        if st.button("训练模型") and 'train_data' in st.session_state:
            with st.spinner("正在训练模型..."):
                try:
                    df = st.session_state['train_data']
                    
                    # 训练随机森林模型
                    rf_reg_basic, rf_reg_opt = train_and_save_models(
                        df, '次日最高涨幅', feature_cols, 'regression', 
                        algorithm='random_forest', date_suffix=date_suffix)
                    rf_cls_basic, rf_cls_opt = train_and_save_models(
                        df, '次日最高涨幅', feature_cols, 'classification', 
                        algorithm='random_forest', date_suffix=date_suffix, threshold=threshold)
                    
                    # 训练XGBoost模型
                    xgb_reg_basic, xgb_reg_opt = train_and_save_models(
                        df, '次日最高涨幅', feature_cols, 'regression', 
                        algorithm='xgboost', date_suffix=date_suffix)
                    xgb_cls_basic, xgb_cls_opt = train_and_save_models(
                        df, '次日最高涨幅', feature_cols, 'classification', 
                        algorithm='xgboost', date_suffix=date_suffix, threshold=threshold)
                    
                    st.success("所有模型训练完成并已保存！")
                    
                    # 显示模型评估结果
                    st.subheader("模型评估结果")
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.write("随机森林回归模型 (基础):", rf_reg_basic['metrics'])
                        st.write("随机森林分类模型 (基础):", rf_cls_basic['metrics'])
                        st.write("随机森林回归模型 (优化):", rf_reg_opt['metrics'])
                        st.write("随机森林分类模型 (优化):", rf_cls_opt['metrics'])
                    
                    with col2:
                        st.write("XGBoost回归模型 (基础):", xgb_reg_basic['metrics'])
                        st.write("XGBoost分类模型 (基础):", xgb_cls_basic['metrics'])
                        st.write("XGBoost回归模型 (优化):", xgb_reg_opt['metrics'])
                        st.write("XGBoost分类模型 (优化):", xgb_cls_opt['metrics'])
                        
                except Exception as e:
                    st.error(f"模型训练失败: {e}")
    
    elif operation == "模型预测":
        st.header("模型预测")
        
        # 文件上传
        uploaded_file = st.file_uploader("上传预测数据文件 (Excel)", type=['xlsx'])
        
        if uploaded_file is not None:
            # 保存上传的文件
            with open(f"temp/uploaded_file.xlsx", "wb") as f:
                f.write(uploaded_file.getbuffer())
            
            st.success("文件上传成功！")
            
            # 选择模型类型
            algorithms = st.multiselect("选择模型算法", ['random_forest', 'xgboost'], default=['random_forest'])
            model_type = st.selectbox("选择模型版本", ['basic', 'optimized'])
            date_suffix = st.text_input("模型日期后缀", "0827")
            
            # 执行预测
            if st.button("开始预测"):
                with st.spinner("正在执行预测..."):
                    try:
                        result, output_path = predict_with_saved_models(
                            "temp/uploaded_file.xlsx",
                            algorithms=algorithms,
                            model=model_type,
                            date_suffix=date_suffix
                        )
                        
                        st.success("预测完成！")
                        
                        # 显示预测结果
                        st.subheader("预测结果")
                        st.dataframe(result)
                        
                        # 提供下载
                        output_file = f"temp/prediction_result.xlsx"
                        result.to_excel(output_file, index=False)
                        
                        with open(output_file, "rb") as file:
                            st.download_button(
                                label="下载预测结果",
                                data=file,
                                file_name="prediction_result.xlsx",
                                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                            )
                            
                        # 显示预测统计
                        st.subheader("预测统计")
                        for algorithm in algorithms:
                            reg_col = f'{algorithm}_Reg'
                            cls_col = f'{algorithm}_Cf'
                            
                            if reg_col in result.columns:
                                st.write(f"{algorithm} 回归预测 - 平均值: {result[reg_col].mean():.2f}")
                            
                            if cls_col in result.columns:
                                positive_count = (result[cls_col] == 1).sum()
                                st.write(f"{algorithm} 分类预测 - 正向预测数量: {positive_count}")
                                
                    except Exception as e:
                        st.error(f"预测失败: {e}")
    
    elif operation == "模型验证":
        st.header("模型验证")
        
        # 选择验证数据
        validation_date = st.text_input("验证数据日期", "0827")
        
        if st.button("验证模型准确率"):
            with st.spinner("正在验证模型准确率..."):
                try:
                    # 准备验证数据
                    df_validation = prepare_all_data(validation_date)
                    feature_cols = ['当日涨幅', '量比','总金额','信号天数','Q','band_width','净额', '净流入', '当日资金流入']
                    
                    # 验证不同模型的准确率
                    accuracy_data = {}
                    
                    algorithms = ['random_forest', 'xgboost']
                    for algorithm in algorithms:
                        # 验证分类模型
                        cls_metrics = validate_model_accuracy(
                            df_validation, '次日最高涨幅', feature_cols, 
                            'classification', threshold=7, algorithm=algorithm, 
                            date_suffix=validation_date)
                        accuracy_data[algorithm] = cls_metrics
                    
                    # 显示准确率结果
                    st.subheader("模型准确率结果")
                    accuracy_df = pd.DataFrame(accuracy_data).T
                    st.dataframe(accuracy_df)
                    
                    # 创建准确率对比图表
                    st.subheader("准确率对比图表")
                    fig = create_accuracy_charts(accuracy_data)
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # 保存验证结果
                    os.makedirs("temp", exist_ok=True)
                    accuracy_df.to_excel(f"temp/model_validation_{validation_date}.xlsx")
                    st.success(f"验证结果已保存至: temp/model_validation_{validation_date}.xlsx")
                    
                except Exception as e:
                    st.error(f"模型验证失败: {e}")
    
    elif operation == "查看模型信息":
        st.header("模型信息")
        
        # 显示已保存的模型
        model_dir = "../models"
        if os.path.exists(model_dir):
            models = os.listdir(model_dir)
            if models:
                st.write("已保存的模型:")
                for model in models:
                    st.write(f"- {model}")
            else:
                st.info("暂无已保存的模型")
        else:
            st.info("模型目录不存在")

if __name__ == "__main__":
    # 检查是否在Streamlit环境中运行
    # 修复: 使用更兼容的方法检测Streamlit环境
    try:
        # 新的检测方法：检查是否在Streamlit中运行
        is_streamlit_run = (
            "STREAMLIT_RUN" in os.environ or 
            any("streamlit" in arg for arg in os.sys.argv) or
            os.environ.get("IS_STREAMLIT", False) or
            # 添加对_streamlit_run属性的检查，这是Streamlit 1.0+的标识
            (hasattr(st, '_is_running_with_streamlit') and st._is_running_with_streamlit)
        )
    except:
        is_streamlit_run = False
    
    if is_streamlit_run:
        streamlit_app()
    else:
        # 原始命令行模式
        # model='basic' or 'optimized'
        main()
        # predict_with_saved_models("../data/predictions/1000/08250950_0952.xlsx", algorithms=['random_forest','xgboost'],model='optimized')
        # predict_with_saved_models("../data/predictions/1200/08251134_1135.xlsx", algorithms=['random_forest','xgboost'],model='optimized')
        # predict_with_saved_models("../data/predictions/1400/08251421_1422.xlsx", algorithms=['random_forest','xgboost'],model='optimized')
        # predict_with_saved_models("../data/predictions/1600/08251518_1520.xlsx", algorithms=['random_forest','xgboost'],model='optimized')

        # predict_with_saved_models("../data/predictions/1000/08260955_0957.xlsx", algorithms=['random_forest','xgboost'],model='optimized')

        # 检查文件是否存在，如果不存在则给出提示
        # base_dir = os.path.dirname(os.path.abspath(__file__))
        # predict_file = os.path.join(base_dir, "..", "data", "predictions", "1600", "08291641_1643.xlsx")
        # if os.path.exists(predict_file):
        #     predict_with_saved_models(predict_file, algorithms=['random_forest','xgboost'],model='optimized', date_suffix='0827')
        # else:
        #     print(f"警告: 预测文件 {predict_file} 不存在，请检查路径或先生成预测数据")
        #     # 列出目录中可用的文件供参考
        #     predict_dir = os.path.join(base_dir, "..", "data", "predictions", "1600")
        #     if os.path.exists(predict_dir):
        #         available_files = [f for f in os.listdir(predict_dir) if f.endswith('.xlsx')]
        #         if available_files:
        #             print(f"在 {predict_dir} 目录中找到以下可用文件:")
        #             for f in available_files:
        #                 print(f"  - {f}")
        #         else:
        #             print(f"在 {predict_dir} 目录中未找到任何Excel文件")
        #     else:
        #         print(f"目录 {predict_dir} 不存在")

# 针对模型进行测试和评估，需要知道模型的所有数据结果，以及模型预测结果。首先，生成该参数threshold某个值 自0812日后的的所有的模型；然后针对该日期以后的数据进行推演，将预测结果和正确结果进行比较，并保存结果到文件中。同时评估模型的准确性
def evaluate_model(model, X_test, y_test):
    """
    评估模型性能
    """
    # 进行预测
    y_pred = model.predict(X_test)
    
    # 计算评估指标
    if hasattr(model, 'predict_proba'):
        # 如果是分类模型，计算分类指标
        y_pred_cls = (y_pred > 0.5).astype(int) if model.__class__.__name__ in ['RandomForestClassifier', 'XGBClassifier'] else y_pred
        accuracy = accuracy_score(y_test, y_pred_cls)
        precision = precision_score(y_test, y_pred_cls, average='weighted', zero_division=0)
        recall = recall_score(y_test, y_pred_cls, average='weighted', zero_division=0)
        f1 = f1_score(y_test, y_pred_cls, average='weighted', zero_division=0)
        
        metrics = {
            'Accuracy': accuracy,
            'Precision': precision,
            'Recall': recall,
            'F1_Score': f1
        }
    else:
        # 如果是回归模型，计算回归指标
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        
        metrics = {
            'MAE': mae,
            'R2': r2,
            'RMSE': rmse
        }
    
    # 创建结果数据框，包含预测值和实际值
    results_df = pd.DataFrame({
        'y_true': y_test,
        'y_pred': y_pred
    })
    
    # 如果是分类任务，添加分类预测结果
    if hasattr(model, 'predict_proba'):
        results_df['y_pred_class'] = y_pred_cls
    
    return metrics, results_df
