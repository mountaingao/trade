import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error, accuracy_score, r2_score, mean_squared_error
import xgboost as xgb
import os
from datetime import datetime
import warnings
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
'''

def prepare_data():
    """
    准备数据集，包括历史数据和预测数据
    """
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
    df = df[['日期', '代码', '当日涨幅', '信号天数', '净额', '净流入', '当日资金流入', '次日涨幅', '次日最高涨幅']]

    # 读取其他数据 每日整理的数据集
    df_other = get_prediction_files_data()
    
    if df_other is not None and not df_other.empty:
        print(f'预测数据量：{len(df_other)}')
        # 提取有效字段
        df_other = df_other[['日期', '代码', '当日涨幅', '信号天数', '净额', '净流入', '当日资金流入', '次日涨幅', '次日最高涨幅']]
        # 合并数据
        df = pd.concat([df, df_other], ignore_index=True)
    
    print(f'总数据量：{len(df)}')
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
        return {'Accuracy': accuracy}

def train_and_evaluate_models(df, target_col, feature_cols, model_type='regression'):
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
    else:  # classification
        # 创建分类目标变量
        if target_col == '次日最高涨幅':
            y_train = (y_train > 7).astype(int)
            y_test = (y_test > 7).astype(int)
        rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
        xgb_model = xgb.XGBClassifier(random_state=42)
    
    # 训练模型
    rf_model.fit(X_train, y_train)
    xgb_model.fit(X_train, y_train)
    
    # 预测
    rf_pred = rf_model.predict(X_test)
    xgb_pred = xgb_model.predict(X_test)
    
    # 评估模型
    rf_metrics = evaluate_model_performance(y_test, rf_pred, model_type)
    xgb_metrics = evaluate_model_performance(y_test, xgb_pred, model_type)
    
    # 获取特征重要性
    rf_importance = rf_model.feature_importances_
    xgb_importance = xgb_model.feature_importances_
    
    # 整理结果
    results = {
        'RandomForest': {
            'model': rf_model,
            'predictions': rf_pred,
            'metrics': rf_metrics,
            'feature_importance': dict(zip(feature_cols, rf_importance))
        },
        'XGBoost': {
            'model': xgb_model,
            'predictions': xgb_pred,
            'metrics': xgb_metrics,
            'feature_importance': dict(zip(feature_cols, xgb_importance))
        },
        'test_data': {
            'X_test': X_test,
            'y_test': y_test
        }
    }
    # print(results)
    return results

def compare_models_performance(results_dict):
    """
    比较不同模型的性能
    """
    comparison_data = []
    
    for task, results in results_dict.items():
        for model_name, model_results in results.items():
            if model_name in ['RandomForest', 'XGBoost']:
                row = {'Task': task, 'Model': model_name}
                row.update(model_results['metrics'])
                comparison_data.append(row)
    
    return pd.DataFrame(comparison_data)

def analyze_feature_importance(results_dict):
    """
    分析特征重要性
    """
    importance_data = []
    
    for task, results in results_dict.items():
        for model_name, model_results in results.items():
            if model_name in ['RandomForest', 'XGBoost']:
                for feature, importance in model_results['feature_importance'].items():
                    importance_data.append({
                        'Task': task,
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
        summary_stats = df_results.groupby(['Task', 'Model']).mean().round(4)
        summary_stats.to_excel(writer, sheet_name='Summary_Stats')
    
    print(f"报表已保存至: {output_file}")
    return output_file

def main():
    """
    主函数
    """
    print("开始数据准备...")
    df = prepare_data()
    
    # 定义多种特征列组合进行测试
    feature_combinations = {
        'Basic_Features': ['当日涨幅', '信号天数', '净额', '净流入', '当日资金流入'],
        'Price_and_Flow_Features': ['当日涨幅', '净额', '净流入', '当日资金流入'],
        'Timing_Features': ['信号天数', '当日涨幅'],
        'Flow_Features': ['净额', '净流入'],
        'All_Features': ['当日涨幅', '信号天数', '净额', '净流入', '当日资金流入']
    }
    
    # 定义目标任务
    tasks = {
        'Next_Day_High_Increase': ('次日最高涨幅', 'regression'),
        'Next_Day_High_Increase_Classification': ('次日最高涨幅', 'classification')
    }
    
    # 存储所有结果
    all_results = {}
    
    print("开始模型训练和评估...")
    for task_name, (target_col, task_type) in tasks.items():
        print(f"处理任务: {task_name}")
        task_results = {}
        for feature_name, feature_cols in feature_combinations.items():
            print(f"  测试特征组合: {feature_name}")
            results = train_and_evaluate_models(df, target_col, feature_cols, task_type)
            task_results[feature_name] = results
        all_results[task_name] = task_results
    
    print("比较模型性能...")
    # 比较模型性能
    df_performance = compare_models_performance(all_results)
    
    print("分析特征重要性...")
    # 分析特征重要性
    df_importance = analyze_feature_importance(all_results)
    print(df_importance)
    # 生成报表文件名
    output_file = f"reports/model_comparison_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx"
    
    print("生成报表...")
    # 生成报表
    generate_report(df_performance, df_importance, output_file)
    
    # 打印结果摘要
    print("\n=== 模型性能比较 ===")
    print(df_performance.to_string(index=False))
    
    print("\n=== 特征重要性分析 ===")
    # 显示每个任务和模型中最重要的3个特征
    for task in df_importance['Task'].unique():
        print(f"\n任务: {task}")
        task_data = df_importance[df_importance['Task'] == task]
        for model in task_data['Model'].unique():
            print(f"  模型: {model}")
            model_data = task_data[task_data['Model'] == model].sort_values('Importance', ascending=False)
            for i, row in model_data.head(3).iterrows():
                print(f"    {row['Feature']}: {row['Importance']:.4f}")
    
    return df_performance, df_importance

if __name__ == "__main__":
    performance_df, importance_df = main()