import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
import os
from collections import Counter
import seaborn as sns

# 设置中文字体支持
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

class StockDataAnalyzer:
    def __init__(self):
        self.dataframes = []
        self.combined_df = None

    def load_data(self, file_paths):
        """加载一个或多个Excel文件"""
        self.dataframes = []
        for file_path in file_paths:
            if os.path.exists(file_path):
                try:
                    df = pd.read_excel(file_path)
                    self.dataframes.append(df)
                    st.write(f"已加载文件: {file_path}")
                except Exception as e:
                    st.write(f"加载文件失败: {file_path}, 错误: {e}")
            else:
                st.write(f"文件不存在: {file_path}")

        # 合并所有数据
        if self.dataframes:
            self.combined_df = pd.concat(self.dataframes, ignore_index=True)
            st.write(f"总共加载了 {len(self.dataframes)} 个文件，合计 {len(self.combined_df)} 行数据")
        else:
            st.write("没有成功加载任何文件")

    def process_data(self):
        """处理数据，增加计算列"""
        if self.combined_df is None or self.combined_df.empty:
            st.write("没有数据可处理")
            return

        # 确保必要的列存在
        required_columns = ['现价', '当日涨幅', '次日涨幅','次日最高涨幅', '日期']
        for col in required_columns:
            if col not in self.combined_df.columns:
                st.write(f"缺少必要列: {col}")
                return

        # 1. 增加两列计算结果
        # ROUND(100000/(F544*100),0)*F544*100*X544/100
        self.combined_df['收盘利润'] = (100000 / (self.combined_df['现价'] * 100)).round() * \
                                     self.combined_df['现价']  * self.combined_df['次日涨幅']

        # ROUND(100000/(F544*100),0)*Z544*F544
        self.combined_df['最高利润'] = (100000 / (self.combined_df['现价'] * 100)).round() * \
                                     self.combined_df['现价'] * self.combined_df['次日最高涨幅']

        st.write("数据处理完成，新增两列计算完成")

    def get_summary_stats(self):
        """获取汇总统计数据"""
        if self.combined_df is None or self.combined_df.empty:
            return None

        stats = {}
        stats['总记录数'] = len(self.combined_df)
        stats['收盘利润总和'] = self.combined_df['收盘利润'].sum()
        stats['最高利润总和'] = self.combined_df['最高利润'].sum()

        # 当日涨幅<19.97时的统计
        filtered_df_low = self.combined_df[self.combined_df['当日涨幅'] < 19.97]
        stats['当日涨幅<19.97记录数'] = len(filtered_df_low)
        stats['当日涨幅<19.97收盘利润总和'] = filtered_df_low['收盘利润'].sum()
        stats['当日涨幅<19.97最高利润总和'] = filtered_df_low['最高利润'].sum()

        # 当日涨幅>=19.97时的统计
        filtered_df_high = self.combined_df[self.combined_df['当日涨幅'] >= 19.97]
        stats['当日涨幅>=19.97记录数'] = len(filtered_df_high)
        stats['当日涨幅>=19.97收盘利润总和'] = filtered_df_high['收盘利润'].sum()
        stats['当日涨幅>=19.97最高利润总和'] = filtered_df_high['最高利润'].sum()

        return stats

    def plot_date_dimension_analysis(self):
        """按日期维度进行分析并绘图"""
        if self.combined_df is None or self.combined_df.empty:
            st.write("没有数据可分析")
            return

        # 按日期统计
        date_stats = self.combined_df.groupby('日期').agg({
            '收盘利润': ['count', 'sum'],
            '最高利润': 'sum',
            '次日最高涨幅': lambda x: (x > 0).sum()  # 盈利次数
        }).reset_index()
        
        date_stats.columns = ['日期', '记录数', '收盘利润和', '最高利润和', '盈利次数']
        date_stats['盈利占比'] = date_stats['盈利次数'] / date_stats['记录数']

        # 创建图表
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # 记录数趋势
        axes[0, 0].plot(date_stats['日期'], date_stats['记录数'], marker='o')
        axes[0, 0].set_title('每日记录数趋势')
        axes[0, 0].set_xlabel('日期')
        axes[0, 0].set_ylabel('记录数')
        axes[0, 0].tick_params(axis='x', rotation=45)
        
        # 利润和趋势
        ax1 = axes[0, 1]
        ax2 = ax1.twinx()
        ax1.plot(date_stats['日期'], date_stats['收盘利润和'], marker='o', color='blue', label='收盘利润和')
        ax2.plot(date_stats['日期'], date_stats['最高利润和'], marker='s', color='red', label='最高利润和')
        ax1.set_title('每日利润和趋势')
        ax1.set_xlabel('日期')
        ax1.set_ylabel('收盘利润和', color='blue')
        ax2.set_ylabel('最高利润和', color='red')
        ax1.tick_params(axis='x', rotation=45)
        
        # 盈利次数趋势
        axes[1, 0].plot(date_stats['日期'], date_stats['盈利次数'], marker='o', color='green')
        axes[1, 0].set_title('每日盈利次数趋势')
        axes[1, 0].set_xlabel('日期')
        axes[1, 0].set_ylabel('盈利次数')
        axes[1, 0].tick_params(axis='x', rotation=45)
        
        # 盈利占比趋势
        axes[1, 1].plot(date_stats['日期'], date_stats['盈利占比'], marker='o', color='orange')
        axes[1, 1].set_title('每日盈利占比趋势')
        axes[1, 1].set_xlabel('日期')
        axes[1, 1].set_ylabel('盈利占比')
        axes[1, 1].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        st.pyplot(fig)

    def plot_code_dimension_analysis(self):
        """按代码维度进行分析并绘图"""
        if self.combined_df is None or self.combined_df.empty or '代码' not in self.combined_df.columns:
            st.write("没有数据或缺少代码列")
            return

        # 按代码统计
        code_stats = self.combined_df.groupby('代码').agg({
            '收盘利润': ['count', 'sum'],
            '最高利润': 'sum'
        }).reset_index()
        
        code_stats.columns = ['代码', '记录数', '收盘利润和', '最高利润和']
        
        # 获取每个代码中当日利润最高的唯一记录数
        unique_profit_records = self.combined_df.loc[self.combined_df.groupby('代码')['收盘利润'].idxmax()]
        unique_code_stats = unique_profit_records.groupby('代码').size().reset_index(name='唯一记录数')

        # 合并统计数据
        merged_stats = pd.merge(code_stats, unique_code_stats, on='代码', how='left').fillna(0)

        # 创建图表
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # 记录数排行（取前20）
        top_codes_count = merged_stats.nlargest(20, '记录数')
        axes[0, 0].bar(range(len(top_codes_count)), top_codes_count['记录数'], color='skyblue')
        axes[0, 0].set_title('代码记录数排行（前20）')
        axes[0, 0].set_xlabel('代码')
        axes[0, 0].set_ylabel('记录数')
        axes[0, 0].set_xticks(range(len(top_codes_count)))
        axes[0, 0].set_xticklabels(top_codes_count['代码'], rotation=45)
        
        # 收盘利润和排行（取前20）
        top_codes_close_profit = merged_stats.nlargest(20, '收盘利润和')
        axes[0, 1].bar(range(len(top_codes_close_profit)), top_codes_close_profit['收盘利润和'], color='lightgreen')
        axes[0, 1].set_title('代码收盘利润和排行（前20）')
        axes[0, 1].set_xlabel('代码')
        axes[0, 1].set_ylabel('收盘利润和')
        axes[0, 1].set_xticks(range(len(top_codes_close_profit)))
        axes[0, 1].set_xticklabels(top_codes_close_profit['代码'], rotation=45)
        
        # 最高利润和排行（取前20）
        top_codes_max_profit = merged_stats.nlargest(20, '最高利润和')
        axes[1, 0].bar(range(len(top_codes_max_profit)), top_codes_max_profit['最高利润和'], color='salmon')
        axes[1, 0].set_title('代码最高利润和排行（前20）')
        axes[1, 0].set_xlabel('代码')
        axes[1, 0].set_ylabel('最高利润和')
        axes[1, 0].set_xticks(range(len(top_codes_max_profit)))
        axes[1, 0].set_xticklabels(top_codes_max_profit['代码'], rotation=45)
        
        # 唯一记录数排行（取前20）
        top_codes_unique = merged_stats.nlargest(20, '唯一记录数')
        axes[1, 1].bar(range(len(top_codes_unique)), top_codes_unique['唯一记录数'], color='gold')
        axes[1, 1].set_title('代码唯一记录数排行（前20）')
        axes[1, 1].set_xlabel('代码')
        axes[1, 1].set_ylabel('唯一记录数')
        axes[1, 1].set_xticks(range(len(top_codes_unique)))
        axes[1, 1].set_xticklabels(top_codes_unique['代码'], rotation=45)
        
        plt.tight_layout()
        st.pyplot(fig)

    def plot_time_dimension_analysis(self):
        """按时间维度进行分析并绘图"""
        if self.combined_df is None or self.combined_df.empty or 'time' not in self.combined_df.columns:
            st.write("没有数据或缺少time列")
            return

        # 按时间统计
        time_stats = self.combined_df.groupby('time').agg({
            '收盘利润': ['count', 'sum'],
            '最高利润': 'sum',
            '次日最高涨幅': lambda x: (x > 0).sum()  # 盈利次数
        }).reset_index()
        
        time_stats.columns = ['time', '记录数', '收盘利润和', '最高利润和', '盈利次数']
        time_stats['盈利占比'] = time_stats['盈利次数'] / time_stats['记录数']

        # 创建图表
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # 记录数分布
        axes[0, 0].bar(time_stats['time'], time_stats['记录数'], color='skyblue')
        axes[0, 0].set_title('各时间点记录数分布')
        axes[0, 0].set_xlabel('时间')
        axes[0, 0].set_ylabel('记录数')
        axes[0, 0].tick_params(axis='x', rotation=45)
        
        # 利润和分布
        ax1 = axes[0, 1]
        ax2 = ax1.twinx()
        ax1.bar(time_stats['time'], time_stats['收盘利润和'], color='lightgreen', alpha=0.7, label='收盘利润和')
        ax2.plot(time_stats['time'], time_stats['最高利润和'], marker='o', color='red', label='最高利润和')
        ax1.set_title('各时间点利润和分布')
        ax1.set_xlabel('时间')
        ax1.set_ylabel('收盘利润和', color='lightgreen')
        ax2.set_ylabel('最高利润和', color='red')
        ax1.tick_params(axis='x', rotation=45)
        
        # 盈利次数分布
        axes[1, 0].bar(time_stats['time'], time_stats['盈利次数'], color='gold')
        axes[1, 0].set_title('各时间点盈利次数分布')
        axes[1, 0].set_xlabel('时间')
        axes[1, 0].set_ylabel('盈利次数')
        axes[1, 0].tick_params(axis='x', rotation=45)
        
        # 盈利占比分布
        axes[1, 1].bar(time_stats['time'], time_stats['盈利占比'], color='purple')
        axes[1, 1].set_title('各时间点盈利占比分布')
        axes[1, 1].set_xlabel('时间')
        axes[1, 1].set_ylabel('盈利占比')
        axes[1, 1].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        st.pyplot(fig)

    def plot_date_time_dimension_analysis(self):
        """按日期+时间维度进行分析并绘图"""
        if self.combined_df is None or self.combined_df.empty or 'time' not in self.combined_df.columns:
            st.write("没有数据或缺少time列")
            return

        # 按日期+时间统计
        datetime_stats = self.combined_df.groupby(['日期', 'time']).agg({
            '收盘利润': ['count', 'sum'],
            '最高利润': 'sum'
        }).reset_index()
        
        datetime_stats.columns = ['日期', 'time', '记录数', '收盘利润和', '最高利润和']

        # 创建热力图数据
        pivot_data = datetime_stats.pivot_table(index='time', columns='日期', values='记录数', fill_value=0)
        
        # 绘制热力图
        fig, ax = plt.subplots(figsize=(12, 8))
        sns.heatmap(pivot_data, annot=True, fmt='g', cmap='YlGnBu', ax=ax)
        ax.set_title('日期+时间记录数热力图')
        ax.set_xlabel('日期')
        ax.set_ylabel('时间')
        plt.xticks(rotation=45)
        plt.yticks(rotation=0)
        st.pyplot(fig)

    def plot_date_code_dimension_analysis(self):
        """按日期+代码维度进行分析并绘图"""
        if self.combined_df is None or self.combined_df.empty or '代码' not in self.combined_df.columns:
            st.write("没有数据或缺少代码列")
            return

        # 按日期+代码统计
        date_code_stats = self.combined_df.groupby(['日期', '代码']).agg({
            '收盘利润': ['count', 'sum'],
            '最高利润': 'sum'
        }).reset_index()
        
        date_code_stats.columns = ['日期', '代码', '记录数', '收盘利润和', '最高利润和']

        # 获取唯一代码数统计
        unique_codes_per_date = self.combined_df.groupby('日期')['代码'].nunique().reset_index()
        unique_codes_per_date.columns = ['日期', '唯一代码数']

        # 创建图表
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        
        # 每日唯一代码数
        axes[0].plot(unique_codes_per_date['日期'], unique_codes_per_date['唯一代码数'], marker='o')
        axes[0].set_title('每日唯一代码数趋势')
        axes[0].set_xlabel('日期')
        axes[0].set_ylabel('唯一代码数')
        axes[0].tick_params(axis='x', rotation=45)
        
        # 每日记录数
        date_record_counts = self.combined_df.groupby('日期').size().reset_index(name='记录数')
        axes[1].plot(date_record_counts['日期'], date_record_counts['记录数'], marker='o', color='orange')
        axes[1].set_title('每日记录数趋势')
        axes[1].set_xlabel('日期')
        axes[1].set_ylabel('记录数')
        axes[1].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        st.pyplot(fig)

def main():
    st.title("股票数据分析仪表板")
    
    # 侧边栏配置
    st.sidebar.header("数据配置")
    
    # 选择数据目录
    data_dir = st.sidebar.text_input("Excel文件目录", "temp")
    
    # 获取目录下的所有Excel文件
    if os.path.exists(data_dir):
        excel_files = [f for f in os.listdir(data_dir) if f.endswith(('.xlsx', '.xls'))]
        if excel_files:
            selected_files = st.sidebar.multiselect("选择Excel文件", excel_files, default=excel_files[:3] if len(excel_files) > 3 else excel_files)
            file_paths = [os.path.join(data_dir, f) for f in selected_files]
        else:
            st.sidebar.warning("目录中没有找到Excel文件")
            file_paths = []
    else:
        st.sidebar.error("指定的目录不存在")
        file_paths = []

    # 初始化分析器
    analyzer = StockDataAnalyzer()
    
    # 加载数据
    if file_paths and st.sidebar.button("加载并分析数据"):
        with st.spinner("正在加载和分析数据..."):
            analyzer.load_data(file_paths)
            analyzer.process_data()
            
            # 显示汇总统计
            st.header("数据汇总统计")
            stats = analyzer.get_summary_stats()
            if stats:
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("总记录数", f"{stats['总记录数']:,}")
                    st.metric("收盘利润总和", f"{stats['收盘利润总和']:,.2f}")
                    st.metric("最高利润总和", f"{stats['最高利润总和']:,.2f}")
                with col2:
                    st.metric("涨幅<19.97记录数", f"{stats['当日涨幅<19.97记录数']:,}")
                    st.metric("涨幅<19.97收盘利润", f"{stats['当日涨幅<19.97收盘利润总和']:,.2f}")
                    st.metric("涨幅<19.97最高利润", f"{stats['当日涨幅<19.97最高利润总和']:,.2f}")
                with col3:
                    st.metric("涨幅>=19.97记录数", f"{stats['当日涨幅>=19.97记录数']:,}")
                    st.metric("涨幅>=19.97收盘利润", f"{stats['当日涨幅>=19.97收盘利润总和']:,.2f}")
                    st.metric("涨幅>=19.97最高利润", f"{stats['当日涨幅>=19.97最高利润总和']:,.2f}")

            # 按日期维度分析
            st.header("按日期维度分析")
            analyzer.plot_date_dimension_analysis()

            # 按代码维度分析
            st.header("按代码维度分析")
            analyzer.plot_code_dimension_analysis()

            # 按时间维度分析
            st.header("按时间维度分析")
            analyzer.plot_time_dimension_analysis()

            # 按日期+时间维度分析
            st.header("按日期+时间维度分析")
            analyzer.plot_date_time_dimension_analysis()

            # 按日期+代码维度分析
            st.header("按日期+代码维度分析")
            analyzer.plot_date_code_dimension_analysis()
    elif not file_paths:
        st.info("请在侧边栏选择数据目录和Excel文件，然后点击'加载并分析数据'按钮")

if __name__ == "__main__":
    main()

    # streamlit run D:/project/trade/dataanalysis/stock/stock_dashboard.py