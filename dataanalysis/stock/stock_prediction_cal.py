import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter
import os
import os
os.environ['MPLBACKEND'] = 'Agg'
import matplotlib
matplotlib.use('Agg')

# 设置中文字体支持
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

class StockPredictorCal:
    def __init__(self):
        self.dataframes = []
        self.combined_df = None

    def load_data(self, file_paths):
        """加载一个或多个Excel文件"""
        for file_path in file_paths:
            if os.path.exists(file_path):
                df = pd.read_excel(file_path)
                self.dataframes.append(df)
                print(f"已加载文件: {file_path}")
            else:
                print(f"文件不存在: {file_path}")

        # 合并所有数据
        if self.dataframes:
            self.combined_df = pd.concat(self.dataframes, ignore_index=True)
            print(f"总共加载了 {len(self.dataframes)} 个文件，合计 {len(self.combined_df)} 行数据")

    def process_data(self):
        """处理数据，增加计算列"""
        if self.combined_df is None or self.combined_df.empty:
            print("没有数据可处理")
            return

        # 确保必要的列存在
        required_columns = ['现价', '当日涨幅', '次日涨幅','次日最高涨幅', '日期']
        for col in required_columns:
            if col not in self.combined_df.columns:
                print(f"缺少必要列: {col}")
                return

        # 1. 增加两列计算结果
        # ROUND(100000/(F544*100),0)*F544*100*X544/100
        self.combined_df['收盘利润'] = (100000 / (self.combined_df['现价'] * 100)).round() * \
                                     self.combined_df['现价']  * self.combined_df['次日涨幅']

        # ROUND(100000/(F544*100),0)*Z544*F544
        self.combined_df['最高利润'] = (100000 / (self.combined_df['现价'] * 100)).round() * \
                                     self.combined_df['现价'] * self.combined_df['次日最高涨幅']

        print("数据处理完成，新增两列计算完成")


    def visualize_data(self):
        """可视化数据"""
        if self.combined_df is None or self.combined_df.empty:
            print("没有数据可可视化")
            return

        # 3. 按日期统计次数并打印出来
        date_counts = Counter(self.combined_df['日期'])

        # 4. 所有数值统计并以图表形式展示
        # 绘制计算列的分布图
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))

        # 收盘利润分布
        axes[0, 0].hist(self.combined_df['收盘利润'], bins=30, alpha=0.7, color='blue')
        axes[0, 0].set_title('收盘利润分布')
        axes[0, 0].set_xlabel('值')
        axes[0, 0].set_ylabel('频次')

        # 最高利润分布
        axes[0, 1].hist(self.combined_df['最高利润'], bins=30, alpha=0.7, color='green')
        axes[0, 1].set_title('最高利润分布')
        axes[0, 1].set_xlabel('值')
        axes[0, 1].set_ylabel('频次')

        # 按日期统计的柱状图
        dates_list = list(date_counts.keys())
        counts_list = list(date_counts.values())
        axes[1, 0].bar(range(len(dates_list)), counts_list, color='orange')
        axes[1, 0].set_title('每日数据量统计')
        axes[1, 0].set_xlabel('日期')
        axes[1, 0].set_ylabel('次数')
        axes[1, 0].set_xticks(range(len(dates_list)))
        axes[1, 0].set_xticklabels([str(d) for d in dates_list], rotation=45)

        # 当日涨幅分布
        axes[1, 1].hist(self.combined_df['当日涨幅'], bins=30, alpha=0.7, color='red')
        axes[1, 1].set_title('当日涨幅分布')
        axes[1, 1].set_xlabel('当日涨幅')
        axes[1, 1].set_ylabel('频次')

        plt.tight_layout()
        # 保存图表到文件而不是显示
        output_path = "temp/visualization_output.png"
        plt.savefig(output_path)
        print(f"图表已保存至: {output_path}")
        plt.close()  # 关闭图表以释放内存

    def save_data(self, output_path="temp/processed_data.xlsx"):
        """保存处理后的数据"""
        if self.combined_df is not None and not self.combined_df.empty:
            self.combined_df.to_excel(output_path, index=False)
            print(f"数据已保存至: {output_path}")
        else:
            print("没有数据可保存")

    def analyze_data(self):
        """统计分析数据"""
        if self.combined_df is None or self.combined_df.empty:
            print("没有数据可分析")
            return

        # 2. 统计分析新增两列的总和，以及当日涨幅 <19.97 时的总和
        sum_col1 = self.combined_df['收盘利润'].sum()
        sum_col2 = self.combined_df['最高利润'].sum()

        # 当日涨幅<19.97时的总和
        filtered_df = self.combined_df[self.combined_df['当日涨幅'] < 19.97]
        sum_col1_filtered = filtered_df['收盘利润'].sum()
        sum_col2_filtered = filtered_df['最高利润'].sum()

        # 当日涨幅<19.97时的总和
        filtered_df = self.combined_df[self.combined_df['当日涨幅'] >= 19.97]
        sum_col1_up_filtered = filtered_df['收盘利润'].sum()
        sum_col2_up_filtered = filtered_df['最高利润'].sum()

        print(f"收盘利润总和: {sum_col1}")
        print(f"最高利润总和: {sum_col2}")
        print(f"当日涨幅<19.97时收盘利润总和: {sum_col1_filtered}")
        print(f"当日涨幅<19.97时最高利润总和: {sum_col2_filtered}")
        print(f"当日涨幅>=19.97时收盘利润总和: {sum_col1_up_filtered}")
        print(f"当日涨幅>=19.97时最高利润总和: {sum_col2_up_filtered}")

        # 3. 按日期、代码统计次数并打印出来
        date_counts = Counter(self.combined_df['日期'])
        print("\n按日期统计次数:")
        # 将date_counts 按照日期排序
        date_counts = {date: count for date, count in sorted(date_counts.items())}
        # date_counts = sorted(date_counts.keys())
        for date in date_counts:
            print(f"{date}: {date_counts[date]}次")

        # 计算相邻两日之和最大的数据
        df_sorted = self.combined_df.sort_values('日期')
        max_sum = 0
        max_dates = None

        dates = list(date_counts.keys())
        dates.sort()

        for i in range(len(dates) - 1):
            current_date = dates[i]
            next_date = dates[i + 1]
            current_sum = date_counts[current_date] + date_counts[next_date]
            if current_sum > max_sum:
                max_sum = current_sum
                max_dates = (current_date, next_date)

        if max_dates:
            print(f"\n相邻两日次数之和最大的是: {max_dates[0]} 和 {max_dates[1]}, 总次数为: {max_sum}")

        # 新增：计算代码+日期唯一组合下的相邻两日次数之和最大值
        print("\n=== 代码+日期唯一组合下的相邻两日次数之和分析 ===")
        if '代码' in self.combined_df.columns:
            # 获取唯一的代码+日期组合数量
            unique_combinations = self.combined_df.groupby(['代码', '日期']).size().reset_index(name='count')
            # 统计每天的唯一组合数
            daily_unique_counts = unique_combinations['日期'].value_counts().sort_index()
            print(f"代码+日期唯一组合: {unique_combinations}")

            print(f"每天唯一组合数: {daily_unique_counts}")

            # 将 unique_combinations 按照日期统计数量
            daily_unique_counts = unique_combinations.groupby('日期').size().sort_index()
            print(f"每天唯一组合数: {daily_unique_counts}")
            # 增加一个新字段，将相邻日期的数量相加
            unique_combinations['日期唯一组合数'] = daily_unique_counts.groupby('日期').transform('size')
            unique_combinations['代码唯一组合数'] = unique_combinations.groupby('代码').transform('size').sort_index()
            print(f"相邻日期唯一组合数: {unique_combinations['相邻日期唯一组合数']}")
            print(f"代码唯一组合数: {unique_combinations['代码唯一组合数']}")

        # 转换为有序列表以便遍历
            sorted_dates = list(daily_unique_counts.index)

            max_adjacent_sum = 0
            max_adjacent_dates = None

            # 遍历相邻两天计算它们的唯一组合数之和
            for i in range(len(sorted_dates)-1):
                current_date = sorted_dates[i]
                next_date = sorted_dates[i+1]
                current_count = daily_unique_counts.get(current_date, 0)
                next_count = daily_unique_counts.get(next_date, 0)
                adjacent_sum = current_count + next_count

                if adjacent_sum > max_adjacent_sum:
                    max_adjacent_sum = adjacent_sum
                    max_adjacent_dates = (current_date, next_date)

            if max_adjacent_dates:
                print(f"相邻两日唯一组合数之和最大的是: {max_adjacent_dates[0]} 和 {max_adjacent_dates[1]}, 总数为: {max_adjacent_sum}")
            else:
                print("未能找到相邻两日的数据")

            # 代码 的统计，统计

        else:
            print("数据中不包含'代码'列，无法进行代码+日期唯一组合分析")


        # 新增分析：按日期和代码分组，找出每个组中最高利润和收盘利润最大的记录,增加去掉>19.97的记录
        print("\n=== 按日期和代码分组的最高利润和收盘利润分析 ===")

        grouped = self.combined_df.groupby(['日期', '代码'])
        # 过滤掉当日涨幅>=19.97的记录
        # grouped = grouped.filter(lambda x: (x['当日涨幅'] < 19.97).all())
        # 检查是否包含代码列
        if '代码' in self.combined_df.columns:
            # 按日期和代码分组
            grouped = self.combined_df.groupby(['日期', '代码'])

            # 找出每组中最高利润最大的记录
            # max_profit_per_group = grouped.apply(lambda x: x.loc[x['最高利润'].idxmax()])
            max_profit_per_group = grouped.apply(lambda x: x.loc[x['最高利润'].idxmax()], include_groups=False)

            # print("\n每个日期+代码组合中最高利润最大的记录:")
            # for index, row in max_profit_per_group.iterrows():
            #     print(f"日期: {row['日期']}, 代码: {row['代码']}, "
            #           f"最高利润: {row['最高利润']}, 收盘利润: {row['收盘利润']}")
            #     if 'time' in row:
            #         print(f"  -> time: {row['time']}")

            # 找出每组中收盘利润最大的记录
            # close_profit_per_group = grouped.apply(lambda x: x.loc[x['收盘利润'].idxmax()])
            close_profit_per_group = grouped.apply(lambda x: x.loc[x['收盘利润'].idxmax()], include_groups=False)

            # print("\n每个日期+代码组合中收盘利润最大的记录:")
            # for index, row in close_profit_per_group.iterrows():
            #     print(f"日期: {row['日期']}, 代码: {row['代码']}, "
            #           f"收盘利润: {row['收盘利润']}, 最高利润: {row['最高利润']}")
            #     if 'time' in row:
            #         print(f"  -> time: {row['time']}")
            # 统计 max_profit_per_group 和 close_profit_per_group 的按照time 的统计和 次数
            max_profit_per_group_time_count = max_profit_per_group.groupby('time')['最高利润'].count()
            max_profit_per_group_time_sum = max_profit_per_group.groupby('time')['最高利润'].sum()
            close_profit_per_group_time_count = close_profit_per_group.groupby('time')['收盘利润'].count()
            close_profit_per_group_time_sum = close_profit_per_group.groupby('time')['收盘利润'].sum()
            print(f"每个time 最高利润总和: {max_profit_per_group_time_sum}")
            print(f"每个time 最高利润次数: {max_profit_per_group_time_count}")

            print(f"每个time 收盘利润总和: {close_profit_per_group_time_sum}")
            print(f"每个time 收盘利润次数: {close_profit_per_group_time_count}")

            # 统计max_profit_per_group 和 close_profit_per_group 的总和，以及按照日期汇总以后的总和排序
            total_sum = max_profit_per_group['最高利润'].sum() + close_profit_per_group['收盘利润'].sum()
            print(f"最高利润总和: {max_profit_per_group['最高利润'].sum()}")
            print(f"收盘利润总和: {close_profit_per_group['收盘利润'].sum()}")
            print(f"总利润总和: {total_sum}")
            # 按照日期汇总最高利润和收盘利润的总和
            daily_total_sum = max_profit_per_group.groupby('日期')['最高利润'].sum() + close_profit_per_group.groupby('日期')['收盘利润'].sum()
            print(f"每天的总利润总和: {daily_total_sum}")
        else:
            print("数据中不包含'代码'列，无法进行按日期和代码分组的分析")





    def analyze_duplicate_codes(self):
        """分析重复代码数据占比及相关统计"""
        if self.combined_df is None or self.combined_df.empty:
            print("没有数据可分析")
            return

        print("\n=== 重复代码数据分析 ===")

        # 检查是否包含必要的列
        if '代码' not in self.combined_df.columns or '日期' not in self.combined_df.columns:
            print("数据中不包含'代码'或'日期'列，无法进行重复代码分析")
            return

        # 按代码和日期分组，统计每组的记录数
        grouped = self.combined_df.groupby(['代码', '日期']).size().reset_index(name='count')

        # 分离重复和非重复数据
        duplicate_groups = grouped[grouped['count'] > 1]  # 有重复的代码+日期组合
        unique_groups = grouped[grouped['count'] == 1]    # 无重复的代码+日期组合

        # 计算占比
        total_groups = len(grouped)
        duplicate_ratio = len(duplicate_groups) / total_groups if total_groups > 0 else 0
        unique_ratio = len(unique_groups) / total_groups if total_groups > 0 else 0

        print(f"总组合数: {total_groups}")
        print(f"有重复的组合数: {len(duplicate_groups)}, 占比: {duplicate_ratio:.2%}")
        print(f"无重复的组合数: {len(unique_groups)}, 占比: {unique_ratio:.2%}")

        # 对于有重复的数据，提取每组中最高利润的最大值
        duplicate_profit_sum = 0
        duplicate_close_profit_sum = 0
        if not duplicate_groups.empty:
            # 获取所有重复的代码+日期组合
            duplicate_keys = duplicate_groups[['代码', '日期']]
            # 筛选出这些组合的所有数据
            duplicate_data = self.combined_df.merge(duplicate_keys, on=['代码', '日期'])
            # 按代码和日期分组，取每组中最高利润的最大值
            max_profit_per_duplicate_group = duplicate_data.groupby(['代码', '日期'])['最高利润'].max()
            duplicate_profit_sum = max_profit_per_duplicate_group.sum()

            # 同样计算收盘利润的最大值总和
            max_close_profit_per_duplicate_group = duplicate_data.groupby(['代码', '日期'])['收盘利润'].max()
            duplicate_close_profit_sum = max_close_profit_per_duplicate_group.sum()

        # 对于无重复的数据，直接求和
        unique_profit_sum = 0
        unique_close_profit_sum = 0
        if not unique_groups.empty:
            # 获取所有无重复的代码+日期组合
            unique_keys = unique_groups[['代码', '日期']]
            # 筛选出这些组合的所有数据
            unique_data = self.combined_df.merge(unique_keys, on=['代码', '日期'])
            # 直接求和
            unique_profit_sum = unique_data['最高利润'].sum()
            unique_close_profit_sum = unique_data['收盘利润'].sum()

        print(f"\n有重复的代码最高利润最大值总和: {duplicate_profit_sum}")
        print(f"有重复的代码收盘利润最大值总和: {duplicate_close_profit_sum}")
        print(f"无重复的代码最高利润总和: {unique_profit_sum}")
        print(f"无重复的代码收盘利润总和: {unique_close_profit_sum}")
def main():
    # 初始化预测器
    predictor = StockPredictorCal()

    # 可以读取多个文件
    file_paths = [
        "temp/0801-0923.xlsx",
        # "temp/another_file.xlsx"  # 可以添加更多文件
    ]

    # 加载数据
    predictor.load_data(file_paths)

    # 处理数据
    predictor.process_data()

    # 分析数据         """统计分析数据"""
    predictor.analyze_data()

    """分析重复代码数据占比及相关统计"""
    predictor.analyze_duplicate_codes()


    # 可视化数据
    predictor.visualize_data()

    # 保存处理后的数据
    predictor.save_data()

if __name__ == "__main__":
    main()