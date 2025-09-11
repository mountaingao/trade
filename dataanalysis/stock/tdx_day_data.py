# day_kline_processor.py
import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
from pytdx.hq import TdxHq_API
from pytdx.params import TDXParams
from tdx_mini_data import get_market_code,get_stock_code

warnings.filterwarnings('ignore')


# 读取每日K线数据
def get_daily_data(code: str) -> pd.DataFrame:
    """使用pytdx获取5分钟K线数据"""
    api = TdxHq_API()
    try:
        # 连接通达信服务器
        # if not api.connect('14.215.128.18', 7709):  # 免费服务器，可替换为其他服务器
        if not api.connect('123.125.108.90', 7709):  # 免费服务器，可替换为其他服务器
            # if not api.connect('183.201.253.76', 7709):  # 免费服务器，可替换为其他服务器
            raise ConnectionError("无法连接到通达信服务器")

        market_code = get_market_code(code)
        stock_code = get_stock_code(code)

        # 获取最近10天的5分钟K线数据
        data = []
        # 修改为获取10天数据，每天48个5分钟周期
        for i in range(10):  # 获取10天的数据
            kline_data = api.get_security_bars(
                TDXParams.KLINE_TYPE_DAILY,
                market_code,
                stock_code,
                (9-i) * 48,  # 每天最多48个5分钟周期，现在获取10天数据
                48
            )
            if kline_data:
                data.extend(kline_data)

        if not data:
            return pd.DataFrame()

        # 转换为DataFrame
        df = pd.DataFrame(data)
        # print( df.columns)
        # print( df.tail(5))
        # 处理日期时间字段
        df['datetime'] = pd.to_datetime(df['datetime'])

        #columns： Index(['open', 'close', 'high', 'low', 'vol', 'amount', 'year', 'month', 'day',
               # 'hour', 'minute', 'datetime'],
        # 重命名列以匹配原代码
        df = df.rename(columns={
            'datetime': 'date',
            'open': 'open',
            'high': 'high',
            'low': 'low',
            'close': 'close',
            'vol': 'volume',
            'amount': 'amount'
        })

        # 只保留需要的列
        required_cols = ['date', 'open', 'high', 'low', 'close', 'volume', 'amount']
        df = df[required_cols]

        # print( df)
        print("获取股票日线数据成功",code)
        return df

    except Exception as e:
        print(f"获取股票{code}数据时出错: {e}")
        return pd.DataFrame()
    finally:
        api.disconnect()

class DayKLineProcessor:
    def __init__(self, data_source='local', data_directory=None):
        """
        初始化日K线处理器

        Parameters:
        data_source: str, 'local' 或 'online'
        data_directory: str, 本地数据目录路径
        """
        self.data_source = data_source
        self.data_directory = data_directory
        self.qu_data = {}  # 存储所有QU数据

    def sma(self, data, period, weight=1):
        """
        计算加权移动平均线 SMA(C, period, weight)

        Parameters:
        data: pandas.Series, 价格数据
        period: float, 周期
        weight: int, 权重

        Returns:
        pandas.Series: SMA数据
        """
        # SMA(X,N,M) = (M*X+(N-M)*SMA(X,N,M).ref(1))/N
        # SMA(C,6.5,1) = (1*C+(6.5-1)*SMA(C,6.5,1).ref(1))/6.5
        alpha = weight / period
        return data.ewm(alpha=alpha, adjust=False).mean()

    def load_local_kline_data(self, file_path):
        """
        从本地文件加载K线数据并处理除权

        Parameters:
        file_path: str, 文件路径

        Returns:
        pandas.DataFrame: K线数据
        """
        try:
            # 根据文件扩展名选择读取方式
            if file_path.endswith('.xlsx') or file_path.endswith('.xls'):
                df = pd.read_excel(file_path)
            elif file_path.endswith('.csv'):
                df = pd.read_csv(file_path)
            else:
                # 尝试以Excel格式读取
                df = pd.read_excel(file_path)

            # 标准化列名（假设包含日期、开盘价、最高价、最低价、收盘价、成交量）
            column_mapping = {
                'date': 'date', 'Date': 'date', 'datetime': 'date', 'DateTime': 'date',
                'open': 'open', 'Open': 'open', 'OPEN': 'open',
                'high': 'high', 'High': 'high', 'HIGH': 'high',
                'low': 'low', 'Low': 'low', 'LOW': 'low',
                'close': 'close', 'Close': 'close', 'CLOSE': 'close',
                'volume': 'volume', 'Volume': 'volume', 'VOLUME': 'volume'
            }

            df.rename(columns=column_mapping, inplace=True)

            # 确保日期列是datetime类型
            df['date'] = pd.to_datetime(df['date'])

            # 按日期排序
            df = df.sort_values('date').reset_index(drop=True)

            # 处理除权数据（这里是一个简化的实现，实际除权处理可能更复杂）
            # 通常需要从专门的除权数据源获取除权信息
            df = self.adjust_for_dividends(df)

            return df
        except Exception as e:
            print(f"读取文件 {file_path} 出错: {e}")
            return None

    def adjust_for_dividends(self, df):
        """
        处理除权数据（简化实现）
        实际应用中需要从专业数据源获取除权信息

        Parameters:
        df: pandas.DataFrame, 原始K线数据

        Returns:
        pandas.DataFrame: 除权调整后的数据
        """
        # 这里只是一个示例实现，实际除权处理需要根据具体的除权数据
        # 假设数据已经是前复权或后复权处理过的
        return df

    def load_online_kline_data(self, symbol):
        """
        从在线数据源加载K线数据

        Parameters:
        symbol: str, 股票代码

        Returns:
        pandas.DataFrame: K线数据
        """
        # 这里只是一个框架，实际需要根据使用的在线数据源实现
        # 可以使用tushare、akshare等库获取在线数据
        print(f"从在线数据源加载 {symbol} 的K线数据")
        kline_data = get_daily_data(symbol)
        # 示例返回空数据框
        return kline_data


    @staticmethod
    def calculate_boll(df: pd.DataFrame, window=20, num_std=2) -> pd.DataFrame:
        """计算布林带指标"""
        # df = df.copy()
        df['ma'] = df['close'].rolling(window=window).mean()
        df['std'] = df['close'].rolling(window=window).std()
        df['upper'] = (df['ma'] + num_std * df['std']).round(2)
        df['lower'] = (df['ma'] - num_std * df['std']).round(2)
        df['band_width'] = 100*(df['upper'] - df['lower']) / df['ma'].round(2)
        return df

    def calculate_bias(data, period=6):
        """
        计算乖离率(BIAS)

        参数:
        data: 价格序列 (pandas Series)
        period: 计算周期 (默认6日)

        返回:
        BIAS值序列
        """
        # 确保只对数值列进行计算
        if isinstance(data, pd.DataFrame):
            # 如果传入的是DataFrame，选择收盘价列进行计算
            price_data = data['close']
        else:
            # 如果传入的是Series，直接使用
            price_data = data

        # 计算移动平均线
        ma = price_data.rolling(window=period).mean()

        # 计算乖离率
        data['bias'] = (100 * (price_data - ma) / ma ).round(2)

        return data
    def SMA(self, X, N, M):
        """
        计算带权重的简单移动平均线

        参数:
        X: 价格序列 (pandas Series)
        N: 移动平均窗口大小 (必须是整数)
        M: 当前价格的权重

        返回:
        SMA值序列
        """
        # 确保N是整数
        N = int(N)

        # 初始化结果数组
        result = np.zeros(len(X))
        result[:] = np.nan  # 将前N-1个值设为NaN

        # 如果数据长度小于窗口大小，直接返回NaN数组
        if len(X) < N:
            return result

        # 计算第一个有效值
        result[N-1] = np.mean(X[:N])

        # 递归计算后续值
        for i in range(N, len(X)):
            result[i] = (X.iloc[i] * M + result[i-1] * (N - M)) / N

        return result
    def calculate_sma(self, df: pd.DataFrame) -> pd.DataFrame:
        """计算SMA指标"""
        # 使用自定义的小数周期移动平均
        df['sma'] = self.SMA(df['close'], 6.5, 1).round(2)
        # 当日sma 值除以前一天 sma 值
        df['sma_ratio'] = (100*(df['sma'] - df['sma'].shift(1)) / df['sma'].shift(1)).round(2)
        return df

    def calculate_qu(self, kline_data):
        """
        根据日K线计算 QU:=SMA(C,6.5,1)

        Parameters:
        kline_data: pandas.DataFrame, K线数据

        Returns:
        pandas.Series: QU数据
        """
        if kline_data is None or len(kline_data) == 0:
            return None

        # 计算QU指标
        close_prices = kline_data['close']
        qu = self.sma(close_prices, 6.5, 1)
        return qu

    def get_qu_data_by_date(self, symbol, target_date):
        """
        获取指定日期和之前的QU数据

        Parameters:
        symbol: str, 股票代码
        target_date: str or datetime, 目标日期

        Returns:
        pandas.DataFrame: 指定日期及之前的QU数据
        """
        target_date = pd.to_datetime(target_date)

        # 获取该股票的K线数据
        if symbol not in self.qu_data:
            print(f"未找到 {symbol} 的数据")
            return None

        df = self.qu_data[symbol]
        df['date'] = pd.to_datetime(df['date'])

        # 筛选指定日期及之前的数据
        result = df[df['date'] <= target_date].copy()
        return result.sort_values('date', ascending=False)

    def predict_qu(self, symbol, days=5):
        """
        基于现有数据预测未来的QU值

        Parameters:
        symbol: str, 股票代码
        days: int, 预测天数

        Returns:
        pandas.DataFrame: 包含预测QU值的数据
        """
        if symbol not in self.qu_data:
            print(f"未找到 {symbol} 的数据")
            return None

        df = self.qu_data[symbol].copy()
        df['date'] = pd.to_datetime(df['date'])

        # 简单的预测方法：使用最后几个数据点的趋势
        if len(df) < 2:
            return None

        # 获取最后几条记录用于预测
        last_data = df.tail(10)  # 使用最近10天的数据

        # 简单线性外推预测（实际应用中可以使用更复杂的模型）
        qu_values = last_data['qu'].values
        dates = last_data['date'].values

        # 计算最近的趋势
        if len(qu_values) >= 2:
            # 简单线性回归预测
            x = np.arange(len(qu_values))
            y = qu_values
            coeffs = np.polyfit(x, y, 1)  # 一次多项式拟合
            slope, intercept = coeffs

            # 预测未来几天
            predictions = []
            last_date = df['date'].max()

            for i in range(1, days + 1):
                next_date = last_date + timedelta(days=i)
                predicted_qu = slope * (len(qu_values) + i - 1) + intercept
                predictions.append({
                    'date': next_date,
                    'qu': predicted_qu,
                    'predicted': True
                })

            prediction_df = pd.DataFrame(predictions)
            return prediction_df
        else:
            return None

    def process_directory_files(self, directory_path, value_filter=None):
        """
        读取指定目录下的文件，筛选符合条件的Excel文件并处理

        Parameters:
        directory_path: str, 目录路径
        value_filter: any, value列需要等于的值，默认为1
        """
        if value_filter is None:
            value_filter = 1

        if not os.path.exists(directory_path):
            print(f"目录 {directory_path} 不存在")
            return

        # 遍历目录下的所有文件和子目录
        for root, dirs, files in os.walk(directory_path):
            for file in files:
                # 检查是否为Excel文件
                if file.endswith(('.xlsx', '.xls')):
                    file_path = os.path.join(root, file)
                    print(f"处理文件: {file_path}")

                    try:
                        # 读取Excel文件
                        excel_data = pd.read_excel(file_path, sheet_name=None)  # 读取所有sheet

                        # 遍历所有sheet
                        for sheet_name, sheet_data in excel_data.items():
                            # 检查是否有value列且包含符合条件的数据
                            if 'value' in sheet_data.columns:
                                filtered_data = sheet_data[sheet_data['value'] == value_filter]

                                if len(filtered_data) > 0:
                                    print(f"  在文件 {file} 的 sheet {sheet_name} 中找到 {len(filtered_data)} 条符合条件的数据")

                                    # 处理每条符合条件的数据
                                    for idx, row in filtered_data.iterrows():
                                        # 假设数据中有股票代码列（code, symbol等）
                                        symbol = None
                                        for col in ['code', 'symbol', 'stock_code', '股票代码']:
                                            if col in row:
                                                symbol = row[col]
                                                break

                                        if symbol:
                                            # 处理该股票的K线数据
                                            self.process_stock_kline(symbol, root)
                                        else:
                                            print(f"    跳过行 {idx}，未找到股票代码")
                    except Exception as e:
                        print(f"处理文件 {file_path} 时出错: {e}")

    def process_stock_kline(self, symbol, data_directory=None):
        """
        处理单个股票的K线数据

        Parameters:
        symbol: str, 股票代码
        data_directory: str, 数据目录
        """
        kline_data = None

        # 根据数据源类型加载数据
        if self.data_source == 'local':
            # 在本地目录中查找该股票的数据文件
            if data_directory is None:
                data_directory = self.data_directory

            if data_directory and os.path.exists(data_directory):
                # 查找匹配的文件
                for file in os.listdir(data_directory):
                    if symbol in file and file.endswith(('.xlsx', '.xls', '.csv')):
                        file_path = os.path.join(data_directory, file)
                        kline_data = self.load_local_kline_data(file_path)
                        break

                if kline_data is None:
                    print(f"未在 {data_directory} 中找到 {symbol} 的数据文件")
            else:
                print("本地数据目录未设置或不存在")

        elif self.data_source == 'online':
            kline_data = self.load_online_kline_data(symbol)

        # 计算QU指标
        if kline_data is not None and len(kline_data) > 0:
            qu_series = self.calculate_qu(kline_data)

            if qu_series is not None:
                # 将日期、收盘价和QU值合并
                result_df = kline_data[['date', 'close']].copy()
                result_df['qu'] = qu_series
                result_df['symbol'] = symbol

                # 存储结果
                self.qu_data[symbol] = result_df
                print(f"成功处理 {symbol} 的数据，共 {len(result_df)} 条记录")
            else:
                print(f"计算 {symbol} 的QU指标失败")
        else:
            print(f"加载 {symbol} 的K线数据失败")

    def save_results(self, output_directory='checking'):
        """
        汇总所有符合条件的数据并保存到Excel文件

        Parameters:
        output_directory: str, 输出目录
        """
        # 创建输出目录
        if not os.path.exists(output_directory):
            os.makedirs(output_directory)

        if not self.qu_data:
            print("没有数据需要保存")
            return

        # 汇总所有数据
        all_data = []
        for symbol, df in self.qu_data.items():
            df_copy = df.copy()
            df_copy['symbol'] = symbol
            all_data.append(df_copy)

        if all_data:
            # 合并所有数据
            combined_df = pd.concat(all_data, ignore_index=True)

            # 按股票代码和日期排序
            combined_df = combined_df.sort_values(['symbol', 'date'])

            # 生成文件名
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = os.path.join(output_directory, f"qu_data_{timestamp}.xlsx")

            # 保存到Excel文件
            combined_df.to_excel(output_file, index=False)
            print(f"数据已保存到: {output_file}")
            print(f"共处理 {len(self.qu_data)} 只股票，总计 {len(combined_df)} 条记录")
        else:
            print("没有数据需要保存")

# data：online  or local
def get_stock_daily_data(code):
    processor = DayKLineProcessor(data_source='online', data_directory='kline_data')

    # 获取K线数据
    if processor.data_source == 'local':
        kline_data = processor.load_local_kline_data(code)
    else:
        kline_data = processor.load_online_kline_data(code)
    # kline_data = processor.load_online_kline_data(code)
    print(kline_data)

    if kline_data is not None and len(kline_data) > 0:
        # 计算QU指标
        kline_data = processor.calculate_sma(kline_data)
        print("sma数据:")
        # print(kline_data.tail())

        # 计算BOLL指标
        kline_data = DayKLineProcessor.calculate_boll(kline_data,29, 2)

        # 2023-09-22 15:00:00 date 字段取 日期 2023-09-22
        kline_data['date'] = pd.to_datetime(kline_data['date'])
        kline_data['date'] = kline_data['date'].dt.date
        # 打印出特定字段
        # print(kline_data[['date', 'close', 'sma', 'sma_ratio', 'upper', 'lower', 'band_width']].tail())


        kline_data = DayKLineProcessor.calculate_bias(kline_data,6)
        print(kline_data[['date', 'close', 'sma', 'sma_ratio', 'upper', 'lower', 'band_width','bias']].tail())


        # 返回 dataFrame 格式
        kline_data = kline_data[['date', 'close', 'sma', 'sma_ratio', 'upper', 'lower', 'band_width','bias']]
        return kline_data

    else:
        print(f"未能获取到 {code} 的K线数据")
        return None



# 使用示例
def main():
    # 创建处理器实例（使用本地数据）
    processor = DayKLineProcessor(data_source='local', data_directory='kline_data')

    # 方式1: 直接处理指定股票
    # processor.process_stock_kline('000001')

    # 方式2: 处理目录中的文件
    processor.process_directory_files('data_directory')

    # 获取指定日期的QU数据
    # qu_data = processor.get_qu_data_by_date('000001', '2023-12-01')
    # print(qu_data)

    # 预测未来QU值
    # prediction = processor.predict_qu('000001', days=5)
    # print(prediction)

    # 保存结果
    processor.save_results()

if __name__ == "__main__":
    # main()
    # get_daily_data('300006')
    # get_stock_daily_data('300006')
    # get_stock_daily_data('300528')
    get_stock_daily_data('688388')
