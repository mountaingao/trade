# 假设有一个包含每日成交量的列表 volume_data
volume_data = [100, 150, 200, 250, 300, 350, 400, 450, 500, 550, 600, 650, 700]

# 计算最近3日、5日、8日、13日的成交量之和
recent_3_days = sum(volume_data[-3:])
recent_5_days = sum(volume_data[-5:])
recent_8_days = sum(volume_data[-8:])
recent_13_days = sum(volume_data[-13:])

# 计算前3日、5日、8日、13日的成交量之和
previous_3_days = sum(volume_data[-6:-3])
previous_5_days = sum(volume_data[-10:-5])
previous_8_days = sum(volume_data[-16:-8])
previous_13_days = sum(volume_data[-26:-13])

# 计算放量比例
volume_ratio_3_days = recent_3_days / previous_3_days
volume_ratio_5_days = recent_5_days / previous_5_days
volume_ratio_8_days = recent_8_days / previous_8_days
volume_ratio_13_days = recent_13_days / previous_13_days

# 输出结果
print(f"最近3日放量比例: {volume_ratio_3_days}")
print(f"最近5日放量比例: {volume_ratio_5_days}")
print(f"最近8日放量比例: {volume_ratio_8_days}")
print(f"最近13日放量比例: {volume_ratio_13_days}")


一、常见技术指标的中英文对照与计算方法
1. 移动平均线 (Moving Average, MA)
简单移动平均线 (SMA):

python
df['SMA_10'] = df['Close'].rolling(window=10).mean()
英文: Simple Moving Average (SMA)

计算: 最近N个收盘价的算术平均值

指数移动平均线 (EMA):

python
df['EMA_12'] = df['Close'].ewm(span=12, adjust=False).mean()
英文: Exponential Moving Average (EMA)

计算: 给予近期价格更高权重的移动平均

2. 相对强弱指数 (Relative Strength Index, RSI)
python
delta = df['Close'].diff()
gain = delta.where(delta > 0, 0)
loss = -delta.where(delta < 0, 0)
avg_gain = gain.rolling(window=14).mean()
avg_loss = loss.rolling(window=14).mean()
rs = avg_gain / avg_loss
df['RSI'] = 100 - (100 / (1 + rs))
英文: Relative Strength Index (RSI)

计算: 100 - (100 / (1 + 平均涨幅/平均跌幅))

3. 移动平均收敛发散 (MACD)
python
df['EMA_12'] = df['Close'].ewm(span=12, adjust=False).mean()
df['EMA_26'] = df['Close'].ewm(span=26, adjust=False).mean()
df['DIF'] = df['EMA_12'] - df['EMA_26']
df['DEA'] = df['DIF'].ewm(span=9, adjust=False).mean()
df['MACD'] = (df['DIF'] - df['DEA']) * 2
英文: Moving Average Convergence Divergence (MACD)

组成:

DIF (Difference Line): 12日EMA - 26日EMA

DEA (Signal Line): DIF的9日EMA

MACD柱状图: (DIF-DEA)×2

4. 布林带 (Bollinger Bands)
python
df['MA_20'] = df['Close'].rolling(window=20).mean()
df['Upper'] = df['MA_20'] + 2 * df['Close'].rolling(window=20).std()
df['Lower'] = df['MA_20'] - 2 * df['Close'].rolling(window=20).std()
英文: Bollinger Bands (BB)

组成:

中轨: N日移动平均线

上轨: 中轨 + K×标准差

下轨: 中轨 - K×标准差

(通常N=20，K=2)

5. 随机指标 (Stochastic Oscillator)
python
low_14 = df['Low'].rolling(window=14).min()
high_14 = df['High'].rolling(window=14).max()
df['%K'] = 100 * ((df['Close'] - low_14) / (high_14 - low_14))
df['%D'] = df['%K'].rolling(window=3).mean()
英文: Stochastic Oscillator

计算:

%K = (当前收盘价 - N日内最低价)/(N日内最高价 - N日内最低价) × 100

%D = %K的M日移动平均

(通常N=14，M=3)

二、技术指标分类英文表达
中文名称	英文表达	主要用途
趋势指标	Trend Indicators	判断市场趋势方向
动量指标	Momentum Indicators	衡量价格变化速度
波动率指标	Volatility Indicators	测量价格波动幅度
成交量指标	Volume Indicators	分析交易量变化
超买超卖指标	Overbought/Oversold	判断市场极端状态
三、技术指标应用场景英文表达
趋势跟踪 (Trend Following):

"We use moving averages to identify the prevailing market trend."

中文: 我们使用移动平均线来识别当前市场趋势。

超买超卖信号 (Overbought/Oversold Signals):

"The RSI above 70 indicates overbought conditions, while below 30 suggests oversold."

中文: RSI高于70表示超买，低于30表示超卖。

背离分析 (Divergence Analysis):

"Bearish divergence occurs when price makes higher highs while the indicator shows lower highs."

中文: 当价格创出更高高点而指标出现更低高点时，形成看跌背离。

交叉信号 (Crossover Signals):

"A bullish signal is generated when the fast MA crosses above the slow MA."

中文: 当快速均线上穿慢速均线时产生看涨信号。