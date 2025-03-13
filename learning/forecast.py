from statsmodels.tsa.arima.model import ARIMA

# 假设已有价格时间序列数据
price_series = minute_data["close"].values

# 训练 ARIMA 模型
model = ARIMA(price_series, order=(5, 1, 0))
model_fit = model.fit()

# 预测未来 30 分钟的价格
forecast = model_fit.forecast(steps=30)
print("未来 30 分钟价格预测：", forecast)