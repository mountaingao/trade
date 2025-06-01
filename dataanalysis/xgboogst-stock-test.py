"""

streamlit run F:/project/trade/dataanalysis/xgboogst-stock-test.py

"""



import streamlit as st
import akshare as ak
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from xgboost import XGBClassifier
import joblib
import talib
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

# 用于技术指标计算
def add_features(df):
    df['MA5'] = talib.SMA(df['close'], timeperiod=5)
    df['RSI'] = talib.RSI(df['close'], timeperiod=14)
    df['MACD'], _, _ = talib.MACD(df['close'], fastperiod=12, slowperiod=26, signalperiod=9)
    return df

def prepare_data(df):
    df['target'] = (df['close'].shift(-1) > df['close']).astype(int)
    df = add_features(df)
    df.dropna(inplace=True)
    return df

# 页面标题
st.set_page_config(page_title="股票涨跌预测", layout="wide")
st.title("股票涨跌预测系统")

# 用户输入
stock_code = st.text_input("请输入股票代码（如000001）", "000001")
start_date = st.date_input("起始日期", pd.to_datetime("2023-01-01"))
end_date = st.date_input("结束日期", pd.to_datetime("2025-05-31"))

if st.button("开始预测"):
    # try:
        st.info("正在获取数据并处理...")

        # 修改后的数据获取逻辑
        market_prefix = "0" if stock_code.startswith(("0", "3")) else "1"
        # 修改后的数据获取逻辑（增加市场前缀判断）
        if stock_code.startswith(("0", "3", "4")):
            secid = f"sz{stock_code}"  # 深市/创业板/北交所
        elif stock_code.startswith(("6", "5", "9")):
            secid = f"sh{stock_code}"  # 沪市/科创板/其他
        else:
            raise ValueError("无效的股票代码格式")

        # {market_prefix}.
        df_raw = ak.stock_zh_a_daily(
            symbol=secid,
            # period="daily",
            start_date=start_date.strftime('%Y%m%d'),
            end_date=end_date.strftime('%Y%m%d'),
            adjust="qfq"
        )

        # 打印数据
        # print(df_raw)
        df = df_raw.rename(columns={
            "开盘": "open",
            "收盘": "close",
            "最高": "high",
            "最低": "low",
            "成交量": "volume",
            "日期": "date"
        })

        df['date'] = pd.to_datetime(df['date'])
        df.set_index("date", inplace=True)
        df = prepare_data(df)

        features = ['MA5', 'RSI', 'MACD']
        X = df[features]
        y = df['target']

        X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=False, test_size=0.2)

        model = XGBClassifier(n_estimators=100, max_depth=3, learning_rate=0.1)
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        print(f"预测准确率：{acc:.2f}")
        st.success(f"预测准确率：{acc:.2f}")
        print(df)
        print(len(y_pred))
        # df['pred'] = y_pred
        # df['correct'] = df['pred'] == df['target']
        # 只给测试集对应的df行赋值
        df.loc[X_test.index, 'pred'] = y_pred
        df.loc[X_test.index, 'correct'] = df.loc[X_test.index, 'pred'] == df.loc[X_test.index, 'target']
        print(df)

        # 绘制K线图
        fig = go.Figure(data=[
            go.Candlestick(
                x=df.index,
                open=df['open'],
                high=df['high'],
                low=df['low'],
                close=df['close'],
                name="K线"
            )
        ])
        df.dropna(inplace=True)
        # 正确上涨（预测1，实际1）
        correct_df = df[df['correct'] & (df['pred'] == 1)]
        fig.add_trace(go.Scatter(
            x=correct_df.index,
            y=correct_df['close'],
            mode='markers',
            marker=dict(symbol='triangle-up', color='green', size=10),
            name='预测正确上涨'
        ))

        # 预测上涨但错了
        wrong_df = df[~df['correct'] & (df['pred'] == 1)]
        fig.add_trace(go.Scatter(
            x=wrong_df.index,
            y=wrong_df['close'],
            mode='markers',
            marker=dict(symbol='triangle-down', color='red', size=10),
            name='预测错误上涨'
        ))

        st.plotly_chart(fig, use_container_width=True)

        st.caption("本应用仅用于技术演示，不构成投资建议。股市有风险，入市需谨慎。")

    # except Exception as e:
    #     print(e)
    #     st.error(f"系统错误：{str(e)}")

