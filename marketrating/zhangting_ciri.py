import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import tushare as ts
import pywencai
from datetime import datetime, timedelta
from functools import lru_cache

# 利用tushare、wencai分析某天涨停股票次日涨跌幅情况
# 使用命令执行
# streamlit run D:/project/trade/marketrating/zhangting_ciri.py
# 打开浏览器就可以看到这个数据 http://172.16.10.23:8501


# Initialize Tushare with your token from environment variable or config file
import os
tushare_token = os.getenv('TUSHARE_TOKEN', '296cdbdc3ad507506ec8785465785f5ea065c81e8ec5b38b8b4e35ba')
ts.set_token(tushare_token)
pro = ts.pro_api()

def get_limit_up_stocks(date):
    try:
        query = f"{date}涨停"
        df = pywencai.get(query=query, sort_key='涨跌幅', sort_order='desc')
        return df[['股票代码', '股票简称', '最新价', '最新涨跌幅']]
    except Exception as e:
        st.error(f"获取涨停股票时发生错误: {e}")
        return pd.DataFrame()

@lru_cache(maxsize=128)
def get_next_trading_day(date):
    next_day = (datetime.strptime(date, '%Y%m%d') + timedelta(days=1)).strftime('%Y%m%d')
    while True:
        try:
            df = pro.trade_cal(exchange='', start_date=next_day, end_date=next_day)
            if df.iloc[0]['is_open'] == 1:
                return next_day
            next_day = (datetime.strptime(next_day, '%Y%m%d') + timedelta(days=1)).strftime('%Y%m%d')
        except Exception as e:
            st.error(f"获取下一个交易日时发生错误: {e}")
            return None

@lru_cache(maxsize=128)
def get_stock_data(stock_code, start_date, end_date):
    try:
        df = pro.daily(ts_code=stock_code, start_date=start_date, end_date=end_date)
        df['trade_date'] = pd.to_datetime(df['trade_date'])
        df = df.sort_values('trade_date')
        return df
    except Exception as e:
        st.error(f"获取股票数据时发生错误: {e}")
        return pd.DataFrame()

def calculate_next_day_performance(limit_up_stocks, date):
    next_trading_day = get_next_trading_day(date)
    if not next_trading_day:
        return pd.DataFrame()

    results = []
    for _, row in limit_up_stocks.iterrows():
        stock_code = row['股票代码']
        stock_name = row['股票简称']

        df = get_stock_data(stock_code, date, next_trading_day)
        if len(df) >= 2:
            limit_up_price = df.iloc[0]['close']
            next_day_price = df.iloc[1]['close']
            change_pct = (next_day_price - limit_up_price) / limit_up_price * 100

            results.append({
                '股票代码': stock_code,
                '股票简称': stock_name,
                '涨停价': limit_up_price,
                '次日收盘价': next_day_price,
                '次日涨跌幅': change_pct
            })

    return pd.DataFrame(results)

def display_stock_analysis(stock_code, selected_date):
    end_date = (datetime.now() + timedelta(days=1)).strftime('%Y%m%d')
    start_date = (datetime.strptime(selected_date, '%Y%m%d') - timedelta(days=60)).strftime('%Y%m%d')

    df = get_stock_data(stock_code, start_date, end_date)
    if not df.empty:
        selected_datetime = pd.to_datetime(selected_date)

        fig = go.Figure()
        fig.add_trace(go.Candlestick(
            x=df['trade_date'],
            open=df['open'],
            high=df['high'],
            low=df['low'],
            close=df['close'],
            increasing_line_color='red',
            decreasing_line_color='green',
            name="股价"
        ))

        if not df[df['trade_date'] == selected_datetime].empty:
            selected_price = df[df['trade_date'] == selected_datetime]['close'].values[0]
            fig.add_trace(go.Scatter(
                x=[selected_datetime],
                y=[selected_price],
                mode='markers',
                marker=dict(size=10, color='blue', symbol='star'),
                name="选择日期"
            ))

        fig.update_layout(
            title=f"{stock_code} 近期走势",
            xaxis_title="日期",
            yaxis_title="价格",
            xaxis_rangeslider_visible=False,
            plot_bgcolor='white',
            paper_bgcolor='white',
        )

        st.plotly_chart(fig)
    else:
        st.error("无法获取股票数据，请检查股票代码是否正确。")

def main():
    st.title("涨停股票分析")

    default_date = datetime(2024, 10, 8).date()
    selected_date = st.date_input("选择日期", value=default_date, min_value=datetime(2020, 1, 1).date(), max_value=datetime.now().date())
    date_str = selected_date.strftime('%Y%m%d')

    limit_up_stocks = get_limit_up_stocks(date_str)
    next_day_performance = calculate_next_day_performance(limit_up_stocks, date_str)

    st.subheader(f"{date_str} 涨停股票及次日表现")

    def style_negative(v, props=''):
        return props if v < 0 else None

    def style_positive(v, props=''):
        return props if v > 0 else None

    styled_df = next_day_performance.style.format({
        '涨停价': '{:.2f}',
        '次日收盘价': '{:.2f}',
        '次日涨跌幅': '{:.2f}%'
    }).applymap(style_negative, props='color:green;', subset=['次日涨跌幅']) \
      .applymap(style_positive, props='color:red;', subset=['次日涨跌幅'])

    st.dataframe(styled_df)

    selected_stock = st.selectbox("选择股票进行详细分析", next_day_performance['股票代码'] + ' - ' + next_day_performance['股票简称'])

    if selected_stock:
        stock_code = selected_stock.split(' - ')[0]
        display_stock_analysis(stock_code, date_str)

if __name__ == "__main__":
    main()
