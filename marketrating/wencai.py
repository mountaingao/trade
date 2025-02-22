import pywencai
import streamlit as st
import pywencai
import requests
import json
import pandas as pd

df = pywencai.get(query="短线复盘")
try:
    # ================== 核心指标看板 ==================
    st.header("📊 核心指标看板")
    cols = st.columns(7)

    zt_data = df.get('涨停梳理', {}).get('newCard', pd.DataFrame())
    if not zt_data.empty:
        metrics = {
            "涨跌停比": (zt_data.iloc[0, 4], "涨停家数与跌停家数比例"),
            "昨日涨停表现": (f"{float(zt_data.iloc[1, 0]):.2f}%", "昨日涨停股票的市场表现"),
            "连板率": (f"{zt_data.iloc[2, 0]}%", "连续涨停板占比"),
            "封板率": (f"{zt_data.iloc[3, 0]}%", "成功封住涨停的比例"),
            "连板收益率": (zt_data.iloc[4, 0], "连板收益率"),
            "连板溢价率": (zt_data.iloc[5, 0], "连板溢价率"),
            "情绪温度": (f"{df['情绪温度']['line3'].iloc[-1]['情绪温度走势']:.1f}℃", "市场情绪综合指标")
        }

        for col, (name, (value, help_text)) in zip(cols, metrics.items()):
                with col:
                    st.metric(name, value, help=help_text)

    # ================== 市场动向分析 ==================
    st.header("🌐 市场动向分析")
    market_trend = df.get('市场动向', '')
    if market_trend and len(market_trend) > 10:
            st.markdown(market_trend, unsafe_allow_html=True)
    else:
        st.info("📅 市场动向数据将于交易日15:30更新，请稍后查看")

    # ================== 情绪温度走势 ==================
    st.header("🌡️ 情绪温度趋势")
    emotion_data = df.get('情绪温度', {}).get('line3', pd.DataFrame())
    if not emotion_data.empty:
        try:
            emotion_data['日期'] = pd.to_datetime(
                emotion_data['时间区间'].astype(str).str[:8],
                format='%Y%m%d',
                errors='coerce'
            )
            valid_data = emotion_data.dropna(subset=['日期'])
            if not valid_data.empty:
                    st.line_chart(
                    valid_data.set_index('日期')['情绪温度走势'].rename("情绪温度"),
                    use_container_width=True
                )
            else:
                st.warning("有效日期数据缺失")
        except Exception as e:
            st.error(f"图表渲染失败: {str(e)}")
    else:
        st.warning("情绪温度数据暂不可用")

    # ================== 详细数据解析 ==================
    st.header("📚 详细数据解析")
    tabs = st.tabs(["连板天梯", "焦点个股", "市场事件"])

    with tabs[0]:
        st.subheader("🏆 连板天梯榜")
        lb_data = df.get('连板天梯', {})
        if lb_data:
            st.write("### 高位连板")
            if'高位板'in lb_data and 'tableV1'in lb_data['高位板']:
                    st.dataframe(lb_data['高位板']['tableV1'], height=400)
            else:
                st.info("暂无高位连板数据")


            st.write("### 两连板")
            if'两连板'in lb_data and 'tableV1'in lb_data['两连板']:
                    st.dataframe(lb_data['两连板']['tableV1'], height=400)
            else:
                st.info("暂无两连板数据")
        else:
            st.warning("连板数据加载失败")

    with tabs[1]:
        st.subheader("🔥 焦点个股分析")
        focus_stocks = df.get('焦点股', pd.DataFrame())
        if not focus_stocks.empty:
                st.dataframe(
                focus_stocks,
                column_config={
                    "股票简称": "名称",
                    "连涨天数": st.column_config.NumberColumn(
                        "连涨",
                        help="连续上涨天数",
                        format="%d 天"
                    ),
                    "涨跌幅:前属同花顺行业": st.column_config.ProgressColumn(
                        "行业涨幅",
                        format="%.2f%%",
                        min_value=-10,
                        max_value=10
                    )
                },
                hide_index=True,
                use_container_width=True
            )
        else:
            st.info("今日无重点监控个股")

    with tabs[2]:
        st.subheader("📅 市场事件追踪")
        event_data = df.get('自选股近15天重要事件', {})
        if event_data:
            col1, col2 = st.columns([3, 2])
            with col1:
                st.write("### 自选股事件")
                if'tableV1'in event_data:
                        st.dataframe(event_data['tableV1'], use_container_width=True)
                else:
                    st.info("近期无自选股相关事件")

            with col2:
                st.write("### 实时要闻")
                if'txt1'in event_data:
                        st.markdown(event_data['txt1'], unsafe_allow_html=True)
        else:
            st.warning("事件数据加载失败")

except Exception as e:
    st.error(f"系统错误: {str(e)}")
    st.button("🔄 重新加载数据", type="primary")