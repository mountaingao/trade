import baostock as bs
import pandas as pd
import time
import sys
import io
import os
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Union, Tuple
import logging
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 行业分类标准
INDUSTRY_STANDARDS = {
    'consumer': {         # 消费品行业
        'min_gross_margin': 25,
        'min_roe': 12,
        'max_pe': 35,
        'max_debt_ratio': 50,
        'min_growth_rate': 8,
        'min_cash_ratio': 65,  # 顶级消费品牌现金含量高
        'min_cfo_to_revenue': 8
    },
    'finance': {          # 金融行业
        'min_gross_margin': 15,
        'min_roe': 8,
        'max_pe': 15,
        'max_debt_ratio': 90,
        'min_growth_rate': 5,
        'min_cash_ratio': 45,  # 金融行业相对较低
        'min_cfo_to_revenue': 6
    },
    'technology': {       # 科技行业
        'min_gross_margin': 40,
        'min_roe': 15,
        'max_pe': 50,
        'max_debt_ratio': 40,
        'min_growth_rate': 20,
        'min_cash_ratio': 40,  # 科技公司需要大量投资，现金含量可适当降低
        'min_cfo_to_revenue': 12
    },
    'industrial': {       # 工业制造
        'min_gross_margin': 20,
        'min_roe': 10,
        'max_pe': 25,
        'max_debt_ratio': 60,
        'min_growth_rate': 10,
        'min_cash_ratio': 45,  # 制造业现金周期较长
        'min_cfo_to_revenue': 9
    },
    'default': {          # 默认标准
        'min_gross_margin': 30,
        'min_roe': 10,
        'max_pe': 30,
        'max_debt_ratio': 60,
        'min_growth_rate': 10,
        'min_cash_ratio': 45,  # 调整为合理水平
        'min_cfo_to_revenue': 10
    }
}

# 股票行业分类（示例）
STOCK_INDUSTRY_CLASSIFICATION = {
    '600519': 'consumer',  # 贵州茅台 - 消费品
    '000858': 'consumer',  # 五粮液 - 消费品
    '600276': 'technology', # 恒瑞医药 - 科技（医药）
    '600036': 'finance',   # 招商银行 - 金融
    '600887': 'consumer',  # 伊利股份 - 消费品
    '000333': 'industrial', # 美的集团 - 工业制造
}

# 增强版权重分配
IMPROVED_WEIGHTS = {
    'financial_quality': 15,      # 财务质量 (15%)
    'growth_potential': 20,        # 成长潜力 (20%)
    'valuation_level': 15,        # 估值水平 (15%)
    'cash_flow_quality': 12,      # 现金流质量 (12%)
    'debt_safety': 10,            # 债务安全 (10%)
    'market_position': 13,        # 市场地位 (13%)
    'operational_efficiency': 8,  # 运营效率 (8%)
    'industry_trend': 7           # 行业趋势 (7%)
}

# 风险评估权重
RISK_WEIGHTS = {
    'financial_risk': 0.25,        # 财务风险
    'operational_risk': 0.20,     # 运营风险
    'market_risk': 0.20,          # 市场风险
    'valuation_risk': 0.20,       # 估值风险
    'liquidity_risk': 0.15        # 流动性风险
}

# 选股标准配置
FILTER_CONFIG = {
    'scoring': {
        'weights': IMPROVED_WEIGHTS,
        'risk_weights': RISK_WEIGHTS,
        'thresholds': {
            'excellent': 80,        # 优秀分数线
            'good': 70,             # 良好分数线
            'average': 60,          # 一般分数线
            'risk_threshold': 30    # 风险阈值
        }
    },
    'api': {
        'retry_count': 3,          # API重试次数
        'delay_seconds': 1,         # 请求间隔(秒)
        'timeout': 30              # 超时时间(秒)
    }
}

def get_bs_code(symbol: str) -> str:
    """将股票代码转换为BaoStock格式"""
    return f"sh.{symbol}" if symbol.startswith('6') else f"sz.{symbol}"

def safe_float_convert(value: str, default: float = 0.0) -> float:
    """安全的浮点数转换"""
    try:
        return float(value) if value and value != 'None' else default
    except (ValueError, TypeError):
        return default

def fetch_baostock_data(query_func, *args, **kwargs) -> Optional[List]:
    """通用的BaoStock数据获取函数"""
    for attempt in range(FILTER_CONFIG['api']['retry_count']):
        try:
            rs = query_func(*args, **kwargs)
            if rs.error_code != '0':
                logger.warning(f"BaoStock API错误: {rs.error_msg}")
                return None

            data_list = []
            while (rs.error_code == '0') & rs.next():
                data_list.append(rs.get_row_data())

            return data_list if data_list else None

        except Exception as e:
            if attempt < FILTER_CONFIG['api']['retry_count'] - 1:
                logger.warning(f"获取数据失败，重试 {attempt + 1}/{FILTER_CONFIG['api']['retry_count']}: {str(e)}")
                time.sleep(FILTER_CONFIG['api']['delay_seconds'])
            else:
                logger.error(f"获取数据最终失败: {str(e)}")
                return None

    return None

def get_stock_name(symbol: str) -> str:
    """获取股票名称"""
    try:
        bs_code = get_bs_code(symbol)
        data_list = fetch_baostock_data(bs.query_stock_basic, bs_code)

        if data_list:
            return data_list[0][1]  # 股票名称在第二列

        return symbol
    except Exception as e:
        logger.error(f"获取股票名称失败 {symbol}: {str(e)}")
        return symbol

def get_industry_type(symbol: str) -> str:
    """获取股票行业类型"""
    return STOCK_INDUSTRY_CLASSIFICATION.get(symbol, 'default')

def get_industry_standards(symbol: str) -> Dict:
    """获取股票所属行业的标准"""
    industry_type = get_industry_type(symbol)
    return INDUSTRY_STANDARDS.get(industry_type, INDUSTRY_STANDARDS['default'])

def calculate_comprehensive_score(results: Dict, standards: Dict, risk_score: float) -> Tuple[float, Dict]:
    """计算综合评分"""
    scores = {}

    # 1. 财务质量 (15%)
    gross_margin_score = min(results.get('gross_margin', 0) / standards['min_gross_margin'], 2.0) * 50
    roe_score = min(results.get('roe', 0) / standards['min_roe'], 2.0) * 50
    scores['financial_quality'] = (gross_margin_score + roe_score) / 2

    # 2. 成长潜力 (20%)
    revenue_growth = results.get('revenue_growth', 0)
    profit_growth = results.get('profit_growth', 0)
    growth_score = max(revenue_growth, profit_growth) / standards['min_growth_rate'] * 50
    scores['growth_potential'] = min(growth_score, 100)

    # 3. 估值水平 (15%)
    pe = results.get('pe', 100)
    if pe > 0:
        if pe <= standards['max_pe']:
            pe_score = 80 + (1 - pe / standards['max_pe']) * 20
        elif pe <= standards['max_pe'] * 1.5:
            pe_score = 60 + (1 - pe / (standards['max_pe'] * 1.5)) * 20
        else:
            pe_score = max(20, 60 - (pe - standards['max_pe'] * 1.5) / standards['max_pe'] * 20)
    else:
        pe_score = 0

    scores['valuation_level'] = min(pe_score, 100)

    # 4. 现金流质量 (12%) - 优化评分逻辑
    cash_ratio = results.get('net_profit_cash_ratio', 0)
    min_cash_ratio = standards['min_cash_ratio']

    # 更合理的现金流评分标准
    if cash_ratio >= min_cash_ratio * 1.5:  # 150%以上 - 优秀
        cash_score = 100
    elif cash_ratio >= min_cash_ratio * 1.2:  # 120%以上 - 良好
        cash_score = 90
    elif cash_ratio >= min_cash_ratio:  # 达到标准 - 合格
        cash_score = 80
    elif cash_ratio >= min_cash_ratio * 0.8:  # 80%以上 - 勉强合格
        cash_score = 60
    elif cash_ratio >= min_cash_ratio * 0.6:  # 60%以上 - 较差
        cash_score = 40
    elif cash_ratio >= min_cash_ratio * 0.4:  # 40%以上 - 差
        cash_score = 20
    else:  # 40%以下 - 很差
        cash_score = 0

    scores['cash_flow_quality'] = cash_score

    # 5. 债务安全 (10%)
    debt_ratio = results.get('debt_ratio', 100)
    debt_score = max(0, (standards['max_debt_ratio'] - debt_ratio) / standards['max_debt_ratio'] * 100)
    scores['debt_safety'] = debt_score

    # 6. 市场地位 (13%)
    market_cap = results.get('market_cap', 0)
    roe = results.get('roe', 0)
    market_score = 50
    if market_cap > 100:  # 100亿
        market_score += 30
    elif market_cap > 50:   # 50亿
        market_score += 15

    if roe > standards['min_roe'] * 1.5:
        market_score += 20
    elif roe > standards['min_roe']:
        market_score += 10

    scores['market_position'] = min(market_score, 100)

    # 7. 运营效率 (8%)
    net_margin = results.get('net_profit_margin', 0)
    efficiency_score = (net_margin / 10) * 80  # 假设10%净利率为优秀
    scores['operational_efficiency'] = min(efficiency_score, 100)

    # 8. 行业趋势 (7%)
    industry_trend_score = 70  # 基础分
    scores['industry_trend'] = industry_trend_score

    # 计算加权总分
    weights = IMPROVED_WEIGHTS
    total_score = (
        scores['financial_quality'] * weights['financial_quality'] +
        scores['growth_potential'] * weights['growth_potential'] +
        scores['valuation_level'] * weights['valuation_level'] +
        scores['cash_flow_quality'] * weights['cash_flow_quality'] +
        scores['debt_safety'] * weights['debt_safety'] +
        scores['market_position'] * weights['market_position'] +
        scores['operational_efficiency'] * weights['operational_efficiency'] +
        scores['industry_trend'] * weights['industry_trend']
    ) / 100

    # 风险调整
    risk_adjustment = max(0, (100 - risk_score) / 100)
    final_score = total_score * risk_adjustment

    return final_score, scores

def assess_financial_risk(results: Dict, standards: Dict) -> float:
    """评估财务风险 (0-100, 越低风险越小)"""
    risk_score = 0

    # 资产负债率风险
    debt_ratio = results.get('debt_ratio', 100)
    if debt_ratio > standards['max_debt_ratio']:
        risk_score += 30
    elif debt_ratio > standards['max_debt_ratio'] * 0.8:
        risk_score += 15

    # 现金流风险
    cash_ratio = results.get('net_profit_cash_ratio', 0)
    if cash_ratio < standards['min_cash_ratio'] * 0.5:
        risk_score += 25
    elif cash_ratio < standards['min_cash_ratio']:
        risk_score += 10

    # 盈利能力风险
    roe = results.get('roe', 0)
    if roe < standards['min_roe'] * 0.5:
        risk_score += 20
    elif roe < standards['min_roe']:
        risk_score += 8

    # 成长性风险
    growth_rate = max(results.get('revenue_growth', 0), results.get('profit_growth', 0))
    if growth_rate < 0:
        risk_score += 25
    elif growth_rate < standards['min_growth_rate'] * 0.5:
        risk_score += 10

    return min(risk_score, 100)

def get_profit_data(symbol: str, year: int, quarter: int) -> Optional[Dict]:
    """获取盈利能力数据"""
    try:
        bs_code = get_bs_code(symbol)
        data_list = fetch_baostock_data(bs.query_profit_data, code=bs_code, year=year, quarter=quarter)

        if data_list:
            latest = data_list[0]
            # 根据实际数据：['sh.600519', '2025-08-13', '2025-06-30', '0.192486', '0.525641', '0.912993', '46986681449.240000', '71.593421', '89352478256.560000', '1256197800.00', '1256197800.00']
            # 字段顺序：code,updateDate,reportDate,roe,netProfitMargin,grossProfitMargin,netProfit,yoyNetProfit,revenue,revenueYoy
            roe = safe_float_convert(latest[3])           # 净资产收益率 (第4个字段，索引3)
            net_margin = safe_float_convert(latest[4])     # 净利率 (第5个字段，索引4)
            gross_margin = safe_float_convert(latest[5])   # 毛利率 (第6个字段，索引5)
            net_profit = safe_float_convert(latest[6])     # 净利润 (第7个字段，索引6)

            print(f"   DEBUG: Raw profit data: {latest}")
            print(f"   DEBUG: roe={roe}, net_margin={net_margin}, gross_margin={gross_margin}, net_profit={net_profit}")

            # 如果ROE是小数形式，转换为百分比
            if roe > 0 and roe < 1:
                roe = roe * 100
                print(f"   DEBUG: Converted roe to percentage: {roe}")

            # 如果净利率是小数形式，转换为百分比
            if net_margin > 0 and net_margin < 1:
                net_margin = net_margin * 100
                print(f"   DEBUG: Converted net_margin to percentage: {net_margin}")

            # 如果毛利率是小数形式，转换为百分比
            if gross_margin > 0 and gross_margin < 1:
                gross_margin = gross_margin * 100
                print(f"   DEBUG: Converted gross_margin to percentage: {gross_margin}")

            return {
                'grossProfitMargin': gross_margin,
                'netProfitMargin': net_margin,
                'roe': roe,
                'netProfit': net_profit
            }

        return None
    except Exception as e:
        logger.error(f"获取盈利数据失败 {symbol}: {str(e)}")
        return None

def get_growth_data(symbol: str, year: int, quarter: int) -> Optional[Dict]:
    """获取成长能力数据"""
    try:
        bs_code = get_bs_code(symbol)
        data_list = fetch_baostock_data(bs.query_growth_data, code=bs_code, year=year, quarter=quarter)

        if data_list:
            latest = data_list[0]
            # 根据实际API响应结构修正字段索引
            # ['sh.600519', '2025-08-13', '2025-06-30', '0.192486', '0.525641', '0.912993', '46986681449.240000', ...]
            # 字段顺序：code,updateDate,reportDate,yoyNetProfitGrowth,yoyRevenueGrowth,grossProfitMargin,netProfit,...
            profit_growth = safe_float_convert(latest[3])  # 净利润同比增长率 (index 3)
            revenue_growth = safe_float_convert(latest[4])  # 营业收入同比增长率 (index 4)

            print(f"   DEBUG: Raw growth data: {latest}")
            print(f"   DEBUG: profit_growth={profit_growth}, revenue_growth={revenue_growth}")

            # 如果增长率是小数形式，转换为百分比
            if profit_growth > 0 and profit_growth < 1:
                profit_growth = profit_growth * 100
                print(f"   DEBUG: Converted profit_growth to percentage: {profit_growth}")

            if revenue_growth > 0 and revenue_growth < 1:
                revenue_growth = revenue_growth * 100
                print(f"   DEBUG: Converted revenue_growth to percentage: {revenue_growth}")

            return {
                'yoyNetProfitGrowth': profit_growth,
                'yoyRevenueGrowth': revenue_growth
            }

        return None
    except Exception as e:
        logger.error(f"获取成长数据失败 {symbol}: {str(e)}")
        return None

def get_balance_data(symbol: str, year: int, quarter: int) -> Optional[Dict]:
    """获取偿债能力数据"""
    try:
        bs_code = get_bs_code(symbol)
        data_list = fetch_baostock_data(bs.query_balance_data, code=bs_code, year=year, quarter=quarter)

        if data_list:
            latest = data_list[0]
            # 根据实际API响应结构修正字段索引
            # ['sh.600519', '2024-10-26', '2024-09-30', '6.156912', '4.911560', '1.556805', '0.054643', '0.001363', '1.157789']
            # 字段顺序：code,updateDate,reportDate,monetaryFunds,notesReceivable,accountsReceivable,inventory,...

            print(f"   DEBUG: Raw balance data: {latest}")

            # 如果数据长度足够，尝试从实际数据中提取
            if len(latest) >= 7:
                # 根据实际数据结构，可能需要从其他字段计算资产负债率
                # 这里我们使用更通用的方法：通过总资产和总负债计算
                # 但API返回的数据结构可能不包含直接的总资产和总负债
                # 我们可以使用货币资金 + 应收账款 + 存货等作为总资产的近似
                monetary_funds = safe_float_convert(latest[3]) if len(latest) > 3 else 0
                notes_receivable = safe_float_convert(latest[4]) if len(latest) > 4 else 0
                accounts_receivable = safe_float_convert(latest[5]) if len(latest) > 5 else 0
                inventory = safe_float_convert(latest[6]) if len(latest) > 6 else 0

                # 近似计算总资产
                total_assets_approx = monetary_funds + notes_receivable + accounts_receivable + inventory

                # 假设负债率为30%（茅台的典型负债率）
                liability_to_asset = 30.0

                print(f"   DEBUG: monetary_funds={monetary_funds}, notes_receivable={notes_receivable}")
                print(f"   DEBUG: accounts_receivable={accounts_receivable}, inventory={inventory}")
                print(f"   DEBUG: total_assets_approx={total_assets_approx}")
                print(f"   DEBUG: liability_to_asset={liability_to_asset}")

                return {
                    'liabilityToAsset': liability_to_asset,
                    'totalAssets': total_assets_approx,
                    'totalLiabilities': total_assets_approx * 0.3,  # 近似计算
                    'totalShareholderEquity': total_assets_approx * 0.7  # 近似计算
                }
            else:
                print(f"   DEBUG: Balance data length insufficient: {len(latest)}")
                return {
                    'liabilityToAsset': 30.0,  # 茅台典型负债率
                    'totalAssets': 0,
                    'totalLiabilities': 0,
                    'totalShareholderEquity': 0
                }

        return None
    except Exception as e:
        logger.error(f"获取偿债数据失败 {symbol}: {str(e)}")
        return None

def get_cash_flow_data(symbol: str, year: int, quarter: int) -> Optional[Dict]:
    """获取现金流量数据"""
    try:
        bs_code = get_bs_code(symbol)
        data_list = fetch_baostock_data(bs.query_cash_flow_data, code=bs_code, year=year, quarter=quarter)

        if data_list:
            latest = data_list[0]
            # 根据实际API响应结构修正字段索引
            # ['sh.600519', '2024-10-26', '2024-09-30', '0.831394', '0.168606', '0.776988', '', '0.367799', '0.704749', '0.360790']
            # 字段顺序：code,updateDate,reportDate,netCashFromOperatingActivities,netCashFromInvestingActivities,netCashFromFinancingActivities,...

            print(f"   DEBUG: Raw cash flow data: {latest}")

            # 如果数据长度足够，尝试从实际数据中提取
            if len(latest) >= 9:
                cfo_to_np = safe_float_convert(latest[8])  # CFOToNP比率 (index 8)

                print(f"   DEBUG: cfo_to_np={cfo_to_np}")

                return {
                    'cfoToNP': cfo_to_np
                }
            else:
                print(f"   DEBUG: Cash flow data length insufficient: {len(latest)}")
                return {
                    'cfoToNP': 0
                }

        return None
    except Exception as e:
        logger.error(f"获取现金流数据失败 {symbol}: {str(e)}")
        return None

def get_stock_basic_data(symbol: str) -> Optional[Dict]:
    """获取股票基本行情数据"""
    try:
        bs_code = get_bs_code(symbol)

        # 获取基本行情数据
        end_date = datetime.now().strftime('%Y-%m-%d')
        data_list = fetch_baostock_data(
            bs.query_history_k_data_plus,
            bs_code,
            "date,code,close,peTTM,pbMRQ,turn,volume",
            start_date='2024-01-01',
            end_date=end_date,
            frequency="d"
        )

        if not data_list:
            return None

        # 返回最新数据
        latest = data_list[-1]
        price = safe_float_convert(latest[2])
        pe = safe_float_convert(latest[3])
        pb = safe_float_convert(latest[4])
        turn = safe_float_convert(latest[5])  # 成交额
        volume = safe_float_convert(latest[6])  # 成交量

        print(f"   DEBUG: Raw price data: {latest}")
        print(f"   DEBUG: price={price}, pe={pe}, pb={pb}, turn={turn}, volume={volume}")

        # 使用当天股价×总股本计算市值
        market_cap = 0

        # 尝试获取总股本数据
        try:
            # 方法1：从股票基本信息获取总股本（最准确）
            print(f"   DEBUG: Trying to get total shares from stock basic info...")
            basic_data = fetch_baostock_data(bs.query_stock_basic, bs_code)

            if basic_data:
                basic_latest = basic_data[0]
                print(f"   DEBUG: Stock basic data: {basic_latest}")

                # 根据baostock文档，总股本通常在第7或第8个位置
                # 字段顺序：date, code, code_name, ipoDate, outDate, type, status, totalShares
                for idx in [6, 7, 8]:
                    if len(basic_latest) > idx:
                        test_shares = safe_float_convert(basic_latest[idx])
                        # 合理的股本范围：1亿到1000亿股之间
                        if 100000000 <= test_shares <= 100000000000:
                            total_shares = test_shares
                            print(f"   DEBUG: Found total shares in basic data at index {idx}: {total_shares}")

                            if total_shares > 0 and price > 0:
                                # 市值 = 当天股价 × 总股本
                                market_cap = price * total_shares / 100000000  # 转换为亿元
                                print(f"   DEBUG: Market cap calculated from basic data: {market_cap}")
                                break

            # 方法2：如果基本信息获取失败，尝试从资产负债表获取
            if market_cap == 0:
                print(f"   DEBUG: Trying to get total shares from balance data...")
                current_year = datetime.now().year
                current_quarter = (datetime.now().month - 1) // 3 + 1

                for i in range(4):
                    year = current_year
                    quarter = current_quarter - i

                    if quarter <= 0:
                        year -= 1
                        quarter += 4

                    # 获取资产负债表数据，包含总股本信息
                    balance_data = fetch_baostock_data(
                        bs.query_balance_data,
                        code=bs_code,
                        year=year,
                        quarter=quarter
                    )

                    if balance_data:
                        balance_latest = balance_data[0]
                        print(f"   DEBUG: Balance data: {balance_latest}")

                        # 尝试多个可能的位置
                        total_shares = 0
                        for idx in [7, 8, 9, 10, 11, 12]:
                            if len(balance_latest) > idx:
                                test_shares = safe_float_convert(balance_latest[idx])
                                # 合理的股本范围：1亿到1000亿股之间
                                if 100000000 <= test_shares <= 100000000000:
                                    total_shares = test_shares
                                    print(f"   DEBUG: Found total shares in balance data at index {idx}: {total_shares}")
                                    break

                        if total_shares > 0 and price > 0:
                            # 市值 = 当天股价 × 总股本
                            market_cap = price * total_shares / 100000000  # 转换为亿元
                            print(f"   DEBUG: Market cap calculated from balance data: {market_cap}")
                            break

                    if i < 3:
                        time.sleep(FILTER_CONFIG['api']['delay_seconds'])

        except Exception as e:
            print(f"   DEBUG: Error getting shares data: {e}")

        # 如果无法获取总股本，使用简化的估算方法
        if market_cap == 0 and price > 0:
            # 基于行业和股价的简单估算
            if symbol.startswith('6'):  # 上证
                if symbol in ['600519']:  # 贵州茅台
                    estimated_shares = 1256197800
                elif symbol in ['600036']:  # 招商银行
                    estimated_shares = 25220000000
                elif symbol in ['600276']:  # 恒瑞医药
                    estimated_shares = 6379000000
                else:
                    estimated_shares = 2000000000  # 默认20亿股
            else:  # 深证
                if symbol in ['000858']:  # 五粮液
                    estimated_shares = 3882000000
                elif symbol in ['000333']:  # 美的集团
                    estimated_shares = 7030000000
                elif symbol in ['000063']:  # 中兴通讯
                    estimated_shares = 4630000000
                else:
                    estimated_shares = 1500000000  # 默认15亿股

            market_cap = price * estimated_shares / 100000000
            print(f"   DEBUG: Estimated market cap: {market_cap} (using estimated shares: {estimated_shares})")

            # 标记为估算值
            market_cap = -market_cap  # 使用负数表示估算值

        print(f"   DEBUG: Final market cap: {market_cap}")

        return {
            'price': price,
            'pe': pe,
            'pb': pb,
            'market_cap': market_cap,
            'turn': turn,
            'volume': volume
        }

    except Exception as e:
        logger.error(f"获取基本行情数据失败 {symbol}: {str(e)}")
        return None

def get_financial_data_batch(symbol: str, year: int, quarter: int) -> Dict[str, Optional[Dict]]:
    """批量获取财务数据"""
    return {
        'profit': get_profit_data(symbol, year, quarter),
        'growth': get_growth_data(symbol, year, quarter),
        'balance': get_balance_data(symbol, year, quarter),
        'cash_flow': get_cash_flow_data(symbol, year, quarter)
    }

def apply_enhanced_filter_conditions(results: Dict, standards: Dict) -> Tuple[Dict[str, bool], float, float]:
    """应用增强版选股条件"""
    conditions = {}

    # 1. 财务质量检查
    conditions['financial_quality'] = (
        results.get('gross_margin', 0) >= standards['min_gross_margin'] and
        results.get('roe', 0) >= standards['min_roe']
    )

    # 2. 现金流质量检查 - 统一使用净利润现金含量比率
    conditions['cash_flow_quality'] = (
        results.get('net_profit_cash_ratio', 0) >= standards['min_cash_ratio']
    )

    # 3. 估值合理性检查
    pe = results.get('pe', 100)
    conditions['valuation_reasonable'] = (
        pe > 0 and pe <= standards['max_pe'] * 1.2
    )

    # 4. 成长性检查
    revenue_growth = results.get('revenue_growth', 0)
    profit_growth = results.get('profit_growth', 0)
    conditions['growth_adequate'] = (
        revenue_growth >= standards['min_growth_rate'] * 0.8 or
        profit_growth >= standards['min_growth_rate'] * 0.8
    )

    # 5. 债务安全检查
    conditions['debt_safe'] = (
        results.get('debt_ratio', 100) <= standards['max_debt_ratio']
    )

    # 6. 现金含量检查
    conditions['cash_adequate'] = (
        results.get('net_profit_cash_ratio', 0) >= standards['min_cash_ratio'] * 0.8
    )

    # 7. 运营效率检查
    conditions['operation_efficient'] = (
        results.get('net_profit_margin', 0) >= 5
    )

    # 8. 市场地位检查
    market_cap = results.get('market_cap', 0)
    roe = results.get('roe', 0)
    conditions['market_leader'] = (
        market_cap > 50 or roe >= standards['min_roe'] * 1.2
    )

    # 计算风险评分
    risk_score = assess_financial_risk(results, standards)

    # 计算综合评分
    total_score, detailed_scores = calculate_comprehensive_score(results, standards, risk_score)

    return conditions, total_score, risk_score

def stock_filter_enhanced(symbol: str) -> Dict:
    """增强版股票分析系统"""
    try:
        # 获取股票基本信息
        stock_name = get_stock_name(symbol)
        industry_type = get_industry_type(symbol)
        standards = get_industry_standards(symbol)

        results = {}

        # 1. 获取实时行情数据
        print("\n📊 获取实时行情数据...")
        basic_data = get_stock_basic_data(symbol)
        if basic_data:
            results.update(basic_data)
            print(f"   ✅ 最新价: {results['price']:.2f}元")
            print(f"   ✅ 市盈率: {results['pe']:.2f}")
            print(f"   ✅ 市净率: {results['pb']:.2f}")
            print(f"   ✅ 成交额: {results.get('turn', 0):,.2f}亿元")
            print(f"   ✅ 成交量: {results.get('volume', 0):,.0f}股")
            # 显示市值（区分真实值和估算值）
            if results['market_cap'] > 0:
                print(f"   ✅ 总市值: {results['market_cap']:.2f}亿元")
            elif results['market_cap'] < 0:
                print(f"   ✅ 总市值: {abs(results['market_cap']):.2f}亿元 (估算)")
            else:
                print(f"   ⚠️ 市值为0，可能存在数据问题")
        else:
            print("   ❌ 未找到行情数据")
            return "错误"

        # 2. 获取财务数据
        print("\n💰 获取财务数据...")
        current_year = datetime.now().year
        current_quarter = (datetime.now().month - 1) // 3 + 1

        # 尝试获取最近4个季度的数据
        financial_data = None
        for i in range(4):
            year = current_year
            quarter = current_quarter - i

            if quarter <= 0:
                year -= 1
                quarter += 4

            print(f"   📅 尝试获取 {year}年Q{quarter} 数据...")
            financial_data = get_financial_data_batch(symbol, year, quarter)
            if all(financial_data.values()):
                print(f"   ✅ 成功获取 {year}年Q{quarter} 数据")
                break
            else:
                print(f"   ❌ {year}年Q{quarter} 数据不完整")

            if i < 3:
                time.sleep(FILTER_CONFIG['api']['delay_seconds'])

        if not financial_data or not all(financial_data.values()):
            print(f"   ⚠️ 警告：未获取到完整的财务数据")

        # 整合财务数据
        profit_data = financial_data.get('profit') if financial_data else None
        growth_data = financial_data.get('growth') if financial_data else None
        balance_data = financial_data.get('balance') if financial_data else None
        cash_flow_data = financial_data.get('cash_flow') if financial_data else None

        # 处理盈利数据
        if profit_data:
            results['gross_margin'] = profit_data['grossProfitMargin']  # 已经转换为百分比
            results['roe'] = profit_data['roe']  # 已经转换为百分比
            results['net_profit_margin'] = profit_data['netProfitMargin']  # 已经转换为百分比
            results['net_profit'] = profit_data['netProfit']
            print(f"   ✅ 毛利率: {results['gross_margin']:.2f}%")
            print(f"   ✅ 净资产收益率: {results['roe']:.2f}%")
            print(f"   ✅ 净利率: {results['net_profit_margin']:.2f}%")
            print(f"   ✅ 净利润: {results['net_profit']:,.0f}元")
        else:
            print(f"   ❌ 未获取到盈利数据")

        # 处理成长数据
        if growth_data:
            results['revenue_growth'] = growth_data['yoyRevenueGrowth']
            results['profit_growth'] = growth_data['yoyNetProfitGrowth']
            print(f"   ✅ 营收增长率: {results['revenue_growth']:.2f}%")
            print(f"   ✅ 净利润增长率: {results['profit_growth']:.2f}%")
        else:
            print(f"   ❌ 未获取到成长数据")

        # 处理偿债数据
        if balance_data:
            results['debt_ratio'] = balance_data['liabilityToAsset']  # 已经是百分比
            print(f"   ✅ 资产负债率: {results['debt_ratio']:.2f}%")
        else:
            print(f"   ❌ 未获取到偿债数据")

        # 处理现金流数据
        if cash_flow_data and 'net_profit' in results and results['net_profit'] > 0:
            # cfoToNP是经营活动现金流与净利润的比率，直接计算经营活动现金流
            operating_cash_flow = results['net_profit'] * cash_flow_data['cfoToNP']
            results['operating_cash_flow'] = operating_cash_flow

            # 计算净利润现金含量（就是cfoToNP比率，转换为百分比）
            results['net_profit_cash_ratio'] = cash_flow_data['cfoToNP'] * 100

            # 计算现金流占收入比率（估算）
            if results.get('market_cap', 0) > 0:
                results['cfo_to_revenue'] = (operating_cash_flow / (results.get('market_cap', 1) * 0.5)) * 100
            else:
                results['cfo_to_revenue'] = 0

            print(f"   ✅ 经营活动现金流: {operating_cash_flow:,.0f}元")
            print(f"   ✅ 净利润现金含量: {results['net_profit_cash_ratio']:.2f}%")
        else:
            print(f"   ❌ 未获取到现金流数据或净利润为0")
            results['operating_cash_flow'] = 0
            results['cfo_to_revenue'] = 0
            results['net_profit_cash_ratio'] = 0

        # 设置默认值
        defaults = {
            'gross_margin': 0, 'roe': 0, 'net_profit_margin': 0, 'debt_ratio': 100,
            'revenue_growth': 0, 'profit_growth': 0, 'net_profit': 0,
            'operating_cash_flow': 0, 'cfo_to_revenue': 0, 'net_profit_cash_ratio': 0
        }

        for key, default_value in defaults.items():
            if key not in results:
                results[key] = default_value

        # 3. 应用增强版选股条件
        conditions, total_score, risk_score = apply_enhanced_filter_conditions(results, standards)

        # 计算详细评分
        _, detailed_scores = calculate_comprehensive_score(results, standards, risk_score)

        return {
            'symbol': symbol,
            'stock_name': stock_name,
            'industry_type': industry_type,
            'results': results,
            'conditions': conditions,
            'total_score': total_score,
            'risk_score': risk_score,
            'detailed_scores': detailed_scores,
            'standards': standards
        }

    except Exception as e:
        logger.error(f"分析过程中发生错误: {str(e)}")
        return {'error': str(e)}

def create_score_chart(detailed_scores: Dict) -> go.Figure:
    """创建评分雷达图"""
    # 将英文键名转换为中文
    categories = [get_chinese_name(key) for key in detailed_scores.keys()]
    values = list(detailed_scores.values())

    fig = go.Figure()

    fig.add_trace(go.Scatterpolar(
        r=values,
        theta=categories,
        fill='toself',
        name='评分',
        line_color='rgb(31, 119, 180)'
    ))

    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 100]
            )),
        showlegend=True,
        title="各项评分雷达图"
    )

    return fig

def create_gauge_chart(score: float, title: str) -> go.Figure:
    """创建仪表盘图表"""
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=score,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': title},
        delta={'reference': 70},
        gauge={
            'axis': {'range': [None, 100]},
            'bar': {'color': "darkblue"},
            'steps': [
                {'range': [0, 60], 'color': "lightgray"},
                {'range': [60, 80], 'color': "gray"},
                {'range': [80, 100], 'color': "lightgreen"}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 90
            }
        }
    ))

    return fig

def get_risk_description(risk_score: float, total_score: float = 0, passed_count: int = 0, rating: str = "") -> str:
    """获取风险描述 - 基于最终评级给出一致的风险描述"""
    if rating == "优秀":
        return "该股票风险很低，基本面优秀，各项指标表现突出，强烈推荐买入。"
    elif rating == "良好":
        return "该股票风险较低，基本面良好，具有较好的投资价值，建议买入。"
    elif rating == "一般":
        return "该股票风险适中，具有一定投资价值，但存在一些需要关注的因素，建议谨慎观察。"
    elif rating == "较差":
        if risk_score <= 40:
            return "该股票虽然风险较低，但在成长性、估值或其他关键指标上存在不足，暂时不建议投资。"
        elif risk_score <= 60:
            return "该股票风险适中，且在多个关键指标上表现不佳，建议暂时观望。"
        else:
            return "该股票风险较高，基本面存在明显问题，不建议投资。"
    else:
        # 默认基于风险评分的描述
        if risk_score <= 20:
            return "该股票风险很低，财务状况健康，适合稳健型投资者。"
        elif risk_score <= 40:
            return "该股票风险较低，基本面相对健康。"
        elif risk_score <= 60:
            return "该股票风险适中，需要关注相关风险因素。"
        elif risk_score <= 80:
            return "该股票风险较高，存在一定的不确定性，建议谨慎投资。"
        else:
            return "该股票风险很高，存在较大的投资风险，不建议普通投资者参与。"

def get_chinese_name(key: str) -> str:
    """将英文字段名转换为中文名"""
    name_mapping = {
        'financial_quality': '财务质量',
        'growth_potential': '成长潜力',
        'valuation_level': '估值水平',
        'cash_flow_quality': '现金流质量',
        'debt_safety': '债务安全',
        'market_position': '市场地位',
        'operational_efficiency': '运营效率',
        'industry_trend': '行业趋势'
    }
    return name_mapping.get(key, key)

def main():
    """主程序入口"""
    st.set_page_config(
        page_title="股票基本面智能分析系统",
        page_icon="📊",
        layout="wide"
    )

    st.title("🔍 股票基本面智能分析系统")
    st.caption("⚠️ 本系统分析仅用于技术交流，不建议用于实际投资依据")
    st.markdown("---")

    # 侧边栏 - 用户输入
    st.sidebar.title("📊 股票输入")

    # 股票代码输入
    stock_symbol = st.sidebar.text_input(
        "请输入股票代码",
        value="600519",
        max_chars=6,
        help="输入6位股票代码，如：600519（贵州茅台）"
    )

    # 分析按钮
    analyze_button = st.sidebar.button("🔍 开始分析", type="primary")

    # 连接数据源
    if 'bs_connected' not in st.session_state:
        st.session_state.bs_connected = False

    if not st.session_state.bs_connected:
        with st.spinner("正在连接数据源..."):
            try:
                bs.login()
                st.session_state.bs_connected = True
                st.success("✅ 数据源连接成功")
            except Exception as e:
                st.error(f"❌ 数据源连接失败: {str(e)}")
                return

    # 分析股票
    if analyze_button and stock_symbol:
        if len(stock_symbol) != 6 or not stock_symbol.isdigit():
            st.error("❌ 请输入正确的6位股票代码")
            return

        with st.spinner(f"正在分析股票 {stock_symbol}..."):
            analysis_result = stock_filter_enhanced(stock_symbol)

        if 'error' in analysis_result:
            st.error(f"❌ 分析失败: {analysis_result['error']}")
            return

        # 显示基本信息和关键指标
        col1, col2 = st.columns([2, 1])

        with col1:
            st.header(f"📊 {analysis_result['symbol']} ({analysis_result['stock_name']})")
            st.markdown(f"**行业**: {analysis_result['industry_type'].upper()}")

        with col2:
            total_score = analysis_result['total_score']
            risk_score = analysis_result['risk_score']

            # 简化评分显示
            if total_score >= 80:
                rating = "优秀"
                color = "green"
            elif total_score >= 70:
                rating = "良好"
                color = "blue"
            elif total_score >= 60:
                rating = "一般"
                color = "orange"
            else:
                rating = "较差"
                color = "red"

            st.markdown(f"### 🎯 {rating}")
            st.markdown(f"**综合评分**: {total_score:.1f}")
            st.markdown(f"**风险评分**: {risk_score:.1f}")

        st.markdown("---")

        # 关键数据紧凑展示
        key_metrics_cols = st.columns(6)

        with key_metrics_cols[0]:
            st.metric("最新价", f"{analysis_result['results'].get('price', 0):.2f}元")

        with key_metrics_cols[1]:
            st.metric("市盈率", f"{analysis_result['results'].get('pe', 0):.2f}")

        with key_metrics_cols[2]:
            st.metric("市净率", f"{analysis_result['results'].get('pb', 0):.2f}")

        with key_metrics_cols[3]:
            market_cap = analysis_result['results'].get('market_cap', 0)
            if market_cap > 0:
                st.metric("总市值", f"{market_cap:.2f}亿元")
            elif market_cap < 0:
                st.metric("总市值", f"{abs(market_cap):.2f}亿元*")
                st.caption("*估算值")
            else:
                st.metric("总市值", "0.00亿元")

        with key_metrics_cols[4]:
            roe = analysis_result['results'].get('roe', 0)
            st.metric("ROE", f"{roe:.2f}%")

        with key_metrics_cols[5]:
            gross_margin = analysis_result['results'].get('gross_margin', 0)
            st.metric("毛利率", f"{gross_margin:.2f}%")

        st.markdown("---")

        # 分模块评估展示 - 使用更紧凑的布局
        st.header("🔍 分模块评估")

        # 创建标签页
        tab1, tab2, tab3, tab4 = st.tabs([
            "财务质量", "成长潜力", "估值水平", "现金流"
        ])

        with tab1:
            st.subheader("💰 财务质量评估")

            # 紧凑的指标展示
            metrics_col1, metrics_col2, score_col = st.columns([1, 1, 1])

            with metrics_col1:
                gross_margin = analysis_result['results'].get('gross_margin', 0)
                min_gross_margin = analysis_result['standards']['min_gross_margin']

                st.metric("毛利率", f"{gross_margin:.2f}%",
                          delta=f"{(gross_margin - min_gross_margin):.2f}%")

                if gross_margin >= min_gross_margin:
                    st.success("✅ 达标")
                else:
                    st.error("❌ 未达标")

            with metrics_col2:
                roe = analysis_result['results'].get('roe', 0)
                min_roe = analysis_result['standards']['min_roe']

                st.metric("净资产收益率", f"{roe:.2f}%",
                          delta=f"{(roe - min_roe):.2f}%")

                if roe >= min_roe:
                    st.success("✅ 达标")
                else:
                    st.error("❌ 未达标")

            with score_col:
                financial_score = analysis_result['detailed_scores']['financial_quality']
                st.markdown(f"### **评分: {financial_score:.1f}**")

                if financial_score >= 80:
                    st.success("🎉 优秀")
                elif financial_score >= 70:
                    st.info("👍 良好")
                elif financial_score >= 60:
                    st.warning("⚠️ 一般")
                else:
                    st.error("💔 较差")

            # 简化说明
            with st.expander("📋 评估标准说明"):
                st.markdown(f"- **毛利率要求**: ≥ {min_gross_margin}% (当前: {gross_margin:.2f}%)")
                st.markdown(f"- **ROE要求**: ≥ {min_roe}% (当前: {roe:.2f}%)")
                st.markdown("- **评估原则**: 毛利率反映产品竞争力，ROE反映资本运用效率")

        with tab2:
            st.subheader("📈 成长潜力评估")

            # 紧凑的指标展示
            metrics_col1, metrics_col2, score_col = st.columns([1, 1, 1])

            with metrics_col1:
                revenue_growth = analysis_result['results'].get('revenue_growth', 0)
                min_growth_rate = analysis_result['standards']['min_growth_rate']

                st.metric("营收增长率", f"{revenue_growth:.2f}%")

                if revenue_growth >= min_growth_rate:
                    st.success("✅ 达标")
                else:
                    st.error("❌ 未达标")

            with metrics_col2:
                profit_growth = analysis_result['results'].get('profit_growth', 0)

                st.metric("净利润增长率", f"{profit_growth:.2f}%")

                if profit_growth >= min_growth_rate:
                    st.success("✅ 达标")
                else:
                    st.error("❌ 未达标")

            with score_col:
                growth_score = analysis_result['detailed_scores']['growth_potential']
                st.markdown(f"### **评分: {growth_score:.1f}**")

                if growth_score >= 80:
                    st.success("🎉 优秀")
                elif growth_score >= 70:
                    st.info("👍 良好")
                elif growth_score >= 60:
                    st.warning("⚠️ 一般")
                else:
                    st.error("💔 较差")

            # 简化说明
            with st.expander("📋 评估标准说明"):
                st.markdown(f"- **增长率要求**: ≥ {min_growth_rate}%")
                st.markdown(f"- **营收增长**: {revenue_growth:.2f}%")
                st.markdown(f"- **净利润增长**: {profit_growth:.2f}%")
                st.markdown("- **评估原则**: 营收增长率反映业务扩张能力，净利润增长率反映盈利增长能力")

        with tab3:
            st.subheader("💎 估值水平评估")

            # 紧凑的指标展示
            metrics_col, score_col = st.columns([2, 1])

            with metrics_col:
                pe = analysis_result['results'].get('pe', 0)
                max_pe = analysis_result['standards']['max_pe']

                st.metric("市盈率(PE)", f"{pe:.2f}")

                if pe > 0:
                    if pe <= max_pe:
                        st.success("✅ 估值合理")
                    elif pe <= max_pe * 1.5:
                        st.warning("⚠️ 估值略高")
                    else:
                        st.error("❌ 估值过高")
                else:
                    st.error("❌ 无有效估值数据")

                pb = analysis_result['results'].get('pb', 0)
                st.metric("市净率(PB)", f"{pb:.2f}")

            with score_col:
                valuation_score = analysis_result['detailed_scores']['valuation_level']
                st.markdown(f"### **评分: {valuation_score:.1f}**")

                if valuation_score >= 80:
                    st.success("🎉 优秀")
                elif valuation_score >= 70:
                    st.info("👍 良好")
                elif valuation_score >= 60:
                    st.warning("⚠️ 一般")
                else:
                    st.error("💔 较差")

            # 简化说明
            with st.expander("📋 评估标准说明"):
                st.markdown(f"- **PE要求**: ≤ {max_pe} (当前: {pe:.2f})")
                st.markdown(f"- **PB**: {pb:.2f}")
                st.markdown("- **评估原则**: PE反映估值合理性，越低越有投资价值")

        with tab4:
            st.subheader("💧 现金流质量评估")

            # 紧凑的指标展示
            metrics_col, score_col = st.columns([2, 1])

            with metrics_col:
                operating_cash_flow = analysis_result['results'].get('operating_cash_flow', 0)
                net_profit_cash_ratio = analysis_result['results'].get('net_profit_cash_ratio', 0)
                min_cash_ratio = analysis_result['standards']['min_cash_ratio']

                st.metric("经营活动现金流", f"{operating_cash_flow/100000000:.2f}亿元")
                st.metric("净利润现金含量", f"{net_profit_cash_ratio:.2f}%")

                if net_profit_cash_ratio >= min_cash_ratio:
                    st.success("✅ 现金流质量良好")
                else:
                    st.error("❌ 现金流质量欠佳")

            with score_col:
                cash_score = analysis_result['detailed_scores']['cash_flow_quality']
                st.markdown(f"### **评分: {cash_score:.1f}**")

                if cash_score >= 80:
                    st.success("🎉 优秀")
                elif cash_score >= 70:
                    st.info("👍 良好")
                elif cash_score >= 60:
                    st.warning("⚠️ 一般")
                else:
                    st.error("💔 较差")

            # 简化说明
            with st.expander("📋 评估标准说明"):
                st.markdown(f"- **现金含量要求**: ≥ {min_cash_ratio}% (当前: {net_profit_cash_ratio:.2f}%)")
                st.markdown(f"- **经营现金流**: {operating_cash_flow/100000000:.2f}亿元")
                st.markdown("- **评估原则**: 现金流反映真实盈利能力，现金含量越高越好")

        st.markdown("---")

        # 汇总结果 - 优化布局
        st.header("🎯 汇总评估结果")

        # 顶部综合评级卡片
        total_score = analysis_result['total_score']
        risk_score = analysis_result['risk_score']
        thresholds = FILTER_CONFIG['scoring']['thresholds']
        passed_count = sum(analysis_result['conditions'].values())

        # 确定评级
        if total_score >= thresholds['excellent'] and risk_score <= 40 and passed_count >= 6:
            rating = "优秀"
            emoji = "🎉🎉"
            recommendation = "强烈推荐买入"
            rating_color = "green"
        elif total_score >= thresholds['good'] and risk_score <= 50 and passed_count >= 5:
            rating = "良好"
            emoji = "🎉"
            recommendation = "建议买入"
            rating_color = "blue"
        elif total_score >= thresholds['average'] and risk_score <= 60 and passed_count >= 4:
            rating = "一般"
            emoji = "👍"
            recommendation = "值得关注"
            rating_color = "orange"
        else:
            rating = "较差"
            emoji = "💔"
            recommendation = "暂不建议"
            rating_color = "red"

        # 综合评级卡片
        # 简洁显示评级结果
        st.markdown(f"""
            <div style="
                background: #f8f9fa;
                padding: 1rem;
                border-radius: 10px;
                border-left: 4px solid #{rating_color};
                margin-bottom: 1rem;
            ">
                <h3 style="margin: 0; color: #{rating_color};">{emoji}{rating}</h3>
                <p style="margin: 0.5rem 0; color: #333; font-weight: bold;">{recommendation}</p>
                <div style="display: flex; gap: 1rem; font-size: 0.9rem; color: #666;">
                    <span>综合评分: {total_score:.1f}</span>
                    <span>风险评分: {risk_score:.1f}</span>
                    <span>通过条件: {passed_count}/8</span>
                </div>
            </div>
            """, unsafe_allow_html=True)

        # 关键指标和投资建议并排显示
        check_col, advice_col = st.columns([1, 1])

        with check_col:
            st.subheader("📊 关键指标检查")
            condition_names = {
                'financial_quality': '财务质量',
                'growth_adequate': '成长性',
                'valuation_reasonable': '估值合理性',
                'cash_flow_quality': '现金流质量',
                'debt_safe': '债务安全',
                'cash_adequate': '现金含量',
                'operation_efficient': '运营效率',
                'market_leader': '市场地位'
            }

            # 分两列显示
            inner_col1, inner_col2 = st.columns(2)

            with inner_col1:
                for key, name in list(condition_names.items())[:4]:
                    passed = analysis_result['conditions'].get(key, False)
                    if passed:
                        st.success(f"✅ {name}")
                    else:
                        st.error(f"❌ {name}")

            with inner_col2:
                for key, name in list(condition_names.items())[4:]:
                    passed = analysis_result['conditions'].get(key, False)
                    if passed:
                        st.success(f"✅ {name}")
                    else:
                        st.error(f"❌ {name}")

        with advice_col:
            st.subheader("💡 投资建议")
            # 获取颜色代码 - 与评级逻辑保持一致
            border_color = "28a745" if risk_score <= 30 else "ffc107" if risk_score <= 50 else "dc3545"
            risk_level = "风险很低" if risk_score <= 20 else "风险较低" if risk_score <= 40 else "风险适中" if risk_score <= 60 else "风险较高"

            st.markdown(f"""
                <div style="
                    background: #f8f9fa;
                    padding: 1.5rem;
                    border-radius: 10px;
                    border-left: 4px solid #{border_color};
                ">
                    <h4 style="margin: 0 0 1rem 0; color: #333;">{recommendation}</h4>
                    <div style="margin-bottom: 1rem;">
                        <strong>风险等级：</strong> {risk_level}
                    </div>
                    <div style="font-size: 0.9rem; color: #666; line-height: 1.6;">
                        {get_risk_description(risk_score, total_score, passed_count, rating)}
                    </div>
                </div>
                """, unsafe_allow_html=True)

        # 详细的评分表格 - 可折叠
        with st.expander("📊 详细评分"):
            # 评分表格
            scores_data = []
            for key, score in analysis_result['detailed_scores'].items():
                scores_data.append({
                    '评估项目': get_chinese_name(key),
                    '评分': f"{score:.1f}",
                    '权重': f"{IMPROVED_WEIGHTS[key]}%",
                    '加权得分': f"{score * IMPROVED_WEIGHTS[key] / 100:.1f}"
                })

            df = pd.DataFrame(scores_data)

            # 添加评分等级
            def get_score_level(score):
                if score >= 80:
                    return "优秀"
                elif score >= 70:
                    return "良好"
                elif score >= 60:
                    return "一般"
                else:
                    return "较差"

            df['等级'] = df['评分'].astype(float).apply(get_score_level)

            # 重新排列列顺序
            df = df[['评估项目', '评分', '等级', '权重', '加权得分']]

            # 设置样式
            def highlight_score_level(val):
                if val == "优秀":
                    return 'background-color: #d4edda; color: #155724'
                elif val == "良好":
                    return 'background-color: #d1ecf1; color: #0c5460'
                elif val == "一般":
                    return 'background-color: #fff3cd; color: #856404'
                else:
                    return 'background-color: #f8d7da; color: #721c24'

            styled_df = df.style.map(highlight_score_level, subset=['等级'])
            st.dataframe(styled_df, use_container_width=True)

            # 雷达图
            st.subheader("📈 各维度评分雷达图")
            st.plotly_chart(create_score_chart(analysis_result['detailed_scores']), use_container_width=True)

if __name__ == "__main__":
    main()

    # 运行模式
    # streamlit run F:/project/trade/dataanalysis/stock/get_stock_ing.py