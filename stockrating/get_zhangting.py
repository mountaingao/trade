import akshare as ak
import pandas as pd
from tqdm import tqdm

def get_limit_up_data():
    """获取当日涨停板数据"""
    # 获取当日股票涨跌停数据
    limit_data = ak.stock_limit(symbol="涨停")

    # 过滤有效数据
    if limit_data.empty:
        print("今日无涨停数据")
        return pd.DataFrame()

    # 提取关键字段
    limit_data = limit_data[['代码', '名称', '涨跌幅', '最新价', '涨停统计', '封单资金']]
    return limit_data

def get_concept_classification(stock_code):
    """获取个股所属概念板块（使用同花顺概念）"""
    try:
        concept_data = ak.stock_board_concept_ths(symbol=stock_code)
        return concept_data['概念名称'].tolist() if not concept_data.empty else []
    except:
        return []

def analyze_concept_limit(limit_data):
    """统计概念板块的涨停分布"""
    # 初始化统计字典
    concept_dict = {}

    # 遍历每只涨停股票
    for idx, row in tqdm(limit_data.iterrows(), total=len(limit_data)):
        stock_code = row['代码']
        concepts = get_concept_classification(stock_code)

        # 更新概念统计
        for concept in concepts:
            if concept not in concept_dict:
                concept_dict[concept] = {
                    '涨停数量': 0,
                    '代表股票': [],
                    '总封单资金(亿)': 0
                }
            concept_dict[concept]['涨停数量'] += 1
            concept_dict[concept]['代表股票'].append(row['名称'])
            concept_dict[concept]['总封单资金(亿)'] += float(row['封单资金'].replace('亿', ''))

    # 转换为 DataFrame 并排序
    df = pd.DataFrame.from_dict(concept_dict, orient='index')
    df = df.sort_values(by='涨停数量', ascending=False)
    return df

def main():
    # 获取涨停数据
    limit_data = get_limit_up_data()
    if limit_data.empty:
        return

    # 统计概念板块
    concept_df = analyze_concept_limit(limit_data)

    # 保存结果
    with pd.ExcelWriter('涨停分析报告.xlsx') as writer:
        limit_data.to_excel(writer, sheet_name='涨停明细', index=False)
        concept_df.to_excel(writer, sheet_name='概念分布')

    # 打印摘要
    print(f"\n【涨停统计报告】")
    print(f"当日涨停总数：{len(limit_data)} 只")
    print(f"涉及概念板块：{len(concept_df)} 个")
    print("Top5热门概念：")
    print(concept_df.head(5)[['涨停数量', '总封单资金(亿)']])

if __name__ == "__main__":
    main()