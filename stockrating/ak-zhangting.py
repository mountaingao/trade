import os
import akshare as ak
import xlsxwriter
import pandas as pd

def setup_pandas_display_options():
    pd.set_option('display.max_rows', None)  # 设置显示无限制行
    pd.set_option('display.max_columns', None)  # 设置显示无限制列
    pd.set_option('display.width', None)  # 自动检测控制台的宽度
    pd.set_option('display.max_colwidth', 50)  # 设置列的最大宽度为50

def fetch_and_process_data(date_str):
    try:
        df = ak.stock_zt_pool_em(date_str)
        if df.empty:
            print(f"No data available for date {date_str}")
            return None

        df['流通市值'] = round(df['流通市值'] / 100000000)
        df['换手率'] = round(df['换手率'])

        selected_columns = ['代码', '名称', '最新价', '流通市值', '换手率', '连板数', '所属行业']
        jj_df = df[selected_columns]

        sorted_temp_df = jj_df.sort_values(by='连板数', ascending=False)
        save_to_excel(sorted_temp_df, f"./{date_str}涨停排序.xlsx")

        temp_df = jj_df.copy()
        industry_count = temp_df['所属行业'].value_counts().to_dict()
        temp_df.loc[:, 'industry_count'] = temp_df['所属行业'].map(industry_count)
        sorted_industry_df = temp_df.sort_values(by=['industry_count', '所属行业', '连板数'], ascending=[False, True, False])
        sorted_industry_df.drop(['industry_count'], axis=1, inplace=True)
        save_to_excel(sorted_industry_df, f"./{date_str}涨停行业排序.xlsx")

        return df

    except Exception as e:
        print(f"Error processing data for date {date_str}: {e}")
        return None

def save_to_excel(df, file_path):
    directory = os.path.dirname(file_path)
    if not os.path.exists(directory):
        os.makedirs(directory)
    df.to_excel(file_path, engine='xlsxwriter')

if __name__ == "__main__":
    setup_pandas_display_options()
    date = "20250221"  # 可以通过命令行参数或配置文件动态设置
    df = fetch_and_process_data(date)
    if df is not None:
        spath = f"./{date}涨停.xlsx"
        save_to_excel(df, spath)
