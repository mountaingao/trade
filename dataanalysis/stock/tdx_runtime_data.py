import pandas as pd
from stockrating.stock_info_pywencai import tdx_get_block_data
import os
import glob
from datetime import datetime, timedelta

# 读取数据并通过下载今日数据，完善数据后写回原文件
def yesteday_data_complete(base_dir="../data/predictions/"):
    """完善昨日预测数据：读取预测目录，匹配昨日文件，合并今日TDX数据后保存"""
    try:
        # 获取昨日日期（YYYY-MM-DD格式）
        yesterday = (datetime.now() - timedelta(days=1)).strftime('%Y-%m-%d')

        # 1. 遍历预测目录下的所有子文件夹
        stock_folders = [f for f in os.listdir(base_dir) 
                        if os.path.isdir(os.path.join(base_dir, f))]
        
        for folder in stock_folders:
            folder_path = os.path.join(base_dir, folder)
            # 2. 获取目录下所有CSV文件
            csv_files = glob.glob(os.path.join(folder_path, "*.csv"))
            
            for file_path in csv_files:
                # 3. 检查文件名是否匹配昨日日期
                file_name = os.path.basename(file_path)
                if os.path.splitext(file_name)[0] == yesterday:
                    print(f"处理昨日文件: {file_path}")
                    
                    # 读取预测文件
                    df_pred = pd.read_csv(file_path)
                    
                    # 4. 获取今日TDX数据文件
                    today_mmdd = datetime.now().strftime('%m%d')
                    tdx_files = glob.glob(f"../data/tdx/{today_mmdd}*.xls")
                    
                    if not tdx_files:
                        print(f"警告: 未找到今日TDX数据文件")
                        continue
                        
                    tdx_file = tdx_files[0]  # 取第一个匹配文件
                    print(f"使用TDX文件: {tdx_file}")
                    
                    # 5. 处理TDX数据
                    # 修改编码为utf-8-sig并添加错误处理
                    df_tdx = pd.read_csv(tdx_file, encoding='utf-8-sig', sep='\t', header=1, errors='replace')
                    df_tdx = df_tdx[:-1]  # 删除最后一行
                    df_tdx.columns = df_tdx.columns.str.replace(' ', '')  # 清理列名空格
                    df_tdx['代码'] = df_tdx['代码'].str.split('=').str[1].str.replace('"', '')
                    
                    # 6. 合并数据（示例：按代码字段合并）
                    # 实际字段名需根据业务调整
                    merged_df = pd.merge(df_pred, df_tdx, on='代码', how='left')
                    
                    # 7. 保存回原文件
                    merged_df.to_csv(file_path, index=False)
                    print(f"已更新文件: {file_path}")
    
    except Exception as e:
        print(f"处理异常: {str(e)}")

# 测试文件内容和格式
def test_tdx_file_data():
    import pandas as pd
    file2 = "../data/tdx/0716142101.xls"
    # 修改编码为utf-8-sig并添加错误处理
    data_02 = pd.read_csv(file2, encoding='utf-8-sig', sep='\t', header=1, errors='replace')
    # 读取数据去掉第一行，第二行是列名
    data_02 = data_02[:-1]
    print( data_02.head(100))
    # 打印列名
    print(data_02.columns)
    # 列名去掉空格
    data_02.columns = data_02.columns.str.replace(' ', '')
    print(data_02.columns)
    # 代码去掉双引号

    data_02['代码']  = data_02['代码'].str.split('=').str[1].str.replace('"', "")
    print( data_02.head(10))

    # 合并new_data和data_02的数据，按照T列进行合并
    # data_02 按字段 代码 排序 倒序
    data_02 = data_02.sort_values(by=['代码'], ascending=False)
    print( data_02.head(100))

    data_02 = data_02.sort_values(by=['代码'])
    print( data_02.head(10))

def tdx_merge_data(blockname,blockname_01):
    import pandas as pd
    file1 = "../data/tdx/"+blockname+'.xls'
    file2 = "../data/tdx/"+blockname_01+'.xls'
    new_data = tdx_get_block_data(blockname)
def tdx_stock_info():
    from pytdx.hq import TdxHq_API
    # 创建 API 对象
    api = TdxHq_API()
    def get_server_ip():
        from pytdx.util.best_ip import select_best_ip
        best_ip = select_best_ip()
        print(f"最优服务器：IP={best_ip['ip']}, 端口={best_ip['port']}")
        # 最优服务器：IP=180.153.18.170, 端口=7709

    # get_server_ip()
    # 连接服务器
    # GOOD RESPONSE 115.238.56.198
    # GOOD RESPONSE shtdx.gtjas.com
    # GOOD RESPONSE sztdx.gtjas.com
    # if api.connect('123.125.108.90', 7709): # 海通
    if api.connect('180.153.18.170', 7709):
    # if api.connect('115.238.56.198', 7709):
        print("连接成功！")
        # 做一些操作
        api.disconnect() #关闭连接
    else:
        print("连接失败！")

    # 获取实时五档行情 TdxHq_API
    data = api.get_security_quotes(0, '000001')
    print(data)
    data = api.get_finance_info(code="600000", market=1)
    print(data)
    # api = TdxHq_API()
    # with api.connect('119.147.212.81', 7709):
    #     # 获取实时五档行情
    #     data = api.get_market_data(code="600000", market=1)
    #     print(data)
    #
    # 获取多股实时报价
    multi_data = api.get_security_quotes(
        [(1, "600000"), (0, "000001")])
    print(multi_data)