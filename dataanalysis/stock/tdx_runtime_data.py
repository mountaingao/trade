


# 测试文件内容和格式
def test_tdx_file_data():
    import pandas as pd
    file2 = "../data/tdx/0716142101.xls"
    # 读取数据去掉第一行，第二行是列名
    data_02 = pd.read_csv(file2,encoding='GBK',sep='\t',header=1)
    # 去掉最后一行
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


from pytdx.hq import TdxHq_API
# 创建 API 对象
api = TdxHq_API()
def get_server_ip():
    from pytdx.util.best_ip import select_best_ip
    best_ip = select_best_ip()
    print(f"最优服务器：IP={best_ip['ip']}, 端口={best_ip['port']}")
    # 最优服务器：IP=180.153.18.170, 端口=7709

get_server_ip()
# 连接服务器
# if api.connect('123.125.108.90', 7709): # 海通
if api.connect('180.153.18.170', 7709):
    print("连接成功！")
    # 做一些操作
    api.disconnect() #关闭连接
else:
    print("连接失败！")

# api = TdxHq_API()
# with api.connect('119.147.212.81', 7709):
#     # 获取实时五档行情
#     data = api.get_market_data(code="600000", market=1)
#     print(data)
#
#     # 获取多股实时报价
#     multi_data = api.get_security_quotes(
#         [(1, "600000"), (0, "000001")])