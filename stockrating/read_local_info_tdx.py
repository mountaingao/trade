from mootdx.quotes import Quotes
from mootdx.reader import Reader
from datetime import datetime, timedelta
import pandas as pd

# 本地日线数据
def get_stock_history_by_local(symbol):
    # 创建 Reader 对象
    reader = Reader.factory(market='std', tdxdir='D:/new_haitong/')
    # reader = Reader.factory(market='std', tdxdir='D:/zd_haitong/')

    # 获取历史日线数据
    daily_data = reader.daily(symbol)
    # print("日线数据：", daily_data.head(5))
    return daily_data


client = Quotes.factory(market='std')
# client = Quotes.factory(market='std', multithread=True, heartbeat=True, bestip=False, timeout=15)

# 远程日线数据
def get_stock_history_by_remote( symbol='', begin=None, end=None, **kwargs):

    """
    读取k线信息

    :param symbol:  股票代码
    :param begin:   开始日期
    :param end:     截止日期
    :return: pd.dataFrame or None
    """
    return client.k(symbol, begin, end)


# 实时行情，列表形式返回
def get_stock_quotes_by_remote(symbols):
    client_data = client.quotes(symbol=symbols)
    print(client_data)
    return client_data



# from mootdx import consts
#
# # client = Quotes.factory(market='std')
# symbol = client.stocks(market=consts.MARKET_SH)

# 历史分时
from mootdx.quotes import Quotes

# client = Quotes.factory(market='std')
# 当日和历史分时
# info = client.F10C(symbol='301210')
# 股票F10数据
def get_stock_f10_by_remote(symbol=''):
    info = client.F10(symbol=symbol, name='最新提示')
    print(info)

def get_stock_minutes_by_remote(symbol='', date=''):
# minu = client.minutes(symbol='301210', date='20250310')
    """
   读取某日分钟数据
   :param symbol:  股票代码
   :param date:   开始日期
   """
    if date == '':
        date = datetime.now().strftime('%Y%m%d')
    minutes = client.minutes(symbol, date)
    # print(minutes)
    return minutes

def calculate_vol_percentage(mini_data):
    """
    计算每分钟成交量占全天总成交量的百分比。

    :param mini_data: 包含分钟数据的 DataFrame
    :return: 包含每分钟成交量百分比的数组
    """
    # 提取成交量列
    vol_data = mini_data['vol']
    
    # 计算总成交量
    total_vol = vol_data.sum()
    print(f"total_vol:{total_vol}")
    # 计算每分钟成交量的累计和
    cumulative_vol = vol_data.cumsum()
    
    # 计算每分钟成交量占全天总成交量的百分比
    vol_percentage = (cumulative_vol / total_vol) * 100
    
    return vol_percentage.tolist()

def calculate_amount_percentage(mini_data):
    """
    计算每分钟成交金额占全天总成交金额的百分比。

    :param mini_data: 包含分钟数据的 DataFrame
    :return: 包含每分钟成交金额的百分比的数组
    """
    # 计算每分钟的成交金额
    amount_data = mini_data['price'] * mini_data['vol']

    # 计算总成交金额
    total_amount = amount_data.sum()
    # print(f"total_amount:{total_amount}")

    # 计算每分钟成交金额的累计和
    cumulative_amount = amount_data.cumsum()

    # 计算每分钟成交金额占全天总成交金额的百分比
    amount_percentage = (cumulative_amount / total_amount)

    return amount_percentage.tolist()

def calculate_total_volume(mini_data, vol_percentage, num):
    """
    计算当日完整的成交量，通过第15个 vol_percentage 的值和前15个 vol 的累计和来反推总成交量。

    :param mini_data: 包含分钟数据的 DataFrame
    :param vol_percentage: 包含每分钟成交量百分比的数组
    :return: 当日完整的成交量
    """
    # 提取前15个 vol 的累计和
    cumulative_vol_15 = mini_data['vol'].iloc[:num].sum()
    
    # 获取第15个 vol_percentage 的值
    percentage_15 = vol_percentage[num-1]
    
    # 反推当日完整的成交量
    total_volume = (cumulative_vol_15 ) / (percentage_15)

    # 获取第num个价格
    price = mini_data['price'].iloc[num-1]

    return total_volume * price * 100

def calculate_total_amount(mini_data, amount_percentage, num):
    """
    计算当日完整的成交量，通过第15个 vol_percentage 的값和前15个 vol 的累计和来反推总成交量。

    :param mini_data: 包含分钟数据的 DataFrame
    :param vol_percentage: 包含每分钟成交量百分比的数组
    :return: 当日完整的成交量
    """
    # 确保 mini_data 的索引是 RangeIndex 或 DatetimeIndex
    if not isinstance(mini_data.index, (pd.RangeIndex, pd.DatetimeIndex)):
        mini_data = mini_data.reset_index(drop=True)

    # 计算每分钟的成交金额
    amount_data = mini_data['price'] * mini_data['vol']

    # print(f"amount_data:{amount_data}")
    # print(f"num:{num}")
    # 提取前num个 amount 的累计和
    current_amount = amount_data.iloc[:num].sum()
    print(f"cumulative_vol_iloc:{current_amount}")
    # 获取第15个 vol_percentage 的값
    percentage_num = amount_percentage[num-1]
    print(f"percentage_num:{percentage_num}")
    # 反推当日完整的成交量
    total_amount = current_amount * 100 / percentage_num

    return total_amount / 1e8,current_amount*100 / 1e8

def expected_calculate_total_amount(symbol, num):
    """
    计算当日预计的成交金额。
    """
    print(f"symbol:{symbol}")
    today = datetime.now().strftime('%Y%m%d')
    print(f"今天的日期: {today}")
    # 获取昨天的日期
    yesterday = (datetime.now() - timedelta(days=1)).strftime('%Y%m%d')
    print(f"昨天的日期: {yesterday}")

    today_mini = get_stock_minutes_by_remote(symbol, today)
    if num == -1:
        num = today_mini['vol'].count()
    # print(num)

    yesterday_mini = get_stock_minutes_by_remote(symbol, yesterday)
    # 计算并打印每分钟成交量百分比
    amount_percentage = calculate_amount_percentage(yesterday_mini)

    # 计算当日完整的成交量
    total_amount,current_amount = calculate_total_amount(today_mini, amount_percentage, num)
    # total_amount = calculate_total_amount(today_mini, amount_percentage, num) / 1e8

    return total_amount,current_amount

def calculate_stock_profit_from_date(symbol, date, price, days=5):
    """
    给出当日信号价格，计算后续第1-days天的最高价、最低价和收盘价的盈利比例。
    :param symbol: 股票代码
    :param date: 信号日期
    :param price: 信号价格
    :param days: 计算后续的天数，默认为5天
    :return: 返回一个字典，包含第1-days天的最高价、最低价和收盘价的盈利比例
    """
    history_data = get_stock_history_by_local(symbol)
    if date not in history_data.index:
        print(f"日期 {date} 不在历史数据中")
        return None

    # 获取日期索引
    date_index = history_data.index.get_loc(date)
    
    # 初始化结果字典
    profit_ratios = {}

    # 计算后续第1-days天的最高价、最低价和收盘价的盈利比例
    for day in range(1, days + 1):
        if date_index + day >= len(history_data):
            print(f"无法计算第 {day} 天的数据，历史数据不足")
            break

        # 获取第day天的数据
        future_data = history_data.iloc[date_index + day]
        # print(f"future_data:{future_data}")
        # 获取最高价、最低价和收盘价
        max_price = future_data['high']
        min_price = future_data['low']
        close_price = future_data['close']
        # print(f"max_price:{max_price}")
        # print(f"min_price:{min_price}")
        # print(f"close_price:{close_price}")

        # 计算盈利比例
        max_profit_ratio = (max_price - price) / price * 100
        min_profit_ratio = (min_price - price) / price * 100
        close_profit_ratio = (close_price - price) / price * 100

        # 存储结果
        profit_ratios[f"{day}_day_max"] = max_profit_ratio
        profit_ratios[f"{day}_day_min"] = min_profit_ratio
        profit_ratios[f"{day}_day_close"] = close_profit_ratio

    return profit_ratios

if __name__ == '__main__':

    list_data = calculate_stock_profit_from_date('300328', '20250303', 9.24)
    print(list_data)
    exit()
    # expected_total_amount = expected_calculate_total_amount(symbol='300565', num=15)
    # expected_total_amount = expected_calculate_total_amount(symbol='301139', num=207)
    expected_total_amount = expected_calculate_total_amount(symbol='301139', num=0)
    print(f"当日预计的成交额: {expected_total_amount}")
    exit()
    # 前一天的分钟数据
    mini = get_stock_minutes_by_remote(symbol='300565', date='20250312')
    # mini = get_stock_minutes_by_remote(symbol='300565', date='20250313')
    # print(mini)
    
    # 计算并打印每分钟成交量百分比
    vol_percentage = calculate_vol_percentage(mini)
    amount_percentage = calculate_amount_percentage(mini)
    print(vol_percentage)
    print(amount_percentage)
    # exit()
    # 得到第15个的值
    print(vol_percentage[14])
    print(amount_percentage[14])


    # 当日的分钟数据
    mini2 = get_stock_minutes_by_remote(symbol='300565', date='20250313')
    mini2 = get_stock_minutes_by_remote(symbol='301139', date='20250313')
    amount_percentage = calculate_amount_percentage(mini2)

    # 计算当日完整的成交量
    total_volume = calculate_total_volume(mini2, vol_percentage, 15)
    print(f"当日预计的成交量: {total_volume}")



    # 计算当日完整的成交量
    total_amount = calculate_total_amount(mini2, amount_percentage, 15)

    print(f"当日预计的成交金额: {total_amount}")

    # 预测当日成交量过10亿


# [3.857529827865681, 5.846775830295749, 6.8861846039774095, 7.587332256309923, 8.47790613659796, 8.988545125386633, 9.572519125283801, 9.86207064515867, 10.208450033420199, 10.793506468905685, 10.965072509653972, 11.548234682859903, 12.042366528926738, 12.305668985784916, 12.974614179364991, 13.412729983736405, 13.710940988317812, 15.751873290090032, 16.794799979433726, 17.281625385279415, 17.537080184122296, 17.969513201655044, 18.80407104024766, 19.80857126620609, 20.80035287400179, 22.14582031028016, 24.265229192205382, 25.517877776785546, 25.937592176155565, 26.089403767417064, 26.57135821311533, 26.853332683871976, 27.14640211940888, 27.255186896034765, 27.891659021965324, 28.123570846762302, 28.673718734524556, 29.01712142491821, 30.100098231029637, 30.329303966855818, 30.85726192505757, 31.032345881467894, 31.284282764648736, 31.386843536641795, 31.892070347488882, 32.06282456154593, 33.10548064199254, 33.527359912539204, 33.88645791896346, 36.95191550507798, 37.644674281601034, 38.48410308034108, 39.138435393478865, 39.44124674931062, 39.73783410050956, 40.143476837231454, 40.459277420123016, 40.71879135242209, 40.82135212441514, 40.91173549603964, 41.15528350341103, 41.32360223739436, 41.378265234604385, 42.76865374779792, 42.93128969494259, 43.22084121481746, 43.55477259381334, 44.28866392269245, 45.49503838587205, 45.75509353596527, 46.09822561746185, 46.65135020309198, 47.07458251812403, 47.33084914365814, 47.533805816467634, 47.88478555597951, 49.80069654730108, 50.200927106081394, 50.54947136551956, 50.67936363611763, 51.18242557578808, 51.27713868976584, 51.33640203822621, 51.51419208360733, 51.996146529305584, 52.22805835410257, 52.407472052866154, 52.60555776552821, 52.73626186281752, 52.876707880401696, 53.100230829389204, 53.21848691741287, 53.26638469219591, 53.86145365687332, 54.770970159956924, 55.09840692542289, 55.221533973593985, 55.33573092816145, 55.52109802266079, 55.72053677980825, 55.922952234823576, 56.08856487983612, 56.345913940958546, 56.74641510863594, 57.00268173417006, 57.074122482999, 57.70788851995876, 57.920316504166024, 58.29240373765009, 58.68641029179757, 58.90100314718148, 59.14915150580321, 59.744761688274785, 59.926881476009164, 60.20560864000086, 60.33279482162814, 60.435626202518286, 61.12107853882019, 61.23500488449059, 61.531862844586605, 62.56098848017925, 63.15253952919464, 63.39852301663974, 63.541945732091776, 63.62610509908345, 63.80064783769961, 63.976543620801166, 64.19573682743541, 64.47284033804463, 65.0240706613952, 65.09902932588618, 65.19293061317269, 65.34284794215465, 65.8391446593981, 66.17713517185018, 66.33462954995034, 66.64474734600324, 66.81685460454568, 66.92320390109786, 67.05471982507841, 67.23521595943032, 67.3323645534818, 67.38838059517721, 67.48607040702285, 67.57943047651521, 67.66521349688935, 68.08384546067106, 68.18207649031085, 68.44240224930115, 68.54739850136792, 68.99363257265172, 69.09294603787983, 69.21444943266845, 69.30239732421923, 69.4257949812874, 69.59600797755029, 69.93724579676731, 70.1269426336199, 70.21759661414147, 70.38402108584526, 70.63622857792319, 70.76882693749205, 70.99289110427372, 71.02103442956998, 71.21776709774664, 71.29976159356167, 71.40502845452554, 71.70486311248942, 71.9405634618455, 72.29100198356322, 77.81575322633458, 79.77875016574795, 80.50533505440592, 81.02598657238653, 81.583711509267, 81.75771303008901, 82.69023128942433, 82.84285470737707, 82.99926664988891, 83.05420025599601, 83.13835962298769, 83.33806898903222, 83.4116746090378, 83.7139447470754, 83.75264181935773, 83.8108227322298, 83.84897858671798, 83.86359146716026, 84.21213572659842, 84.26815176829385, 84.40751535028969, 84.53280726963742, 84.59721218714229, 84.8128874781145, 84.97714707864165, 85.21149438351235, 85.38495468654018, 85.54217845574327, 85.84959016282538, 86.12858793571415, 86.31557868359596, 86.58077540273369, 86.71310315340548, 87.17097340726369, 87.71002633024568, 88.16194318836816, 88.31213112624717, 88.50209857199684, 88.60844786854902, 88.980264493136, 89.35370477110547, 89.49009165523344, 89.81346928724324, 89.99369481269805, 90.16309598226971, 90.3370975030917, 90.48349691641162, 90.73570440848954, 90.80119176158273, 90.96761623328652, 91.36189339633108, 91.65252735179428, 92.14476493558155, 92.29549409125474, 92.43133975758855, 92.67651141834241, 92.99285321902813, 93.63230204282657, 94.12345719102552, 94.59566971642893, 95.33578504994087, 95.56093165231087, 96.42498586068513, 97.27550962420543, 97.72526161115125, 98.24916043589681, 98.75195176667019, 98.75195176667019, 98.75195176667019, 100.0]
