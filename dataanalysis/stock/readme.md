

1、通过公式选出适合的个股
选股：

上轨:= MA(CLOSE,60)+ 2*STD(CLOSE,60),LINETHICK1,COLORFFFF00;
主板:=  (SUBSTR(CODE,1,2) == '60' OR SUBSTR(CODE,1,2) == '00')  AND C*1.1>= HHV(C,50) AND C*1.1>= 上轨 AND COUNT(C/REF(C,1)>=1.095,50) >1;
创业板:= (SUBSTR(CODE,1,2) == '30' OR SUBSTR(CODE,1,2) == '68')  AND C*1.2>= HHV(C,50)  AND C*1.2>= 上轨 AND COUNT(C/REF(C,1)>=1.145,200)>=1  ;
北证:= (SUBSTR(CODE,1,1) == '8' OR SUBSTR(CODE,1,1) == '9' OR SUBSTR(CODE,1,1) == '4' ) AND C*1.3>= HHV(C,50)   AND C*1.3>= 上轨  ;
周上轨:="BOLL.UB#WEEK"(29),LINETHICK2,COLORGREEN;
周:=(CLOSE-周上轨)/周上轨 >= -0.1;

主板涨停:= COUNT(C/REF(C,1)>=1.095,50) >=2 AND 主板  ;
创业:= COUNT(C/REF(C,1)>=1.145,200)>1 ; {AND 创业板 ;}

最高价:= HHV(C,50) >=  C;
成交额:=( AMOUNT>100000000 OR C/REF(C,1)>1.095);
SMAUP:= C > SMA(C, 6.5, 1);
SMADOWN:=  C > SMA(C, 13.5, 1);
趋:=(BARSCOUNT(C)> 453 AND C> MA(C,453)) OR BARSCOUNT(C) <= 453 ;
上:= C <= 上轨;
{涨幅: C/REF(C,1) >0.95;}
涨10日:= C/REF(C,10) <1.70;
MACD1:= MACD.MACD > 0  OR MACD.MACD >REF(MACD.MACD,1);

{强势的情况}
强: (创业板 OR 主板)
AND 最高价
AND 成交额
AND 涨10日
AND ( SMAUP AND SMADOWN)
AND MACD1
AND 上
AND 周
;

预选副图
上轨:= MA(CLOSE,60)+ 2*STD(CLOSE,60),LINETHICK1,COLORFFFF00;
主板:  (SUBSTR(CODE,1,2) == '60' OR SUBSTR(CODE,1,2) == '00')  AND C*1.1>= HHV(C,50) AND C*1.1>= 上轨 AND COUNT(C/REF(C,1)>=1.095,50) >1;
创业板: (SUBSTR(CODE,1,2) == '30' OR SUBSTR(CODE,1,2) == '68')  AND C*1.2>= HHV(C,50)  AND C*1.2>= 上轨 AND COUNT(C/REF(C,1)>=1.145,200)>=1  ;
北证:= (SUBSTR(CODE,1,1) == '8' OR SUBSTR(CODE,1,1) == '9' OR SUBSTR(CODE,1,1) == '4' ) AND C*1.3>= HHV(C,50)   AND C*1.3>= 上轨  ;
周上轨:="BOLL.UB#WEEK"(29),LINETHICK2,COLORGREEN;
周:(CLOSE-周上轨)/周上轨 >= -0.1;

主板涨停:= COUNT(C/REF(C,1)>=1.095,50) >=2 AND 主板  ;
创业: COUNT(C/REF(C,1)>=1.145,200)>1 ; {AND 创业板 ;}

最高价: HHV(C,50) >=  C;
成交额: ( AMOUNT>100000000 OR C/REF(C,1)>1.095);
SMAUP: C > SMA(C, 6.5, 1);
SMADOWN:  C > SMA(C, 13.5, 1);
趋:(BARSCOUNT(C)> 453 AND C> MA(C,453)) OR BARSCOUNT(C) <= 453 ;
上: C>上轨;
{涨幅: C/REF(C,1) >0.95;}
涨10日: C/REF(C,10) <1.70;
MACD1: MACD.MACD > 0  OR MACD.MACD >REF(MACD.MACD,1);


{强势的情况}
强: (创业板 OR 主板)
AND 最高价
AND 成交额
AND 涨10日
AND ( SMAUP AND SMADOWN)
AND MACD1
AND 周
;

2、程序跑分，符合的预警 ，预警条件加上周线上轨

周UP:"BOLL.UB#WEEK"(29),LINETHICK2,COLORGREEN;
周:CLOSE >= 周上轨;


3、根据测算结果，进行推演，选出符合的条件股



4、 jupyter notebook  检测参数和条件
启动 jupyter notebook ，在命令行输入：
cd .\dataanalysis\stock\
jupyter notebook



pip install pynput 




几个程序的说明：
1、auto_shoupan.py  每日运行的程序，每天可以运行多次，生成不同时间节点的数据，供日后分析使用， todo 调用时间节点模型还有问题，增加记录点以后无法正确运行
2、auto_shoupan_test.py   运行某一段未完成的程序，处理数据
3、auto_shoupan_yesteday.py   处理昨日收盘数据
4、auto_shoupan_stat.py  统计预测的成功率，应该结合回归和值来分析，todo 此处需完善
5、model_xunlian.py   训练模型,各个时间段的单独模型需要每日训练
使用随机森林来预测分类

todo 
写一个自动运行脚本，在启动时就开始运行



// AE列：提取创科北涨停数（取+前数字）
=IF(ISNUMBER(FIND("+",J3)), VALUE(LEFT(J3, FIND("+",J3)-1)), VALUE(J3))

// AF列：提取跌停数（取+前数字）  
=IF(ISNUMBER(FIND("+",K3)), VALUE(LEFT(K3, FIND("+",K3)-1)), VALUE(K3))

// AG列：提取连板高度（取第一个数字）
=IF(ISNUMBER(FIND("+",P3)), VALUE(LEFT(P3, FIND("+",P3)-1)), VALUE(P3))

// AH列：处理炸板率空值
=IF(ISBLANK(M3), 0, M3)

// AI列：处理连板晋级率空值
=IF(ISBLANK(N3), 0, N3)


excel 公式 

=IF(ISNUMBER(FIND("+",T3)), VALUE(LEFT(T3, FIND("+",T3)-1)), VALUE(T3))


=IF(ISNUMBER(FIND("+",U3)), VALUE(LEFT(U3, FIND("+",U3)-1)), VALUE(U3))

=IF(ISNUMBER(FIND("+",Z3)), VALUE(LEFT(Z3, FIND("+",Z3)-1)), VALUE(Z3))


=IF(ISBLANK(M3), 0, M3)


=MIN(100, MAX(0,
(红盘率*100*0.25) +                           -- 市场广度基础分
(MIN(非ST涨停数量/总家数*800, 20)) +          -- 涨停强度分  
(MIN(涨幅>6%/总家数*600, 15)) +               -- 涨幅广度分
(MIN(创科北涨停数/总家数*400, 10)) +          -- 创新板强度分
(MAX(0, 20 - 跌停数*1.2)) +                   -- 跌停风险扣分
(MAX(0, 15 - 跌幅>6%*0.6)) +                  -- 大跌风险扣分
(MAX(0, 10 - 炸板率*25)) +                    -- 炸板风险扣分
(连板晋级率*100*0.12) +                       -- 晋级质量分
(MIN(连板数量*0.7, 8)) +                      -- 连板数量分
(MIN(连板高度*2.5, 6)) +                      -- 连板高度分
IF(两市总额万亿>1.2, 3, 0) +                  -- 成交额加分
IF(开盘金额>120, 2, 0)                        -- 开盘金额加分
))



=MIN(100, MAX(0,
(M3*100*0.25) +
(MIN(R3/L3*800, 20)) +
(MIN(S3/L3*600, 15)) +
(MIN(AS3/L3*400, 10)) +
(MAX(0, 20 - U3*1.2)) +
(MAX(0, 15 - V3*0.6)) +
(MAX(0, 10 - W3*25)) +
(X3*100*0.12) +
(MIN(Y3*0.7, 8)) +
(MIN(AU3*2.5, 6)) +
IF(O3>1.2, 3, 0) +
IF(N3>120, 2, 0)
))