

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

