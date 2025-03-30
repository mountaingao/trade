自动交易的几种方式
1、pyautogui 快捷键和截屏方式处理（先研究这个 ）

2、通过easytrader 实现查询、下单等功能



easytrader

https://easytrader.readthedocs.io/zh-cn/master/usage/

支持券商¶
海通客户端(海通网上交易系统独立委托)
华泰客户端(网上交易系统（专业版Ⅱ）)
国金客户端(全能行证券交易终端PC版)
通用同花顺客户端(同花顺免费版)
其他券商专用同花顺客户端(需要手动登陆)


{一目均衡表}

N1:=9;N2:=26;N3:=52;

RH:=REFX(H,N2);

RO:=REFX(O,N2);

RC:=REFX(C,N2);

RL:=REFX(L,N2);

DRAWKLINE(RH,RO,RL,RC);

AA:=(HHV(RH,120)-LLV(RL,120))/30,LINETHICK;

VAR1:=(2*RC+RH+RL)/4;

转换线:=(HHV(H,N1)+LLV(L,N1))/2,COLORAAFF99,LINETHICK;

基准线:=(HHV(H,N2)+LLV(L,N2))/2,COLORFF6DD8,LINETHICK;

迟行带:REFX(CLOSE,N3),COLORFF9224;

先行带A:REF((转换线+基准线)/2,N2),COLORYELLOW,LINETHICK;

先行带B:REF((HHV(H,N3)+LLV(H,N3))/2,N2),COLOR909090;

STICKLINE(先行带A<先行带B,先行带A,先行带B,0,-1),COLOR339933;

STICKLINE(先行带A>=先行带B,先行带A,先行带B,0,-1),COLOR0033CC;

PLOYLINE(1,先行带A),COLORYELLOW;

REFX(PLOYLINE(1,基准线),N2),COLORFF6DD8;

REFX(PLOYLINE(1,转换线),N2),COLORAAFF99;

DRAWTEXT(CROSS(转换线,基准线) AND VAR1<MIN(先行带A,先行带B),MIN(基准线,RL)*0.995,'▲'),COLORRED;

DRAWTEXT(CROSS(转换线,基准线) AND RANGE(VAR1,MIN(先行带A,先行带B),MAX(先行带A,先行带B)),RL*0.995,'▲▲'),COLORRED;

DRAWTEXT(CROSS(转换线,基准线) AND VAR1>MAX(先行带A,先行带B),RL-AA*0.5,'▲▲▲'),COLORRED;

DRAWTEXT(CROSS(基准线,转换线) AND VAR1<MIN(先行带A,先行带B),MAX(基准线,RH)+AA*2,'▼\\N▼\\N▼'),COLORFF9966;

DRAWTEXT(CROSS(基准线,转换线) AND RANGE(VAR1,MIN(先行带A,先行带B),MAX(先行带A,先行带B)),MAX(基准线,RH)+AA,'▼\\N▼'),COLORFF9966;

DRAWTEXT(CROSS(基准线,转换线) AND VAR1>MAX(先行带A,先行带B),MAX(基准线,RH)+AA,'▼'),COLORFF9966;