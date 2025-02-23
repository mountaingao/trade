from bs4 import BeautifulSoup

# 假设html_content是上述HTML文档的内容
html_content = """
<!DOCTYPE HTML>
<html>
<head id="head">
		    <title>贵州茅台(600519) 最新动态_F10_同花顺金融服务网</title>
	<meta http-equiv="X-UA-Compatible" content="IE=EmulateIE7;IE=9"/>
	<meta http-equiv="Content-Type" content="text/html; charset=gbk"/>
	<meta name="keywords" content="贵州茅台最新动态,贵州茅台公司概况,贵州茅台财务分析,贵州茅台股东研究,贵州茅台股本结构,贵州茅台公司大事,贵州茅台分红融资"/>
	<meta name="description" content="暂无"/>
	<!-- 券商url参数添加，落地只落broker-pcf10.html -->
	
	<script type="text/javascript" src="//s.thsi.cn/js/chameleon/chameleon.min.1740313.js"></script> <script>
    function getUrlParams(key, url) {
        var href = url ? url : window.location.href;
        if (href.indexOf("?") < 0 && href.indexOf("#") < 0) {
            return false;
        }
        var hrefStr = href.replace(/#/g, "&");
        hrefStr = hrefStr.replace(/\?/g, "&");
        var paramsObj= {};
        hrefStr.split("&").forEach(function(i) {
            if (i.indexOf("=") > -1) {
                paramsObj[i.split("=")[0]] = i.split("=")[1];
            }
        });
        if (key) {
            return paramsObj[key] || false;
        } else {
            return paramsObj;
        }
    }

    function addParam(url, obj) {
        var str = "",res;
        for (var key in obj) {
            str += (key + "=" + obj[key]);
        }
        if (url.indexOf("?") > -1) {
            res = url.split('?').join('?' + str + '&');
        } else if (url.indexOf("#") > -1) {
            res = url.replace(/#/g, "?" + str + "#");
        } else {
            res = url + "?" + str;
        }
        return res;
    }

    var historyUrl = document.referrer;
    if (historyUrl) {
        var broker = getUrlParams('broker', historyUrl);
        if (broker && window.history) {
            var url = window.location.href;
            url = addParam(url, broker ? {broker: broker} : {});
            window.history.pushState('', '', url);
        }
    }
</script>
	<!-- f10工具箱，工具函数修改，落地只落utils.html -->
	
	<script type="text/javascript" src="//s.thsi.cn/js/chameleon/chameleon.min.1740313.js"></script> <script type="text/javascript" charset="utf-8" src="//s.thsi.cn/cb?cd/website-thsc-f10-utils/1.4.8/f10-polyfill.js" crossorigin></script>
<script type="text/javascript" charset="utf-8" src="//s.thsi.cn/cd/kernel-thslc-component-materials-container/F10/static/css-vars-ponyfill.js" ></script>
<script type="text/javascript" charset="utf-8" src="//s.thsi.cn/cb?cd/website-thsc-f10-utils/1.6.0/thsc-f10-utils.js" crossorigin></script>
<script>
    function isIEInClient(){
        return !!(window.external && 'getUserPath' in window.external && 'createObject' in window.external);
    }

    function isInTongYiClient(){
      return !!(window.external && 'createObject' in window.external);
    }

    function isInYuanHangClient(){
        return !!(window.HevoCef && 'IsHevoCef' in window.HevoCef);
    }
    
    function isInIEEnv(){
        var ua = window.navigator.userAgent.toLowerCase();
        return ua.indexOf('msie') > -1 || ua.indexOf('trident/') > -1;
    }

</script>
		<script type="text/javascript" src="//s.thsi.cn/js/chameleon/chameleon.min.1740313.js"></script> <script src="//s.thsi.cn/js/m/common/bridge.js" type="text/javascript"  crossorigin></script>
	<script type="text/javascript"  crossorigin  src="//s.thsi.cn/js/jquery-1.8.3.min.js"></script>
	<!-- 新版埋点cdn -->
	<script crossorigin src="//s.thsi.cn/cb?cd/weblog/0.0.1-alpha.27/weblog.js"></script>
		<script>
		var xhrProto = XMLHttpRequest.prototype;
		var origOpen = xhrProto.open;
		xhrProto.open = function (method, url) {
			if (location.host.indexOf('0033.cn') > -1 && arguments[1] !== undefined && arguments[1].indexOf('10jqka.com.cn') > -1) {
				arguments[1] = arguments[1].replace('10jqka.com.cn', '0033.cn');
			}
			return origOpen.apply(this, arguments);
		};
	</script>
	<script id="monitor-script" api_key="ths_f10" src="//s.thsi.cn/cb?cd/website-thsc-f10-utils/1.6.37/monitor.js" crossorigin></script>
	<script type="text/javascript"  crossorigin src="//s.thsi.cn/cb?js/common/cefapi/1.5.5/cefApi.min.js;js/basic/stock/newPcJs/common.js"></script>
	<script>
		try{
			feMonitor.setConfig(
				{
				slientDev:false,//开发环境(localhost)是否发送报错请求，默认不发送为true
				sampleRate:0.1, //采样率,可设置0-1之间，越小采集的频率越低
				isCollectPerformance:true, //是否采集性能
				title:'pc-F10'
				}
			);
		}catch(e){

		}
	</script>
	<style>
        .wencaibs{color:#008BFF!important;cursor:pointer;}
        .wd-tb .ptxt { word-break: break-word;}
        .comment-chart-head{color: #89a;}
        .comment-chart-head span{float: right;}
        .comment-chart-head span i{vertical-align: middle; display: inline-block; width: 20px; height: 20px;margin: 0 2px 0 6px; border-radius: 20px; overflow: hidden;}
        .comment-chart-head .imp-news{background: #da2e1f;}
        .comment-chart-head .yidong-news{background: #e3bf1a;}

        #ckg_table1 tr td{text-align:center}
        #ckg_table2 tr td{text-align:center}
        #ckg_table3 tr td{text-align:center}
        #ckg_table4 tr td{text-align:center}
        #ckg_table5 tr td{text-align:center}
		li{list-style-type:none;
			list-style-position:outside;}
		#jpjl-input:focus { outline: none; }
		.black.inClient .clientJump {
			cursor: pointer;
			color: #0199fa;
			text-decoration: underline;
		}
		.white.inClient .clientJump {
			color: #07519a;
			text-decoration: underline;
			cursor: pointer;
		}
		.managelist .name a {    color: #0199fa;
			margin: 0 3px 2px;
			text-decoration: underline;
			border-bottom: none !important;
		}
	</style>
	<!--统计页面加载时间-->
	<script type="text/javascript">
		var loadTimer= new Date().getTime();
	</script>
    <style>
        #search_cyter{
            display: block !important;
        }
		#profile .tickLabels .tickLabel{
			padding-right:12px;
			box-sizing:border-box;
		}
	  .a_cursor {
		pointer: cursor;
	}
    </style>
</head>
<script type="text/javascript">

//cookie设置
function setReviewJumpState(state) {
	document.cookie = 'reviewJump='
					+ state
					+ ';path=/;domain='
					+ window.location.host;
}
//取得对应名字cookie
function getCookie(name) {
	var cookieValue = "";
	var search = name + "=";
	if (document.cookie.length > 0) {
	    offset = document.cookie.indexOf(search);
	    if (offset != -1)    {
	        offset += search.length;
	        end = document.cookie.indexOf(";", offset);
	        if (end == -1) end = document.cookie.length;
	        cookieValue = unescape(document.cookie.substring(offset, end))
	    }
	}
	return cookieValue;
}
//栏目提前跳转
var reviewJumpState = getCookie('reviewJump');
if (!!reviewJumpState && reviewJumpState != 'nojump') {
	setReviewJumpState('nojump');
	window.location.href = unescape(reviewJumpState);
} else {
	setReviewJumpState('nojump');
}
//动态添加link-同步-会阻塞
function addCssByLink(url){
    document.write('<link rel="stylesheet" type="text/css"');
    document.write(' href="' + url + '">');
}
function isMac() {
        return /macintosh|mac os x/i.test(navigator.userAgent);
};
 var hash = window.location.hash.substring(1);
hashArray = hash.split('=');
if (hashArray[0] == 'stockpage') {
    var STOCK_SKIN = 'white';
} else {
    var STOCK_SKIN = getCookie('skin_color');
}
//肤色与客户端同步
function syncClientSkin(){
	try {//取得客户端肤色
		if (isMac()) {
			return 'white';
		}
		return 'black';
	} catch(e) {}
}
if (!STOCK_SKIN) {
	STOCK_SKIN = syncClientSkin();
}
document.documentElement.setAttribute("class",STOCK_SKIN);
if (STOCK_SKIN == 'white') {
	// addCssByLink('/f10/css/company-compare.css');
	addCssByLink('//s.thsi.cn/css/basic/stock/company-compare-202006082100.css');
 
    addCssByLink('//s.thsi.cn/cb?css/basic/stock/white/20200604153837/style_v4-2.css;css/basic/stock/white/20200604153837/chart_v2.min.20180110.css;css/basic/stock/white/20200604153837/dupont.css;css/basic/stock/white/20200604153837/black_v2-4.20190711.css;css/basic/stock/white/custom.min.css;css/basic/stock/white/20200604153837/operations.css;css/basic/stock/white/20200604153837/writecsskzd.min.css');
	addCssByLink('//s.thsi.cn/cb?css/basic/stock/20200604153837/survey-w.css;css/basic/stock/20200604153837/wxgcl_mod.css;css/basic/stock/provider-w_v2.css;css/basic/stock/20200604153837/searchbar-w_v2.css;css/basic/stock/20200604153837/wccept.css;css/basic/stock/wtgfx.css');
	addCssByLink('//s.thsi.cn/css/basic/stock/fschool/20200604153837/f_school_w.css');
	addCssByLink('//s.thsi.cn/css/basic/stock/white/wsplpager.css');
	addCssByLink('//s.thsi.cn/css/basic/stock/wchartsdom.css');
	addCssByLink('//s.thsi.cn/cb?css/basic/stock/white/20200604153837/industrydata.css;css/basic/stock/white/20200604153837/main_v1-2.css;css/basic/stock/white/20200604153837/xgcl.css;css/basic/stock/white/20200604153837/courier.css;css/basic/stock/white/20200604153837/recommend.css;css/basic/stock/white/20200604153837/housedata.css;css/basic/stock/white/202005071600/20200604153837/longhu.css;css/basic/stock/white/20200604153837/operate_white.css');
	addCssByLink('//s.thsi.cn/css/basic/stock/white/remind_white.css');
	addCssByLink('//s.thsi.cn/css/basic/stock/index_v1.css');
	addCssByLink('//s.thsi.cn/js/home/v5/thirdpart/scrollbar/jquery.mCustomScrollbar.css');
	addCssByLink('//s.thsi.cn/css/basic/stock/20200604153837/instition_white_v2.css');
	if (!!window.ActiveXObject&&!window.XMLHttpRequest) {
		addCssByLink('//s.thsi.cn/css/basic/stock/white/20200604153837/expression.css');
	}
					addCssByLink("//s.thsi.cn/css/basic/common/white/title.css")
} else {
	// addCssByLink('/f10/css/company-compare.css');
	addCssByLink('//s.thsi.cn/css/basic/stock/company-compare-202006082100.css');
 	addCssByLink('//s.thsi.cn/cb?css/basic/stock/black/20200604153837/style_v3-5.min.20200426.css;css/basic/stock/black/20200604153837/chart_v2.min.20190410.css;css/basic/stock/black/20200604153837/black_v2-4.20190711.css;css/basic/stock/black/custom.min.css;css/basic/stock/black/20200604153837/operations.css;css/basic/stock/black/20200604153837/csskzd.min.20150608.css;/css/basic/stock/20200604153837/survey.20150929.css;/css/basic/stock/provider_v2.css;/css/basic/stock/20200604153837/searchbar_v2.css;/css/basic/stock/20200604153837/ccept.css');
	addCssByLink('//s.thsi.cn/css/basic/stock/fschool/20200604153837/f_school.css');
	addCssByLink('//s.thsi.cn/css/basic/stock/black/splpager.20170504.css');
	addCssByLink('//s.thsi.cn/cb?css/basic/stock/chartsdom.css;css/basic/stock/20200604153837/xgcl_mod.css;/css/basic/stock/tgfx.css');
	addCssByLink('//s.thsi.cn/cb?css/basic/stock/black/20200604153837/industrydata.css;css/basic/stock/black/20200604153837/main_v1-2.css;css/basic/stock/black/20200604153837/xgcl.css;css/basic/stock/black/20200604153837/courier.css;css/basic/stock/black/20200604153837/funcRecommend.css;css/basic/stock/black/20200604153837/dupont.css;css/basic/stock/black/202005071600/20200604153837/longhu.css;css/basic/stock/black/20200604153837/operate.css');
	addCssByLink('//s.thsi.cn/css/basic/stock/black/remind_black.css');
	addCssByLink('//s.thsi.cn/js/home/v5/thirdpart/scrollbar/jquery.mCustomScrollbar.css');

	addCssByLink('//s.thsi.cn/css/basic/stock/index_v1.css');
	addCssByLink('//s.thsi.cn/css/basic/stock/20200604153837/instition_black_v2.css');
	if (!!window.ActiveXObject&&!window.XMLHttpRequest) {
		addCssByLink('//s.thsi.cn/css/basic/stock/black/20200604153837/expression.min.20140801.css');
	}
					addCssByLink("//s.thsi.cn/css/basic/common/black/title.css")
}
/*calendar*/
function  minDate(){
 	var data = $('#inte_json').text();
	var datajson =new Object();
	if (data) {
		data = eval('('+data+')');
	}
	for (var x in data) {
		datajson = data[x];
	}
	var minDate = datajson.year+'-'+datajson.month+'-'+datajson.day;
	return minDate;
 }
 Date.prototype.Format = function(fmt)
{ //author: meizz
  var o = {
    "M+" : this.getMonth()+1,                 //月份
    "d+" : this.getDate(),                    //日
    "h+" : this.getHours(),                   //小时
    "m+" : this.getMinutes(),                 //分
    "s+" : this.getSeconds(),                 //秒
    "q+" : Math.floor((this.getMonth()+3)/3), //季度
    "S"  : this.getMilliseconds()             //毫
  };
  if(/(y+)/.test(fmt))
    fmt=fmt.replace(RegExp.$1, (this.getFullYear()+"").substr(4 - RegExp.$1.length));
  for(var k in o)
    if(new RegExp("("+ k +")").test(fmt))
  fmt = fmt.replace(RegExp.$1, (RegExp.$1.length==1) ? (o[k]) : (("00"+ o[k]).substr((""+ o[k]).length)));
  return fmt;
}
 function  displayDate(){
 	var data = $('#inte_json').text();
 	var date =new Array();
	var datajson =new Object();
	var i=0;
	if (data) {
		data = eval('('+data+')');
	}
	for (var x in data) {
		datajson[x] = data[x];
		dates = datajson[x].year+'-'+datajson[x].month+'-'+datajson[x].day;
		if (x==0|| date[i-1] != dates) {
			date[i] = dates;
			i++;
		}
	}
	var count =date.length;
	var date2 = new Date();
	var month =date2.getMonth()+1;
	var day =date2.getDate();
	if ((date2.getMonth()+1)>9) {
	} else {
	 	var month ="0"+month;
	}
	if (date2.getDate()>9) {
	} else {
	 	var day ="0"+day;
	}
	year = date2.getFullYear();
	date2 = year+'-'+month+'-'+day;
	var result = "['";
	for (i=0;i<count;i++) {
		if (date2 == date[i]) {
			dateArr = date2.split('-');
			newdate = new Date(new Date(dateArr[0], dateArr[1]-1,  dateArr[2])-24*60*60*1000);
			date2 = newdate.Format("yyyy-MM-dd");
		} else {
			result +=date2+"',";
			dateArr = date2.split('-');
			newdate = new Date(new Date(dateArr[0], dateArr[1]-1,  dateArr[2])-24*60*60*1000);
			date2 = newdate.Format("yyyy-MM-dd");
			i--;
		}
	}
	result = result.substr(0,result.length-1)+']';
	return result;
 }
    function showDatePicker1(color) {
        WdatePicker({
            selfCssUrl:'//s.thsi.cn/css/basic/stock/'+STOCK_SKIN+'/mydatepicker.css',
            eCont:'WdateDiv1',
            vel:'elDate1',
            minDate:minDate(),
            maxDate:'%y-%M-%d',
            onpicked:function(dp){
                $('#elDate1').text(dp.cal.getNewDateStr());
                var invalider = $(".J_calendar").find("iframe").contents().find(".WinvalidDay");
            },
            Mchanged:function(dp){
                var invalider = $(".J_calendar").find("iframe").contents().find(".WinvalidDay");
            }
        });
    }
    function showDatePicker2(color) {

        WdatePicker({
            selfCssUrl:'//s.thsi.cn/css/basic/stock/'+STOCK_SKIN+'/mydatepicker.css',
            eCont:'WdateDiv2',
            vel:'elDate2',
            maxDate:'%y-%M-%d',
            onpicked:function(dp){
                $('#elDate2').text(dp.cal.getNewDateStr());
                var invalider = $(".J_calendar").find("iframe").contents().find(".WinvalidDay");
            },
            Mchanged:function(dp){
                var invalider = $(".J_calendar").find("iframe").contents().find(".WinvalidDay");
            }
        });
    }
  function isIE() {
	// 判断IE浏览器及其版本
	if(document.documentMode) {
		return document.documentMode;
	}
	var userAgent = navigator.userAgent.toLowerCase();
	if (userAgent.indexOf('edge') > -1) {
		return 'edge';
	}
	return false;
  }
	 // 初始化财务
	 var initFinance = window.localStorage.getItem('initFinance');
	 if (!initFinance) {
		 window.localStorage.setItem('initFinance', 'init');
		 window.localStorage.setItem('memoryPage', 'old');
	  }
	  // 跳转地址
    function jumpToUrl(url, name, anchor) {
	  if (!url || !name) {
			return;
		}
		// 新老地址切换
		var oldPage = {
			'financen': true
		}
		var pageLocation = window.localStorage.getItem('memoryPage');
      //在IE环境中
      if (isInIEEnv() || (pageLocation && pageLocation == 'old' && oldPage[name])) {
        location.href = url;
	    return;
     }
     var code = $('#stockCode').val();
	 var codeName = $('#stockName').val();
	 var marketid = $('#marketId').val();
	 var pageAnchor = anchor ? '&anchor=' + anchor : '';
	 var host = window.location.host;
     location.href = '//' + host + '/astockpc/astockmain/index.html#/' + name + '?code=' + code + '&marketid=' + marketid + '&code_name=' + codeName + pageAnchor;
    }
	// 跳转新地址
	function jumpToNew() {
       var code = $('#stockCode').val();
	   var codeName = $('#stockName').val();
	   var marketid = $('#marketId').val();
	   var url = window.location.href;
	   var module = url.match(/\/(\w+).html?/);
	   var moduleName = '';
	   var host = window.location.host;
	   if (module && module[1]) {
		   moduleName = module[1];
		   if (module[1] == 'finance') {
			   moduleName = 'financen';
		   }
	   }
	   window.TA.log('F10hs_cwfx.tyxbb.click');
	   window.localStorage.setItem('memoryPage', 'new');
	   setTimeout(function (){
		  location.href = '//' + host + '/astockpc/astockmain/index.html#/' + moduleName + '?code=' + code + '&marketid=' + marketid + '&code_name=' + codeName;
	   }, 10)
	}
</script>
    <script type="text/javascript" charset="utf-8" src="//s.thsi.cn/js/basic/stockph_v2/qdc_ad-510a70.js"></script>
	<link href="//s.thsi.cn/css/basic/stockph/qdc_ad-6e8683.css" rel="stylesheet" type="text/css">
 
<body>
<div class="votepopbox votesuccess" style="display: none">
		<div class="vpopicon"></div>
		<p class="vpoptxt fz18">谢谢您的宝贵意见</p>
		<p class="vpoptxt">您的支持是我们最大的动力</p>
	</div>
	<div class="votepopbox voteinfo" style="display: none">
		<div class="vpopicon"></div>
		<p class="vpoptxt fz18">谢谢您的支持</p>
		<p class="vpoptxt"></p>
	</div>
    <div class="wrapper">
<!-- F10 Header Start -->
		<div class="header">
		    <div class="hd">
		        <div class="logo fl" id="logo_self"><a title="同花顺F10">同花顺F10</a></div>

															<div onclick="sendTalog('f10_click_quot')" style="width:260px; margin-left:60px; font-size:14px;cursor:pointer;margin-right:10px;display:none;text-align:center" class="fl tip " id="quotedata">
							<div  class="fl">最新价： <span style="width:40px" class="upcolor" id="zxj">--</span></div>
							<div  class="fr">涨跌幅： <span style="width:40px;" class="upcolor" id="zdf">--</span></div>
						</div>
											        <span class="fr skin-change"><a href="###">换肤</a></span>
		        <div class="search fr" id="search_cyter">
		            <div id="updownchange" type="once" class="codeChange fl" style="display: none;">
		                <a href="javascript:void(0)" class="per">上一个股</a>
		                <a href="javascript:void(0)" class="next">下一个股</a>
		            </div>
		            <div class="fl searchbar">
			            <div class="text fl">
			                <input type="text" id="jpjl-input" value="输入股票名称或代码" /><!--文本框获得焦点给.searchbar添加hover类-->
			             </div>
		                <input type="button" value="搜索" class="btn" id="submit"/>
		            </div>
		        </div>
		    </div>
		    <div class="bd clear">
		        <div class="code fl">
					<div><h1 style="margin:3px 0px 0px 0px">
												贵州茅台</h1></div>
					<div>
						<a href="interactive.html" tid="wdm" posid="r1c1" class="iwen" onclick="TA.log({'id':'F10_review','nj':1})">i问董秘</a>
						<h1 style="margin:3px 0px 0px 0px">
						600519						</h1>
					</div>
		        </div>
		        <div class="nav">
		            <ul>
		                <li><a href="./"              target="_self" tid="zxdt" posid="r1c3"  class="cur">最新动态</a></li>
		                <li><a href="./company.html"  target="_self" tid="gszl" posid="r1c4" > 公司资料</a></li>
		                <li><a href="./holder.html"   target="_self" tid="gdyj" posid="r1c5" >股东研究</a></li>
		                <li><a href="./operate.html"  target="_self" tid="jyfx" posid="r1c6" > 经营分析</a></li>
		                <li><a href="./equity.html"   target="_self" tid="gbjg" posid="r1c7" >股本结构</a></li>
		                <li><a href="./capital.html"  target="_self" tid="zbyz" posid="r1c8" > 资本运作</a></li>
		                <li><a href="./worth.html"    target="_self" tid="ylyc" posid="r1c9"  >盈利预测</a></li>
		                <li><a href="./news.html"     target="_self" tid="xwgg" posid="r2c3" >新闻公告</a></li>
		                <li><a href="./concept.html"  target="_self" tid="gntc" posid="r2c4" >概念题材</a></li>
		                <li><a  href="javascript:void(0)"  onclick="jumpToUrl('./position.html', 'position', '');return false;"target="_self" tid="zlcc" posid="r2c5" >主力持仓</a></li>
		                <li><a href="javascript:void(0)"  onclick="jumpToUrl('./finance.html', 'financen', '');return false;" target="_self" tid="cwgk" posid="r2c6" > 财务分析</a></li>
						<li><a href="javascript:void(0)"  onclick="jumpToUrl('./bonus.html', 'bonus', '');return false;" target="_self" tid="fhrz" posid="r2c7" >分红融资</a></li>
						<li><a href="./event.html"    target="_self" tid="gsds" posid="r2c8" >公司大事</a></li>
		                <li><a href="./field.html"    target="_self" tid="hydb" posid="r2c9" >行业对比</a></li>
		            </ul>
		        </div>
		        <div class="subnav">
					<ul>
					   						<li>
							<a style="position:relative;height:25px;" class="skipto" type="" nav="profile" taid="f10_click_profile" name="index.html#profile" href="###">公司概要							</a>
						</li>
												<li>
							<a style="position:relative;height:25px;" class="skipto" type="" nav="sick" taid="f10_click_sick" name="index.html#sick" href="###">直击疫情							</a>
						</li>
												<li>
							<a style="position:relative;height:25px;" class="skipto" type="" nav="pointnew" taid="f10_click_pointnew" name="index.html#pointnew" href="###">近期重要事件							</a>
						</li>
												<li>
							<a style="position:relative;height:25px;" class="skipto" type="" nav="compare" taid="f10_click_compare" name="index.html#compare" href="###">A股&GDR股价对比							</a>
						</li>
												<li>
							<a style="position:relative;height:25px;" class="skipto" type="" nav="news" taid="f10_click_news" name="index.html#news" href="###">新闻公告							</a>
						</li>
												<li>
							<a style="position:relative;height:25px;" class="skipto" type="" nav="finance" taid="f10_click_finance" name="index.html#finance" href="###">财务指标							</a>
						</li>
												<li>
							<a style="position:relative;height:25px;" class="skipto" type="" nav="main" taid="f10_click_main" name="index.html#main" href="###">主力控盘							</a>
						</li>
												<li>
							<a style="position:relative;height:25px;" class="skipto" type="" nav="movie" taid="f10_click_movie" name="index.html#movie" href="###">电影票房							</a>
						</li>
												<li>
							<a style="position:relative;height:25px;" class="skipto" type="" nav="material" taid="f10_click_material" name="index.html#material" href="###">题材要点							</a>
						</li>
												<li>
							<a style="position:relative;height:25px;" class="skipto" type="" nav="directional" taid="f10_click_directional" name="index.html#directional" href="###">定增融资							</a>
						</li>
												<li>
							<a style="position:relative;height:25px;" class="skipto" type="" nav="payback" taid="f10_click_payback" name="index.html#payback" href="###">龙虎榜							</a>
						</li>
												<li>
							<a style="position:relative;height:25px;" class="skipto" type="" nav="deal" taid="f10_click_deal" name="index.html#deal" href="###">大宗交易							</a>
						</li>
												<li>
							<a style="position:relative;height:25px;" class="skipto" type="" nav="margin" taid="f10_click_margin" name="index.html#margin" href="###">融资融券							</a>
						</li>
												<li>
							<a style="position:relative;height:25px;" class="skipto" type="" nav="interactive" taid="f10_click_interactive" name="interactive.html#interactive" href="###">投资者互动							</a>
						</li>
								            </ul>
		        </div>
		    </div>
		</div>
<!-- F10 Header End -->
<!-- F10 Content Start -->
		<style>
		.iwc_searchbar{display: none;}
	</style>
		<div class="iwc_searchbar clearfix" style="">
    <div class="tips-box" style="display: none;">
        <div class="t-hd"></div>
        <div class="t-bd">
            <div class="info">
                <p>F10 功能找不到？在搜索框里直接输入您想要的功能！比如输入“龙虎榜”，赶快试一下哦！</p>
                <div class="btn">
                    <a href="###" class="a-left close">跳过</a>
                    <a href="###" class="a-right next">下一步</a>
                </div>
            </div>
            <div class="info" style="display: none;">
                <p>选股选不好？可以直接输入你想要的问句啦，比如输入“近一周涨幅超过30%的股票”赶紧行动吧！</p>
                <div class="btn">
                    <a href="###" class="a-left prev">上一步</a>
                    <a href="###" class="a-right close">完成</a>
                </div>
            </div>
        </div>
        <div class="t-ft"></div>
   	 </div>
     <div class="searchfx">
        <input id="search_input" x-webkit-speech="" x-webkit-grammar="builtin:search" lang="zh_CN" class="fillbox" autocomplete="off" type="text" value="输入问句，你想知道的都在这里">
        <input class="action_btn" id="search_submit" type="button" value="搜索">
        <span  class="tips-icon"></span>
     </div>
	</div>
	<div class="content page_event_content" >
<!-- 公司概要 -->
<style type="text/css">
#norm_mgsy,#norm_mggjj,#norm_mgxjl,#norm_jlr,#norm_jzcsy,#norm_mgwfp,#norm_yysr,#norm_mgjzc,#norm_mgwfplr,#norm_xsmll,#norm_zzygf{height:200px;}
#company-pk {
	visibility: hidden;
    font-size: 12px;
    padding: 2px 12px;
    border-radius: 2px;
    float: right;
    margin-right: 10px;
    cursor: pointer;
    margin-top: 5px;
	letter-spacing: 1px;
	background: #0C84E3;
    color: #ffffff;
}
</style>
<div class="m_box popp_box event new_msg z102" id="profile" stat="index_profile">
	<div class="hd flow_index search_z">
        <h2>公司概要</h2>
    </div>
	<div class="bd" style="padding-bottom: 30px;">
		<table class="m_table m_table_db" style="table-layout:fixed">
			<tbody>
				<tr>
					<td width="620">
							<span class="hltip f12 fl">公司亮点：</span>
						<span class="tip f14 fl core-view-text" style="width:560px;" title="白酒第一品牌，茅台酒是世界三大名酒之一">
						白酒第一品牌，茅台酒是世界三大名酒之一						</span>
					</td>
					<td class="rank-td">
						<span class="hltip f12 f1">市场人气排名：</span>
						<span class="f14 popularing-rank popularing-rank" id="profileMarketBtn">
							
						</span>
						<span class="hltip f12 f1 industry-pupular-rank">行业人气排名：</span>
						<span class="f14 popularing-rank" id="profileIndustryBtn"></span>
					</td>
				</tr>
				 <tr>
                	<td width="620">
                   	 	<span class="hltip f12 fl">主营业务：</span>
                    	<span class="tip f14 fl main-bussiness-text" style="width:560px;">
							<a class='newtaid' ifjump='1'  newtaid='f10_zxdt_code-gsgy-r1c1-clickcontent-quota-zyyewu' title="茅台酒及系列酒的生产与销售。" href="operate.html#intro" >茅台酒及系列酒的生产与销售。</a>
						</span>
					</td>
					<td ><span class="hltip f12 f1">所属申万行业：</span>
						<span class="tip f14">白酒Ⅱ</span>
					</td>
				</tr>
				<tr>
					<td >
												<div class="hltip f12 fl iwcclick newtaid" newtaid='f10_zxdt_code-gsgy-r2c1-clickword-quota-gainian' taname='f10_stock_iwc_gnqrqm' style="width:96px;position: relative;" content="概念贴合度是指每支股票所涉及的概念与该股之间的走势贴合程度，按照其排名标注1、2、3，其中1表示此概念是与该股走势最贴合的概念，2次之，依次类推。" data-url='http://www.iwencai.com/yike/detail/auid/31280d5188b3ecd7?qs=client_f10_baike_1  '>概念贴合度排名：</div>
																		<div class="f14  newconcept" cont="yes" style="overflow: hidden;">
																																																<a class='newtaid'  ifjump='1' newtaid='f10_zxdt_code-gsgy-r2c1-clickcontent-atgn-306750' title="此概念在该股票中走势贴合度排名第一" href="./concept.html?cid=306750#ifind">超级品牌<em class="ccept_top1"></em></a>，																																																			<a class='newtaid'  ifjump='1' newtaid='f10_zxdt_code-gsgy-r2c1-clickcontent-atgn-301496' title="此概念在该股票中走势贴合度排名第二" href="./concept.html?cid=301496#ifind">白酒概念<em class="ccept_top2"></em></a>，																																																			<a class='newtaid'  ifjump='1' newtaid='f10_zxdt_code-gsgy-r2c1-clickcontent-atgn-301715' title="此概念在该股票中走势贴合度排名第三" href="./concept.html?cid=301715#ifind">证金持股<em class="ccept_top3"></em></a>，																																																			<a class='newtaid'  ifjump='1' newtaid='f10_zxdt_code-gsgy-r2c1-clickcontent-atgn-309034' href="./concept.html?cid=309034#ifind">国企改革</a>，																																																			<a class='newtaid'  ifjump='1' newtaid='f10_zxdt_code-gsgy-r2c1-clickcontent-atgn-301490' href="./concept.html?cid=301490#ifind">沪股通</a>，																																																			<a class='newtaid'  ifjump='1' newtaid='f10_zxdt_code-gsgy-r2c1-clickcontent-atgn-300900' href="./concept.html?cid=300900#ifind">融资融券</a>，																																																			<a class='newtaid'  ifjump='1' newtaid='f10_zxdt_code-gsgy-r2c1-clickcontent-atgn-308718' href="./concept.html?cid=308718#ifind">同花顺漂亮100</a>，																																																			<a class='newtaid'  ifjump='1' newtaid='f10_zxdt_code-gsgy-r2c1-clickcontent-atgn-309154' href="./concept.html?cid=309154#ifind">西部大开发</a>																															 							<a class="alltext newtaid"   ifjump='1' newtaid='f10_zxdt_code-gsgy-r2c1-clickxqcont-quota-gainian' style="color:#0199fa;font-size:12px;margin-left: 6px;" href="concept.html#ifind" >详情>></a>
												    </div>
					</td>
										<td style="vertical-align:top">
                    	<div class="hltip f12 fl">财务分析：</div>
                    	<div class="f14" style="overflow: hidden;">
                    		                        		<a   ifjump='1' newtaid='f10_zxdt_code-gsgy-r2c2-clickcontent-quota-caiwu' href="##" class="newtaid tip fBox_trigger" tag="0">权重股</a>，                        	                        		<a   ifjump='1' newtaid='f10_zxdt_code-gsgy-r2c2-clickcontent-quota-caiwu' href="##" class="newtaid tip fBox_trigger" tag="1">一线蓝筹</a>，                        	                        		<a   ifjump='1' newtaid='f10_zxdt_code-gsgy-r2c2-clickcontent-quota-caiwu' href="##" class="newtaid tip fBox_trigger" tag="2">绩优股</a>                        	                    	</div>
                	</td>
                	                										</tr>
				<!-- 可比公司 -->
				<tr id="compareCompanyTr">
					<td colspan="2" style="padding: 0 8px;">
						<div class="compare-btn-box" id="compareBtn"><a href="###" class="f12 f1">对比>></a></div>
						<div class="compare-company-title">
							<span class="hltip f12 " id="compareCompanyTitle">可比公司<br/>()</span>
						</div>
						<div class="compare-company-list" id="compareCompanyList">
							<div class="china-company-list close">
								<span class="china-company-title" id="chinaCompanyTitle"></span>
								<span class="more-company-btn">全部</span>
								<span class="close-company-btn">收起</span>
								<span class="listing-company-num" id="listingCompanyNum">家未上市公司</span>
								<div class="company-list" id="chinaCompanyList"></div>
								<div class="company-list-all" id="chinaCompanyListAll"></div>
							</div>
							<div class="aborad-company-list close">
								<span class="china-company-title" id="abroadCompanyTitle"></span>
								<span class="more-company-btn">全部</span>
								<span class="close-company-btn">收起</span>
								<div class="company-list" id="abroadCompanyList"></div>
								<div class="company-list-all" id="abroadCompanyListAll">
								</div>
							</div>
						</div>
					</td>
				</tr>
			</tbody>
		</table>
				<table class="m_table m_table_db mt10">	
			<tbody>							
				<tr>
					<td>
						<span newtaid='f10_zxdt_code-gsgy-r3c1-clickword-quota-dtshiyinglv' class="newtaid hltip f12 iwcclick" taname='f10_stock_iwc_gzgyck' data-url='http://www.iwencai.com/yike/detail/auid/2e874be2278ef2a8?qs=client_f10_baike_1  '  content='总市值除以预估全年净利润，例如当前公布一季度净利润1000万，则预估全年净利润4000万。' >市盈率(动态)：</span>
						<span class="tip f12" id="dtsyl">23.051</span>
					</td>
					<td><span class="hltip f12 iwcclick newtaid" newtaid='f10_zxdt_code-gsgy-r3c2-clickword-quota-eps' taname='f10_stock_iwc_gzgyck' data-url='http://www.iwencai.com/yike/detail/auid/e8047e835e78ef8f?qs=client_f10_baike_1  ' content='税后利润除以股本总数 。该指标综合反映公司获利能力，可以用来判断和评价管理层的经营业绩。'>每股收益：</span><span class="tip f12">48.42元</span>
												<div class="popp_box" style="display:inline;">
															<a href="javascript:void(0)"  ifjump='1'  newtaid='f10_zxdt_code-gsgy-r3c2-clickplus-quota-eps' class="newtaid flashtab m_more popwin" targ="norm_mgsy" onclick="TA.log({'id':'f10_open_mgsy','fid':'f10_click_gsgy','nj':1})"><span class="none falshData">[[[0,"49.93"],[1,"16.55"],[2,"28.64"],[3,"42.09"],[4,"59.49"],[5,"19.16"],[6,"33.19"],[7,"48.42"]],[[0,"2022\u5e74\u62a5"],[1,"2023\u4e00\u5b63\u62a5"],[2,"2023\u4e2d\u62a5"],[3,"2023\u4e09\u5b63\u62a5"],[4,"2023\u5e74\u62a5"],[5,"2024\u4e00\u5b63\u62a5"],[6,"2024\u4e2d\u62a5"],[7,"2024\u4e09\u5b63\u62a5"]],"\u5143"]</span></a>
							 
							<div class="rp_tipbox flash_tipbox none norm_mgsy" style="z-index:200;width:660px;">
								<p class="pd5 fl">
									<span><strong class="the_tit">每股收益</strong></span><span><strong class="hltip f12 legend_label">单位：元</strong></span>
									<a href="/600519/field.html#position-0"><span class="hltip f12">同行业排名第<em class="the_rank">1</em>位</span></a>																			
								</p>
								<span class="fr"><a class="gray f12" href="javascript:void(0)" onclick="jumpToUrl('./finance.html', 'financen', 'cwzb');return false;" taid="f10_click_ckmx#f10_click_gsgy">查看明细>></a></span>	
								<div class="flash_wraper pd5 cb" id="norm_mgsy"></div>
								<s class="left_arrow"></s>
							</div>
						</div>
					</td>
					<td>
						<span class="hltip f12" taname='f10_stock_iwc_gzgyck' >每股资本公积金：</span><span class="tip f12">1.09元</span>
						<div class="popp_box" style="display:inline;">
															<a href="javascript:void(0)"  ifjump='1'  newtaid='f10_zxdt_code-gsgy-r3c3-clickplus-quota-mgggj' class="newtaid flashtab m_more popwin" targ="norm_mggjj" onclick="TA.log({'id':'f10_open_mgjzc','fid':'f10_click_gsgy','nj':1})"><span class="none falshData">[[[0,"1.09"],[1,"1.09"],[2,"1.09"],[3,"1.09"],[4,"1.09"],[5,"1.09"],[6,"1.09"],[7,"1.09"]],[[0,"2022\u5e74\u62a5"],[1,"2023\u4e00\u5b63\u62a5"],[2,"2023\u4e2d\u62a5"],[3,"2023\u4e09\u5b63\u62a5"],[4,"2023\u5e74\u62a5"],[5,"2024\u4e00\u5b63\u62a5"],[6,"2024\u4e2d\u62a5"],[7,"2024\u4e09\u5b63\u62a5"]],"\u5143"]</span>
								</a>
														<div class="rp_tipbox flash_tipbox none norm_mggjj" style="z-index:209;width:660px;">
								<p class="pd5 fl">
									<span><strong class="the_tit">每股资本公积金</strong></span><span><strong class="hltip f12 legend_label">单位：元</strong></span>
									<a href="/600519/field.html#position-1"><span class="hltip f12">同行业排名第<em class="the_rank">14</em>位</span></a>												
								</p>
								<span class="fr"><a class="gray f12" href="javascript:void(0)" onclick="jumpToUrl('./finance.html', 'financen', 'cwzb');return false;" taid="f10_click_ckmx#f10_click_gsgy">查看明细>></a></span>	
								<div class="flash_wraper pd5 cb" id="norm_mggjj">
								</div>
								<s class="left_arrow"></s>
							</div>
						</div>
					</td>
					<td>
												<span class="hltip f12">分类：</span>
						<span class="hltip f12 newtaid iwcclick" newtaid='f10_zxdt_code-gsgy-r3c4-clickcontent-quota-devide' taname='f10_stock_iwc_gzgyck' data-url="http://www.iwencai.com/yike/detail/auid/716981f756614a79?qs=client_f10_baike_1  " content="按照流通市值从大到小排名，排名在前的公司流通市值之和占A股市场总流通市值的一定比例（70%、90%）来划分大中小盘股。">
													超大盘股												</span>
											</td>
				</tr>
				<tr>
					<td>
						<span class="hltip f12 iwcclick newtaid" newtaid='f10_zxdt_code-gsgy-r4c1-clickword-quota-jtshiyinglv' taname='f10_stock_iwc_gzgyck' data-url='http://www.iwencai.com/yike/detail/auid/030cdc5aa4b28858?qs=client_f10_baike_1  ' content='总市值除以上年度净利润。市盈率是最常用来评估股价水平是否合理的指标之一。一般市盈率越小越好，动态市盈率小于静态市盈率说明这个股票的成长性好。'>市盈率(静态)：</span>
						<span class="tip f12" id="jtsyl">25.02</span>
					</td>
					<td>
						<span class="hltip f12">
                            营业总收入：                        </span><span class="tip f12">
                            1231.23亿元
						同比增长16.91%						
						</span>
						
						<div class="popp_box" style="display:inline;">
															<a href="javascript:void(0)"   ifjump='1'  newtaid='f10_zxdt_code-gsgy-r4c2-clickplus-quota-income' class="newtaid flashtab m_more popwin" targ="norm_yysr" onclick="TA.log({'id':'f10_open_yysr','fid':'f10_click_gsgy','nj':1})"><span class="none falshData">[[[0,"1275.54"],[1,"393.79"],[2,"709.87"],[3,"1053.16"],[4,"1505.60"],[5,"464.85"],[6,"834.51"],[7,"1231.23"]],[[0,"2022\u5e74\u62a5"],[1,"2023\u4e00\u5b63\u62a5"],[2,"2023\u4e2d\u62a5"],[3,"2023\u4e09\u5b63\u62a5"],[4,"2023\u5e74\u62a5"],[5,"2024\u4e00\u5b63\u62a5"],[6,"2024\u4e2d\u62a5"],[7,"2024\u4e09\u5b63\u62a5"]],"\u4ebf\u5143"]</span></a>							<div class="rp_tipbox flash_tipbox none norm_yysr" style="z-index:199;width:660px;">
								<p class="pd5 fl">
									<span><strong class="the_tit">营业总收入</strong></span><span><strong class="hltip f12 legend_label">单位：亿元</strong></span>
									<a href="/600519/field.html#position-4"><span class="hltip f12">同行业排名第<em class="the_rank">1</em>位</span></a>								</p>
								<span class="fr"><a class="gray f12" href="javascript:void(0)" onclick="jumpToUrl('./finance.html', 'financen', 'cwzb');return false;" taid="f10_click_ckmx#f10_click_gsgy">查看明细>></a></span>	
								<div class="flash_wraper pd5 cb" id="norm_yysr">
								</div>
								<s class="left_arrow"></s>
							</div>		
						</div>							
					</td>
					
					<td>
						<span class="hltip f12 " taname='f10_stock_iwc_gzgyck' >每股未分配利润：</span><span class="tip f12">153.56元</span>
												<div class="popp_box" style="display:inline;">
														<a href="javascript:void(0)"   ifjump='1' newtaid='f10_zxdt_code-gsgy-r4c3-clickplus-quota-udpps' class="newtaid flashtab m_more popwin" targ="norm_mgwfplr" onclick="TA.log({'id':'f10_open_mgwfplr','fid':'f10_click_gsgy','nj':1})"><span class="none falshData">[[[0,"128.40"],[1,"144.94"],[2,"129.94"],[3,"143.39"],[4,"137.70"],[5,"156.86"],[6,"138.33"],[7,"153.56"]],[[0,"2022\u5e74\u62a5"],[1,"2023\u4e00\u5b63\u62a5"],[2,"2023\u4e2d\u62a5"],[3,"2023\u4e09\u5b63\u62a5"],[4,"2023\u5e74\u62a5"],[5,"2024\u4e00\u5b63\u62a5"],[6,"2024\u4e2d\u62a5"],[7,"2024\u4e09\u5b63\u62a5"]],"\u5143"]</span></a>							<div class="rp_tipbox flash_tipbox none norm_mgwfplr" style="z-index:209;width:660px;">
								<p class="pd5 fl">
									<span><strong class="the_tit">每股未分配利润</strong></span><span><strong class="hltip f12 legend_label">单位：元</strong></span>
									<a href="/600519/field.html#position-1"><span class="hltip f12">同行业排名第<em class="the_rank">1</em>位</span></a>												
								</p>
								<span class="fr"><a class="gray f12" href="javascript:void(0)" onclick="jumpToUrl('./finance.html', 'financen', 'cwzb');return false;" taid="f10_click_ckmx#f10_click_gsgy">查看明细>></a></span>	
								<div class="flash_wraper pd5 cb" id="norm_mgwfplr">
								</div>
								<s class="left_arrow"></s>
							</div>
						</div>
					</td>
					<td>	
						<span class="hltip f12 iwcclick newtaid" newtaid='f10_zxdt_code-gsgy-r4c4-clickword-quota-zongguben'  taname='f10_stock_iwc_gzgyck' content='总股本,包括新股发行前的股份和新发行的股份的数量的总和。'  data-url='http://www.iwencai.com/yike/detail/auid/49a7ab6c3635d98a?qs=client_f10_baike_1  ' >总股本：</span>
						<span class="tip f12" ><input type="hidden" value="12.56" id="stockzgb" />12.56亿股</span>
					</td>
				</tr>
				<tr>
					<td>
						<span class="hltip f12 iwcclick newtaid" newtaid='f10_zxdt_code-gsgy-r5c1-clickword-quota-shijinglv'  taname='f10_stock_iwc_gzgyck' content='每股股价除以每股净资产。一般来说市净率较低的股票，投资价值较高，相反，则投资价值较低。市净率小于1，股票市价低于每股净资产，称为破净。' data-url='http://www.iwencai.com/yike/detail/auid/8edcac9853037bde?qs=client_f10_baike_1  ' >市净率：</span>
						<span class="tip f12" id="sjl">7.86</span>
					</td>
					<td>
						<span class="hltip f12">净利润：</span>
                        <span class="tip f12">608.28亿元
						
						同比增长15.04%						</span>
						
						<div class="popp_box" style="display:inline;">
														<a href="javascript:void(0)"   ifjump='1'  newtaid='f10_zxdt_code-gsgy-r5c2-clickplus-quota-netprofit' class="newtaid flashtab m_more popwin" targ="norm_jlr" onclick="TA.log({'id':'f10_open_jlr','fid':'f10_click_gsgy','nj':1})"><span class="none falshData">[[[0,"627.17"],[1,"207.95"],[2,"359.80"],[3,"528.76"],[4,"747.34"],[5,"240.65"],[6,"416.96"],[7,"608.28"]],[[0,"2022\u5e74\u62a5"],[1,"2023\u4e00\u5b63\u62a5"],[2,"2023\u4e2d\u62a5"],[3,"2023\u4e09\u5b63\u62a5"],[4,"2023\u5e74\u62a5"],[5,"2024\u4e00\u5b63\u62a5"],[6,"2024\u4e2d\u62a5"],[7,"2024\u4e09\u5b63\u62a5"]],"\u4ebf\u5143"]</span></a>							<div class="rp_tipbox flash_tipbox none norm_jlr" style="z-index:198;width:900px;left:18px;">
								<p class="pd5 fl">
									<span><strong class="the_tit">净利润</strong></span><span><strong class="hltip f12 legend_label">单位：亿元</strong></span>
									<a href="/600519/field.html#position-3"><span class="hltip f12">同行业排名第<em class="the_rank">1</em>位</span></a>															
								</p>
								<span class="fr"><a class="gray f12" href="javascript:void(0)" onclick="jumpToUrl('./finance.html', 'financen', 'cwzb');return false;" taid="f10_click_ckmx#f10_click_gsgy">查看明细>></a></span>
								<div class="flash_wraper pd5 cb" id="norm_jlr">
								</div>
								<s class="left_arrow"></s>
							</div>
						</div>
					</td>
					<td>
						<span class="hltip f12 iwcclick newtaid" newtaid='f10_zxdt_code-gsgy-r5c3-clickword-quota-cfps' taname='f10_stock_iwc_gzgyck' content='经营活动产生现金流量净额除以年度末普通股总股本，反映了每股发行在外的普通股票所平均占有的现金流量。' data-url='http://www.iwencai.com/yike/detail/auid/80b26f89d653f5c9?qs=client_f10_baike_1  '>每股经营现金流：</span><span class="tip f12">35.36元</span>
												<div class="popp_box" style="display:inline;">
														<a href="javascript:void(0)"   ifjump='1'  newtaid='f10_zxdt_code-gsgy-r5c3-clickplus-quota-cfps' class="newtaid flashtab m_more popwin" targ="norm_mgxjl" onclick="TA.log({'id':'f10_open_mgxjl','fid':'f10_click_gsgy','nj':1})"><span class="none falshData">[[[0,"29.21"],[1,"4.18"],[2,"24.19"],[3,"39.80"],[4,"53.01"],[5,"7.31"],[6,"29.15"],[7,"35.36"]],[[0,"2022\u5e74\u62a5"],[1,"2023\u4e00\u5b63\u62a5"],[2,"2023\u4e2d\u62a5"],[3,"2023\u4e09\u5b63\u62a5"],[4,"2023\u5e74\u62a5"],[5,"2024\u4e00\u5b63\u62a5"],[6,"2024\u4e2d\u62a5"],[7,"2024\u4e09\u5b63\u62a5"]],"\u5143"]</span></a>							<div class="rp_tipbox flash_tipbox none norm_mgxjl" style="z-index:188;width:660px;">
								<p class="pd5 fl">
									<span><strong class="the_tit">每股经营现金流</strong></span><span><strong class="hltip f12 legend_label">单位：元</strong></span>
									<a href="/600519/field.html#position-2"><span class="hltip f12">同行业排名第<em class="the_rank">1</em>位</span></a>								</p>
								<span class="fr"><a class="gray f12" href="javascript:void(0)" onclick="jumpToUrl('./finance.html', 'financen', 'cwzb');return false;" taid="f10_click_ckmx#f10_click_gsgy">查看明细>></a></span>	
								<div class="flash_wraper pd5 cb" id="norm_mgxjl">
								</div>
								<s class="left_arrow"></s>
							</div>
						</div>
					</td>
					<td>
						<span class="hltip f12"  taname='f10_stock_iwc_gzgyck'>总市值：</span><span class="tip f12 " id="stockzsz">18691亿</span>
					</td>
				</tr>
				<tr>
					<td>
						<span class="hltip f12 iwcclick newtaid" newtaid='f10_zxdt_code-gsgy-r6c1-clickword-quota-pne' taname='f10_stock_iwc_gzgyck' content='股东权益除以总股数。反映了每股股票所拥有的资产现值，每股净资产越少，股东拥有的资产现值越少，通常每股净资产越高越好。' data-url='http://www.iwencai.com/yike/detail/auid/6fba27adc0f9439b?qs=client_f10_baike_1  '>每股净资产：</span><span class="tip f12">189.23元</span>
						
						<div class="popp_box" style="display:inline;">
															<a href="javascript:void(0)"   ifjump='1' newtaid='f10_zxdt_code-gsgy-r6c1-clickplus-quota-pne' class="newtaid flashtab m_more popwin" targ="norm_mgjzc" onclick="TA.log({'id':'f10_open_mgjzc','fid':'f10_click_gsgy','nj':1})"><span class="none falshData">[[[0,"157.23"],[1,"173.76"],[2,"159.94"],[3,"173.39"],[4,"171.68"],[5,"190.84"],[6,"174.00"],[7,"189.23"]],[[0,"2022\u5e74\u62a5"],[1,"2023\u4e00\u5b63\u62a5"],[2,"2023\u4e2d\u62a5"],[3,"2023\u4e09\u5b63\u62a5"],[4,"2023\u5e74\u62a5"],[5,"2024\u4e00\u5b63\u62a5"],[6,"2024\u4e2d\u62a5"],[7,"2024\u4e09\u5b63\u62a5"]],"\u5143"]</span></a>
															<div class="rp_tipbox flash_tipbox none norm_mgjzc" style="z-index:209;width:660px;">
								<p class="pd5 fl">
									<span><strong class="the_tit">每股净资产</strong></span><span><strong class="hltip f12 legend_label">单位：元</strong></span>
									<a href="/600519/field.html#position-1"><span class="hltip f12">同行业排名第<em class="the_rank">1</em>位</span></a>								</p>
								<span class="fr"><a class="gray f12" href="javascript:void(0)" onclick="jumpToUrl('./finance.html', 'financen', 'cwzb');return false;" taid="f10_click_ckmx#f10_click_gsgy">查看明细>></a></span>	
								<div class="flash_wraper pd5 cb" id="norm_mgjzc">
								</div>
								<s class="left_arrow"></s>
							</div>
						</div>
					</td>
					<td>
						<span class="hltip f12" taname='f10_stock_iwc_gzgyck' >毛利率：</span><span class="tip f12">91.53%</span>
						<div class="popp_box" style="display:inline;">
															<a href="javascript:void(0)"   ifjump='1' newtaid='f10_zxdt_code-gsgy-r6c2-clickplus-quota-mll' class="newtaid flashtab m_more popwin" targ="norm_xsmll" onclick="TA.log({'id':'f10_open_mgjzc','fid':'f10_click_gsgy','nj':1})"><span class="none falshData">[[[0,"91.87"],[1,"92.60"],[2,"91.80"],[3,"91.71"],[4,"91.96"],[5,"92.61"],[6,"91.76"],[7,"91.53"]],[[0,"2022\u5e74\u62a5"],[1,"2023\u4e00\u5b63\u62a5"],[2,"2023\u4e2d\u62a5"],[3,"2023\u4e09\u5b63\u62a5"],[4,"2023\u5e74\u62a5"],[5,"2024\u4e00\u5b63\u62a5"],[6,"2024\u4e2d\u62a5"],[7,"2024\u4e09\u5b63\u62a5"]],""]</span>
								</a>
														<div class="rp_tipbox flash_tipbox none norm_xsmll" style="z-index:209;width:660px;">
								<p class="pd5 fl">
									<span><strong class="the_tit">毛利率</strong></span><span><strong class="hltip f12 legend_label">单位：%</strong></span>
									<a href="/600519/field.html#position-1"><span class="hltip f12">同行业排名第<em class="the_rank">1</em>位</span></a>												
								</p>
								<span class="fr"><a class="gray f12" href="javascript:void(0)" onclick="jumpToUrl('./finance.html', 'financen', 'cwzb');return false;" taid="f10_click_ckmx#f10_click_gsgy">查看明细>></a></span>	
								<div class="flash_wraper pd5 cb" id="norm_xsmll">
								</div>
								<s class="left_arrow"></s>
							</div>
						</div>
					</td>
					
					<td >
						<span class="hltip f12 iwcclick newtaid" newtaid='f10_zxdt_code-gsgy-r6c3-clickword-quota-roe' taname='f10_stock_iwc_gzgyck' data-url='http://www.iwencai.com/yike/detail/auid/d2edf9bd2e6a92ce?qs=client_f10_baike_1  ' content='净利润除以平均股东权益，该指标反映股东权益的收益水平。指标值越高，说明投资带来的收益越高。'>净资产收益率：</span><span class="tip f12">26.09%</span>
						
						<div class="popp_box" style="display:inline;">
															<a href="javascript:void(0)"   ifjump='1'  newtaid='f10_zxdt_code-gsgy-r6c3-clickplus-quota-roe' class="newtaid flashtab m_more popwin" targ="norm_jzcsy" onclick="TA.log({'id':'f10_open_jzcsyl','fid':'f10_click_gsgy','nj':1})"><span class="none falshData">[[[0,"30.26"],[1,"10.00"],[2,"16.70"],[3,"24.82"],[4,"34.19"],[5,"10.57"],[6,"17.63"],[7,"26.09"]],[[0,"2022\u5e74\u62a5"],[1,"2023\u4e00\u5b63\u62a5"],[2,"2023\u4e2d\u62a5"],[3,"2023\u4e09\u5b63\u62a5"],[4,"2023\u5e74\u62a5"],[5,"2024\u4e00\u5b63\u62a5"],[6,"2024\u4e2d\u62a5"],[7,"2024\u4e09\u5b63\u62a5"]],""]</span></a>
														<div class="rp_tipbox flash_tipbox none norm_jzcsy" style="z-index:208;width:660px;">
								<p class="pd5 fl">
									<span><strong class="the_tit">净资产收益率</strong></span><span><strong class="hltip f12 legend_label">单位：%</strong></span>
									<a href="/600519/field.html#position-6"><span class="hltip f12">同行业排名第<em class="the_rank">2</em>位</span></a>											
								</p>
								<span class="fr"><a class="gray f12" href="javascript:void(0)" onclick="jumpToUrl('./finance.html', 'financen', 'cwzb');return false;" taid="f10_click_ckmx#f10_click_gsgy">查看明细>></a></span>
								<div class="flash_wraper pd5 cb" id="norm_jzcsy"></div>
								<s class="left_arrow"></s>
							</div>
						</div>
					</td>
					<td>
						<span class="hltip f12  iwcclick newtaid" newtaid='f10_zxdt_code-gsgy-r6c4-clickword-quota-liutongagu' taname='f10_stock_iwc_gzgyck' content='流通股是指上市公司股份中，可以在交易所流通的股份数量。'  data-url='http://www.iwencai.com/yike/detail/auid/e17f528dc6567415?qs=client_f10_baike_1  '>流通A股：</span><span class="tip f12">12.56亿股</span>
					</td>
				</tr>
				
				<tr>
  <td>
    <span class="hltip f12">更新日期：</span>
    <span class="tip f12">2025-02-21</span>
  </td>
  <td>
    <span class="hltip f12">总质押股份数量：</span> 
    <span class="tip f12">40.78万股</span> 
        </td>
  <td colspan="2">
    <span class="hltip f12">质押股份占A股总股本比：</span>
    <span class="tip f12">
        0.03%    </span>
    <div class="popp_box" style="display:inline;">
      <a href="javascript:void(0)"   ifjump='1'  newtaid='f10_zxdt_code-gsgy-r8c3-clickplus-quota-zybl' class="newtaid flashtab m_more popwin" targ="norm_zzygf" onclick="TA.log({'id':'f10_open_zysl','fid':'f10_click_gsgy','nj':1})"><span class="none gqzy falshData">
              [["0.0100","0.0100","0.0100","0.0100","0.0100","0.0100","0.0400","0.0400","0.0400","0.0400","0.0300","0.0300"],["2024-12-06","2024-12-13","2024-12-20","2024-12-27","2025-01-03","2025-01-10","2025-01-17","2025-01-24","2025-01-27","2025-02-07","2025-02-14","2025-02-21"]]          </span>
      </a>
      <div class="rp_tipbox flash_tipbox none norm_zzygf" style="z-index:208;">
        <p class="pd5 fl">
          <span><strong class="the_tit">质押股份占A股总股本比</strong></span><span><strong class="hltip f12 legend_label">单位：%</strong></span>
        </p>
        <div class="flash_wraper pd5 cb" id="zzygfzx" style="width:580px;height:200px;"></div>
        <div class="tl f12" style="padding-bottom:5px;padding-left:10px;">*数据来源：中登公司及上市公司最新公布的质押公告</div>
        <s class="left_arrow"></s>
      </div>
    </div>
  </td>
</tr>
    			</tbody>
		</table>
		
				<span class="fr" style="margin-top: 4px;margin-right: 10px;color:#666">
		     以上为三季报		</span>
				<div class="financeBox" style="display: none;">
            					<div class="tabCont" style="display:none">
					<div class="tipbox_hd">
		                <h3>
		                    <span class="tip">权重股条件</span><a onclick="TA.log({'id':'f10_stock_index_financetab_jump','nj':1})" href="http://www.iwencai.com/stockpick/search?typed=0&preParams=&ts=1&f=1&qs=f10_caiwu_tag&selfsectsn=&querytype=&searchfilter=&tid=stockpick&w=  权重股"  target="_blank" >[成分股]</a>
		                </h3>
		                <span class="msg  zx"></span>
		            </div>
					<div class="tipbox_bd">
                		                    	<div class="bk">
                        	<span class="tip">问财百科：</span>
                        	                        		权重股就是总股本巨大的上市公司股票，他的股票总数占股票市场股票总数的比重很大，也就权重很大，他的涨跌对股票指数的影响很大。
                        	                   	 	</div>
                	</div>
                </div>
                
							<div class="tabCont" style="display:none">
					<div class="tipbox_hd">
		                <h3>
		                    <span class="tip">一线蓝筹条件</span><a onclick="TA.log({'id':'f10_stock_index_financetab_jump','nj':1})" href="http://www.iwencai.com/stockpick/search?typed=0&preParams=&ts=1&f=1&qs=f10_caiwu_tag&selfsectsn=&querytype=&searchfilter=&tid=stockpick&w=  一线蓝筹"  target="_blank" >[成分股]</a>
		                </h3>
		                <span class="msg  plh"></span>
		            </div>
					<div class="tipbox_bd">
                		                    	<div class="bk">
                        	<span class="tip">问财百科：</span>
                        	                        		蓝筹股(blue chip)是指稳定的现金股利政策对公司现金流管理有较高的要求，通常将那些经营业绩较好，具有稳定且较高的现金股利支付的公司股票，多指长期稳定增长的、大型的、传统工业股及金融股。
                        	                   	 	</div>
                	</div>
                </div>
                
							<div class="tabCont" style="display:none">
					<div class="tipbox_hd">
		                <h3>
		                    <span class="tip">绩优股条件</span><a onclick="TA.log({'id':'f10_stock_index_financetab_jump','nj':1})" href="http://www.iwencai.com/stockpick/search?typed=0&preParams=&ts=1&f=1&qs=f10_caiwu_tag&selfsectsn=&querytype=&searchfilter=&tid=stockpick&w=  绩优股"  target="_blank" >[成分股]</a>
		                </h3>
		                <span class="msg good"></span>
		            </div>
					<div class="tipbox_bd">
                		                    	<table>
                       		<thead>
		                        <tr>
		                            <th width="130">指标名称</th>
		                            <th width="80" class="tc">指标数据</th>
		                            <th class="pl20">判断条件</th>
		                        </tr>
		                    </thead>
                        	<tbody>
                        					                        <tr>
			                            <td>每股收益(ttm)</td>
			                            <td class="tc">65.82元</td>
			                            <td class="pl20">
			                            				                            		每股收益大于0.5
			                            				                            </td>
			                        </tr>
		                        		                       		                         
		                        	<tr>
			                            <td>销售毛利率</td>
			                            <td class="tc">91.96</td>
			                            <td class="pl20">
			                            				                            		毛利率大于30
			                            											</td>
			                        </tr>
		                       		                         
		                        	<tr>
			                            <td>市盈率(上个交易日)</td>
			                            <td class="tc">23.05</td>
			                            <td class="pl20">
			                            				                            		市盈率小于30
			                            											</td>
			                        </tr>
		                       		                       		                       		                                                	</tbody>
                    	</table>
                    	                    	<div class="bk">
                        	<span class="tip">问财百科：</span>
                        	                        		绩优股具有较高的投资回报和投资价值。其公司拥有资金、市场、信誉等方面的优势，对各种市场变化具有较强的承受和适应能力，绩优股的股价一般相对稳定且呈长期上升趋势。
                        	                   	 	</div>
                	</div>
                </div>
                
		         
             <div class="opinion_opts_wraper fr evaluate" tag="financetab">
					<div class="oow_cont fl">
						<span>您对此栏目的评价：</span>
						<a class="opinion_up" tag="support" href="javascript:void(0)"> 有用 <span class="opi_up">0</span></a>
						<a class="opinion_down" tag="opposition" href="javascript:void(0)"> 没用 <span class="opi_down">0</span></a>
						<span class="btn_idea"><a onclick="$('.f10_append_advice').click();" title="我要提建议" href="javascript:void(0)">提建议</a></span>
					</div>
			</div>
            <div class="arrow"></div>
        </div>
	</div>
	<!-- 公司信息 -->
	<div class="rp_tipbox company-info-box" id="companyInfoBox">
		<table class="m_table m_table_db">
			<tr>
				<td rowspan="3" class="logo-box" width="100">
					<img src="//i.thsi.cn/images/basic/client/company/logo1.png">
				</td>
				<td>
					<span class="hltip f12 fl">公司名称：</span>
					<span class="tip f14 fl info-text" id="companyInfoName" title="">
					
					</span>
				</td>
			</tr>
			<tr>
				<td>
					<span class="hltip f12 fl">公司简称：</span>
					<span class="tip f14 fl info-text" id="companyInfoNickName" title="">
					
					</span>
				</td>
			</tr>
			<tr>
				<td>
					<span class="hltip f12 fl">上市场所：</span>
					<span class="tip f14 fl info-text" id="companyInfoMarket" title="">
					
					</span>
				</td>
			</tr>
			<tr>
				<td colspan="2" class="tip-td">
					<div class="industry-info">
						<span class="hltip f12 fl">所属行业：</span>
						<span class="tip f14 fl info-text" id="companyInfoIndustry" title="">
						
						</span>
					</div>
					<div class="place-info">
						<span class="hltip f12 fl">所属地域：</span>
						<span class="tip f14 fl info-text" id="companyInfoArea" title="">
						
						</span>
					</div>
				</td>
			</tr>
			<tr style="display: none" id="companyMainBussiness">
				<td colspan="2">
					<span class="hltip f12 fl">主营业务：</span>
					<span class="tip f14 fl info-text" id="companyInfoBussiness" title="">
					
					</span>
				</td>
			</tr>
			<tr>
				<td colspan="2">
					<span class="hltip f12 fl">公司简介：</span>
					<span class="tip f14 fl info-text" id="companyInfoIntro" title="">
					
					</span>
					<span class="show-more" id="companyInfoMore">了解更多>></span>
				</td>
			</tr>
		</table>
	</div>
	<!-- 人气对比 -->
	<div class="rp_tipbox compare-list-box" id="compareListBox" >
		<div class="compare-header">
			<span class="compare-title"></span>
			<span id="company-pk">A股PK</span>
		</div>
		<div class="tool-box clearfix">
			<!-- <input type="text" readonly name="" value="全球" class="compare-select-input"> -->
			<span class="compare-mid-count"></span>
			<ul class="compare-select close">
				<li class="selected-item">全球</li>
				<li class="select-choose">全球</li>
				<li class="select-choose">国内</li>
				<li class="select-choose">国外</li>
			</ul>
			<ul class="compare-choose-list">
				<li class="active">总市值</li>
				<li>市盈率(TTM)</li>
				<li>市净率</li>
			</ul>
		</div>
		<div class="compare-content" id="compareContent">
			<ul class="default-list" id="compareComtentDefault">
			</ul>
			<ul class="default-list sort-list" id="compareComtentSort">
			</ul>
		</div>
		<p class="remark-text">注：市值数据为截止上一交易日并已进行货币转换</p>
	</div>
	<!-- 人气排名 -->
	<iframe class="rp_tipbox market-compare-box" id="marketCompareBox" src="">
	</iframe>
	<div class="ft"></div>
</div>

<!-- 直击疫情 -->
<!--  -->
<!--  -->
<!--  -->

<!-- 近期重要事件 -->
<!-- 近期重要事件 -->
<!-- 2013-04-09 将“最新提示”修改为“特别提示” -->
<div class="m_box event new_msg z101" id="pointnew" stat="index_pointnew">
	<div class="hd flow_index">
		<h2>近期重要事件</h2>
	</div>
	
	           <div class="scrollWrap bd">
				<div id="scrollbar1">
					<div class="scrollbar">
						<div class="track none">
							<div class="thumb">
								<div class="end"></div>
							</div>
						</div>
					</div>
					<div class="viewport">
						<div class="overview">

                		<!-- 2014-03-14 add-->
                		                		<!-- 2014-03-14 end -->
		
            		<!-- 2013-05-22 add -->
            		                                    				<!-- end 2013-05-22 add-->
            				
            		<table class="m_table m_table_db" id="tableList">
            			<tbody>
                                                				<tr >
                    					<td class="hltip tc f12">2025-04-03</td>
                    					
	<td>

		<strong class="hltip fl">披露时间：</strong>


		<a ifjump='1' newtaid='f10_zxdt_code-rctevt-r1c2-clickxqcont-quota-pltime' onclick="clickTalogStat('f10_tbts_plsj','F10new_tbts');" class="newtaid fr hla gray f12 stat_jumptodata" title="到数据中心查看更多" href="http://data.10jqka.com.cn/financial/yypl/op/code/code/600519/#refCountId=basic_50f3c8ec_832  " target="_blank">更多>></a>


		<span>将于2025-04-03披露《2024年年报》</span>


	</td>


                    				</tr>
            	                    			
            				                    				<tr >
                    					<td class="hltip tc f12">2025-02-21</td>
                    					<td>


	<strong class="hltip fl">融资融券：</strong> 


	<a   ifjump='1' taid="f10_tbts_rzrq#F10new_tbts" newtaid='f10_zxdt_code-rctevt-r2c2-clickxqcont-quota-rzrq' class="newtaid fr hla gray skipto f12" name="index.html#margin" title="到融资融券栏目查看">详情&gt;&gt;</a> 
    
	<span>

		<a   ifjump='1' taid="f10_tbts_rzrq#F10new_tbts" class="newtaid skipto" newtaid='f10_zxdt_code-rctevt-r2c2-clickcontent-quota-rzrq' name="index.html#margin" title="">
            融资余额166.3亿元，融资净买入额-1.077亿元		</a>


	</span>


</td>


                    				</tr>
            	                    			
            				                    				<tr >
                    					<td class="hltip tc f12">2025-02-18</td>
                    					<td><strong class="hltip fl">大宗交易：</strong>
    
	<a  ifjump='1'  newtaid='f10_zxdt_code-rctevt-r3c2-clickxqcont-quota-dzjy' taid="f10_tbts_dzjy#F10new_tbts" class="newtaid fr hla gray skipto f12" name="index.html#deal" title="到大宗交易栏目查看">详情&gt;&gt;</a> 

	<span>

	<a  ifjump='1'  newtaid='f10_zxdt_code-rctevt-r3c2-clickcontent-quota-dzjy' taid="f10_tbts_dzjy#F10new_tbts" class="newtaid skipto" name="index.html#deal" title="到大宗交易栏目查看">
        2025-02-18共发生3笔交易，成交均价1475.00元，平均溢价率0.00%，总成交量6.37万股，总成交金额9396万元	</a>

	</span>

</td>                    				</tr>
            	                    			
            				                    				<tr >
                    					<td class="hltip tc f12">2025-02-10</td>
                    					<td><strong class="hltip fl">大宗交易：</strong>
    
	<a  ifjump='1'  newtaid='f10_zxdt_code-rctevt-r4c2-clickxqcont-quota-dzjy' taid="f10_tbts_dzjy#F10new_tbts" class="newtaid fr hla gray skipto f12" name="index.html#deal" title="到大宗交易栏目查看">详情&gt;&gt;</a> 

	<span>

	<a  ifjump='1'  newtaid='f10_zxdt_code-rctevt-r4c2-clickcontent-quota-dzjy' taid="f10_tbts_dzjy#F10new_tbts" class="newtaid skipto" name="index.html#deal" title="到大宗交易栏目查看">
        成交均价1431.51元，溢价率0.00%，成交量2.55万股，成交金额3650万元	</a>

	</span>

</td>                    				</tr>
            	                    			
            				                    				<tr >
                    					<td class="hltip tc f12">2025-02-07</td>
                    						<td>


		<strong class="hltip fl">发布公告：</strong>


		<a   ifjump='1'  newtaid='f10_zxdt_code-rctevt-r5c2-clickxqcont-quota-notice'  taid="f10_tbts_fbgg#F10new_tbts" class="newtaid fr hla gray f12 skipto" name="news.html#pub" title="到公告列表查看更多">更多&gt;&gt;</a>


		<span>

            

			<a   ifjump='1' newtaid='f10_zxdt_code-rctevt-r5c2-clickcontent-quota-notice'  class="newtaid client" target="_blank" onclick="clickTalogStat('f10_tbts_fbgg','F10new_tbts');" href="http://news.10jqka.com.cn/field/sn/20250207/50728759.shtml  " title="贵州茅台：贵州茅台关于回购股份实施进展的公告">《贵州茅台：贵州茅台关于回购股份实施进展的公告》</a>
                        	<a class="client remind_pubnote tip_button" onclick="TA.log({'id':'f10_tbts_ckdygg','fid':'F10new_tbts','nj':1})" href="http://news.10jqka.com.cn/field/sn/20250207/50728759.shtml  " target="_blank" title="点击查看对应公告"></a>
            
			

		</span>


	</td>


                    				</tr>
            	                    			
            				                    				<tr >
                    					<td class="hltip tc f12">2025-01-24</td>
                    					<td><strong class="hltip fl">大宗交易：</strong>
    
	<a  ifjump='1'  newtaid='f10_zxdt_code-rctevt-r6c2-clickxqcont-quota-dzjy' taid="f10_tbts_dzjy#F10new_tbts" class="newtaid fr hla gray skipto f12" name="index.html#deal" title="到大宗交易栏目查看">详情&gt;&gt;</a> 

	<span>

	<a  ifjump='1'  newtaid='f10_zxdt_code-rctevt-r6c2-clickcontent-quota-dzjy' taid="f10_tbts_dzjy#F10new_tbts" class="newtaid skipto" name="index.html#deal" title="到大宗交易栏目查看">
        成交均价1436.00元，溢价率0.00%，成交量4.5万股，成交金额6462万元	</a>

	</span>

</td>                    				</tr>
            	                    			
            				                    				<tr  class="none">
                    					<td class="hltip tc f12">2025-01-22</td>
                    					<td><strong class="hltip fl">大宗交易：</strong>
    
	<a  ifjump='1'  newtaid='f10_zxdt_code-rctevt-r7c2-clickxqcont-quota-dzjy' taid="f10_tbts_dzjy#F10new_tbts" class="newtaid fr hla gray skipto f12" name="index.html#deal" title="到大宗交易栏目查看">详情&gt;&gt;</a> 

	<span>

	<a  ifjump='1'  newtaid='f10_zxdt_code-rctevt-r7c2-clickcontent-quota-dzjy' taid="f10_tbts_dzjy#F10new_tbts" class="newtaid skipto" name="index.html#deal" title="到大宗交易栏目查看">
        成交均价1441.00元，溢价率0.00%，成交量1.94万股，成交金额2796万元	</a>

	</span>

</td>                    				</tr>
            	                    			
            				                    				<tr  class="none">
                    					<td class="hltip tc f12">2025-01-21</td>
                    					<td><strong class="hltip fl">大宗交易：</strong>
    
	<a  ifjump='1'  newtaid='f10_zxdt_code-rctevt-r8c2-clickxqcont-quota-dzjy' taid="f10_tbts_dzjy#F10new_tbts" class="newtaid fr hla gray skipto f12" name="index.html#deal" title="到大宗交易栏目查看">详情&gt;&gt;</a> 

	<span>

	<a  ifjump='1'  newtaid='f10_zxdt_code-rctevt-r8c2-clickcontent-quota-dzjy' taid="f10_tbts_dzjy#F10new_tbts" class="newtaid skipto" name="index.html#deal" title="到大宗交易栏目查看">
        成交均价1468.15元，溢价率0.00%，成交量21.78万股，成交金额3.198亿元	</a>

	</span>

</td>                    				</tr>
            	                    			
            				                    				<tr  class="none">
                    					<td class="hltip tc f12">2025-01-09</td>
                    						<td>


		<strong class="hltip fl">发布公告：</strong>


		<a   ifjump='1'  newtaid='f10_zxdt_code-rctevt-r9c2-clickxqcont-quota-notice'  taid="f10_tbts_fbgg#F10new_tbts" class="newtaid fr hla gray f12 skipto" name="news.html#pub" title="到公告列表查看更多">更多&gt;&gt;</a>


		<span>

            

			<a   ifjump='1' newtaid='f10_zxdt_code-rctevt-r9c2-clickcontent-quota-notice'  class="newtaid client" target="_blank" onclick="clickTalogStat('f10_tbts_fbgg','F10new_tbts');" href="http://news.10jqka.com.cn/field/sn/20250109/50479333.shtml  " title="贵州茅台：贵州茅台第四届董事会2025年度第一次会议决议公告">《贵州茅台：贵州茅台第四届董事会2025年度第一次会议决议公告》</a>
                        	<a class="client remind_pubnote tip_button" onclick="TA.log({'id':'f10_tbts_ckdygg','fid':'F10new_tbts','nj':1})" href="http://news.10jqka.com.cn/field/sn/20250109/50479333.shtml  " target="_blank" title="点击查看对应公告"></a>
            
			

		</span>


	</td>


                    				</tr>
            	                    			
            				                    				<tr  class="none">
                    					<td class="hltip tc f12">2025-01-03</td>
                    						<td>


		<strong class="hltip fl">发布公告：</strong>


		<a   ifjump='1'  newtaid='f10_zxdt_code-rctevt-r10c2-clickxqcont-quota-notice'  taid="f10_tbts_fbgg#F10new_tbts" class="newtaid fr hla gray f12 skipto" name="news.html#pub" title="到公告列表查看更多">更多&gt;&gt;</a>


		<span>

            

			<a   ifjump='1' newtaid='f10_zxdt_code-rctevt-r10c2-clickcontent-quota-notice'  taid="f10_tbts_fbgg#F10new_tbts" class="newtaid skipto" name="news.html#pub" title="贵州茅台：贵州茅台关于首次回购公司股份暨回购进展的公告">


				《贵州茅台：贵州茅台关于首次回购公司股份暨回购进展的公告》&nbsp;等2篇公告


			</a>


            

		</span>


	</td>


                    				</tr>
            	                    			
            				                    				<tr  class="none">
                    					<td class="hltip tc f12">2025-01-03</td>
                    					<td>

	<strong class="hltip title fl">业绩预告：</strong> 

	<span class="hltip_cont">预计年报业绩：净利润857.0亿元左右，增长幅度为14.67%左右
		<span class="performance_trailer"> 

			<a   ifjump='1' newtaid='f10_zxdt_code-rctevt-r11c2-expand-quota-yejiyugao' class="newtaid check_details f12" href="javascript:void(0);" onclick="clickTalogStat('f10_tbts_yjyg','F10new_tbts');">

				
			</a>
             			<div class="check_else">

				<dl>

					<dt class="fl">原因：</dt>

					<dd class="tip"></dd>

				</dl>

			</div> 

		</span>									

	</span>

</td>                    				</tr>
            	                    			
            				                    				<tr  class="none">
                    					<td class="hltip tc f12">2024-12-28</td>
                    						<td>


		<strong class="hltip fl">发布公告：</strong>


		<a   ifjump='1'  newtaid='f10_zxdt_code-rctevt-r12c2-clickxqcont-quota-notice'  taid="f10_tbts_fbgg#F10new_tbts" class="newtaid fr hla gray f12 skipto" name="news.html#pub" title="到公告列表查看更多">更多&gt;&gt;</a>


		<span>

            

			<a   ifjump='1' newtaid='f10_zxdt_code-rctevt-r12c2-clickcontent-quota-notice'  taid="f10_tbts_fbgg#F10new_tbts" class="newtaid skipto" name="news.html#pub" title="贵州茅台：贵州茅台关于以集中竞价交易方式回购公司股份的回购报告书">


				《贵州茅台：贵州茅台关于以集中竞价交易方式回购公司股份的回购报告书》&nbsp;等3篇公告


			</a>


            

		</span>


	</td>


                    				</tr>
            	                    			
            				                    				<tr  class="none">
                    					<td class="hltip tc f12">2024-12-28</td>
                    					<td>
    <strong class="hltip title fl">股票回购：</strong>
    <span class="hltip_cont">拟回购不超过338.6万股，进度：实施回购；已累计回购68.51万股，均价为1460元</span>
</td>                    				</tr>
            	                    			
            				                    				<tr  class="none">
                    					<td class="hltip tc f12">2024-12-19</td>
                    					

	<td><strong class="hltip fl">实施分红：</strong> 


		<a   ifjump='1'  newtaid='f10_zxdt_code-rctevt-r14c2-clickxqcont-quota-fhzzplan' class="newtaid fr hla gray f12 a_cursor" onclick="jumpToUrl('./bonus.html', 'bonus', '');return false;" taid="f10_tbts_fhrz#F10new_tbts" title="到分红融资栏目查看">详情&gt;&gt;</a> 


		<span>


			<a   ifjump='1'  class="newtaid" onclick="jumpToUrl('./bonus.html', 'bonus', '');return false;" newtaid='f10_zxdt_code-rctevt-r14c2-clickcontent-quota-fhzzplan' taid="f10_tbts_fhrz#F10new_tbts">10派238.82元(含税)，股权登记日为2024-12-19，除权除息日为2024-12-20，派息日为2024-12-20</a>


		</span>


	</td>



                    				</tr>
            	                    			
            				                    				<tr  class="none">
                    					<td class="hltip tc f12">2024-12-14</td>
                    						<td>


		<strong class="hltip fl">发布公告：</strong>


		<a   ifjump='1'  newtaid='f10_zxdt_code-rctevt-r15c2-clickxqcont-quota-notice'  taid="f10_tbts_fbgg#F10new_tbts" class="newtaid fr hla gray f12 skipto" name="news.html#pub" title="到公告列表查看更多">更多&gt;&gt;</a>


		<span>

            

			<a   ifjump='1' newtaid='f10_zxdt_code-rctevt-r15c2-clickcontent-quota-notice'  class="newtaid client" target="_blank" onclick="clickTalogStat('f10_tbts_fbgg','F10new_tbts');" href="http://news.10jqka.com.cn/field/sn/20241214/50212685.shtml  " title="贵州茅台：贵州茅台2024年中期权益分派实施公告">《贵州茅台：贵州茅台2024年中期权益分派实施公告》</a>
                        	<a class="client remind_pubnote tip_button" onclick="TA.log({'id':'f10_tbts_ckdygg','fid':'F10new_tbts','nj':1})" href="http://news.10jqka.com.cn/field/sn/20241214/50212685.shtml  " target="_blank" title="点击查看对应公告"></a>
            
			

		</span>


	</td>


                    				</tr>
            	                    			
            				                    				<tr  class="none">
                    					<td class="hltip tc f12">2024-11-27</td>
                    					<td>
	<strong class="hltip title fl">股东大会：</strong> 
	<span class="hltip_cont">召开临时股东大会，审议相关议案
		<span class="performance_trailer"> 
			<a   ifjump='1' newtaid='f10_zxdt_code-rctevt-r16c2-expand-quota-gddh' class="newtaid check_details f12" href="javascript:void(0);" onclick="clickTalogStat('f10_tbts_gddh','F10new_tbts');">
				详细内容&nbsp<span class="open_btn">▼</span><span class="close_btn hidden">▲</span>
			</a>
							<a class="client remind_pubnote tip_button" onclick="TA.log({'id':'f10_tbts_ckdygg','fid':'F10new_tbts','nj':1})" href="http://news.10jqka.com.cn/field/sn/20241127/49836401.shtml  " target="_blank" title="点击查看对应公告"></a>
						<div class="check_else">  
				<dl>
					<dd class="tip">1.审议《2024-2026年度现金分红回报规划》
2.审议《2024年中期利润分配方案》
3.审议《关于以集中竞价交易方式回购公司股份的方案》
4.审议《关于调整酱香型系列酒制酒技改工程及配套设施项目建设规模及总投资的议案》
5.审议《关于选举监事的议案》</dd>
				</dl>
			</div> 
		</span>									
	</span>

</td>                    				</tr>
            	                    			
            				                    				<tr  class="none">
                    					<td class="hltip tc f12">2024-10-26</td>
                    					

	<td><strong class="hltip fl">业绩披露：</strong> 


		<a ifjump='1' newtaid='f10_zxdt_code-rctevt-r17c2-clickxqcont-quota-yejipilu' class="newtaid fr hla gray f12" href="javascript:void(0)" onclick="jumpToUrl('./finance.html', 'financen', '');return false;" taid="f10_tbts_yjpl#F10new_tbts" title="到财务指标栏目查看">详情&gt;&gt;</a> 


		<span><a ifjump='1'  class="newtaid" newtaid='f10_zxdt_code-rctevt-r17c2-clickcontent-quota-yejipilu' href="javascript:void(0)" onclick="jumpToUrl('./finance.html', 'financen', '');return false;" taid="f10_tbts_yjpl#F10new_tbts">2024年三季报每股收益48.42元，净利润608.28亿元，同比去年增长15.04%</a></span>

                    <a class="client remind_pubnote tip_button" onclick="TA.log({'id':'f10_tbts_ckdygg','fid':'F10new_tbts','nj':1})" href="http://news.10jqka.com.cn/field/sn/20241026/49640507.shtml  " target="_blank" title="点击查看对应公告"></a>
        
	</td>


                    				</tr>
            	                    			
            				                    				<tr  class="none">
                    					<td class="hltip tc f12">2024-10-26</td>
                    					<td>

	<strong class="hltip fl">股东人数变化：</strong> 

	<a   ifjump='1'  newtaid='f10_zxdt_code-rctevt-r18c2-clickxqcont-quota-holdernum' taid="f10_tbts_gdrs#F10new_tbts" class="newtaid fr hla gray f12 skipto" name="holder.html" title="到股东研究栏目查看">详情&gt;&gt;</a> 

	<span>

		<a   ifjump='1' newtaid='f10_zxdt_code-rctevt-r18c2-clickcontent-quota-holdernum'  taid="f10_tbts_gdrs#F10new_tbts" class="newtaid skipto" name="holder.html">

		截止2024-09-30，公司股东人数比上期（2024-06-30）增长2373户，幅度1.19%
		</a>
		
	</span>

</td>                    				</tr>
            	                    			
            				                    				<tr  class="none">
                    					<td class="hltip tc f12">2024-08-23</td>
                    					<td>
	<strong class="hltip title fl">新增概念：</strong> 
	<span class="hltip_cont">增加同花顺概念“西部大开发”概念解析
	    		<span class="performance_trailer"> 
			<a   ifjump='1' newtaid='f10_zxdt_code-rctevt-r19c2-expand-quota-xzgn'  class="newtaid check_details f12" href="javascript:void(0);" onclick="clickTalogStat('f10_tbts_xzgn','F10new_tbts');">
				详细内容&nbsp<span class="open_btn">▼</span><span class="close_btn hidden">▲</span>
			</a>
			<div class="check_else">  
				<dl>
					<dd class="tip"><span>西部大开发：</span>公司注册地址为贵州省遵义市仁怀市茅台镇</dd>
				</dl>
			</div> 
		</span>	
										
	</span>
</td>
                    				</tr>
            	                    			
            				                    				<tr  class="none">
                    					<td class="hltip tc f12">2024-08-09</td>
                    						<td><strong class="hltip fl">分配预案：</strong> 


		<a   ifjump='1'  newtaid='f10_zxdt_code-rctevt-r20c2-clickxqcont-quota-fpplan' class="newtaid fr hla gray f12 a_cursor" onclick="jumpToUrl('./bonus.html', 'bonus', '');return false;" taid="f10_tbts_fhrz#F10new_tbts" title="到分红融资栏目查看">详情&gt;&gt;</a> 


		<span>


			<a   ifjump='1'  class="newtaid" newtaid='f10_zxdt_code-rctevt-r20c2-clickcontent-quota-fpplan' onclick="jumpToUrl('./bonus.html', 'bonus', '');return false;" taid="f10_tbts_fhrz#F10new_tbts">
			
			2024年中报分配方案：不分配不转增，方案进度：董事会通过			
			</a>
                        <a class="client remind_pubnote tip_button" onclick="TA.log({'id':'f10_tbts_ckdygg','fid':'F10new_tbts','nj':1})" href="http://news.10jqka.com.cn/field/sn/20240809/48741901.shtml  " target="_blank" title="点击查看对应公告"></a>
            
		</span>


	</td>




                    				</tr>
            	                    			
            				                    				<tr  class="none">
                    					<td class="hltip tc f12">2024-08-09</td>
                    					

	<td><strong class="hltip fl">业绩披露：</strong> 


		<a ifjump='1' newtaid='f10_zxdt_code-rctevt-r21c2-clickxqcont-quota-yejipilu' class="newtaid fr hla gray f12" href="javascript:void(0)" onclick="jumpToUrl('./finance.html', 'financen', '');return false;" taid="f10_tbts_yjpl#F10new_tbts" title="到财务指标栏目查看">详情&gt;&gt;</a> 


		<span><a ifjump='1'  class="newtaid" newtaid='f10_zxdt_code-rctevt-r21c2-clickcontent-quota-yejipilu' href="javascript:void(0)" onclick="jumpToUrl('./finance.html', 'financen', '');return false;" taid="f10_tbts_yjpl#F10new_tbts">2024年中报每股收益33.19元，净利润416.96亿元，同比去年增长15.88%</a></span>

                    <a class="client remind_pubnote tip_button" onclick="TA.log({'id':'f10_tbts_ckdygg','fid':'F10new_tbts','nj':1})" href="http://news.10jqka.com.cn/field/sn/20240809/48741905.shtml  " target="_blank" title="点击查看对应公告"></a>
        
	</td>


                    				</tr>
            	                    			
            				                    				<tr  class="none">
                    					<td class="hltip tc f12">2024-08-09</td>
                    					<td>

	<strong class="hltip fl">股东人数变化：</strong> 

	<a   ifjump='1'  newtaid='f10_zxdt_code-rctevt-r22c2-clickxqcont-quota-holdernum' taid="f10_tbts_gdrs#F10new_tbts" class="newtaid fr hla gray f12 skipto" name="holder.html" title="到股东研究栏目查看">详情&gt;&gt;</a> 

	<span>

		<a   ifjump='1' newtaid='f10_zxdt_code-rctevt-r22c2-clickcontent-quota-holdernum'  taid="f10_tbts_gdrs#F10new_tbts" class="newtaid skipto" name="holder.html">

		截止2024-06-30，公司股东人数比上期（2024-03-31）增长3.82万户，幅度23.73%
		</a>
		
	</span>

</td>                    				</tr>
            	                    			
            				                    				<tr  class="none">
                    					<td class="hltip tc f12">2024-08-09</td>
                    					<td>
	<strong class="hltip title fl">参控公司：</strong> 
	<span class="hltip_cont">
	<a   ifjump='1' newtaid='f10_zxdt_code-rctevt-r23c2-clickxqcont-quota-ckgs' taid="f10_tbts_ckgs#F10new_tbts" class="newtaid fr hla gray f12 skipto" name="company.html#share" title="到公司资料—参控股公司查看">详情&gt;&gt;</a> 
	参控北京友谊使者商贸有限公司，参控比例为70.0000%，参控关系为子公司		<span class="performance_trailer"> 
			<a   ifjump='1' newtaid='f10_zxdt_code-rctevt-r23c2-expand-quota-ckgs'  class="newtaid check_details f12" style="width:auto" href="javascript:void(0);" onclick="clickTalogStat('f10_tbts_qtckgs','F10new_tbts');">
				其它参控公司&nbsp<span class="open_btn">▼</span><span class="close_btn hidden">▲</span>
			</a>
			<div class="check_else">  
				<dl>
				    					<dd class="tip">参控贵州茅台酒巴黎贸易有限公司，参控比例为100.0000%，参控关系为子公司					</dd><br>
										<dd class="tip">参控贵州茅台酒进出口有限责任公司，参控比例为70.0000%，参控关系为子公司					</dd><br>
										<dd class="tip">参控贵州茅台酒销售有限公司，参控比例为95.0000%，参控关系为子公司					</dd><br>
										<dd class="tip">参控贵州茅台酱香酒营销有限公司，参控比例为100.0000%，参控关系为子公司					</dd><br>
										<dd class="tip">参控贵州茅台集团财务有限公司，参控比例为51.0000%，参控关系为子公司					</dd><br>
										<dd class="tip">参控贵州赖茅酒业有限公司，参控比例为43.0000%，参控关系为子公司					</dd><br>
									</dl>
			</div> 
		</span>									
	</span>

</td>

                    				</tr>
            	                    			
            				                    				<tr  class="none">
                    					<td class="hltip tc f12">2024-06-18</td>
                    					

	<td><strong class="hltip fl">实施分红：</strong> 


		<a   ifjump='1'  newtaid='f10_zxdt_code-rctevt-r24c2-clickxqcont-quota-fhzzplan' class="newtaid fr hla gray f12 a_cursor" onclick="jumpToUrl('./bonus.html', 'bonus', '');return false;" taid="f10_tbts_fhrz#F10new_tbts" title="到分红融资栏目查看">详情&gt;&gt;</a> 


		<span>


			<a   ifjump='1'  class="newtaid" onclick="jumpToUrl('./bonus.html', 'bonus', '');return false;" newtaid='f10_zxdt_code-rctevt-r24c2-clickcontent-quota-fhzzplan' taid="f10_tbts_fhrz#F10new_tbts">10派308.76元(含税)，股权登记日为2024-06-18，除权除息日为2024-06-19，派息日为2024-06-19</a>


		</span>


	</td>



                    				</tr>
            	                    			
            				                    				<tr  class="none">
                    					<td class="hltip tc f12">2024-05-29</td>
                    					<td>
	<strong class="hltip title fl">股东大会：</strong> 
	<span class="hltip_cont">召开年度股东大会，审议相关议案
		<span class="performance_trailer"> 
			<a   ifjump='1' newtaid='f10_zxdt_code-rctevt-r25c2-expand-quota-gddh' class="newtaid check_details f12" href="javascript:void(0);" onclick="clickTalogStat('f10_tbts_gddh','F10new_tbts');">
				详细内容&nbsp<span class="open_btn">▼</span><span class="close_btn hidden">▲</span>
			</a>
						<div class="check_else">  
				<dl>
					<dd class="tip">1.审议《2023年度董事会工作报告》
2.审议《2023年度监事会工作报告》
3.审议《2023年度独立董事述职报告》
4.审议《2023年年度报告(全文及摘要)》
5.审议《2023年度财务决算报告》
6.审议《2024年度财务预算方案》
7.审议《2023年度利润分配方案》
8.审议《关于聘请2024年度财务审计机构和内控审计机构的议案》
9.审议《关于选举董事的议案》
10.审议《关于贵州茅台集团财务有限公司日常关联交易的议案》
11.审议《关于修订<公司独立董事制度>的议案》</dd>
				</dl>
			</div> 
		</span>									
	</span>

</td>                    				</tr>
            	                    			
            				                    				<tr  class="none">
                    					<td class="hltip tc f12">2024-04-27</td>
                    					

	<td><strong class="hltip fl">业绩披露：</strong> 


		<a ifjump='1' newtaid='f10_zxdt_code-rctevt-r26c2-clickxqcont-quota-yejipilu' class="newtaid fr hla gray f12" href="javascript:void(0)" onclick="jumpToUrl('./finance.html', 'financen', '');return false;" taid="f10_tbts_yjpl#F10new_tbts" title="到财务指标栏目查看">详情&gt;&gt;</a> 


		<span><a ifjump='1'  class="newtaid" newtaid='f10_zxdt_code-rctevt-r26c2-clickcontent-quota-yejipilu' href="javascript:void(0)" onclick="jumpToUrl('./finance.html', 'financen', '');return false;" taid="f10_tbts_yjpl#F10new_tbts">2024年一季报每股收益19.16元，净利润240.65亿元，同比去年增长15.73%</a></span>

                    <a class="client remind_pubnote tip_button" onclick="TA.log({'id':'f10_tbts_ckdygg','fid':'F10new_tbts','nj':1})" href="http://news.10jqka.com.cn/field/sn/20240427/47366233.shtml  " target="_blank" title="点击查看对应公告"></a>
        
	</td>


                    				</tr>
            	                    			
            				                    				<tr  class="none">
                    					<td class="hltip tc f12">2024-04-27</td>
                    					<td>

	<strong class="hltip fl">股东人数变化：</strong> 

	<a   ifjump='1'  newtaid='f10_zxdt_code-rctevt-r27c2-clickxqcont-quota-holdernum' taid="f10_tbts_gdrs#F10new_tbts" class="newtaid fr hla gray f12 skipto" name="holder.html" title="到股东研究栏目查看">详情&gt;&gt;</a> 

	<span>

		<a   ifjump='1' newtaid='f10_zxdt_code-rctevt-r27c2-clickcontent-quota-holdernum'  taid="f10_tbts_gdrs#F10new_tbts" class="newtaid skipto" name="holder.html">

		截止2024-03-31，公司股东人数比上期（2023-12-31）减少631户，幅度-0.39%
		</a>
		
	</span>

</td>                    				</tr>
            	                    			
            				                    				<tr  class="none">
                    					<td class="hltip tc f12">2024-04-03</td>
                    					

	<td><strong class="hltip fl">业绩披露：</strong> 


		<a ifjump='1' newtaid='f10_zxdt_code-rctevt-r28c2-clickxqcont-quota-yejipilu' class="newtaid fr hla gray f12" href="javascript:void(0)" onclick="jumpToUrl('./finance.html', 'financen', '');return false;" taid="f10_tbts_yjpl#F10new_tbts" title="到财务指标栏目查看">详情&gt;&gt;</a> 


		<span><a ifjump='1'  class="newtaid" newtaid='f10_zxdt_code-rctevt-r28c2-clickcontent-quota-yejipilu' href="javascript:void(0)" onclick="jumpToUrl('./finance.html', 'financen', '');return false;" taid="f10_tbts_yjpl#F10new_tbts">2023年年报每股收益59.49元，净利润747.34亿元，同比去年增长19.16%</a></span>

                    <a class="client remind_pubnote tip_button" onclick="TA.log({'id':'f10_tbts_ckdygg','fid':'F10new_tbts','nj':1})" href="http://news.10jqka.com.cn/field/sn/20240403/46711917.shtml  " target="_blank" title="点击查看对应公告"></a>
        
	</td>


                    				</tr>
            	                    			
            				                    				<tr  class="none">
                    					<td class="hltip tc f12">2024-04-03</td>
                    					<td>

	<strong class="hltip fl">股东人数变化：</strong> 

	<a   ifjump='1'  newtaid='f10_zxdt_code-rctevt-r29c2-clickxqcont-quota-holdernum' taid="f10_tbts_gdrs#F10new_tbts" class="newtaid fr hla gray f12 skipto" name="holder.html" title="到股东研究栏目查看">详情&gt;&gt;</a> 

	<span>

		<a   ifjump='1' newtaid='f10_zxdt_code-rctevt-r29c2-clickcontent-quota-holdernum'  taid="f10_tbts_gdrs#F10new_tbts" class="newtaid skipto" name="holder.html">

		截止2023-12-31，公司股东人数比上期（2023-09-30）增长1.16万户，幅度7.74%
		</a>
		
	</span>

</td>                    				</tr>
            	                    			
            				                    				<tr  class="none">
                    					<td class="hltip tc f12">2024-04-03</td>
                    					<td>
	<strong class="hltip title fl">参控公司：</strong> 
	<span class="hltip_cont">
	<a   ifjump='1' newtaid='f10_zxdt_code-rctevt-r30c2-clickxqcont-quota-ckgs' taid="f10_tbts_ckgs#F10new_tbts" class="newtaid fr hla gray f12 skipto" name="company.html#share" title="到公司资料—参控股公司查看">详情&gt;&gt;</a> 
	参控北京友谊使者商贸有限公司，参控比例为70.0000%，参控关系为子公司		<span class="performance_trailer"> 
			<a   ifjump='1' newtaid='f10_zxdt_code-rctevt-r30c2-expand-quota-ckgs'  class="newtaid check_details f12" style="width:auto" href="javascript:void(0);" onclick="clickTalogStat('f10_tbts_qtckgs','F10new_tbts');">
				其它参控公司&nbsp<span class="open_btn">▼</span><span class="close_btn hidden">▲</span>
			</a>
			<div class="check_else">  
				<dl>
				    					<dd class="tip">参控贵州茅台酒巴黎贸易有限公司，参控比例为100.0000%，参控关系为子公司					</dd><br>
										<dd class="tip">参控贵州茅台酒进出口有限责任公司，参控比例为70.0000%，参控关系为子公司					</dd><br>
										<dd class="tip">参控贵州茅台酒销售有限公司，参控比例为95.0000%，参控关系为子公司					</dd><br>
										<dd class="tip">参控贵州茅台酱香酒营销有限公司，参控比例为100.0000%，参控关系为子公司					</dd><br>
										<dd class="tip">参控贵州茅台集团财务有限公司，参控比例为51.0000%，参控关系为子公司					</dd><br>
										<dd class="tip">参控贵州赖茅酒业有限公司，参控比例为43.0000%，参控关系为子公司					</dd><br>
									</dl>
			</div> 
		</span>									
	</span>

</td>

                    				</tr>
            	                    			
            				                    				<tr  class="none">
                    					<td class="hltip tc f12">2023-06-28</td>
                    					
<td>
	<strong class="hltip title fl">股东增持：</strong> 
	<span class="hltip_cont">中国贵州茅台酒厂(集团)有限责任公司、贵州茅台酒厂(集团)技术开发有限公司于2023.02.13至2023.06.26期间累计增持80.27万股，占流通股本比例0.06%		<span class="performance_trailer"> 
			<a   ifjump='1' newtaid='f10_zxdt_code-rctevt-r31c2-expand-quota-chigu'  class="newtaid check_details f12" href="javascript:void(0);" onclick="clickTalogStat('f10_tbts_gddh','F10new_tbts');">
				详细内容&nbsp<span class="open_btn">▼</span><span class="close_btn hidden">▲</span>
			</a>
						<div class="check_else">  
				<dl>
				    					<dd class="tip">中国贵州茅台酒厂(集团)有限责任公司                    					于2023.02.13至2023.06.26期间										增持77.13万股，占流通股本比例0.06%
					</dd><br>
										<dd class="tip">贵州茅台酒厂(集团)技术开发有限公司                    					于2023.02.13至2023.06.26期间										增持3.14万股，占流通股本比例0.0025%
					</dd><br>
									</dl>
			</div> 
		</span>									
	</span>

</td>

                    				</tr>
            	                    			
            				                    				<tr  class="none">
                    					<td class="hltip tc f12">2023-02-11</td>
                    					
<td>
	<strong class="hltip title fl">股东增持：</strong> 
	<span class="hltip_cont">中国贵州茅台酒厂(集团)有限责任公司、贵州茅台酒厂(集团)技术开发有限公司于2023.02.10累计增持15.45万股，占流通股本比例0.01%		<span class="performance_trailer"> 
			<a   ifjump='1' newtaid='f10_zxdt_code-rctevt-r32c2-expand-quota-chigu'  class="newtaid check_details f12" href="javascript:void(0);" onclick="clickTalogStat('f10_tbts_gddh','F10new_tbts');">
				详细内容&nbsp<span class="open_btn">▼</span><span class="close_btn hidden">▲</span>
			</a>
						<div class="check_else">  
				<dl>
				    					<dd class="tip">中国贵州茅台酒厂(集团)有限责任公司                    					于2023.02.10										增持14.83万股，占流通股本比例0.01%
					</dd><br>
										<dd class="tip">贵州茅台酒厂(集团)技术开发有限公司                    					于2023.02.10										增持6200股，占流通股本比例0.0005%
					</dd><br>
									</dl>
			</div> 
		</span>									
	</span>

</td>

                    				</tr>
            	                    			
            				                    				<tr  class="none">
                    					<td class="hltip tc f12">2022-11-29</td>
                    					    <td>
        <strong class="hltip title fl">增减持计划：</strong>
        <span>公司实际控制人中国贵州茅台酒厂(集团)有限责任公司、贵州茅台酒厂(集团)技术开发有限公司计划自2022-12-27起至2023-06-26，拟使用不超过30.94亿元进行增持</span>
    </td>
                    				</tr>
            	                    			
            				                    				<tr  class="none">
                    					<td class="hltip tc f12">2021-09-10</td>
                    					<td>
        <strong class="hltip title fl">股权转让：</strong>
    <a ifjump='1' newtaid='f10_zxdt_code-rctevt-r34c2-clickxqcont-quota-shougou' taid="f10_tbts_sgjb#F10new_tbts" class="newtaid fr hla gray skipto f12" name="capital.html#transfer" title="到公司大事的资本运作栏目查看">详情&gt;&gt;</a>
    <span>
        贵州省人民政府国有资产监督管理委员会拟转让中国贵州茅台酒厂(集团)有限责任公司10.00%股权给贵州金融控股集团有限责任公司(贵州贵民投资集团有限责任公司)，进度：完成
        <span class="performance_trailer">
                    <a class="check_details f12" href="javascript:void(0);" onclick="clickTalogStat('f10_tbts_ztyyxl');">
            <span class="open_btn">详细内容▼</span><span class="close_btn hidden">收起▲</span>
        </a>
                <div class="check_else">
        贵州茅台酒股份有限公司(以下简称公司)于2021年9月9日收到公司控股股东中国贵州茅台酒厂(集团)有限责任公司(以下简称茅台集团公司)通知:根据贵州省财政厅、贵州省人力资源和社会保障厅、贵州省人民政府国有资产监督管理委员会(以下简称贵州省国资委)《关于划转部分国有资本充实社保基金有关事项的通知》(黔财工〔2020〕286号)和贵州省国资委《关于做好我委所持有关企业部分股权划转金控集团公司持有有关事项的通知》(黔国资通产权〔2021〕3号),贵州省国资委将其持有的茅台集团公司10%股权无偿划转给贵州金融控股集团有限责任公司(贵州贵民投资集团有限责任公司)。转让完成后受让方持有标的公司10%股权。        </div>
        </span>
    </span>

</td>                    				</tr>
            	                    			
            				                    				<tr  class="none">
                    					<td class="hltip tc f12">2021-02-22</td>
                    					<td>
	<strong class="hltip title fl">概念动态：</strong> 
	<span class="hltip_cont">“白酒概念”概念有解析内容更新
	    		<span class="performance_trailer"> 
			<a ifjump='1' newtaid='f10_zxdt_code-rctevt-r35c2-expand-quota-xzgn' class="newtaid check_details f12" href="javascript:void(0);" onclick="clickTalogStat('f10_tbts_gndt','F10new_tbts');">
				详细内容&nbsp<span class="open_btn">▼</span><span class="close_btn hidden">▲</span>
			</a>
			<div class="check_else">  
				<dl>
					<dd class="tip"><span>白酒概念：</span>全国两大高端白酒龙头之一(行业第一)，酱香型白酒代表，拥有八百年历史，被尊称国酒”，曾于1915年获“巴拿马万国博览会”金奖。公司主要业务是茅台酒及系列酒的生产与销售。 主导产品“贵州茅台酒”是世界三大蒸馏名酒之一，也是集国家地理标志产品、有机食品和国家非物质文化遗产于一身的白酒品牌。</dd>
				</dl>
			</div> 
		</span>	
										
	</span>
</td>
                    				</tr>
            	                    			
            				                    				<tr  class="none">
                    					<td class="hltip tc f12">2020-12-31</td>
                    					<td>
        <strong class="hltip title fl">股权转让：</strong>
    <a ifjump='1' newtaid='f10_zxdt_code-rctevt-r36c2-clickxqcont-quota-shougou' taid="f10_tbts_sgjb#F10new_tbts" class="newtaid fr hla gray skipto f12" name="capital.html#transfer" title="到公司大事的资本运作栏目查看">详情&gt;&gt;</a>
    <span>
        中国贵州茅台酒厂(集团)有限责任公司拟转让公司4.00%股权给贵州省国有资本运营有限责任公司，进度：完成
        <span class="performance_trailer">
                    <a class="check_details f12" href="javascript:void(0);" onclick="clickTalogStat('f10_tbts_ztyyxl');">
            <span class="open_btn">详细内容▼</span><span class="close_btn hidden">收起▲</span>
        </a>
                <div class="check_else">
        贵州茅台酒股份有限公司(以下简称本公司)于2020年12月23日接到本公司控股股东中国贵州茅台酒厂(集团)有限责任公司(以下简称茅台集团)《关于无偿划转贵州茅台酒股份有限公司国有股份的通知》,根据贵州省人民政府国有资产监督管理委员会的相关通知要求,茅台集团拟通过无偿划转方式将持有的本公司50,240,000股股份(占本公司总股本的4.00%)划转至贵州省国有资本运营有限责任公司。本次无偿划转完成后,茅台集团将持有本公司678,291,955股股份,占本公司总股本的54.00%;贵州省国有资本运营有限责任公司和贵州金融控股集团有限责任公司(贵州贵民投资集团有限责任公司)合计持有本公司62,311,148股股份,占本公司总股本的4.96%,其中,贵州省国有资本运营有限责任公司持有本公司58,823,928股股份,占本公司总股本的4.68%,贵州金融控股集团有限责任公司(贵州贵民投资集团有限责任公司)持有本公司3,487,220股股份,占本公司总股本的0.28%。        </div>
        </span>
    </span>

</td>                    				</tr>
            	                    			
            				                    				<tr  class="none">
                    					<td class="hltip tc f12">2020-01-02</td>
                    					<td>
        <strong class="hltip title fl">股权转让：</strong>
    <a ifjump='1' newtaid='f10_zxdt_code-rctevt-r37c2-clickxqcont-quota-shougou' taid="f10_tbts_sgjb#F10new_tbts" class="newtaid fr hla gray skipto f12" name="capital.html#transfer" title="到公司大事的资本运作栏目查看">详情&gt;&gt;</a>
    <span>
        中国贵州茅台酒厂(集团)有限责任公司拟转让公司4.00%股权给贵州省国有资本运营有限责任公司，进度：完成
        <span class="performance_trailer">
                    <a class="check_details f12" href="javascript:void(0);" onclick="clickTalogStat('f10_tbts_ztyyxl');">
            <span class="open_btn">详细内容▼</span><span class="close_btn hidden">收起▲</span>
        </a>
                <div class="check_else">
        　　贵州茅台酒股份有限公司（以下简称本公司）于2019年12月25日接到本公司控股股东中国贵州茅台酒厂（集团）有限责任公司（以下简称茅台集团）《关于无偿划转贵州茅台酒股份有限公司国有股份的通知》，根据贵州省人民政府国有资产监督管理委员会的相关通知要求，茅台集团拟通过无偿划转方式将持有的本公司50,240,000股股份（占本公司总股本的4.00%）划转至贵州省国有资本运营有限责任公司。　　本次无偿划转完成后,茅台集团将持有本公司728,531,955股股份,占本公司总股本的58.00%;贵州省国有资本运营有限责任公司将持有本公司50,240,000股股份,占本公司总股本的4.00%。        </div>
        </span>
    </span>

</td>                    				</tr>
            	                    			
            				            			</tbody>
            		</table>

		                </div>
            </div>
        </div>
           <div class="part_all_show_btn">
           <a   ifjump='1'  newtaid="f10_zxdt_code-rctevt-s-expand-sta-rctevt" onclick="TA.log({'id':'f10_click_zysj','nj':1})" class="newtaid arrow_btn btndown" href="javascript:void(0);"></a>
       </div>
    	</div>
	<div class="ft"></div>
</div>
<!-- end 特别提示 -->


<!-- 热点新闻 -->
    <!-- 新闻公告 -->
    <div class="m_box  clearfix" id="news" stat="index_news">
        <div class="hd">
            <h2>新闻公告</h2>
        </div>
        <div class="bd pt5 pr clearfix">
            <div class="clearfix index">
                <div class="m_box post_news fl post clearfix">
                    <div class="bd clearfix">
                        <div class="newslist clearfix">
                            <div class="news_title">热点新闻 <a   ifjump='1' newtaid="f10_zxdt_code-rdxw-nw-clickmore-sta-rdxw" class="newtaid gray f12 news_more skipto" title="更多热点新闻" href="###" name="news.html#mine" taid="f10_click_rdxw#f10_click_xwgg"> 更多>></a></div>
                            <!-- 新闻数据改为ajax请求 -->
                            <ul id="news_ajax_content"></ul>
                        </div>
                    </div>
                    <div class="ft">
                        <div class="ftin"></div>
                    </div>
                </div>
                <div class="m_box comp_post fr post">
                    <div class="bd clearfix">
                        <div class="newslist clearfix">
                            <div class="news_title">公司公告 <a   ifjump='1' newtaid="f10_zxdt_code-gsgg-ne-clickmore-sta-gsgg" class="newtaid gray f12 news_more skipto" title="更多公司公告" href="###" name="news.html#pub" taid="f10_click_gsgg#f10_click_xwgg"> 更多>></a></div>
                            <ul>
                                                                                                                                                <li>
                                            <a   ifjump='1'  newtaid="f10_zxdt_code-gsgg-r1c1-click-atgg-50728759" onclick="TA.log({'id':'f10_home_gsgg','fid':'f10_click_xwgg','nj':1})" href="http://news.10jqka.com.cn/field/sn/20250207/50728759.shtml  " class="newtaid client" target="_blank" title="贵州茅台：贵州茅台关于回购股份实施进展的公告">
                                                <span class="gray" sctime="02/06" todaytime="16:06">02/07</span>
                                                贵州茅台：贵州茅台关于回购股份实施进展的公告                                            </a>
                                        </li>
                                                                            <li>
                                            <a   ifjump='1'  newtaid="f10_zxdt_code-gsgg-r2c1-click-atgg-50479333" onclick="TA.log({'id':'f10_home_gsgg','fid':'f10_click_xwgg','nj':1})" href="http://news.10jqka.com.cn/field/sn/20250109/50479333.shtml  " class="newtaid client" target="_blank" title="贵州茅台：贵州茅台第四届董事会2025年度第一次会议决议公告">
                                                <span class="gray" sctime="01/08" todaytime="18:13">01/09</span>
                                                贵州茅台：贵州茅台第四届董事会2025年度第一次会议决议公告                                            </a>
                                        </li>
                                                                            <li>
                                            <a   ifjump='1'  newtaid="f10_zxdt_code-gsgg-r3c1-click-atgg-50414615" onclick="TA.log({'id':'f10_home_gsgg','fid':'f10_click_xwgg','nj':1})" href="http://news.10jqka.com.cn/field/sn/20250103/50414615.shtml  " class="newtaid client" target="_blank" title="贵州茅台：贵州茅台关于首次回购公司股份暨回购进展的公告">
                                                <span class="gray" sctime="01/02" todaytime="18:01">01/03</span>
                                                贵州茅台：贵州茅台关于首次回购公司股份暨回购进展的公告                                            </a>
                                        </li>
                                                                            <li>
                                            <a   ifjump='1'  newtaid="f10_zxdt_code-gsgg-r4c1-click-atgg-50414611" onclick="TA.log({'id':'f10_home_gsgg','fid':'f10_click_xwgg','nj':1})" href="http://news.10jqka.com.cn/field/sn/20250103/50414611.shtml  " class="newtaid client" target="_blank" title="贵州茅台：贵州茅台2024年度生产经营情况公告">
                                                <span class="gray" sctime="01/02" todaytime="18:01">01/03</span>
                                                贵州茅台：贵州茅台2024年度生产经营情况公告                                            </a>
                                        </li>
                                                                            <li>
                                            <a   ifjump='1'  newtaid="f10_zxdt_code-gsgg-r5c1-click-atgg-50360279" onclick="TA.log({'id':'f10_home_gsgg','fid':'f10_click_xwgg','nj':1})" href="http://news.10jqka.com.cn/field/sn/20241228/50360279.shtml  " class="newtaid client" target="_blank" title="贵州茅台：贵州茅台关于以集中竞价交易方式回购公司股份的回购报告书">
                                                <span class="gray" sctime="12/27" todaytime="21:48">12/28</span>
                                                贵州茅台：贵州茅台关于以集中竞价交易方式回购公司股份的回购报告书                                            </a>
                                        </li>
                                                                            <li>
                                            <a   ifjump='1'  newtaid="f10_zxdt_code-gsgg-r6c1-click-atgg-50360275" onclick="TA.log({'id':'f10_home_gsgg','fid':'f10_click_xwgg','nj':1})" href="http://news.10jqka.com.cn/field/sn/20241228/50360275.shtml  " class="newtaid client" target="_blank" title="贵州茅台：贵州茅台关于回购股份事项通知债权人公告">
                                                <span class="gray" sctime="12/27" todaytime="21:48">12/28</span>
                                                贵州茅台：贵州茅台关于回购股份事项通知债权人公告                                            </a>
                                        </li>
                                                                            <li>
                                            <a   ifjump='1'  newtaid="f10_zxdt_code-gsgg-r7c1-click-atgg-50360277" onclick="TA.log({'id':'f10_home_gsgg','fid':'f10_click_xwgg','nj':1})" href="http://news.10jqka.com.cn/field/sn/20241228/50360277.shtml  " class="newtaid client" target="_blank" title="贵州茅台：贵州茅台关于实施2024年中期权益分派后调整回购股份价格上限的公告">
                                                <span class="gray" sctime="12/27" todaytime="21:48">12/28</span>
                                                贵州茅台：贵州茅台关于实施2024年中期权益分派后调整回购股份价格上限的公告                                            </a>
                                        </li>
                                                                                                </ul>
                        </div>
                    </div>
                    <div class="ft">
                        <div class="ftin"></div>
                    </div>
                </div>
            </div>
        </div>
        <div class="ft"></div>
    </div>

<!-- 财务指标 -->
<!-- 财务指标 -->
<div class="m_box new_msg z100" id="finance" stat="index_finance">
	<div class="hd flow_index">
		<h2>财务指标</h2>
	</div>	
	<div class="bd">						
		<table class="m_table m_hl fixtable">
			<thead>
				<tr>
					<th width="95px">报告期\指标</th>
					<th>基本每股收益(元)</th>
					<th>每股净资产(元)</th>
					<th newtaid="f10_zxdt_code-cwzb-r1c4-clickword-quota-mggjj" class='newtaid iwcclick' taname='f10_stock_iwc_cwzbck' data-url='http://www.iwencai.com/yike/detail/auid/ead9ab6046834303?qs=client_f10_baike_2  ' content='每股公积金就是公积金除股票总股数。是公司未来扩张的物质基础，也可以是股东未来转赠红股的希望之所在。没有公积金的上市公司，就是没有希望的上市公司。'>每股资本公积金(元)</th>
					<th newtaid="f10_zxdt_code-cwzb-r1c5-clickword-quota-udpps" class='newtaid iwcclick' taname='f10_stock_iwc_cwzbck' data-url='http://www.iwencai.com/yike/detail/auid/0f7d51cf91a3a0c0?qs=client_f10_baike_2  ' content='每股未分配利润=企业当期未分配利润总额/总股本。每股未分配利润反映的是公司平均每股所占有的留存于公司内部的未分配利润的多少' >每股未分配利润(元)</th>
					<th newtaid="f10_zxdt_code-cwzb-r1c6-clickword-quota-ocfps" class='newtaid iwcclick' taname='f10_stock_iwc_cwzbck' data-url='http://www.iwencai.com/yike/detail/auid/80b26f89d653f5c9?qs=client_f10_baike_2  ' content='指用公司经营活动的现金流入-经营活动的现金流出再除以总股本。反映每股发行在外的普通股票所平均占有的现金流量，或者说是反映公司为每一普通股获取的现金流入量的指标'>每股经营现金流(元)</th>
					<th width="80px">营业总收入(元)</th>
					<th width="80px">净利润(元)</th>
					<th width="72px">净资产收益率</th>
					<th width="90px">变动原因</th>
				</tr>
			</thead>
			<tbody>
							<tr>
					<td class="tc f12">2024-09-30</td>
					<td>48.42</td>
					<td>189.23</td>
					<td>1.09</td>
					<td>153.56</td>
					<td>35.36</td>
					<td>1231.23亿</td>
					<td>608.28亿</td>
					<td>26.09%</td>
					<td class="tl f12">
						<div class="popp_box" style="display: inline">
												</div>
						三季报					</td>
				</tr>
							<tr>
					<td class="tc f12">2024-06-30</td>
					<td>33.19</td>
					<td>174.00</td>
					<td>1.09</td>
					<td>138.33</td>
					<td>29.15</td>
					<td>834.51亿</td>
					<td>416.96亿</td>
					<td>17.63%</td>
					<td class="tl f12">
						<div class="popp_box" style="display: inline">
												</div>
						中报					</td>
				</tr>
							<tr>
					<td class="tc f12">2024-03-31</td>
					<td>19.16</td>
					<td>190.84</td>
					<td>1.09</td>
					<td>156.86</td>
					<td>7.31</td>
					<td>464.85亿</td>
					<td>240.65亿</td>
					<td>10.57%</td>
					<td class="tl f12">
						<div class="popp_box" style="display: inline">
												</div>
						一季报					</td>
				</tr>
							<tr>
					<td class="tc f12">2023-12-31</td>
					<td>59.49</td>
					<td>171.68</td>
					<td>1.09</td>
					<td>137.70</td>
					<td>53.01</td>
					<td>1505.60亿</td>
					<td>747.34亿</td>
					<td>34.19%</td>
					<td class="tl f12">
						<div class="popp_box" style="display: inline">
												</div>
						年报					</td>
				</tr>
							<tr>
					<td class="tc f12">2023-09-30</td>
					<td>42.09</td>
					<td>173.39</td>
					<td>1.09</td>
					<td>143.39</td>
					<td>39.80</td>
					<td>1053.16亿</td>
					<td>528.76亿</td>
					<td>24.82%</td>
					<td class="tl f12">
						<div class="popp_box" style="display: inline">
												</div>
						三季报					</td>
				</tr>
						</tbody>
		</table>	
		<p class="p5_0 tr">
			<a   ifjump='1' newtaid="f10_zxdt_code-cwzb-se-clickmore-sta-caiwu" href="javascript:void(0)" onclick="jumpToUrl('./finance.html', 'financen', 'cwzb');return false;" taid="f10_click_cwsj#f10_click_cwzb" class="newtaid hla f12" title="到财务分析栏目查看">查看更多财务数据&gt;&gt;</a>
		</p>
	</div>
	<div class="ft">
		<div class="ftin"></div>
	</div>
</div>

<!-- 主力控盘 -->
<div class="m_box clearfix" id="main" stat="index_main">
	<div class="hd">
		<h2>主力控盘</h2>
	</div>
	<div class="bd clearfix">
				<table class="m_table m_hl">
			<thead>
				<tr>
					<th>指标/日期</th>
										<th>2024-09-30</th>
										<th>2024-06-30</th>
										<th>2024-03-31</th>
										<th>2023-12-31</th>
										<th>2023-09-30</th>
										<th>2023-06-30</th>
									</tr>
			</thead>
			<tbody>
				<tr>
					<th class="tl">股东总数</th>
										<td class="tr">201582</td>
										<td class="tr">199209</td>
										<td class="tr">161009</td>
										<td class="tr">161640</td>
										<td class="tr">150025</td>
										<td class="tr">161750</td>
									</tr>
				<tr>
					<th class="tl">较上期变化</th>
										<td class="tr"><span style="color:#FB0000">+1.19%</span></td>
										<td class="tr"><span style="color:#FB0000">+23.73%</span></td>
										<td class="tr"><span class="downcolor">-0.39%</span></td>
										<td class="tr"><span style="color:#FB0000">+7.74%</span></td>
										<td class="tr"><span class="downcolor">-7.25%</span></td>
										<td class="tr"><span style="color:#FB0000">+3.56%</span></td>
									</tr>
				<tr>
					<td class="tl" colspan="7"><span class="gray f12">提示：股票价格通常与股东人数成反比，股东人数越少，则代表筹码越集中，股价越有可能上涨</span></td>
				</tr>						
			</tbody>
		</table>
						<div class="flash_box_cont fl">
						          			 <p class="topcolor f14">
                                            截止2024-11-18，前十大流通股东持有<strong>9.01亿</strong>股，占流通盘<strong>71.72%</strong>，主力控盘度非常高。
                     </p>
							
			<div class="f10_label_box" style="display: none;">
                <div class="head">
                    <span class="close">X</span>
                    <span class="tip">贵州茅台</span>十大流通股东<span class="tip">机构持股成本</span>估算
                    <span class="gd_ques iwcclick" taname="f10_stock_iwc_cbgsbk" data-url="http://www.iwencai.com/yike/detail/auid/fd14fbc180f05d69?qs=client_f10_baike_14  " tag="syl" content="机构持股成本是计算机构（暂不含个人）从进入到现在的持仓成本价。通俗的算法就是，每个季度买卖股票所花的钱/当前股份数量。"></span>
                    
                </div>
                <div class="title">
                    <table>
                        <thead>
                        <tr>
			    <th>序号</th>
                            <th class="tl" width="225">机构名称</th>
                            <th>持股比例</th>
                            <th>成本估算</th>
                        </tr>
                        </thead>
                    </table>
                </div>
                <div class="cont" tag="-1">
                                        <table style="">
                        <tbody>
                                                </tbody>
                    </table>
                    
                    <table style="display: none;">
                        <tbody>
                                                 </tbody>               
                        </tbody>
                    </table>
                </div>
                <div class="totalinfo">
                    <span class="tip"></span>家机构平均持股成本<span class="tip"></span>元，最近收盘价<span class="tip">--</span>元 <a href="holder.html#flowholder" style="color: #0199fa;">[详细]</a>
                </div>
                <div class="clearfix">
                    <div class="opinion_opts_wraper fr evaluate " tag="main">
                    <b class="oow_left fl"></b>
                    <div class="oow_cont fl">
                       <span>您对此栏目的评价：</span>
                       <a class="opinion_up" tag="support" href="javascript:void(0)">有用 <span class="opi_up">0</span></a>
                       <a class="opinion_down" tag="opposition" href="javascript:void(0)"> 没用 <span class="opi_down">0</span></a>
                       <span class="btn_idea">
                       <a onclick="$('.f10_append_advice').click();" title="我要提建议" href="javascript:void(0)">提建议</a></span>
                       <!--<a title="参与调查" href="http://vote.10jqka.com.cn/webvote/089bbf82354d38796389130318308dd1  "  target="_blank" class="btn_survey">参与调查</a>-->
                    </div>
                    <b class="oow_right fl"></b>
                </div>
                </div>
            </div>
            
						
			<dl class="f14">
                		    		<dt class="fl">截止
                    2024-11-18</dt>
                				<dd class="fl">
					<ul class="extra_list">
																		<li>
							<span class="fl">暂无基金、社保、信托、QFII等机构持仓
							<a class="c_text2 f12" href="javascript:void(0)" onclick="jumpToUrl('./position.html', 'position', 'position_holdDetail');return false;" taid="f10_click_zlkpmx#f10_click_zlkp">明细 ></a></span>
						</li>					
											</ul>
				</dd>
			</dl>
			<a   ifjump='1' newtaid="f10_zxdt_code-zlkp-se1-click-sta-gudong" style="bottom:0;right:0" class="newtaid shareholders skipto" href="###" name="holder.html#holdernum" taid="f10_click_zlgdyj#f10_click_zlkp">股东研究</a>
		</div>
		
		
	</div>
	<div class="ft"></div>
</div>
<!-- 电影票房 -->
<!-- content -->

<!-- 概念题材 -->
<!-- 概念题材 -->
<div class="m_box" id="material" stat="index_concept">
    <div class="hd">
        <h2> 题材要点</h2>
        <a ifjump='1' newtaid="f10_zxdt_code-tcyd-ne-clickmore-sta-ticai" class="newtaid fr more skipto" href="###" name="concept.html" taid="f10_gntc_gd" title="查看更多概念题材">更多&gt;&gt;</a>
    </div>
    <div class="bd pr clearfix">
        <div class="gntc" id="gntc-content">
            <!-- 数据将在这里动态加载 -->
        </div>
    </div>
    <div class="ft"></div>
</div>
<script type="text/javascript" src="//s.thsi.cn/js/chameleon/chameleon.min.1740313.js"></script> <script>
    $(document).ready(function(){
     // 发送请求获取题材要点数据
     var code = $('#stockCode').val();
     var market = $('#marketId').val();
     $.ajax({
        url: '/fuyao/f10_migrate/company/v1/theme_key_points?subject=' + market + '-' + code,
        method: 'GET',
        dataType:'json',
        success: function(val) {
            var status_code = val.status_code;
            if( status_code == 0 ){
                var data = val.data;
                var gntcContent = $('#gntc-content');
                data.forEach(function(item, index) {
                   var p = $('<p>');
                   if (index > 5) {
                      p.css('display', 'none'); 
                   }
                   p.append('<span href="concept.html" title="查看详细"><span class="gnmc f16">' + item.title + '</span></span><br>&nbsp;&nbsp;&nbsp;&nbsp;' + item.content);
                  gntcContent.append(p);
                });
                return;
            }
             // 清空目标容器的内容
             $('#material').empty();
             //屏蔽子导航栏中的财务要点tab
             $(".subnav [nav='material']").hide();
        },
        error: function(error) {
            console.log('Failed to fetch material data',error);
            // 清空目标容器的内容
            $('#material').empty();
            //屏蔽子导航栏中的财务要点tab
            $(".subnav [nav='material']").hide();
        }
    });
})
</script>


<!-- 龙虎榜 -->
<div class="m_box" id="payback" stat="index_payback">
	<div class="hd">
		<h2><span class="iwcclick"  taname='f10_stock_iwc_lhb' content='龙虎榜指每日两市中涨跌幅、换手率等由大到小的排名榜单。'  data-url='http://www.iwencai.com/yike/detail/auid/341edcdcaaee5c49?qs=client_f10_baike_7  '>龙虎榜</span></h2>	
	</div>
	<div class="bd">
		    		
    		<p class="tip"><a onclick="TA.log({'id':'f10_click_lhb','nj':1})" target="_blank" href="http://data.10jqka.com.cn/market/lhbgg/code/600519#refCountId=basic_50d2bfa6_569  " class="fr hla stat_jumptodata">查看历史龙虎榜&gt;&gt;</a><span class="f14">最近1年内该股未能登上龙虎榜。</span></p>
 
				</div>
	<div class="ft"></div>
</div>
<!-- 大宗交易 -->
<!-- 大宗交易 -->
<div class="m_box" id="deal" stat="index_deal">
	<div class="hd">
		<h2><span class="iwcclick" taname='f10_stock_iwc_dzjyck' content='大宗交易又称为大宗买卖，一般是指交易规模、数量和金额都非常大，远远超过市场的平均交易规模。'  data-url='http://www.iwencai.com/yike/detail/auid/6b3ef1de5effbe4f?qs=client_f10_baike_3  '  >大宗交易</span></h2>
	</div>
	<div class="bd">
				<p class="over">
			<a   ifjump='1' newtaid="f10_zxdt_code-dzjy-ne-clickmore-sta-lishi" onclick="TA.log({'id':'f10_click_dzjy','nj':1})" href="http://data.10jqka.com.cn/market/dzjy/op/code/code/600519/#refCountId=basic_50d2c0da_234  " target="_blank" class="newtaid fr hla stat_jumptodata">查看历史大宗交易信息&gt;&gt;</a>
		</p>
		<table class="m_table m_hl mt10">
			<thead>
				<tr>
					<th width="80px">交易日期</th>
					<th>成交价(元)</th>
					<th>成交金额(元)</th>
					<th>成交量(股)</th>
					<th newtaid="f10_zxdt_code-dzjy-r1c5-clickword-quota-yijialv" class='newtaid iwcclick' taname='f10_stock_iwc_dzjyck' data-url='http://www.iwencai.com/yike/detail/auid/e106225fd9e07626?qs=client_f10_baike_3  ' content='是指正股的价格还要上涨多少百分比才可让认购权证持有人达到盈亏平衡'>溢价率</th>
					<th width="220px">买入营业部</th>
					<th width="220px">卖出营业部</th>
				</tr>
			</thead>
			<tbody>
														<tr>
					<td class="tc">2025-02-18</td>
					<td>1475.00</td>
					<td>545.75万</td>
					<td>3700.00</td>
					<td>0.00%</td>
					<td class="tl">华泰证券股份有限公司深圳前海证券营业部</td>
					<td class="tl">华泰证券股份有限公司深圳前海证券营业部</td>
				</tr>
															<tr>
					<td class="tc">2025-02-18</td>
					<td>1475.00</td>
					<td>2950.00万</td>
					<td>2.00万</td>
					<td>0.00%</td>
					<td class="tl">中信证券股份有限公司上海分公司</td>
					<td class="tl">华林证券股份有限公司河南分公司</td>
				</tr>
															<tr>
					<td class="tc">2025-02-18</td>
					<td>1475.00</td>
					<td>5900.00万</td>
					<td>4.00万</td>
					<td>0.00%</td>
					<td class="tl">国泰君安证券股份有限公司总部</td>
					<td class="tl">华林证券股份有限公司河南分公司</td>
				</tr>
															<tr>
					<td class="tc">2025-02-10</td>
					<td>1431.51</td>
					<td>3650.35万</td>
					<td>2.55万</td>
					<td>0.00%</td>
					<td class="tl">华泰证券股份有限公司总部</td>
					<td class="tl">申万宏源证券有限公司国际部</td>
				</tr>
															<tr>
					<td class="tc">2025-01-24</td>
					<td>1436.00</td>
					<td>6462.00万</td>
					<td>4.50万</td>
					<td>0.00%</td>
					<td class="tl">海通证券股份有限公司上海黄浦区中山南路证券营业部</td>
					<td class="tl">中国国际金融股份有限公司上海分公司</td>
				</tr>
															<tr>
					<td class="tc">2025-01-22</td>
					<td>1441.00</td>
					<td>2795.54万</td>
					<td>1.94万</td>
					<td>0.00%</td>
					<td class="tl">国投证券股份有限公司总部</td>
					<td class="tl">华泰证券股份有限公司总部</td>
				</tr>
															<tr>
					<td class="tc">2025-01-21</td>
					<td>1468.15</td>
					<td>3.20亿</td>
					<td>21.78万</td>
					<td>0.00%</td>
					<td class="tl">中信证券股份有限公司总部(非营业场所)</td>
					<td class="tl">国泰君安证券股份有限公司总部</td>
				</tr>
															<tr>
					<td class="tc">2025-01-07</td>
					<td>1440.20</td>
					<td>1008.14万</td>
					<td>7000.00</td>
					<td>0.00%</td>
					<td class="tl">兴业证券股份有限公司厦门建业路证券营业部</td>
					<td class="tl">国泰君安证券股份有限公司厦门国际金融中心证券营业部</td>
				</tr>
															<tr>
					<td class="tc">2025-01-02</td>
					<td>1458.24</td>
					<td>4083.07万</td>
					<td>2.80万</td>
					<td>-2.00%</td>
					<td class="tl">申万宏源证券有限公司证券投资总部</td>
					<td class="tl">华林证券股份有限公司河南分公司</td>
				</tr>
															<tr>
					<td class="tc">2024-12-30</td>
					<td>1412.00</td>
					<td>1412.00万</td>
					<td>1.00万</td>
					<td>-7.41%</td>
					<td class="tl">华泰证券股份有限公司深圳深南大道证券营业部</td>
					<td class="tl">华泰证券股份有限公司深圳深南大道证券营业部</td>
				</tr>
																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																							</tbody>
		</table>
				
			</div>
	<div class="ft"></div>
</div>
<!-- 融资融券 -->
<div class="m_box" id="margin" stat="index_margin">
	<div class="hd">
		<h2><span class="iwcclick" taname='f10_stock_iwc_rzrq' content='融资融券是指投资者向具有深圳证券交易所会员资格的证券公司提供担保物，借入资金买入本所上市证券或借入本所上市证券并卖出的行为。'  data-url='http://www.iwencai.com/yike/detail/auid/59841202292da8c6?qs=client_f10_baike_8  ' >融资融券</span></h2>
	</div>
	<div class="bd">
				<p>
			<a   ifjump='1' newtaid="f10_zxdt_code-rzrq-ne-clickmore-sta-lishi" onclick="TA.log({'id':'f10_click_rzrq','nj':1})" href="http://data.10jqka.com.cn/market/rzrqgg/op/code/code/600519/#refCountId=basic_50d2c037_178  " target="_blank" class="newtaid fr hla stat_jumptodata">查看历史融资融券信息&gt;&gt;</a>
			<span class="tip f14">融资余额若长期增加时表示投资者心态偏向买方，市场人气旺盛属强势市场；反之则属弱势市场。</span>
		</p>
		<table class="m_table m_hl mt10">
			<thead>
				<tr>
					<th width="80px">交易日期</th>
					<th><span newtaid="f10_zxdt_code-rzrq-r1c2-clickword-quota-rzyue" class="newtaid iwcclick" taname='f10_stock_iwc_rzrq' content='融资余额指投资者每日融资买进与归还借款间的差额。融券余额增加，表示市场趋向卖方；反之则趋向买方。'  data-url='http://www.iwencai.com/yike/detail/auid/d27e9b9e5d23d33f?qs=client_f10_baike_8  '>融资余额</span>(元)</th>
					<th><span style="color:#FB0000">融资余额/流通市值</span></th>
					<th><span newtaid="f10_zxdt_code-rzrq-r1c4-clickword-quota-rzmairu" class="newtaid iwcclick" taname='f10_stock_iwc_rzrq' content='是指每天收盘后累计发生的向券商借钱买股票尚未偿还的资金，也就是融资买入后未偿还的金额。'  data-url='http://www.iwencai.com/yike/detail/auid/ca38b49b0d3cbfc2?qs=client_f10_baike_8  '>融资买入额</span>(元)</th>
					<th><span newtaid="f10_zxdt_code-rzrq-r1c5-clickword-quota-rzmaichu" class="newtaid iwcclick" taname='f10_stock_iwc_rzrq' content='融券卖出量，是指某一股票每天收盘后累计发生的向证券公司借的股票卖出总和。'  data-url='http://www.iwencai.com/yike/detail/auid/7bc03e9c150af4e2?qs=client_f10_baike_8  '>融券卖出量</span>(股)</th>
					<th><span newtaid="f10_zxdt_code-rzrq-r1c6-clickword-quota-rqyuliang" class="newtaid iwcclick" taname='f10_stock_iwc_rzrq' content='融券余量，是指每天收盘后累计发生的向券商借股票卖出尚未偿还的股票数量。'  data-url='http://www.iwencai.com/yike/detail/auid/9c4a232daafff976?qs=client_f10_baike_8  '>融券余量</span>(股)</th>
					<th><span newtaid="f10_zxdt_code-rzrq-r1c7-clickword-quota-rqyue" class="newtaid iwcclick" taname='f10_stock_iwc_rzrq' content='融券余额指投资者每日融券卖出与买进还券间的差额。'  data-url='http://www.iwencai.com/yike/detail/auid/06701d417a274584?qs=client_f10_baike_8  '>融券余额</span>(元)</th>
					<th><span newtaid="f10_zxdt_code-rzrq-r1c8-clickword-quota-rzrqyue" class="newtaid iwcclick" taname='f10_stock_iwc_rzrq' content='融资融券余额=融资余额+融券余额'  data-url='http://www.iwencai.com/yike/detail/auid/3354aeff90ee0a08?qs=client_f10_baike_8  '>融资融券余额</span>(元)</th>
				</tr>
			</thead>
			<tbody>
							<tr>
					<td class="tc">2025-02-21</td>
					<td>
														166.33亿											</td>
					
					<td>
													0.89%							
					</td>
					
					<td>
					  								4.62亿											</td>
					<td>6200.00</td>
					<td>7.84万</td>
					<td>1.17亿</td>
					<td>								167.50亿											</td>
				</tr>
							<tr>
					<td class="tc">2025-02-20</td>
					<td>
														167.41亿											</td>
					
					<td>
													0.90%							
					</td>
					
					<td>
					  								2.26亿											</td>
					<td>6400.00</td>
					<td>7.59万</td>
					<td>1.12亿</td>
					<td>								168.52亿											</td>
				</tr>
							<tr>
					<td class="tc">2025-02-19</td>
					<td>
														167.27亿											</td>
					
					<td>
													0.89%							
					</td>
					
					<td>
					  								3.07亿											</td>
					<td>9500.00</td>
					<td>7.86万</td>
					<td>1.17亿</td>
					<td>								168.44亿											</td>
				</tr>
							<tr>
					<td class="tc">2025-02-18</td>
					<td>
														168.09亿											</td>
					
					<td>
													0.91%							
					</td>
					
					<td>
					  								2.31亿											</td>
					<td>1.04万</td>
					<td>7.52万</td>
					<td>1.11亿</td>
					<td>								169.19亿											</td>
				</tr>
							<tr>
					<td class="tc">2025-02-17</td>
					<td>
														168.65亿											</td>
					
					<td>
													0.91%							
					</td>
					
					<td>
					  								3.07亿											</td>
					<td>4600.00</td>
					<td>6.96万</td>
					<td>1.02亿</td>
					<td>								169.67亿											</td>
				</tr>
							<tr>
					<td class="tc">2025-02-14</td>
					<td>
														168.52亿											</td>
					
					<td>
													0.91%							
					</td>
					
					<td>
					  								1.93亿											</td>
					<td>7700.00</td>
					<td>6.76万</td>
					<td>9964.36万</td>
					<td>								169.51亿											</td>
				</tr>
							<tr>
					<td class="tc">2025-02-13</td>
					<td>
														169.16亿											</td>
					
					<td>
													0.92%							
					</td>
					
					<td>
					  								4.13亿											</td>
					<td>6700.00</td>
					<td>6.45万</td>
					<td>9443.04万</td>
					<td>								170.10亿											</td>
				</tr>
							<tr>
					<td class="tc">2025-02-12</td>
					<td>
														169.79亿											</td>
					
					<td>
													0.94%							
					</td>
					
					<td>
					  								2.31亿											</td>
					<td>9800.00</td>
					<td>7.06万</td>
					<td>1.02亿</td>
					<td>								170.81亿											</td>
				</tr>
							<tr>
					<td class="tc">2025-02-11</td>
					<td>
														170.92亿											</td>
					
					<td>
													0.96%							
					</td>
					
					<td>
					  								2.50亿											</td>
					<td>2800.00</td>
					<td>6.30万</td>
					<td>8927.02万</td>
					<td>								171.82亿											</td>
				</tr>
							<tr>
					<td class="tc">2025-02-10</td>
					<td>
														170.69亿											</td>
					
					<td>
													0.95%							
					</td>
					
					<td>
					  								2.38亿											</td>
					<td>4100.00</td>
					<td>6.74万</td>
					<td>9641.94万</td>
					<td>								171.65亿											</td>
				</tr>
						</tbody>
		</table>
				  
	</div>
	<div class="ft"></div>
</div>
<div class="m_box" id="interactive" stat="index_interactive">
</div>
</div>
		<!-- F10 Content End -->
<!-- F10 Footer Start -->
		<div class="footer">
		<div class="links">
		    <a href="http://stockpage.10jqka.com.cn  " target="_blank" onclick="clickTalogStat('jptostock');">同花顺个股页面</a> |
		    <a href="http://moni.10jqka.com.cn/  " target="_blank">模拟炒股</a> |
		    <a href="http://www.10jqka.com.cn/school/  " target="_blank">股民学校</a> |
		    <a href="http://mobile.10jqka.com.cn/?req=pcf10  " target="_blank">手机炒股</a>|
		    <a target="_blank" href=" http://wpa.qq.com/msgrd?v=3&amp  ;uin=2270716061&amp;site=qq&amp;menu=yes">联系我们</a>
		</div>
		<p>免责声明:本信息由同花顺金融研究中心提供，仅供参考，同花顺金融研究中心力求但不保证数据的完全准确，如有错漏请以中国证监会指定上市公司信息披露媒体为准，同花顺金融研究中心不对因该资料全部或部分内容而引致的盈亏承担任何责任。用户个人对服务的使用承担风险。同花顺对此不作任何类型的担保。同花顺不担保服务一定能满足用户的要求，也不担保服务不会受中断，对服务的及时性，安全性，出错发生都不作担保。同花顺对在同花顺上得到的任何信息服务或交易进程不作担保。同花顺提供的包括同花顺理财的所有文章，数据，不构成任何的投资建议，用户查看或依据这些内容所进行的任何行为造成的风险和结果都自行负责，与同花顺无关。 </p>
		</div>
<!-- F10 Footer End -->
    </div>
    <div class="r-go-top" id="r-go-top"></div>
</body>
<input id="stockCode" type="hidden" value="600519">
<input id="F001" type="hidden" value="A">
<input id="F002" type="hidden" value="上海证券交易所">
<input id="F004" type="hidden" value="1">
<input id="stockName" type="hidden" value="贵州茅台">
<input id="marketId" type="hidden" value="17">
<input id="cateName"  type="hidden" value="最新动态">
<input id="catecode"  type="hidden" value="index">
<input id="fcatecode"  type="hidden" value="index">
<input id="sid" type="hidden" value="F10new_zxdt">
<input id="fid" type="hidden" value="F10,F10master,F10main,F10new">
<input id="qid" type="hidden" value="14585021">
<div style="display:none" id="wordraddom">
</div>
<!-- title 容器 -->
	<div class="titleBox" id="altlayer" style="display:none; width: 260px;">
		<div class="tipbox_bd p0_5"><span style="line-height:24px;" class="tip f14">
		<span id="altlayer_content"></span>
		</div>
	</div>
	<!--pc-New头部入口跳转-->
		
	
<script type="text/javascript"  crossorigin src="//s.thsi.cn/cb?js/common/cefapi/1.5.5/cefApi.min.js;js/basic/stock/newPcJs/common.js"></script>
<script>
try {
    external.createObject('Util');
    //document.getElementById("updownchange").style.display = "block";
} catch (e) {}
</script>

<script type="text/javascript"  crossorigin src="//s.thsi.cn/cb?js/jquery-1.8.3.min.js;js/basic/common/inheritance.js;js/basic/stock/popWin.js;js/basic/common/Model_v2.js;js/basic/stock/20200604153837/po_v2.js"></script>
    					<script type="text/javascript" charset="utf-8" src="//s.thsi.cn/js/basic/stockph_v2/stock/companyCompare-cbe62d.js"></script>
		        
    <script type="text/javascript" crossorigin charset="utf-8" src="//s.thsi.cn/js/basic/stockph_v2/stock/common_v3-2-a58b25.js"></script>
<!--[if IE ]><script type="text/javascript"  crossorigin src="//s.thsi.cn/js/excanvas.min.js"></script><![endif]-->
<!--[if IE 6]>
<script type="text/javascript"  crossorigin src="//s.thsi.cn/js/basic/DD_belatedPNG_0.0.8a-min.js"></script>
<script type="text/javascript">DD_belatedPNG.fix('.company_logo,.btn-gnjx-left, .btn-gnjx-right, .concept_hot_icons,.iwc_searchbar,.iwc_searchbar .tips-icon,#autocomplete_search_input dd .icona,#autocomplete_search_input dd .iconb');</script>
<![endif]-->
<script type="text/javascript" crossorigin src="//s.thsi.cn/js/basic/jquery.pos-fixed.js"></script>
<script type="text/javascript" crossorigin src="//s.thsi.cn/cb?js/ta.min.js;js/pa.min.js;js/basic/stock/StatLoad.js;js/basic/stock/20200605135220/xgcl_mod.js"></script>
<script type="text/javascript" crossorigin src="//s.thsi.cn/js/basic/stock/tongji.min.js"></script>
<!-- 统计 -->
<script type="text/javascript"  crossorigin src="//s.thsi.cn/cb?js/home/ths_core.min.js;js/home/ths_quote.min.js;js/home/tongji.min.js" charset="utf-8"></script>
<script type="text/javascript"  crossorigin src="//s.thsi.cn/js/basic/stock/commonLog_v3.js" charset="utf-8"></script>
<!--F10运营位脚本-->
<script type="text/javascript" crossorigin charset="utf-8" src="//s.thsi.cn/js/basic/stockph_v2/stock/append-8f7ed2.js"></script>
<script type="text/javascript" crossorigin charset="utf-8" src="//s.thsi.cn/js/basic/stockph_v2/stock/f10widget_v3-5ec039.js"></script>
<script type="text/javascript"  crossorigin src="//s.thsi.cn/cb?/js/navigation/jquery.bgiframe.min.js;/js/navigation/jquery.ui.position.min.js"></script>
<script type="text/javascript"  crossorigin src="//s.thsi.cn/cb?js/basic/stock/json2.js"></script>
<script type="text/javascript" charset="utf-8" src="//s.thsi.cn/js/basic/stockph_v2/stock/search_v3-fec53c.js"></script>
<script>
//---------------自定义title弹窗 start----------
var tempalt='';//临时存储title
document.body.onmousemove = function(event) {
	if(altlayer.style.display==''){
		$("#altlayer").css({
			left: event.pageX,
			top: event.pageY+10
		});
	}
};
document.body.onmouseover = function(event) {
	if (event.srcElement.className !== 'titleIcon') {
		return false;
	}
	if(event.srcElement.title && (event.srcElement.title!='' || (event.srcElement.title=='' && tempalt!=''))){
		$("#altlayer").css({
			left: event.pageX,
			top: event.pageY+20,
			display: ''
		});
		$("#altlayer_content").html(event.srcElement.title)
		tempalt = event.srcElement.title;
		event.srcElement.title='';
	}
};
document.body.onmouseout = function(event) {
	if (tempalt != '') {
		event.srcElement.title = tempalt;	
	}
	tempalt = '';
	$("#altlayer").hide();
};
//---------------自定义title弹窗 end----------
//F10客户端结点设置
function setClientCacheData() {
    try { //判断是否在客户端，存入结点
        external.createObject('Util');
        window.API.use({
            method: 'Util.getHxVer',
            success: function(data) {
                var hexinVer = getVersionStr(data);
                var pathname = window.location.pathname
                pathname = pathname.replace(/\/\d{2,3}\//, '/%4/');
                pathname = pathname.replace(/\/[\d|A]\d{5}\//, '/%6/');
                //dalert(hexinVer);//上线注意，测试版版本86080  正式版 86090 请注意
                if (hexinVer >= 86080) {
                    window.API.use({
                        method: 'Info.cacheInfoData',
                        data: ['f10_client_node', 'http://basic.10jqka.com.cn  ' + pathname],
                        success: function() {
                        }
                    })
                } else {
                    window.API.use({
                        method: 'Info.cacheInfoData',
                        data: ['', 'http://basic.10jqka.com.cn/  ' + pathname],
                        success: function() {
                        }
                    })
                }
            }
        })
        document.getElementById("updownchange").style.display = "block";
    } catch (e) {
        //远航版隐藏header
        $("#quotedata").empty()
        $("#updownchange").empty()
    }
	try {
		callNativeHandler('getMacVer', {}, function(data){
			if (data.plaform == 'mac' && data.version>='1.3.0' ) {
				var pathname = window.location.pathname
                pathname = pathname.replace(/\/\d{2,3}\//, '/%4/');
                pathname = pathname.replace(/\/[\d|A]\d{5}\//, '/%6/');
				callNativeHandler('cache_info_Data', {
					'f10_client_node': 'http://basic.10jqka.com.cn  ' + pathname
				}, function(data){
				});
			}
		});
	} catch (e) {
    }
}
setClientCacheData();
</script>
<script type="text/javascript" crossorigin src="//s.thsi.cn/cb?js/basic/stock/highcharts_v2.js;js/basic/stock/tableSorter_v2-2.js"></script>
<script type="text/javascript" crossorigin src="//s.thsi.cn/cb?js/basic/stock/jquery.flot.min.js;js/basic/stock/jquery.flot.tooltip.min.js;js/basic/stock/jskzd.20151120.js"></script>
<script type="text/javascript" crossorigin src="//s.thsi.cn/cb?js/basic/stock/iwencai_v2-4.201605172.js;js/basic/stock/userevaluate_v3.20170811.js;js/basic/stock/pubtime_v2.js"></script>
<script type="text/javascript" crossorigin src="//s.thsi.cn/cb?js/basic/stock/modpager.js;js/basic/stock/jquery.pager.js"></script>
<script type="text/javascript" crossorigin src="//s.thsi.cn/js/clientinfo/nav/jquery.tinyscrollbar.js"></script>
<script type="text/javascript" crossorigin src="//s.thsi.cn/cb?js/basic/stock/202005271800/20200604153837/remind_v6.js"></script>
<script type="text/javascript" crossorigin charset="utf-8" src="//s.thsi.cn/js/basic/stockph_v2/stock/gdr-ce9bc8.js"></script>
<script type="text/javascript" charset="utf-8" crossorigin src="//s.thsi.cn/cb?js/basic/stockph_v2/stock/index-afb812.js"></script>
<script type="text/javascript" charset="utf-8" crossorigin src="//s.thsi.cn/cb?js/basic/stockph_v2/stock/dongtai-dd4014.js"></script>
<script type="text/javascript"  crossorigin src="//s.thsi.cn/js/basic/stock/20200604153837/stockpage_v3.20190506.js"></script>
<script type="text/javascript" charset="utf-8" crossorigin id="ths-parser-script" app_key="f10_astockpc_config" src="//s.thsi.cn/cd/website-thsc-f10-utils/1.4.39/parser.js"></script>
<script type="text/javascript">
var sid = 'F10new_zxdt';
var fid = 'F10,F10master,F10main,F10new';
var stockcode = '600519';
if (window.location.href.indexOf('interactive')<0
		&&window.location.href.indexOf('dupont')<0) {
    $(".subnav .skipto").each(function(){
    	sonnav = $(this).attr('nav');
    	if ($("#"+sonnav).length>0 || $("[idchange='"+sonnav+"']").length>0) {
    	} else {
    		$(this).hide();
    	}
    })
}
if (stockcode.substr(0,2) == '43' || stockcode.substr(0,2) == '83') {
	var third_sid = sid+'_thirdboard';
	var third_fid = fid+'_thirdboard';
    PA.setStartTime(loadTimer);
	PA.init({'id':third_sid, 'fid':third_fid, 'stockcode':stockcode});
}
$(document).ready(function(){
var hash = window.location.hash.substring(1);
var hashArray = hash.split('-');
if(hashArray[0]  == 'position'){
	$("#sortNav li").eq(hashArray[1]).find("a").click();
}
});
try {
    external.createObject('Util');
    window.API.use({
        method: 'Passport.get',
        data: ['m_qs'],
        success: function(data) {
            var qsid = data
            if (qsid == '114') {
                $("#f10_top_ad").hide();
                $("#r-go-top").hide();
            }
        }
    })
} catch (e) {}
if (!isIE6()) {
    $(".m_box").statload();
}
try {
    external.createObject('Util');
    window.API.use({
        method: 'Passport.get',
        data: 'm_qs',
        success: function(data) {
            var qsid = data
            if (qsid < 800) {
                fid += ',f10new_qs';
            } else {
                fid += ',f10new_fqs';
            }
            PA.setStartTime(loadTimer);
            setTimeout(function(){
                PA.init({
                    'id': sid,
                    'fid': fid,
                    'stockcode': stockcode,
                    'qsid': qsid
                });
            },100)
        }
    })
} catch (e) {
    PA.setStartTime(loadTimer);
    PA.init({
        'id': sid,
        'fid': fid,
        'stockcode': stockcode
    });
}
try {
    external.createObject('Util');
    var externalSessionId = window.API.createSessionId('external');
    window.API.use({
        method: 'external.registerEvent',
        data: 'onshow',
        sessionId: externalSessionId,
        persistent: true,
        callbackName: 'onshow',
        success: function(data) {
            if (!data) {
                 PA.setStartTime(loadTimer);
                 PA.init({
                    'id': sid,
                    'fid': fid,
                    'stockcode': stockcode,
                    'hide': 1,
                    'nj': 1,
                    _sid: "__ths_onshow"
                });
            }
        }
    })
} catch (e) {}
</script>
<div style="display:none"><script type="text/javascript">
// 在客户端内，html做个区分，解决端内的一些特性
if ((window.external && 'createObject' in window.external) || (window.HevoCef && "IsHevoCef" in window.HevoCef)) {
	$('html').addClass("inClient");
}
(function(){
var _bdhmProtocol = (("https:" == document.location.protocol) ? " https://" : " http://");
document.write(unescape("%3Cscript src='" + _bdhmProtocol + "hm.baidu.com/h.js%3F78c58f01938e4d85eaf619eae71b4ed1' type='text/javascript'%3E%3C/script%3E"));
});
</script>

<script type="text/javascript" charset="utf-8" crossorigin src="//s.thsi.cn/js/basic/stockph_v2/stock/f10hq_v4-5-576b58.js"></script>
<script type="text/javascript" charset="utf-8" crossorigin src="//s.thsi.cn/js/basic/stockph_v2/stock/onlyBAuthorize-4a6ce9.js"></script>
</div>
<!--F10头部广告-->

<!--<div id="f10_top_ad" class="f10_ad_top none"><script type="text/javascript">CNZZ_SLOT_RENDER('297563');</script></div>-->
</html>

"""



def stock_extract_concept_ranking(html_content):
    """
    从HTML内容中提取概念贴合度排名信息。

    参数:
        html_content (str): HTML文档内容。

    返回:
        list: 包含概念名称和链接的列表，格式为 [(概念名称, 链接), ...]。
              如果未找到相关信息，返回空列表。
    """
    # 使用BeautifulSoup解析HTML
    soup = BeautifulSoup(html_content, 'html.parser')

    # 查找包含“概念贴合度排名”的标签
    concept_ranking = soup.find('div', class_='newconcept')

    # 提取所有概念
    concepts = []
    if concept_ranking:
        for a_tag in concept_ranking.find_all('a'):
            concept_name = a_tag.text.strip()
            # concept_link = a_tag.get('href', '')
            # concepts.append((concept_name, concept_link))
            concepts.append((concept_name))
        if concepts:
            concepts.pop()  # 移除最后一个元素
    return concepts

# 示例用法
if __name__ == "__main__":

    # 调用函数提取概念贴合度排名
    concepts = stock_extract_concept_ranking(html_content)

    # 输出结果
    if concepts:
        print("概念贴合度排名：")

        for concept in concepts:
            print(f"{concept}")
    else:
        print("未找到概念贴合度排名信息")