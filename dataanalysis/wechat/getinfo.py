# 使用requests-html模拟手机访问
from requests_html import HTMLSession

session = HTMLSession()
url = "https://mp.weixin.qq.com/mp/profile_ext?action=home&__biz=[公众号biz参数]#wechat_redirect"
r = session.get(url, headers={"User-Agent": "Mozilla/5.0"})
r.html.render(sleep=3)  # 需要安装chromium

articles = r.html.find('.weui_media_box')[:5]
for item in articles:
    title = item.find('h4', first=True).text
    print(title)


exit()

import feedparser

# 示例：订阅「华尔街见闻」公众号
rss_url = "https://rsshub.app/wechat/wallstreetcn"
feed = feedparser.parse(rss_url)

for entry in feed.entries[:5]:  # 打印最新5篇文章
    print(f"标题：{entry.title}")
    print(f"发布时间：{entry.published}")
    print(f"摘要：{entry.summary[:100]}...")
    print(f"原文链接：{entry.link}\n")