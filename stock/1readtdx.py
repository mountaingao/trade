import time
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
import smtplib
from email.mime.text import MIMEText

# 邮件配置
EMAIL_HOST = 'smtp.example.com'  # SMTP服务器
EMAIL_PORT = 587  # 端口
EMAIL_USER = 'your_email@example.com'  # 发件邮箱
EMAIL_PASSWORD = 'your_password'  # 邮箱密码
TO_EMAIL = 'recipient@example.com'  # 收件邮箱

# 监控的目录
WATCH_DIR = 'D:\BaiduSyncdisk\个人\通达信\ALERT'

# 文件更改处理类
class FileChangeHandler(FileSystemEventHandler):
    def on_modified(self, event):
        if not event.is_directory:
            print(f'文件已更改: {event.src_path}')
            # send_notification(f'文件已更改: {event.src_path}')
            change_ths(event.src_path)
#更改同花顺数据
def change_ths(message):
    change_ths_data()


# 调用函数
show_alert_with_sound()

# 发送邮件通知
def send_notification(message):
    msg = MIMEText(message, 'plain', 'utf-8')
    msg['From'] = EMAIL_USER
    msg['To'] = TO_EMAIL
    msg['Subject'] = '文件更改通知'

    try:
        server = smtplib.SMTP(EMAIL_HOST, EMAIL_PORT)
        server.starttls()
        server.login(EMAIL_USER, EMAIL_PASSWORD)
        server.sendmail(EMAIL_USER, [TO_EMAIL], msg.as_string())
        server.quit()
        print('邮件通知已发送')
    except Exception as e:
        print(f'邮件发送失败: {e}')

# 启动监控
if __name__ == "__main__":
    event_handler = FileChangeHandler()
    observer = Observer()
    observer.schedule(event_handler, path=WATCH_DIR, recursive=True)
    observer.start()

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()
    observer.join()


