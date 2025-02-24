from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
import time

# 设置浏览器驱动路径（例如ChromeDriver）
driver_path = "path/to/chromedriver"  # 替换为你的ChromeDriver路径
driver = webdriver.Chrome(driver_path)

try:
    # 打开同花顺网页版
    driver.get("https://www.10jqka.com.cn")

    # 等待页面加载
    time.sleep(5)

    # 点击登录按钮
    login_button = driver.find_element(By.CLASS_NAME, "login-btn")
    login_button.click()

    # 输入用户名和密码（替换为你的同花顺账号）
    username_input = driver.find_element(By.NAME, "username")
    password_input = driver.find_element(By.NAME, "password")
    username_input.send_keys("your_username")  # 替换为你的用户名
    password_input.send_keys("your_password")  # 替换为你的密码

    # 点击登录
    submit_button = driver.find_element(By.CLASS_NAME, "submit-btn")
    submit_button.click()

    # 等待登录完成
    time.sleep(5)

    # 搜索股票代码（例如贵州茅台：600519）
    search_input = driver.find_element(By.CLASS_NAME, "search-input")
    search_input.send_keys("600519")
    search_input.send_keys(Keys.RETURN)

    # 等待搜索结果加载
    time.sleep(5)

    # 点击“加自选”按钮
    add_button = driver.find_element(By.CLASS_NAME, "add-to-favorites")
    add_button.click()

    # 等待操作完成
    time.sleep(3)

    print("股票已成功添加到自选股！")

finally:
    # 关闭浏览器
    driver.quit()