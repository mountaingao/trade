import pyautogui
import time

pyautogui.PAUSE = 1     #执行动作后暂停的秒数
#不设置则 执行动作后默认延迟时间是0.1秒
#只能在执行一些pyautogui动作后才能使用，建议用time.sleep
#最好在代码开始时就设置，否则无法使用ctrl c打断

#如果函数运行期间想要停止，请把鼠标移动到屏幕得左上角（0，0）位置，
#这触发pyautogui.FaailSafeException异常，从而终止程序运行。
pyautogui.FAILSAFE = True     #默认True则鼠标(0,0)可触发异常；False不触发

#坐标+RGB实时显示
# pyautogui.displayMousePosition()    #官方自带的函数

#sleep，可用time.sleep()代替
pyautogui.sleep(1)

# #获取中心点
# pyautogui.center(coords)#coords is a 4-integer tuple of (left, top, width, height)
# pyautogui.center((o,o,100,100))
#
# #倒计时
# pyautogui.countdown(seconds)
# >>> ag.countdown(5)
# 5 4 3 2 1
#
# #直接执行 pyautogui 在后台实际调用的命令
# pyautogui.run(commandStr, _ssCount=None)
# >>> ag.run('ccg-20,+0c')    这个命令表示
# 单击鼠标两次，然后鼠标光标向左移动20个像素，然后再次单击。
# 可忽略命令和参数之间的空白。命令字符必须是小写。引号必须是单引号。
# >>> ag.run('c c g -20,+0 c')
'''
`c` => `click(button=PRIMARY)`
`l` => `click(button=LEFT)`
`m` => `click(button=MIDDLE)`
`r` => `click(button=RIGHT)`
`su` => `scroll(1) # scroll up`
`sd` => `scroll(-1) # scroll down`
`ss` => `screenshot('screenshot1.png') # filename number increases on its own`
`gX,Y` => `moveTo(X, Y)`
`g+X,-Y` => `move(X, Y) # The + or - prefix is the difference between move() and moveTo()`
`dX,Y` => `dragTo(X, Y)`
`d+X,-Y` => `drag(X, Y) # The + or - prefix is the difference between drag() and dragTo()`
`k'key'` => `press('key')`
`w'text'` => `write('text')`
`h'key,key,key'` => `hotkey(*'key,key,key'.replace(' ', '').split(','))`
`a'hello'` => `alert('hello')`
`sN` => `sleep(N) # N can be an int or float`
`pN` => `PAUSE = N # N can be an int or float`
`fN(commands)` => for i in range(N): run(commands)
'''


x,y = pyautogui.position()    #当前鼠标的位置
print(x,y)
x,y = pyautogui.size()        #当前屏幕分辨率
print(x,y)
#判断坐标是否在屏幕内
pyautogui.onScreen(-1,-1)    #False
pyautogui.onScreen(1,1)        #True

#点击
pyautogui.click(x=None,y=None,clicks=1,interval=0.0,duration=0.0,button='primary')
#在xy位置单击，interval 单击之间等待的秒数,duration是从当前鼠标位置 移动到xy位置所用的时间
pyautogui.leftClick(x=None, y=None, interval=0.0, duration=0.0)
pyautogui.rightClick(x=None, y=None, interval=0.0, duration=0.0)
pyautogui.middleClick(x=None, y=None, interval=0.0, duration=0.0)    #中键单击
pyautogui.doubleClick(x=None, y=None, interval=0.0, button='left', duration=0.0,)
pyautogui.tripleClick(x=None, y=None, interval=0.0, button='left', duration=0.0,)

#拖动1
pyautogui.mouseDown(x=None, y=None, button='primary', duration=0.0,)    #移动到xy按下
pyautogui.mouseUp(x=None, y=None, button='primary', duration=0.0,)    #移动到xy松开

#拖动2
pyautogui.dragTo(x=None, y=None, duration=0.0, button='primary',  mouseDownUp=True)
pyautogui.dragRel(xOffset=0, yOffset=0, duration=0.0, button='primary', mouseDownUp=True)
#如果最后的参数mouseDownUp设置为False则鼠标只是单纯的移动，不执行按下或者松开操作!
#另外如果duration设置为0或者不设置，拖动也不会成功,只是单纯的移动!
num_seconds = 1
#绝对移动，左上角坐标是（0,0）
#若duration小于pyautogui.MINIMUM_DURATION=0.1（默认）,则移动是即时的
pyautogui.moveTo(x,y,duration=num_seconds)

#相对于当前位置移动，+x向右，+y向下
pyautogui.moveRel(0,0,duration=num_seconds)
pyautogui.move(x,y)

#滑轮滚动
pyautogui.scroll(-10) #向下10，且使用当前鼠标位置
pyautogui.scrolll(10,x=10,y=100) # 将鼠标移到x,y, 向上滚动10格

#进阶：用 缓动/渐变函数 控制光标移动的速度和方向
# PyAutoGUI有30种缓动/渐变函数，可以通过pyautogui.ease*?查看
# 开始很慢，不断加速
pyautogui.moveTo(100, 100, 2, pyautogui.easeInQuad)
# 开始很快，不断减速            pyautogui.easeOutQuad
# 开始和结束都快，中间比较慢    pyautogui.easeInOutQuad
# 一步一徘徊前进                pyautogui.easeInBounce
# 徘徊幅度更大，甚至超过起点和终点    pyautogui.easeInElastic
