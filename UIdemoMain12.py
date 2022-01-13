# GUIdemo11.py
# Demo11 of GUI by PyQt5
# Copyright 2021 youcans, XUPT
# Crated：2021-10-20

import sys, math, sip
import numpy as np      # 导入 numpy 并简写成 np
import cv2
from scipy import signal, fftpack     # 导入 scipy 工具包 signal, fftpack 模块

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
matplotlib.use("Qt5Agg")  # 声明使用 QT5

from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import QApplication, QMainWindow, QMessageBox, QFileDialog
from uiDemo12 import Ui_MainWindow  # 导入 uiDemo9.py 中的 Ui_MainWindow 界面类


PI = math.pi

class MyFigure(FigureCanvas):  # 窗口部件，继承FigureCanvas基类
    def __init__(self, width=5, height=4, dpi=100):
        # 1. 配置中文显示
        plt.rcParams["font.family"] = ["SimHei"]  # 用来正常显示中文标签
        plt.rcParams["axes.unicode_minus"] = False  # 用来正常显示负号
        # 2.创建一个 Figure
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        # 3. 在父类中激活Figure窗口
        super(MyFigure, self).__init__(self.fig)  # 此句必不可少，否则不能显示图形


class MyMainWindow(QMainWindow, Ui_MainWindow):  # 继承 QMainWindow 类和 Ui_MainWindow 界面类
    def __init__(self, parent=None):
        super(MyMainWindow, self).__init__(parent)  # 初始化父类
        self.setupUi(self)  # 继承 Ui_MainWindow 界面类

        # 堆叠布局控制：所有的 pushButton 点击动作都连接到堆叠布局控制函数 stackController()
        listBtn = [self.pushButton_1, self.pushButton_2, self.pushButton_3,
                   self.pushButton_4, self.pushButton_5, self.pushButton_6]
        for snBtn in listBtn:
            snBtn.clicked.connect(self.stackController)  # 连接到堆叠布局控制函数 stackController()

    def stackController(self):  # 堆叠页面控制函数
        sender = self.sender().objectName()  # 获取当前信号 sender
        index = {                    # 定义每个按键对应的堆叠页面
                "pushButton_1": 0,  # page_0
                "pushButton_2": 0,  # page_0
                "pushButton_3": 0,  # page_0
                "pushButton_4": 0,  # page_0
                "pushButton_5": 1,  # page_1
                "pushButton_6": 1}  # page_1
        self.stackedWidget.setCurrentIndex(index[sender])  # 根据信号 index 设置所显示的页面


    def click_pushButton_1(self):  # 点击 pushButton_01 触发
        self.lineEdit.setText("1. 阈值抠图")
        self.plainTextEdit.appendPlainText("开始阈值抠图")

        listLabel = [self.label_1, self.label_2, self.label_3, self.label_4, self.label_5, self.label_6]
        for snLabel in listLabel:
            snLabel.setPixmap(QtGui.QPixmap(""))  # 删除图片 snLabel

        try:
            hImg, wImg, cImg = self.img.shape  # 获取图片 Img 的 height, width, channel
            self.plainTextEdit.appendPlainText("图片原始尺寸为：{},{},{}".format(hImg,wImg,cImg))
            imgOri = self.img.copy()
            # imgGreen = imgOri[:, :, 1]  # imgGreen 为 绿色通道的 色彩强度图 (注意不是原图的灰度转换结果)

            QImg1 = cv2.cvtColor(imgOri, cv2.COLOR_BGR2RGB)  # 图片格式转换：BGR(OpenCV) -> RGB(PyQt5)
            QImg1 = QtGui.QImage(QImg1.data, wImg, hImg, wImg*cImg,  # 创建并读入 QImage 图像
                     QtGui.QImage.Format_RGB888)  # QImage.Format_RGB888：图像采用 24位RGB(8/8/8)存储格式
            self.label_1.setPixmap(QtGui.QPixmap.fromImage(QImg1))  # 将QImage 显示在 label_1 控件中
            self.label_1.setScaledContents(True)  # 图片自适应 QLabel 区域大小

            QImg2 = cv2.cvtColor(imgOri, cv2.COLOR_BGR2GRAY)  # 图片格式转换：BGR(OpenCV) -> RGB(PyQt5)
            QImg2 = QtGui.QImage(QImg2.data, wImg, hImg,  # 创建并读入 QImage 图像
                     QtGui.QImage.Format_Indexed8)  # QImage.Format_RGB888：图像采用 24位RGB(8/8/8)存储格式
            self.label_2.setPixmap(QtGui.QPixmap.fromImage(QImg2))  # 将QImage 显示在 label_1 控件中
            self.label_2.setScaledContents(True)  # 图片自适应 QLabel 区域大小

        except Exception as e:
            self.lineEdit.setText(str(e))
            QMessageBox.about(self, "About",
                              """抠图图像没有读入。""")
            return
        return

    def click_pushButton_2(self):  # 点击 pushButton_02 触发
        self.lineEdit.setText("1. 几何变换")
        self.plainTextEdit.appendPlainText("1.2 图像缩放")
        self.label_1.setPixmap(QtGui.QPixmap("../image/fractal02.png"))
        return

    def click_pushButton_3(self):  # 点击 pushButton_03 触发
        self.lineEdit.setText("1. 几何变换")
        self.plainTextEdit.appendPlainText("1.3 图像转置")
        # self.stackedWidget.setCurrentIndex(0)  # 打开 stackedWidget > page_0

        # (1) 生成信号: 周期性方波 (square-wave waveform)
        t = np.linspace(0, 1, 500, endpoint=False)
        sig = np.sin(2 * PI * t)  # 控制方波的占空比 duty cycle 随时间变化
        pwm = signal.square(2 * PI * 30 * t, duty=(sig + 1) / 2)
        # (2) 绘图
        Canvas = MyFigure(3, 3, 100)  # width=3, height=3, dpi=100
        ax1 = Canvas.fig.add_subplot(211)  # 上下 2 行的第 1 行
        ax1.plot(t, sig)  # 绘制占空比波形图
        ax2 = Canvas.fig.add_subplot(212)  # 上下 2 行的第 2 行
        ax2.plot(t, pwm)  # 绘制 PWM 波形图
        Canvas.fig.suptitle("Square-wave waveform (PWM)")
        # (3) 向 label_1 加载新图片
        self.label_1.setPixmap(QtGui.QPixmap(""))  # 删除原有 label_1 显示图片
        QtWidgets.QVBoxLayout(self.label_1).addWidget(Canvas)
        return

    def click_pushButton_4(self):  # 点击 pushButton_04 触发
        self.lineEdit.setText("1. 几何变换")
        self.plainTextEdit.appendPlainText("1.4 图像旋转")
        # self.label_1.clear()  # 清空内容
        # self.label_1.setPixmap(QtGui.QPixmap(""))  # 删除原有 label_1 显示图片
        Img = cv2.imread("../image/fractal02.png")  # Opencv 读入图片
        hImg, wImg, bImg = Img.shape  # 获取图片 Img 的 height, width, bytesPerComponent
        QImg = cv2.cvtColor(Img, cv2.COLOR_BGR2RGB)  # 图片格式转换：BGR(OpenCV) -> RGB(PyQt5)
        # QImg = QImage(img.data, width, height, bytesPerLine, QImage.Format_RGB888)
        QImg = QtGui.QImage(QImg.data, wImg, hImg, wImg*bImg,  # 创建并读入 QImage 图像
                     QtGui.QImage.Format_RGB888)  # QImage.Format_RGB888：图像采用 24位RGB(8/8/8)存储格式
        self.label_1.setPixmap(QtGui.QPixmap.fromImage(QImg))  # 将QImage 显示在 label_1 控件中
        return

    def click_pushButton_5(self):  # 点击 pushButton_05 触发
        self.lineEdit.setText("2. 图像增强")
        self.plainTextEdit.appendPlainText("2.1 灰度变换")
        self.label_2.setPixmap(QtGui.QPixmap("../image/fractal01.png"))
        self.label_2.setScaledContents(True)  # 图片自适应 QLabel 区域大小
        return

    def click_pushButton_6(self):  # 点击 pushButton_06 触发
        self.plainTextEdit.appendPlainText("2.2 直方图修正")
        self.label_3.setPixmap(QtGui.QPixmap("../image/fractal02.png"))
        return

    def trigger_actHelp(self):  # 动作 actHelp 触发
        QMessageBox.about(self, "About",
                          """数字图像处理工具箱 v1.0\nCopyright YouCans, XUPT 2021""")
        return

    def trigger_actOpen(self):  # 动作 actOpen 触发
        self.lineEdit.setText("<font color=\"#0000FF\">\n 导入图像...</font>")  # 显示蓝色
        print("导入图像...")
        self.fileRead, ok = QFileDialog.getOpenFileName(self, "打开", "../images/", "All Files (*);;Text Files (*.txt)")
        self.img = cv2.imread(self.fileRead,flags=1)  # Opencv 读入图片

        # 在状态栏显示文件地址
        self.plainTextEdit.appendPlainText(self.fileRead)
        return

    def trigger_actSave(self):  # 导出图像
        self.lineEdit.setText("<font color=\"#0000FF\">\n 保存图像...</font>")  # 显示蓝色
        print("保存图像...")
        file, ok = QFileDialog.getOpenFileName(self, "打开", "../images/", "All Files (*);;Text Files (*.txt)")
        # 在状态栏显示文件地址
        self.plainTextEdit.appendPlainText(file)

        # self.statusbar.showMessage(file)
        return

if __name__ == '__main__':
    app = QApplication(sys.argv)  # 在 QApplication 方法中使用，创建应用程序对象
    myWin = MyMainWindow()  # 实例化 MyMainWindow 类，创建主窗口
    myWin.show()  # 在桌面显示控件 myWin
    sys.exit(app.exec_())  # 结束进程，退出程序
