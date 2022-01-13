# GUI_Matting_v4.py
# Matting GUI by PyQt5
# Copyright 2021 youcans, XUPT
# Crated：2021-12-10
# 版本说明：
# v1: 基于 PyQt5 建立 GUI 框架
# v2: (1) 读取图片功能
#     (2) 实现阈值抠图功能
# v3: (1) GUI 图像显示函数 imgShowLabel
#     (2) 选择图片放大
#     (3) 实现自适应阈值抠图功能
# v4: (1) 实现HSV颜色范围抠图功能


import sys, math, sip
import numpy as np      # 导入 numpy 并简写成 np
import cv2

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
matplotlib.use("Qt5Agg")  # 声明使用 QT5

from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import QApplication, QMainWindow, QMessageBox, QFileDialog
from uiMatting2 import Ui_MainWindow  # 导入 uiMatting1.py 中的 Ui_MainWindow 界面类

def imgShowLabel(img, label):  # 显示图像

    if img.ndim==3:  # 彩色图像
        hImg, wImg, cImg = img.shape  # 获取图片 Img 的 height, width, bytesPerComponent
        QImg = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # 图片格式转换：BGR(OpenCV) -> RGB(PyQt5)
        QImg = QtGui.QImage(QImg.data, wImg, hImg, wImg * cImg,  # 创建并读入 QImage 图像
                QtGui.QImage.Format_RGB888)  # QImage.Format_RGB888：图像采用 24位RGB(8/8/8)存储格式
        label.setPixmap(QtGui.QPixmap.fromImage(QImg))  # 将QImage 显示在 label_1 控件中
        label.setScaledContents(True)  # 图片自适应 QLabel 区域大小
    elif img.ndim==2:  # 灰度图像
        hImg, wImg = img.shape  # 获取图片 Img 的 height, width, bytesPerComponent
        QImg = QtGui.QImage(img.data, wImg, hImg,  # 创建并读入 QImage 图像
                QtGui.QImage.Format_Indexed8)  # QImage.Format_Indexed8：图像采用 8位Gray存储格式
        label.setPixmap(QtGui.QPixmap.fromImage(QImg))  # 将QImage 显示在 label_1 控件中
        label.setScaledContents(True)  # 图片自适应 QLabel 区域大小
    return


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

        ## --- GUI 定义动作 ---
        # 建立信号与槽的连接
        self.pushButton_p11.clicked.connect(self.click_pushButton_p11)
        self.pushButton_p12.clicked.connect(self.click_pushButton_p12)
        self.pushButton_p13.clicked.connect(self.click_pushButton_p13)
        self.pushButton_p14.clicked.connect(self.click_pushButton_p14)
        ## --- GUI 定义动作 完成---

        # ## --- GUI 加载初始数据 ---
        self.img3 = np.zeros((600, 400), np.uint8)
        self.img4 = np.ones((600, 400), np.uint8)*64
        self.img5 = np.ones((600, 400), np.uint8)*127
        self.img6 = np.ones((600, 400), np.uint8)*255
        # ## --- GUI 加载初始数据 完成---

        return

    def click_pushButton_1(self):  # 点击 pushButton_01 触发
        self.stackedWidget.setCurrentIndex(0)  # 选择堆叠布局页面 stackedWidget > page_0
        self.lineEdit.setText("1. 阈值抠图")
        self.plainTextEdit.appendPlainText("对原始图像采用固定阈值处理，生成遮罩进行抠图")

        # listLabel = [self.label_1, self.label_2, self.label_3, self.label_4, self.label_5, self.label_6]
        # for snLabel in listLabel:
        #     snLabel.setPixmap(QtGui.QPixmap(""))  # 删除图片 snLabel

        # 1) 获取原始图像
        try:
            imgOri = self.img.copy()
            imgGray = cv2.cvtColor(imgOri, cv2.COLOR_BGR2GRAY)  # 图片格式转换：BGR(OpenCV) -> RGB(PyQt5)
            hImg, wImg, cImg = imgOri.shape  # 获取图片 Img 的 height, width, channel
            self.plainTextEdit.appendPlainText("图片原始尺寸为：{},{},{}".format(hImg,wImg,cImg))
        except Exception as e:  # 错误处理
            self.lineEdit.setText(str(e))
            QMessageBox.about(self, "About", """<font color=\"#0000FF\">请读入抠图图像。</font>""")
            return

        # 2) 绿色通道转换为二值图像，生成遮罩 Mask、逆遮罩 MaskInv
        # 如果背景不是绿屏而是其它颜色，可以采用对应的颜色通道进行阈值处理 (不宜基于灰度图像进行固定阈值处理，性能差异很大)
        # imgGreen = imgOri[:, :, 1]  # imgGreen 为 绿色通道的 色彩强度图 (注意不是原图的灰度转换结果)
        # colorThresh = 245  # 绿屏背景的颜色阈值 (注意研究阈值的影响)
        # ret, binary = cv2.threshold(imgGreen, colorThresh, 255, cv2.THRESH_BINARY)  # 转换为二值图像，生成遮罩，抠图区域黑色遮盖
        ret, binary = cv2.threshold(imgGray, 127, 255, cv2.THRESH_BINARY)  # 转换为二值图像，生成遮罩，抠图区域黑色遮盖
        binaryInv = cv2.bitwise_not(binary)  # 按位非(黑白转置)，生成逆遮罩，抠图区域白色开窗，抠图以外区域黑色

        # 3) 用遮罩进行抠图和更换背景
        # 生成抠图图像 (前景保留，背景黑色)
        imgMatte = cv2.bitwise_and(imgOri, imgOri, mask=binaryInv)  # 生成抠图前景，标准抠图以外的逆遮罩区域输出黑色

        # 4) 将背景颜色更换为红色: 修改逆遮罩 (抠图以外区域黑色)
        imgReplace = imgOri.copy()
        imgReplace[binaryInv == 0] = [0, 0, 255]  # 黑色区域(0/0/0)修改为红色(BGR:0/0/255)

        # 5) 图像显示到 GUI label
        self.img1 = imgOri
        self.img2 = imgGray
        self.img3 = binary
        self.img4 = binaryInv
        self.img5 = imgMatte
        self.img6 = imgReplace
        listImg = [self.img1, self.img2, self.img3, self.img4, self.img5, self.img6]
        listLabel = [self.label_1, self.label_2, self.label_3, self.label_4, self.label_5, self.label_6]
        for i in range(len(listLabel)):
            imgShowLabel(listImg[i], listLabel[i])  # 图像显示函数
        return

    def click_pushButton_2(self):  # 点击 pushButton_02 触发
        self.stackedWidget.setCurrentIndex(0)  # 选择堆叠布局页面 stackedWidget > page_0
        self.lineEdit.setText("2. 自适应阈值抠图")
        self.plainTextEdit.appendPlainText("对原始图像采用自适应阈值处理，生成遮罩进行抠图")

        # listLabel = [self.label_1, self.label_2, self.label_3, self.label_4, self.label_5, self.label_6]
        # for snLabel in listLabel:
        #     snLabel.setPixmap(QtGui.QPixmap(""))  # 删除图片 snLabel

        # 1) 获取原始图像
        try:
            imgOri = self.img.copy()
            imgGray = cv2.cvtColor(imgOri, cv2.COLOR_BGR2GRAY)  # 图片格式转换：BGR(OpenCV) -> RGB(PyQt5)
            hImg, wImg, cImg = imgOri.shape  # 获取图片 Img 的 height, width, channel
            self.plainTextEdit.appendPlainText("图片原始尺寸为：{},{},{}".format(hImg,wImg,cImg))
        except Exception as e:  # 错误处理
            self.lineEdit.setText(str(e))
            QMessageBox.about(self, "About", """<font color=\"#0000FF\">请读入抠图图像。</font>""")
            return

        # 2) 从原始图像提取绿色通道
        # 如果背景不是绿屏而是其它颜色，可以采用对应的颜色通道进行阈值处理 (不宜基于灰度图像进行固定阈值处理，性能差异很大)
        imgGreen = imgOri[:, :, 1]  # imgGreen 为 绿色通道的 色彩强度图 (注意不是原图的灰度转换结果)

        # 3) 自适应阈值化能够根据图像不同区域亮度分布自适应地改变阈值
        # cv.adaptiveThreshold(src, maxValue, adaptiveMethod, thresholdType, blockSize, C[, dst]) -> dst
        # 参数 adaptiveMethod: ADAPTIVE_THRESH_MEAN_C(均值法), ADAPTIVE_THRESH_GAUSSIAN_C(高斯法)
        # 参数 thresholdType: THRESH_BINARY(小于阈值为0), THRESH_BINARY_INV(大于阈值为0)
        # 参数 blockSize: 邻域大小，正奇数
        binary = cv2.adaptiveThreshold(imgGreen, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 5, 0)
        binaryInv = cv2.bitwise_not(binary)  # 按位非(黑白转置)，生成逆遮罩，抠图区域白色开窗，抠图以外区域黑色

        # 4) 用遮罩进行抠图和更换背景
        # 生成抠图图像 (前景保留，背景黑色)
        imgMatte = cv2.bitwise_and(imgOri, imgOri, mask=binaryInv)  # 生成抠图前景，标准抠图以外的逆遮罩区域输出黑色

        # 5) 将背景颜色更换为红色: 修改逆遮罩 (抠图以外区域黑色)
        imgReplace = imgOri.copy()
        imgReplace[binaryInv == 0] = [0, 0, 255]  # 黑色区域(0/0/0)修改为红色(BGR:0/0/255)

        # 5) 图像显示到 GUI label
        self.img1 = imgOri
        self.img2 = imgGray
        self.img3 = binary
        self.img4 = binaryInv
        self.img5 = imgMatte
        self.img6 = imgReplace
        listImg = [self.img1, self.img2, self.img3, self.img4, self.img5, self.img6]
        listLabel = [self.label_1, self.label_2, self.label_3, self.label_4, self.label_5, self.label_6]
        for i in range(len(listLabel)):
            imgShowLabel(listImg[i], listLabel[i])  # 图像显示函数

        return

    def click_pushButton_3(self):  # 点击 pushButton_03 触发
        self.stackedWidget.setCurrentIndex(0)  # 选择堆叠布局页面 stackedWidget > page_0
        self.lineEdit.setText("3. HSV 阈值抠图")
        self.plainTextEdit.appendPlainText(" 转换到 HSV 空间，对背景颜色范围进行阈值处理，生成遮罩进行抠图")

        # listLabel = [self.label_1, self.label_2, self.label_3, self.label_4, self.label_5, self.label_6]
        # for snLabel in listLabel:
        #     snLabel.setPixmap(QtGui.QPixmap(""))  # 删除图片 snLabel

        # 1) 获取原始图像
        try:
            imgOri = self.img.copy()
            imgGray = cv2.cvtColor(imgOri, cv2.COLOR_BGR2GRAY)  # 图片格式转换：BGR(OpenCV) -> RGB(PyQt5)
            hImg, wImg, cImg = imgOri.shape  # 获取图片 Img 的 height, width, channel
            self.plainTextEdit.appendPlainText("图片原始尺寸为：{},{},{}".format(hImg,wImg,cImg))
        except Exception as e:  # 错误处理
            self.lineEdit.setText(str(e))
            QMessageBox.about(self, "About", """<font color=\"#0000FF\">请读入抠图图像。</font>""")
            return

        # 2) 转换到 HSV 空间，对背景颜色范围进行阈值处理，生成遮罩 Mask、逆遮罩 MaskInv
        # 使用 cv.nrange 函数在 HSV 空间检查设定的颜色区域范围，转换为二值图像，生成遮罩
        # cv.inRange(src, lowerb, upperb[, dst]	) -> dst
        # inRange(frame,Scalar(low_b,low_g,low_r), Scalar(high_b,high_g,high_r))
        hsv = cv2.cvtColor(imgOri, cv2.COLOR_BGR2HSV)  # 将图片转换到 HSV 色彩空间
        lowerColor = np.array([35, 43, 46])  # (下限: 绿色33/43/46,红色156/43/46,蓝色100/43/46)
        upperColor = np.array([77, 255, 255])  # (上限: 绿色77/255/255,红色180/255/255,蓝色124/255/255)
        binary = cv2.inRange(hsv, lowerColor, upperColor)  # 对指定颜色区域进行阈值处理，生成遮罩，抠图区域黑色遮盖
        binaryInv = cv2.bitwise_not(binary)  # 按位非(黑白转置)，生成逆遮罩，抠图区域白色开窗，抠图以外区域黑色

        # 4) 用遮罩进行抠图和更换背景
        # 生成抠图图像 (前景保留，背景黑色)
        imgMatte = cv2.bitwise_and(imgOri, imgOri, mask=binaryInv)  # 生成抠图前景，标准抠图以外的逆遮罩区域输出黑色

        # 5) 将背景颜色更换为红色: 修改逆遮罩 (抠图以外区域黑色)
        imgReplace = imgOri.copy()
        imgReplace[binaryInv == 0] = [0, 0, 255]  # 黑色区域(0/0/0)修改为红色(BGR:0/0/255)

        # 5) 图像显示到 GUI label
        self.img1 = imgOri
        self.img2 = imgGray
        self.img3 = binary
        self.img4 = binaryInv
        self.img5 = imgMatte
        self.img6 = imgReplace
        listImg = [self.img1, self.img2, self.img3, self.img4, self.img5, self.img6]
        listLabel = [self.label_1, self.label_2, self.label_3, self.label_4, self.label_5, self.label_6]
        for i in range(len(listLabel)):
            imgShowLabel(listImg[i], listLabel[i])  # 图像显示函数

        return

    def click_pushButton_4(self):  # 点击 pushButton_04 触发
        self.stackedWidget.setCurrentIndex(0)  # 选择堆叠布局页面 stackedWidget > page_0
        self.lineEdit.setText("4. 提取轮廓")
        self.plainTextEdit.appendPlainText(" 转换到 HSV 空间，对背景颜色范围进行阈值处理，生成遮罩进行抠图")

        # listLabel = [self.label_1, self.label_2, self.label_3, self.label_4, self.label_5, self.label_6]
        # for snLabel in listLabel:
        #     snLabel.setPixmap(QtGui.QPixmap(""))  # 删除图片 snLabel

        # 1) 获取原始图像
        try:
            imgOri = self.img.copy()
            imgGray = cv2.cvtColor(imgOri, cv2.COLOR_BGR2GRAY)  # 图片格式转换：BGR(OpenCV) -> RGB(PyQt5)
            hImg, wImg, cImg = imgOri.shape  # 获取图片 Img 的 height, width, channel
            self.plainTextEdit.appendPlainText("图片原始尺寸为：{},{},{}".format(hImg,wImg,cImg))
        except Exception as e:  # 错误处理
            self.lineEdit.setText(str(e))
            QMessageBox.about(self, "About", """<font color=\"#0000FF\">请读入抠图图像。</font>""")
            return

        # 2) 边缘检测
        # cv.GaussianBlur(src, ksize, sigmaX[, dst[, sigmaY[, borderType]]]) -> dst
        # cv.Canny(image, threshold1, threshold2[, edges[, apertureSize[, L2gradient]]]	) -> edges
        # cv.dilate(src, kernel[, dst[, anchor[, iterations[, borderType[, borderValue]]]]]	) -> dst
        blur = cv2.GaussianBlur(imgGray, (3, 3), 1)  # 高斯平滑
        canny = cv2.Canny(blur, 127, 255, apertureSize=3)  # 二值图像
        kernel = np.ones((3, 3), np.uint8)

        # 生成逆遮罩用于图片合成
        binaryInv = cv2.bitwise_not(canny)  # 按位非(黑白转置)，生成逆遮罩，抠图区域白色开窗，抠图以外区域黑色
        # 生成抠图图像 (前景保留，背景黑色)
        recovery = cv2.bitwise_and(imgOri, imgOri, mask=binaryInv)  # 恢复原始图像，边缘增强
        gray = cv2.cvtColor(recovery, cv2.COLOR_BGR2GRAY)  # 彩色图像转换为灰度图像

        # 闭运算
        # cv.morphologyEx( src, op, kernel[, dst[, anchor[, iterations[, borderType[, borderValue]]]]] ) ->	dst
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        closing = cv2.morphologyEx(recovery, cv2.MORPH_CLOSE, kernel)  # cv.MORPH_CLOSE
        exGray = cv2.cvtColor(closing, cv2.COLOR_BGR2GRAY)  # 彩色图像转换为灰度图像
        _, mask = cv2.threshold(exGray, 127, 255, cv2.THRESH_BINARY_INV)

        # # 4. 最大轮廓检测
        ret, thresh = cv2.threshold(imgGray, 127, 255, cv2.THRESH_BINARY_INV)
        # 寻找二值化图中的轮廓
        _, contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        imgC = imgOri.copy()
        imgContour = cv2.drawContours(imgC, contours, -1, (0, 0, 255), 2)

        # 4) 用遮罩进行抠图和更换背景
        # 生成抠图图像 (前景保留，背景黑色)
        imgMatte = cv2.bitwise_and(imgOri, imgOri, mask=binaryInv)  # 生成抠图前景，标准抠图以外的逆遮罩区域输出黑色

        # 5) 图像显示到 GUI label
        self.img1 = imgOri
        self.img2 = imgGray
        self.img3 = canny
        self.img4 = mask
        self.img5 = imgContour
        self.img6 = imgMatte
        listImg = [self.img1, self.img2, self.img3, self.img4, self.img5, self.img6]
        listLabel = [self.label_1, self.label_2, self.label_3, self.label_4, self.label_5, self.label_6]
        for i in range(len(listLabel)):
            imgShowLabel(listImg[i], listLabel[i])  # 图像显示函数
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

    def click_pushButton_p11(self):  # 点击 pushButton_p11 触发
        self.lineEdit.setText("放大显示 图 3")
        imgShowLabel(self.img3, self.label_2)  # 图像显示函数, flag=0 灰度图像
        return

    def click_pushButton_p12(self):  # 点击 pushButton_p12 触发
        self.lineEdit.setText("放大显示 图 4")
        imgShowLabel(self.img4, self.label_2)  # 图像显示函数, flag=0 灰度图像
        return

    def click_pushButton_p13(self):  # 点击 pushButton_p13 触发
        self.lineEdit.setText("放大显示 图 5")
        imgShowLabel(self.img5, self.label_2)  # 图像显示函数, flag=0 灰度图像
        return

    def click_pushButton_p14(self):  # 点击 pushButton_p14 触发
        self.lineEdit.setText("放大显示 图 6")
        imgShowLabel(self.img6, self.label_2)  # 图像显示函数, flag=0 灰度图像
        return

    def trigger_actHelp(self):  # 动作 actHelp 触发
        QMessageBox.about(self, "About",
                          """<font color=\"#0000FF\">数字图像处理工具箱 v1.0<br>
                          Copyright YouCans, XUPT 2021</font>""")
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

        QMessageBox.about(self, "About",
                          """程序开发中...\nCopyright YouCans, XUPT 2021""")

        # file, ok = QFileDialog.getOpenFileName(self, "打开", "../images/", "All Files (*);;Text Files (*.txt)")
        # # 在状态栏显示文件地址
        # self.plainTextEdit.appendPlainText(file)
        return

if __name__ == '__main__':
    app = QApplication(sys.argv)  # 在 QApplication 方法中使用，创建应用程序对象
    myWin = MyMainWindow()  # 实例化 MyMainWindow 类，创建主窗口
    myWin.show()  # 在桌面显示控件 myWin
    sys.exit(app.exec_())  # 结束进程，退出程序
