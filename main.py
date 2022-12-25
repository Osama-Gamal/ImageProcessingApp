import PIL
from PyQt5 import QtWidgets, uic, QtGui
import sys
import numpy as np
from PyQt5.QtGui import QPixmap, QImage, QColor
from PyQt5.QtWidgets import QFileDialog, QPushButton
import cv2
from PIL import Image, ImageFilter
from skimage import exposure
from skimage.util import random_noise
from scipy import ndimage
from matplotlib import pyplot as plt, cm


class Ui(QtWidgets.QMainWindow):
    def __init__(self):
        super(Ui, self).__init__()
        uic.loadUi('mainWindow.ui', self)
        self.loadImgBtn.clicked.connect(self.loadImgFunc)
        self.medianBtn.clicked.connect(self.medianProcess)
        self.minFilterBtn.clicked.connect(self.minFilterProcess)
        self.maxFilterBtn.clicked.connect(self.maxFilterProcess)
        self.addNoiseBtn.clicked.connect(self.addNoise)
        self.reobertBtn.clicked.connect(self.reobertBorders)
        self.SobelBtn.clicked.connect(self.sobelProcess)
        self.AverageBtn.clicked.connect(self.averageColour)
        self.laplicianBtn.clicked.connect(self.laplacian)
        self.ButterLowBtn.clicked.connect(self.butterLow)
        self.GaussianBtn.clicked.connect(self.gaussain)
        self.foureirBtn.clicked.connect(self.fourier)
        self.grayBtn.clicked.connect(self.grayLevel)
        self.negativeBtn.clicked.connect(self.negativeImage)
        self.ThresholdingBtn.clicked.connect(self.tresholding)
        self.logBtn.clicked.connect(self.logProcess)
        self.powerBtn.clicked.connect(self.powerLaw)
        self.contrastBtn.clicked.connect(self.contrastStretching)
        self.HistgoramBtn.clicked.connect(self.histgoram)
        self.resizeBtn.clicked.connect(self.resizeProcess)
        self.BinaryBtn.clicked.connect(self.convertToBinary)
        self.rgbBTn.clicked.connect(self.convertToRGB)
        self.additionBtn.clicked.connect(self.additionImage)
        self.subtractBtn.clicked.connect(self.subtractImage)
        self.andBtn.clicked.connect(self.bitWiseAndImage)
        self.xorBtn.clicked.connect(self.bitWiseXorImage)
        self.notBtn.clicked.connect(self.bitWiseNotImage)
        self.cropBtn.clicked.connect(self.extractRegionImage)
        self.matchingBtn.clicked.connect(self.histogramMatching)
        self.equalizationBtn.clicked.connect(self.histogramEqualization)
        self.saveBtn.clicked.connect(self.saveImg)
        self.show()


    def loadImgFunc(self):
        image = QFileDialog.getOpenFileName(None, 'OpenFile', '/home/',
                                            "Images (*.png *.jpeg *.jpg)")
        if len(image[0]) != 0:
            print(image)
            global imagePath
            imagePath = image[0]
            if imagePath is not None:
                pixmap = QPixmap(imagePath)
                global originalImage,pil_image
                originalImage = cv2.imread(imagePath, cv2.IMREAD_UNCHANGED)
                pil_image = Image.open(imagePath)
                self.imagePathTxt.setText(imagePath)
                self.mainImage.setPixmap(self.convertcvImgToQtImg(originalImage))
                self.mainImage.setScaledContents(1)


    def additionImage(self):
        image = QFileDialog.getOpenFileName(None, 'OpenFile', '/home/',
                                            "Images (*.png *.jpeg *.jpg)")

        if len(image[0]) != 0:
            SecondimagePath = image[0]
            Secondimage = cv2.imread(SecondimagePath, cv2.IMREAD_UNCHANGED)
            # هنا انا لازم اعمل تغيير لحجم الصوره التانيه علشان اقدر اطيق مفهوم الاضافه لان الصورتين لازم يكونوا نفس الحجم
            dimensions = (originalImage.shape[1],originalImage.shape[0])
            Secondimage = cv2.resize(Secondimage, dimensions, interpolation=cv2.INTER_AREA)
            weightedSum = cv2.addWeighted(originalImage, 0.5, Secondimage, 0.4, 0)
            self.processedImage.setPixmap(self.convertcvImgToQtImg(weightedSum))
            self.processedImage.setScaledContents(1)

    def subtractImage(self):
        image = QFileDialog.getOpenFileName(None, 'OpenFile', '/home/',
                                            "Images (*.png *.jpeg *.jpg)")
        if len(image[0]) != 0:
            SecondimagePath = image[0]
            Secondimage = cv2.imread(SecondimagePath, cv2.IMREAD_UNCHANGED)
            # هنا انا لازم اعمل تغيير لحجم الصوره التانيه علشان اقدر اطيق مفهوم الاضافه لان الصورتين لازم يكونوا نفس الحجم
            dimensions = (originalImage.shape[1], originalImage.shape[0])
            Secondimage = cv2.resize(Secondimage, dimensions, interpolation=cv2.INTER_AREA)
            sub = cv2.subtract(originalImage, Secondimage)
            self.processedImage.setPixmap(self.convertcvImgToQtImg(sub))
            self.processedImage.setScaledContents(1)

    def bitWiseAndImage(self):
        image = QFileDialog.getOpenFileName(None, 'OpenFile', '/home/',
                                            "Images (*.png *.jpeg *.jpg)")
        if len(image[0]) != 0:
            SecondimagePath = image[0]
            Secondimage = cv2.imread(SecondimagePath, cv2.IMREAD_UNCHANGED)
            dimensions = (originalImage.shape[1], originalImage.shape[0])
            Secondimage = cv2.resize(Secondimage, dimensions, interpolation=cv2.INTER_AREA)
            bitwiseAnd = cv2.bitwise_and(originalImage, Secondimage)
            self.processedImage.setPixmap(self.convertcvImgToQtImg(bitwiseAnd))
            self.processedImage.setScaledContents(1)

    def bitWiseXorImage(self):
        image = QFileDialog.getOpenFileName(None, 'OpenFile', '/home/',
                                            "Images (*.png *.jpeg *.jpg)")
        if len(image[0]) != 0:
            SecondimagePath = image[0]
            Secondimage = cv2.imread(SecondimagePath, cv2.IMREAD_UNCHANGED)
            dimensions = (originalImage.shape[1], originalImage.shape[0])
            Secondimage = cv2.resize(Secondimage, dimensions, interpolation=cv2.INTER_AREA)
            bitwiseXor = cv2.bitwise_xor(originalImage, Secondimage)
            self.processedImage.setPixmap(self.convertcvImgToQtImg(bitwiseXor))
            self.processedImage.setScaledContents(1)

    def bitWiseNotImage(self):
        bitwiseNot = cv2.bitwise_not(originalImage)
        self.processedImage.setPixmap(self.convertcvImgToQtImg(bitwiseNot))
        self.processedImage.setScaledContents(1)

    def extractRegionImage(self):
        blank = np.zeros(originalImage.shape[:2], dtype='uint8')
        #بجيب حجم الصوره الاصليه في الطول والعرض علشان لما اعمل الدايره كماسك اخلي الاحداثيات في النص بالظبط
        center_coordinates = (int(originalImage.shape[1]/2), int(originalImage.shape[0]/2))
        radius = 350
        mask = cv2.circle(blank, center_coordinates, radius, 255, -1)
        masked = cv2.bitwise_and(originalImage, originalImage, mask=mask)
        self.processedImage.setPixmap(self.convertcvImgToQtImg(masked))
        self.processedImage.setScaledContents(1)

    def histogramMatching(self):
        image = QFileDialog.getOpenFileName(None, 'OpenFile', '/home/',
                                            "Images (*.png *.jpeg *.jpg)")
        if len(image[0]) != 0:
            SecondimagePath = image[0]
            Secondimage = cv2.imread(SecondimagePath, cv2.IMREAD_UNCHANGED)
            dimensions = (originalImage.shape[1], originalImage.shape[0])
            Secondimage = cv2.resize(Secondimage, dimensions, interpolation=cv2.INTER_AREA)
            multi = True if Secondimage.shape[-1] > 1 else False
            matched = exposure.match_histograms(originalImage, Secondimage, multichannel=multi)
            self.processedImage.setPixmap(self.convertcvImgToQtImg(matched))
            self.processedImage.setScaledContents(1)

    def histogramEqualization(self):
        src = cv2.cvtColor(originalImage, cv2.COLOR_BGR2GRAY)
        dst = cv2.equalizeHist(src)
        self.processedImage.setPixmap(self.convertcvImgToQtImg(dst))
        self.processedImage.setScaledContents(1)
    def medianProcess(self):
        median = cv2.medianBlur(originalImage, int(self.medianValBox.value()))
        self.processedImage.setPixmap(self.convertcvImgToQtImg(median))
        self.processedImage.setScaledContents(1)

    def minFilterProcess(self):
        min_ = pil_image.filter(ImageFilter.MinFilter(size=int(self.minfilterInput.value())))
        self.processedImage.setPixmap(self.pil2pixmap(min_))
        self.processedImage.setScaledContents(1)

    def maxFilterProcess(self):
        max_ = pil_image.filter(ImageFilter.MaxFilter(size=int(self.maxfilterInput.value())))
        self.processedImage.setPixmap(self.pil2pixmap(max_))
        self.processedImage.setScaledContents(1)

    def addNoise(self):
        noise_img = random_noise(originalImage, mode='s&p', amount=self.noiseInput.value())
        noise_img = np.array(255 * noise_img, dtype='uint8')
        self.processedImage.setPixmap(self.convertcvImgToQtImg(noise_img))
        self.processedImage.setScaledContents(1)

    def reobertBorders(self):
        roberts_cross_v = np.array([[1, 0],
                                    [0, -1]])
        roberts_cross_h = np.array([[0, 1],
                                    [-1, 0]])
        img = cv2.imread(imagePath,0).astype('float64')
        img /= 255.0
        vertical = ndimage.convolve(img, roberts_cross_v)
        horizontal = ndimage.convolve(img, roberts_cross_h)
        edged_img = np.sqrt(np.square(horizontal) + np.square(vertical))
        edged_img *= 255
        self.processedImage.setPixmap(self.convertcvImgToQtImg(edged_img))
        self.processedImage.setScaledContents(1)

    def sobelProcess(self):
        gray = cv2.cvtColor(originalImage, cv2.COLOR_BGR2GRAY)
        x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3, scale=1)
        y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3, scale=1)
        absx = cv2.convertScaleAbs(x)
        absy = cv2.convertScaleAbs(y)
        edge = cv2.addWeighted(absx, 0.5, absy, 0.5, 0)
        self.processedImage.setPixmap(self.convertcvImgToQtImg(edge))
        self.processedImage.setScaledContents(1)

    def averageColour(self):
        average_color_row = np.average(originalImage, axis=0)
        average_color = np.average(average_color_row, axis=0)
        d_img = np.ones((312, 312, 3), dtype=np.uint8)
        d_img[:, :] = average_color
        self.processedImage.setPixmap(self.convertcvImgToQtImg(d_img))
        self.processedImage.setScaledContents(1)

    def laplacian(self):
        blur = cv2.GaussianBlur(originalImage, (int(self.verticalLaplace.value()), int(self.horiaontalLaplace.value())), 0)
        laplacian = cv2.Laplacian(blur, cv2.CV_64F)
        self.processedImage.setPixmap(self.convertcvImgToQtImg(laplacian))
        self.processedImage.setScaledContents(1)

    def butterLow(self):
        f = cv2.imread(imagePath, 0)
        F = np.fft.fft2(f)
        Fshift = np.fft.fftshift(F)
        M, N = f.shape
        H = np.zeros((M, N), dtype=np.float32)
        D0 = self.butterDoinput.value()  # cut of frequency
        n = self.butterOrderInput.value()  # order
        for u in range(M):
            for v in range(N):
                D = np.sqrt((u - M / 2) ** 2 + (v - N / 2) ** 2)
                H[u, v] = 1 / (1 + (D / D0) ** n)
        self.processedImage.setPixmap(self.convertcvImgToQtImg(H))
        self.processedImage.setScaledContents(1)

    def convertToBinary(self):
        gray = cv2.cvtColor(originalImage, cv2.COLOR_BGR2GRAY)
        ret, thresh = cv2.threshold(gray, 170, 255, 0)
        self.processedImage.setPixmap(self.convertcvImgToQtImg(thresh))
        self.processedImage.setScaledContents(1)
    def convertToRGB(self):
        image_rgb = cv2.cvtColor(originalImage, cv2.COLOR_BGR2RGB)
        self.processedImage.setPixmap(self.convertcvImgToQtImg(image_rgb))
        self.processedImage.setScaledContents(1)

    def gaussain(self):
        dst = cv2.GaussianBlur(originalImage, (int(self.uguassianInput.value()), int(self.vguassianInput.value())), cv2.BORDER_DEFAULT)
        self.processedImage.setPixmap(self.convertcvImgToQtImg(dst))
        self.processedImage.setScaledContents(1)

    def fourier(self):
        img = cv2.imread(imagePath, 0)
        dft = cv2.dft(np.float32(img), flags=cv2.DFT_COMPLEX_OUTPUT)
        dft_shift = np.fft.fftshift(dft)
        magnitude_spectrum = self.magnitudeInput.value() * np.log(cv2.magnitude(
            dft_shift[:, :, 0],
            dft_shift[:, :, 1])
        )
        plt.subplot(121), plt.imshow(img, cmap='gray')
        plt.title('Input Image'), plt.xticks([]), plt.yticks([])
        plt.subplot(122), plt.imshow(magnitude_spectrum, cmap='gray')
        plt.title('Magnitude Spectrum'), plt.xticks([]), plt.yticks([])

        figure = plt.gcf()
        figure.canvas.draw()
        b = figure.axes[0].get_window_extent()
        img = np.array(figure.canvas.buffer_rgba())
        img = img[int(b.y0):int(b.y1), int(b.x0):int(b.x1), :]
        img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGRA)
        self.processedImage.setPixmap(self.convertcvImgToQtImg(img))
        self.processedImage.setScaledContents(1)

    def grayLevel(self):
        img = cv2.imread(imagePath)
        (row, col) = img.shape[0:2]
        for i in range(row):
            for j in range(col):
                img[i, j] = sum(img[i, j]) * self.grayInput.value()
        self.processedImage.setPixmap(self.convertcvImgToQtImg(img))
        self.processedImage.setScaledContents(1)

    def negativeImage(self):
        img_neg = 255 - originalImage
        #img_neg = cv2.bitwise_not(originalImage)
        self.processedImage.setPixmap(self.convertcvImgToQtImg(img_neg))
        self.processedImage.setScaledContents(1)

    def tresholding(self):
        img = cv2.imread(imagePath, 0)
        img = cv2.medianBlur(img, 5)
        ret, th1 = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
        th2 = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, \
                                   cv2.THRESH_BINARY, 11, 2)
        th3 = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, \
                                   cv2.THRESH_BINARY, 11, 2)
        titles = ['Original Image', 'Global Thresholding (v = 127)',
                  'Adaptive Mean Thresholding', 'Adaptive Gaussian Thresholding']
        images = [img, th1, th2, th3]
        for i in range(4):
            plt.subplot(2, 2, i + 1), plt.imshow(images[i], 'gray')
            plt.title(titles[i])
            plt.xticks([]), plt.yticks([])
        figure = plt.gcf()
        figure.canvas.draw()
        b = figure.axes[0].get_window_extent()
        img = np.array(figure.canvas.buffer_rgba())
        img = img[int(b.y0):int(b.y1), int(b.x0):int(b.x1), :]
        img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGRA)
        self.processedImage.setPixmap(self.convertcvImgToQtImg(img))
        self.processedImage.setScaledContents(1)

    def logProcess(self):
        c = 255 / np.log(1 + np.max(originalImage))
        log_image = c * (np.log(originalImage + 1))
        log_image = np.array(log_image, dtype=np.uint8)
        self.processedImage.setPixmap(self.convertcvImgToQtImg(log_image))
        self.processedImage.setScaledContents(1)


    def powerLaw(self):
        gamma_two_point_two = np.array(255 * (originalImage / 255) ** 2.2, dtype='uint8')
        gamma_point_four = np.array(255 * (originalImage / 255) ** self.powerInput.value(), dtype='uint8')
        img3 = cv2.hconcat([gamma_two_point_two, gamma_point_four])
        self.processedImage.setPixmap(self.convertcvImgToQtImg(img3))
        self.processedImage.setScaledContents(1)

    def contrastStretching(self):
        r1 = int(self.contrastr1.value())
        s1 = 0
        r2 = int(self.contrastr2.value())
        s2 = 255
        pixelVal_vec = np.vectorize(self.pixelVal)
        contrast_stretched = pixelVal_vec(originalImage, r1, s1, r2, s2)
        self.processedImage.setPixmap(self.convertcvImgToQtImg(contrast_stretched))
        self.processedImage.setScaledContents(1)

    def histgoram(self):
       # hist = cv2.calcHist([originalImage], [0], None, [256], [0, 256])
        img = cv2.imread(imagePath)
        color = ('b', 'g', 'r')
        for i, col in enumerate(color):
            histr = cv2.calcHist([img], [i], None, [256], [0, 256])
            plt.plot(histr, color=col)
            plt.xlim([0, 256])

        figure = plt.gcf()
        figure.canvas.draw()
        b = figure.axes[0].get_window_extent()
        img = np.array(figure.canvas.buffer_rgba())
        img = img[int(b.y0):int(b.y1), int(b.x0):int(b.x1), :]
        img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGRA)
        self.processedImage.setPixmap(self.convertcvImgToQtImg(img))
        self.processedImage.setScaledContents(1)

    def resizeProcess(self):
        width = int(self.heightInput.value())
        height = int(self.widthInput.value())
        dim = (width, height)
        resized = cv2.resize(originalImage, dim, interpolation=cv2.INTER_AREA)
        self.processedImage.setPixmap(self.convertcvImgToQtImg(resized))
        self.processedImage.setScaledContents(1)
    def saveImg(self):
        self.processedImage.pixmap().save('ProcessedImage.png')


    def pixelVal(self,pix, r1, s1, r2, s2):
        if (0 <= pix and pix <= r1):
            return (s1 / r1) * pix
        elif (r1 < pix and pix <= r2):
            return ((s2 - s1) / (r2 - r1)) * (pix - r1) + s1
        else:
            return ((255 - s2) / (255 - r2)) * (pix - r2) + s2


    def convertcvImgToQtImg(self,cvImage):
        cvImage = QtGui.QImage(cvImage, cvImage.shape[1], cvImage.shape[0], cvImage.shape[1] * 3, QtGui.QImage.Format_BGR888)
        pix = QtGui.QPixmap(cvImage)
        return QtGui.QPixmap(pix)

    def pil2pixmap(self, im):

        if im.mode == "RGB":
            r, g, b = im.split()
            im = Image.merge("RGB", (b, g, r))
        elif im.mode == "RGBA":
            r, g, b, a = im.split()
            im = Image.merge("RGBA", (b, g, r, a))
        elif im.mode == "L":
            im = im.convert("RGBA")
        # Bild in RGBA konvertieren, falls nicht bereits passiert
        im2 = im.convert("RGBA")
        data = im2.tobytes("raw", "RGBA")
        qim = QtGui.QImage(data, im.size[0], im.size[1], QtGui.QImage.Format_ARGB32)
        pixmap = QtGui.QPixmap.fromImage(qim)
        return pixmap



app = QtWidgets.QApplication(sys.argv)
window = Ui()





app.exec_()

