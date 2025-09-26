from PyQt6.QtCore import Qt
from PyQt6.QtGui import QPixmap
from PyQt6.QtWidgets import *

from main_ui import Ui_MainWindow
import imagealgorithm as imalg


class MainWindow(QMainWindow, Ui_MainWindow):
    def __init__(self):
        super().__init__()
        self.setupUi(self)

        self.imgPath = None
        self.imgMatrix = None
        self.imgMatrixAdjusted = None

        self.mainImageLabel.mouseMoved.connect(self.updatePointerInfo)
        self.updatePointerInfo(*[None] * 8)
        self.updateImageInfo()

        self.brightnessValue = 0
        self.redValue = 0
        self.greenValue = 0
        self.blueValue = 0
        self.contrastValue = 0

        for slider, handler in (
            (self.brightnessSlider, self.onBrightnessChanged),
            (self.redSlider, self.onRedChanged),
            (self.greenSlider, self.onGreenChanged),
            (self.blueSlider, self.onBlueChanged),
            (self.contrastSlider, self.onContrastChanged)
        ):
            slider.valueChanged.connect(handler)
            slider.sliderReleased.connect(self.onSliderReleased)

        self.loadImageAction.triggered.connect(self.loadImage)
        self.exitApplicationAction.triggered.connect(self.destroy)
        self.exitApplicationAction.setShortcut('Alt+F4')

        self.statusbar.showMessage("Готово!")

    def updateImageInfo(self):
        """ Update info labels about image dimensions. """
        if self.imgMatrix is not None:
            height, width, channels = self.imgMatrix.shape
            self.statLabel.setText(f"Размер: {width}×{height}×{channels}")
        else:
            self.statLabel.setText("Размер: Н/Д")

    def updatePointerInfo(self, x, y, r, g, b, intensity, window_avg, window_std):
        """ Update info labels about pointed pixel and its window. """
        if any((x is None, y is None, r is None, g is None, b is None,
                intensity is None, window_avg is None, window_std is None)):
            self.pointerPositionLabel.setText(f"Позиция: Н/Д")
            self.pointerRgbLabel.setText(f"RGB: Н/Д")
            self.pointerIntensityLabel.setText(f"Интенсивность: Н/Д")
            self.windowAverageLabel.setText(f"Среднее окна: Н/Д")
            self.windowStdDevLabel.setText(f"Ст. отклонение: Н/Д")
        else:
            self.pointerPositionLabel.setText(f"Позиция: ({x}, {y})")
            self.pointerRgbLabel.setText(f"RGB: ({r}, {g}, {b})")
            self.pointerIntensityLabel.setText(f"Интенсивность: {intensity:.2f}")
            self.windowAverageLabel.setText(f"Среднее окна: {window_avg:.2f}")
            self.windowStdDevLabel.setText(f"Ст. отклонение: {window_std:.2f}")

    @staticmethod
    def setImagePixmap(label: QLabel, pixmap: QPixmap):
        label.setPixmap(pixmap.scaled(
            label.width(),
            label.height(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation
        ))

    def updateInfoImages(self):
        grayMatrix = imalg.toGrayscale(self.imgMatrixAdjusted)
        grayPixmap = imalg.pixmapFromMatrix(grayMatrix)
        self.setImagePixmap(self.blackWhiteImageLabel, grayPixmap)

        redMatrix = imalg.toChannel(self.imgMatrixAdjusted, 0)
        redPixmap = imalg.pixmapFromMatrix(redMatrix)
        self.setImagePixmap(self.redImageLabel, redPixmap)

        greenMatrix = imalg.toChannel(self.imgMatrixAdjusted, 1)
        greenPixmap = imalg.pixmapFromMatrix(greenMatrix)
        self.setImagePixmap(self.greenImageLabel, greenPixmap)

        blueMatrix = imalg.toChannel(self.imgMatrixAdjusted, 2)
        bluePixmap = imalg.pixmapFromMatrix(blueMatrix)
        self.setImagePixmap(self.blueImageLabel, bluePixmap)

        redHistPixmap = imalg.getChannelHistPixmap(self.imgMatrixAdjusted, 0, 'red')
        self.setImagePixmap(self.redHistLabel, redHistPixmap)

        greenHistPixmap = imalg.getChannelHistPixmap(self.imgMatrixAdjusted, 1, 'green')
        self.setImagePixmap(self.greenHistLabel, greenHistPixmap)

        blueHistPixmap = imalg.getChannelHistPixmap(self.imgMatrixAdjusted, 2, 'blue')
        self.setImagePixmap(self.blueHistLabel, blueHistPixmap)

    def loadImage(self):
        dial = QFileDialog(self)
        dial.setAcceptMode(QFileDialog.AcceptMode.AcceptOpen)
        dial.setFileMode(QFileDialog.FileMode.ExistingFile)
        dial.setNameFilter("Изображение (*.bmp *.png *.tiff)")
        dial.setViewMode(QFileDialog.ViewMode.List)

        if dial.exec():
            file = dial.selectedFiles()[0]
            self.imgPath = file
            self.imgMatrix = imalg.loadImagePixmap(file)
            self.imgMatrixAdjusted = self.imgMatrix
            imgPixmap = imalg.pixmapFromMatrix(self.imgMatrixAdjusted)
            self.setImagePixmap(self.mainImageLabel, imgPixmap)
            self.mainImageLabel.setImageMatrix(self.imgMatrixAdjusted)

            self.updateImageInfo()
            self.updateInfoImages()

            self.statusbar.showMessage(f"Изображение загружено: {file}")

    def onBrightnessChanged(self, value):
        self.brightnessValue = value
        self.adjustImage()

    def onRedChanged(self, value):
        self.redValue = value
        self.adjustImage()

    def onGreenChanged(self, value):
        self.greenValue = value
        self.adjustImage()

    def onBlueChanged(self, value):
        self.blueValue = value
        self.adjustImage()

    def onContrastChanged(self, value):
        self.contrastValue = value
        self.adjustImage()

    def adjustImage(self):
        if self.imgMatrix is None:
            return

        self.imgMatrixAdjusted = imalg.applyContrast(self.imgMatrix, self.contrastValue)
        self.imgMatrixAdjusted = imalg.applyBrightness(self.imgMatrixAdjusted, self.brightnessValue)
        self.imgMatrixAdjusted = imalg.applyChannelAdjustment(
            self.imgMatrixAdjusted, self.redValue, self.greenValue, self.blueValue
        )

        imgPixmap = imalg.pixmapFromMatrix(self.imgMatrixAdjusted)
        self.setImagePixmap(self.mainImageLabel, imgPixmap)

    def onSliderReleased(self):
        self.updateInfoImages()


if __name__ == '__main__':
    import sys
    app = QApplication(sys.argv)
    wnd = MainWindow()
    wnd.show()
    sys.exit(app.exec())
