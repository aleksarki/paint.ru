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
        self.imgPixmap = None
        self.imgMatrix = None

        self.mainImageLabel.mouseMoved.connect(self.updatePointerInfo)
        self.updatePointerInfo(*[None] * 8)
        self.updateImageInfo()

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
            self.imgPixmap = imalg.pixmapFromMatrix(self.imgMatrix)
            self.setImagePixmap(self.mainImageLabel, self.imgPixmap)
            self.mainImageLabel.setImageMatrix(self.imgMatrix)

            self.updateImageInfo()

            grayMatrix = imalg.toGrayscale(self.imgMatrix)
            grayPixmap = imalg.pixmapFromMatrix(grayMatrix)
            self.setImagePixmap(self.blackWhiteImageLabel, grayPixmap)

            redMatrix = imalg.toChannel(self.imgMatrix, 0)
            redPixmap = imalg.pixmapFromMatrix(redMatrix)
            self.setImagePixmap(self.redImageLabel, redPixmap)

            greenMatrix = imalg.toChannel(self.imgMatrix, 1)
            greenPixmap = imalg.pixmapFromMatrix(greenMatrix)
            self.setImagePixmap(self.greenImageLabel, greenPixmap)

            blueMatrix = imalg.toChannel(self.imgMatrix, 2)
            bluePixmap = imalg.pixmapFromMatrix(blueMatrix)
            self.setImagePixmap(self.blueImageLabel, bluePixmap)

            redHistPixmap = imalg.getChannelHistPixmap(self.imgMatrix, 0, 'red')
            self.setImagePixmap(self.redHistLabel, redHistPixmap)

            greenHistPixmap = imalg.getChannelHistPixmap(self.imgMatrix, 1, 'green')
            self.setImagePixmap(self.greenHistLabel, greenHistPixmap)

            blueHistPixmap = imalg.getChannelHistPixmap(self.imgMatrix, 2, 'blue')
            self.setImagePixmap(self.blueHistLabel, blueHistPixmap)

            self.statusbar.showMessage(f"Изображение загружено: {file}")


if __name__ == '__main__':
    import sys
    app = QApplication(sys.argv)
    wnd = MainWindow()
    wnd.show()
    sys.exit(app.exec())
