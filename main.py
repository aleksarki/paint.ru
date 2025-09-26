from PyQt6.QtCore import Qt
from PyQt6.QtGui import QPixmap, QImage
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

        self.loadImageAction.triggered.connect(self.loadImage)
        self.exitApplicationAction.triggered.connect(self.destroy)
        self.exitApplicationAction.setShortcut('Alt+F4')
        self.statusbar.showMessage("Готово!")

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
            self.imgMatrix = imalg.loadImage(file)
            self.imgPixmap = imalg.pixmapFromMatrix(self.imgMatrix)
            self.setImagePixmap(self.mainImageLabel, self.imgPixmap)

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

            redHistPixmap = imalg.getChannelHist(self.imgMatrix, 0, 'red')
            self.setImagePixmap(self.redHistLabel, redHistPixmap)

            greenHistPixmap = imalg.getChannelHist(self.imgMatrix, 1, 'green')
            self.setImagePixmap(self.greenHistLabel, greenHistPixmap)

            blueHistPixmap = imalg.getChannelHist(self.imgMatrix, 2, 'blue')
            self.setImagePixmap(self.blueHistLabel, blueHistPixmap)



            self.statusbar.showMessage(f"Изображение загружено: {file}")


if __name__ == '__main__':
    import sys
    app = QApplication(sys.argv)
    wnd = MainWindow()
    wnd.show()
    sys.exit(app.exec())
