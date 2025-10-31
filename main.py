from PyQt6.QtCore import Qt
from PyQt6.QtGui import QPixmap
from PyQt6.QtWidgets import *

from dialogs import ParameterDialog, ThresholdDialog, BrightnessRangeDialog, UnsharpMaskingDialog, ConvolutionDialog
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

        self.brightnessNegationButton.clicked.connect(self.doBrightnessNegation)
        self.pillowNegationButton.clicked.connect(self.doBrightnessNegationPIL)
        self.rNegationButton.clicked.connect(lambda: self.doChannelNegation(0))
        self.gNegationButton.clicked.connect(lambda: self.doChannelNegation(1))
        self.bNegationButton.clicked.connect(lambda: self.doChannelNegation(2))

        self.rgExchangeButton.clicked.connect(lambda: self.exchangeChannels(0, 1))
        self.gbExchangeButton.clicked.connect(lambda: self.exchangeChannels(1, 2))
        self.rbExchangeButton.clicked.connect(lambda: self.exchangeChannels(0, 2))

        self.horisontalMirrorButton.clicked.connect(lambda: self.doMirror(0))
        self.verticalMirrorButton.clicked.connect(lambda: self.doMirror(1))
        self.neighbourAverageButton.clicked.connect(self.doNeighbourAverage)

        self.logatithmTransformAction.triggered.connect(self.doLogarithmicTransform)
        self.exponentTransformAction.triggered.connect(self.doPowerTransform)
        self.binaryTransformAction.triggered.connect(self.doBinaryTransform)
        self.brightnessRangeAction.triggered.connect(self.doBrightnessRangeCut)
        self.blurredMaskingAction.triggered.connect(self.doUnsharpMasking)
        

        self.loadImageAction.triggered.connect(self.loadImage)
        self.saveImageAction.triggered.connect(self.saveImage)
        self.exitApplicationAction.triggered.connect(self.destroy)
        self.exitApplicationAction.setShortcut('Alt+F4')

        # --- Сглаживание ---
        self.rectangleFilterAction.triggered.connect(self.doMeanFilter)
        self.medianFilterAction.triggered.connect(self.doMedianFilter)
        self.gaussFilterAction.triggered.connect(self.doGaussianFilter)
        self.sigmaFilterAction.triggered.connect(self.doSigmaFilter)
        self.customFilterAction.triggered.connect(self.doConvolution)
        #self.absoluteDiffAction.triggered.connect(self.doAbsDiffMap)

        # --- Высокие частоты ---
        self.highMeanAction.triggered.connect(self.doHighPassMean)
        self.highGaussianAction.triggered.connect(self.doHighPassGaussian)

        self.cornerDetectionAction.triggered.connect(self.doCornerDetection)
        self.cornerHarrisAction.triggered.connect(self.doCornerHarris)
        self.sobelEdgesAction.triggered.connect(self.doSobelEdgeDetection)

        self.statusbar.showMessage("Готово!")

    def updateImageInfo(self):
        """ Update info labels about image dimensions. """
        if self.imgMatrix is not None:
            height, width, channels = self.imgMatrix.shape
            self.statLabel.setText(f"Размер: {height}×{width}×{channels}")
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
            self.imgMatrix = imalg.loadImageMatrix(file)
            self.imgMatrixAdjusted = self.imgMatrix
            imgPixmap = imalg.pixmapFromMatrix(self.imgMatrixAdjusted)
            self.setImagePixmap(self.mainImageLabel, imgPixmap)
            self.mainImageLabel.setImageMatrix(self.imgMatrixAdjusted)

            self.updateImageInfo()
            self.updateInfoImages()

            self.statusbar.showMessage(f"Изображение загружено: {file}")

    def saveImage(self):
        if self.imgMatrix is None:
            self.statusbar.showMessage("Нет открытого изображения")
            return

        path, selectedFilter = QFileDialog.getSaveFileName(
            self,
            "Сохранить изображение",
            "",  # folder
            "PNG Images (*.png)"
        )
        if not path:
            return

        if not path.lower().endswith('.png'):
            path += '.png'

        success = imalg.saveImageMatrix(path, self.imgMatrixAdjusted)
        if success:
            self.statusbar.showMessage(f"Сохранено изображение {path}")
        else:
            self.statusbar.showMessage("Ошибка сохранения")

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
        self.imgMatrixAdjusted = self.matrixAdjustmentSequence(self.imgMatrix)
        imgPixmap = imalg.pixmapFromMatrix(self.imgMatrixAdjusted)
        self.setImagePixmap(self.mainImageLabel, imgPixmap)

    def matrixAdjustmentSequence(self, matrix):
        adjusted = imalg.applyContrast(matrix, self.contrastValue)
        adjusted = imalg.applyBrightness(adjusted, self.brightnessValue)
        adjusted = imalg.applyChannelAdjustment(
            adjusted, self.redValue, self.greenValue, self.blueValue
        )
        return adjusted

    def onSliderReleased(self):
        self.updateInfoImages()

    def doBrightnessNegation(self):
        if self.imgMatrix is None:
            self.statusbar.showMessage("Нет открытого изображения")
            return
        self.imgMatrix = imalg.applyRgbNegation(self.imgMatrix)
        self.imgMatrixAdjusted = self.matrixAdjustmentSequence(self.imgMatrix)
        imgPixmap = imalg.pixmapFromMatrix(self.imgMatrixAdjusted)
        self.setImagePixmap(self.mainImageLabel, imgPixmap)
        self.updateInfoImages()
        self.statusbar.showMessage("Негатив")

    def doBrightnessNegationPIL(self):
        if self.imgMatrix is None:
            self.statusbar.showMessage("Нет открытого изображения")
            return
        self.imgMatrix = imalg.applyRgbNegation(self.imgMatrix)
        self.imgMatrixAdjusted = self.matrixAdjustmentSequence(self.imgMatrix)
        imgPixmap = imalg.pixmapFromMatrix(self.imgMatrixAdjusted)
        self.setImagePixmap(self.mainImageLabel, imgPixmap)
        self.updateInfoImages()
        self.statusbar.showMessage("Негатив (Pillow)")

    def doChannelNegation(self, channel):
        if self.imgMatrix is None:
            self.statusbar.showMessage("Нет открытого изображения")
            return
        self.imgMatrix = imalg.applyChannelNegation(self.imgMatrix, channel)
        self.imgMatrixAdjusted = self.matrixAdjustmentSequence(self.imgMatrix)
        imgPixmap = imalg.pixmapFromMatrix(self.imgMatrixAdjusted)
        self.setImagePixmap(self.mainImageLabel, imgPixmap)
        self.updateInfoImages()
        self.statusbar.showMessage(f"Негатив канала {channel}")

    def exchangeChannels(self, channel1, channel2):
        if self.imgMatrix is None:
            self.statusbar.showMessage("Нет открытого изображения")
            return
        self.imgMatrix = imalg.applyChannelExchange(self.imgMatrix, channel1, channel2)
        self.imgMatrixAdjusted = self.matrixAdjustmentSequence(self.imgMatrix)
        imgPixmap = imalg.pixmapFromMatrix(self.imgMatrixAdjusted)
        self.setImagePixmap(self.mainImageLabel, imgPixmap)
        self.updateInfoImages()
        self.statusbar.showMessage(f"Обмен каналов {channel1}, {channel2}")

    def doMirror(self, axis):
        if self.imgMatrix is None:
            self.statusbar.showMessage("Нет открытого изображения")
            return
        self.imgMatrix = imalg.applyMirror(self.imgMatrix, axis)
        self.imgMatrixAdjusted = self.matrixAdjustmentSequence(self.imgMatrix)
        imgPixmap = imalg.pixmapFromMatrix(self.imgMatrixAdjusted)
        self.setImagePixmap(self.mainImageLabel, imgPixmap)
        self.updateInfoImages()
        self.statusbar.showMessage(f"Отражение по оси {axis}")

    def doNeighbourAverage(self):
        if self.imgMatrix is None:
            self.statusbar.showMessage("Нет открытого изображения")
            return
        self.imgMatrix = imalg.applyNeighbourAverage(self.imgMatrix)
        self.imgMatrixAdjusted = self.matrixAdjustmentSequence(self.imgMatrix)
        imgPixmap = imalg.pixmapFromMatrix(self.imgMatrixAdjusted)
        self.setImagePixmap(self.mainImageLabel, imgPixmap)
        self.updateInfoImages()
        self.statusbar.showMessage("Усреднение соседних пикселей")

    def doLogarithmicTransform(self):
        if self.imgMatrix is None:
            self.statusbar.showMessage("Нет открытого изображения")
            return

        dialog = QMessageBox(self)
        dialog.setWindowTitle("Логарифмическое преобразование")
        dialog.setText("Применить логарифмическое преобразование?")
        dialog.setStandardButtons(QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)

        if dialog.exec() == QMessageBox.StandardButton.Yes:
            self.imgMatrix = imalg.applyLogarithmicTransform(self.imgMatrix)
            # self.imgMatrix = imalg.applyLogarithmicTransform(self.imgMatrixAdjusted)  # !!! dope !!!
            self.imgMatrixAdjusted = self.matrixAdjustmentSequence(self.imgMatrix)
            imgPixmap = imalg.pixmapFromMatrix(self.imgMatrixAdjusted)
            self.setImagePixmap(self.mainImageLabel, imgPixmap)
            self.updateInfoImages()
            self.statusbar.showMessage("Логарифмическое преобразование применено")

    def doPowerTransform(self):
        if self.imgMatrix is None:
            self.statusbar.showMessage("Нет открытого изображения")
            return

        dialog = ParameterDialog(
            "Степенное преобразование",
            "Значение гаммы:",
            default_value=1.0,
            min_val=0.1,
            max_val=5.0,
            step=0.1,
            parent=self
        )
        if dialog.exec() == QDialog.DialogCode.Accepted:
            gamma = dialog.getValue()
            self.imgMatrix = imalg.applyPowerTransform(self.imgMatrix, gamma)
            self.imgMatrixAdjusted = self.matrixAdjustmentSequence(self.imgMatrix)
            imgPixmap = imalg.pixmapFromMatrix(self.imgMatrixAdjusted)
            self.setImagePixmap(self.mainImageLabel, imgPixmap)
            self.updateInfoImages()
            self.statusbar.showMessage(f"Степенное преобразование (γ={gamma:.2f}) применено")

    def doBinaryTransform(self):
        if self.imgMatrix is None:
            self.statusbar.showMessage("Нет открытого изображения")
            return

        dialog = ThresholdDialog(self)
        if dialog.exec() == QDialog.DialogCode.Accepted:
            threshold = dialog.getValue()
            self.imgMatrix = imalg.applyBinaryTransform(self.imgMatrix, threshold)
            self.imgMatrixAdjusted = self.matrixAdjustmentSequence(self.imgMatrix)
            imgPixmap = imalg.pixmapFromMatrix(self.imgMatrixAdjusted)
            self.setImagePixmap(self.mainImageLabel, imgPixmap)
            self.updateInfoImages()
            self.statusbar.showMessage(f"Бинарное преобразование (порог={threshold}) применено")

    def doBrightnessRangeCut(self):
        if self.imgMatrix is None:
            self.statusbar.showMessage("Нет открытого изображения")
            return

        dialog = BrightnessRangeDialog(self)
        if dialog.exec() == QDialog.DialogCode.Accepted:
            params = dialog.getValues()

            self.imgMatrix = imalg.applyBrightnessRangeCut(
                self.imgMatrix,
                params['low_threshold'],
                params['high_threshold'],
                params['constant_value'],
                params['keep_others']
            )
            self.imgMatrixAdjusted = self.matrixAdjustmentSequence(self.imgMatrix)
            imgPixmap = imalg.pixmapFromMatrix(self.imgMatrixAdjusted)
            self.setImagePixmap(self.mainImageLabel, imgPixmap)
            self.updateInfoImages()

            approach = "сохранены" if params['keep_others'] else f"приведены к {params['constant_value']}"
            self.statusbar.showMessage(
                f"Вырезан диапазон [{params['low_threshold']}-{params['high_threshold']}], "
                f"внешние пиксели {approach}"
            )

    # ============================================================
    # === 2. СГЛАЖИВАНИЕ ========================================
    # ============================================================

    def showDifferenceDialog(self, original, filtered):
        """Открыть окно с картой абсолютной разности между двумя изображениями."""
        diff = imalg.absoluteDifference(original, filtered)
        diffPixmap = imalg.pixmapFromMatrix(diff)

        dialog = QDialog(self)
        dialog.setWindowTitle("Карта абсолютной разности")
        dialog.resize(500, 500)

        label = QLabel(dialog)
        label.setPixmap(diffPixmap.scaled(
            480, 480,
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation
        ))
        label.setAlignment(Qt.AlignmentFlag.AlignCenter)

        layout = QVBoxLayout(dialog)
        layout.addWidget(label)
        dialog.setLayout(layout)

        dialog.exec()

    def doMeanFilter(self):
        if self.imgMatrix is None:
            self.statusbar.showMessage("Нет открытого изображения")
            return

        original = self.imgMatrix.copy()
        filtered = imalg.meanFilter(original, k=3)

        # обновляем изображение
        self.imgMatrix = filtered
        self.imgMatrixAdjusted = self.matrixAdjustmentSequence(self.imgMatrix)
        self.setImagePixmap(self.mainImageLabel, imalg.pixmapFromMatrix(self.imgMatrixAdjusted))
        self.updateInfoImages()

        # показываем карту разности
        self.showDifferenceDialog(original, filtered)

        self.statusbar.showMessage("Прямоугольный фильтр (3x3) применён")

    def doMedianFilter(self):
        if self.imgMatrix is None:
            self.statusbar.showMessage("Нет открытого изображения")
            return
        
        original = self.imgMatrix.copy()
        filtered = imalg.medianFilter(original, k=3)

        self.imgMatrix = imalg.medianFilter(self.imgMatrix, k=3)
        self.imgMatrixAdjusted = self.matrixAdjustmentSequence(self.imgMatrix)
        self.setImagePixmap(self.mainImageLabel, imalg.pixmapFromMatrix(self.imgMatrixAdjusted))
        self.updateInfoImages()

        # показываем карту разности
        self.showDifferenceDialog(original, filtered)

        self.statusbar.showMessage("Медианный фильтр (3x3) применён")

    def doGaussianFilter(self):
        if self.imgMatrix is None:
            self.statusbar.showMessage("Нет открытого изображения")
            return
        sigma, ok = QInputDialog.getDouble(self, "Гауссов фильтр", "Введите σ:", 1.5, 0.1, 10.0, 1)
        if not ok:
            return
        
        original = self.imgMatrix.copy()
        filtered = imalg.gaussianFilter(original, sigma)

        self.imgMatrix = imalg.gaussianFilter(self.imgMatrix, sigma)
        self.imgMatrixAdjusted = self.matrixAdjustmentSequence(self.imgMatrix)
        self.setImagePixmap(self.mainImageLabel, imalg.pixmapFromMatrix(self.imgMatrixAdjusted))
        self.updateInfoImages()

        # показываем карту разности
        self.showDifferenceDialog(original, filtered)

        self.statusbar.showMessage(f"Гауссов фильтр применён (σ={sigma})")

    def doSigmaFilter(self):
        if self.imgMatrix is None:
            self.statusbar.showMessage("Нет открытого изображения")
            return
        sigma, ok = QInputDialog.getDouble(self, "Сигма-фильтр", "Введите порог σ:", 20.0, 1.0, 100.0, 1)
        if not ok:
            return
        
        original = self.imgMatrix.copy()
        filtered = imalg.sigmaFilter(original, k=3, sigma_threshold=sigma)
        
        self.imgMatrix = imalg.sigmaFilter(self.imgMatrix, k=3, sigma_threshold=sigma)
        self.imgMatrixAdjusted = self.matrixAdjustmentSequence(self.imgMatrix)
        self.setImagePixmap(self.mainImageLabel, imalg.pixmapFromMatrix(self.imgMatrixAdjusted))
        self.updateInfoImages()

        # показываем карту разности
        self.showDifferenceDialog(original, filtered)

        self.statusbar.showMessage(f"Сигма-фильтр применён (σ={sigma})")
    """ 
    def doAbsDiffMap(self):
        if self.imgMatrix is None or self.imgMatrixAdjusted is None:
            self.statusbar.showMessage("Нет изображения для сравнения")
            return
        diff = imalg.absoluteDifference(self.imgMatrix, self.imgMatrixAdjusted)
        diffPixmap = imalg.pixmapFromMatrix(diff)
        self.setImagePixmap(self.mainImageLabel, diffPixmap)
        self.statusbar.showMessage("Показана карта абсолютной разности")
    """

    # Резкость

    def doUnsharpMasking(self):
        """ Apply unsharp masking for sharpness enhancement """
        if self.imgMatrix is None:
            self.statusbar.showMessage("Нет открытого изображения")
            return

        dialog = UnsharpMaskingDialog(self)
        if dialog.exec() == QDialog.DialogCode.Accepted:
            params = dialog.getValues()

            # Apply unsharp masking
            self.imgMatrix = imalg.applyUnsharpMasking(
                self.imgMatrix,
                params['blur_radius'],
                params['strength']
            )

            self.imgMatrixAdjusted = self.matrixAdjustmentSequence(self.imgMatrix)
            imgPixmap = imalg.pixmapFromMatrix(self.imgMatrixAdjusted)
            self.setImagePixmap(self.mainImageLabel, imgPixmap)
            self.updateInfoImages()

            self.statusbar.showMessage(
                f"Нерезкое маскирование: радиус={params['blur_radius']}, "
                f"сила={params['strength']:.2f}"
            )

    # ============================================================
    # === 3. ВЫСОКОЧАСТОТНАЯ ОБРАБОТКА ==========================
    # ============================================================

    def doHighPassMean(self):
        if self.imgMatrix is None:
            self.statusbar.showMessage("Нет изображения для обработки")
            return

        c, ok = QInputDialog.getDouble(self, "High-pass (усреднение)", "Введите коэффициент c:", 1.0, 0.0, 3.0, 1)
        if not ok:
            return

        original = self.imgMatrix.copy()
        filtered = imalg.highPassFilter(original, method="mean", c=c, k=3)

        self.imgMatrix = filtered
        self.imgMatrixAdjusted = self.matrixAdjustmentSequence(self.imgMatrix)
        self.setImagePixmap(self.mainImageLabel, imalg.pixmapFromMatrix(self.imgMatrixAdjusted))
        self.updateInfoImages()

        self.statusbar.showMessage(f"High-pass (усреднение), c={c}")

    def doHighPassGaussian(self):
        if self.imgMatrix is None:
            self.statusbar.showMessage("Нет изображения для обработки")
            return

        sigma, ok1 = QInputDialog.getDouble(self, "High-pass (Гаусс)", "Введите σ:", 1.0, 0.1, 10.0, 1)
        if not ok1:
            return
        c, ok2 = QInputDialog.getDouble(self, "High-pass (Гаусс)", "Введите коэффициент c:", 1.0, 0.0, 3.0, 1)
        if not ok2:
            return

        original = self.imgMatrix.copy()
        filtered = imalg.highPassFilter(original, method="gaussian", sigma=sigma, c=c)

        self.imgMatrix = filtered
        self.imgMatrixAdjusted = self.matrixAdjustmentSequence(self.imgMatrix)
        self.setImagePixmap(self.mainImageLabel, imalg.pixmapFromMatrix(self.imgMatrixAdjusted))
        self.updateInfoImages()

        self.statusbar.showMessage(f"High-pass (Гаусс), σ={sigma}, c={c}")

    def doCornerDetection(self):
        if self.imgMatrix is None:
            self.statusbar.showMessage("Нет изображения для обработки")
            return

        threshold, ok = QInputDialog.getDouble(self, "Выделение углов", "Введите порог:", 10000.0, 1000.0, 100000.0, 0)
        if not ok:
            return

        original = self.imgMatrix.copy()
        filtered = imalg.cornerDetectionHessian(original, threshold=threshold)

        self.imgMatrix = filtered
        self.imgMatrixAdjusted = self.matrixAdjustmentSequence(self.imgMatrix)
        self.setImagePixmap(self.mainImageLabel, imalg.pixmapFromMatrix(self.imgMatrixAdjusted))
        self.updateInfoImages()

        self.statusbar.showMessage(f"Выделение углов (порог={threshold})")

    def doCornerHarris(self):
        if self.imgMatrix is None:
            self.statusbar.showMessage("Нет изображения для обработки")
            return

        k, ok1 = QInputDialog.getDouble(self, "Выделение углов (Харрис)", "Введите параметр k (0.04–0.06):", 0.05, 0.01, 0.2, 2)
        if not ok1:
            return

        threshold, ok2 = QInputDialog.getDouble(self, "Выделение углов (Харрис)", "Введите порог:", 1e6, 1e4, 1e8, 0)
        if not ok2:
            return

        original = self.imgMatrix.copy()
        filtered = imalg.cornerDetectionHarris(original, k=k, threshold=threshold)

        self.imgMatrix = filtered
        self.imgMatrixAdjusted = self.matrixAdjustmentSequence(self.imgMatrix)
        self.setImagePixmap(self.mainImageLabel, imalg.pixmapFromMatrix(self.imgMatrixAdjusted))
        self.updateInfoImages()

        self.statusbar.showMessage(f"Выделение углов (Харрис), k={k}, threshold={threshold}")



    def doConvolution(self):
        """Применить произвольную свёртку"""
        if self.imgMatrix is None:
            self.statusbar.showMessage("Нет открытого изображения")
            return

        dialog = ConvolutionDialog(self)
        if dialog.exec() == QDialog.DialogCode.Accepted:
            kernel = dialog.get_matrix()
            options = dialog.get_options()

            # Применить свёртку
            filtered = imalg.applyConvolution(
                self.imgMatrix,
                kernel,
                options['normalize'],
                options['add_128'],
                options['abs_value']
            )

            # Обновить изображение
            original = self.imgMatrix.copy()
            self.imgMatrix = filtered
            self.imgMatrixAdjusted = self.matrixAdjustmentSequence(self.imgMatrix)
            self.setImagePixmap(self.mainImageLabel, imalg.pixmapFromMatrix(self.imgMatrixAdjusted))
            self.updateInfoImages()

            # Показать карту разности
            # self.showDifferenceDialog(original, filtered)

            self.statusbar.showMessage("Свёртка применена")

    def doSobelEdgeDetection(self):
        """Выделение границ оператором Собеля"""
        if self.imgMatrix is None:
            self.statusbar.showMessage("Нет открытого изображения")
            return

        # Применить оператор Собеля
        edges = imalg.applySobelEdgeDetection(self.imgMatrix)

        # Показать результат
        self.imgMatrixAdjusted = edges
        self.setImagePixmap(self.mainImageLabel, imalg.pixmapFromMatrix(self.imgMatrixAdjusted))
        self.updateInfoImages()

        self.statusbar.showMessage("Границы выделены оператором Собеля")


if __name__ == '__main__':
    import sys

    app = QApplication(sys.argv)
    wnd = MainWindow()
    wnd.show()
    sys.exit(app.exec())
