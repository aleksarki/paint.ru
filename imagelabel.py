import numpy as np
from PyQt6.QtCore import QRect, pyqtSignal, Qt
from PyQt6.QtGui import QPainter, QPen, QColor
from PyQt6.QtWidgets import QLabel


class ImageLabel(QLabel):
    # signal to transfer pixel coords, rgb, & other
    # x, y, r, g, b, intensity, window avg, window std
    mouseMoved = pyqtSignal(int, int, int, int, int, float, float, float)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.mousePos = None
        self.windowSize = 11
        self.imageMatrix = None
        self.setMouseTracking(True)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)

    def setImageMatrix(self, matrix):
        self.imageMatrix = matrix

    def mouseMoveEvent(self, event):
        self.mousePos = event.pos()
        self.update()

        # if there is image
        if self.imageMatrix is not None and self.pixmap() and not self.pixmap().isNull():
            img_x, img_y = self.getImageCoordinates(event.pos())  # mouse coords to image coords
            if img_x is not None and img_y is not None:
                r, g, b = self.imageMatrix[img_y, img_x]
                intensity = (r + g + b) / 3.0
                window_avg, window_std = self.calculateWindowStatistics(img_x, img_y)
                self.mouseMoved.emit(img_x, img_y, r, g, b, intensity, window_avg, window_std)  # send signal

        super().mouseMoveEvent(event)

    def getImageCoordinates(self, mouse_pos):
        """ Convert mouse coordinates to image coordinates. """
        if not self.pixmap() or self.pixmap().isNull():
            return None, None

        pixmap = self.pixmap()
        label_size = self.size()
        pixmap_size = pixmap.size()

        # Вычисляем область отображения pixmap (центрирован)
        pixmap_rect = QRect(
            (label_size.width() - pixmap_size.width()) // 2,
            (label_size.height() - pixmap_size.height()) // 2,
            pixmap_size.width(),
            pixmap_size.height()
        )

        # Проверяем, находится ли мышь в области изображения
        if pixmap_rect.contains(mouse_pos):
            # Преобразуем координаты мыши в координаты изображения
            img_x = int((mouse_pos.x() - pixmap_rect.x()) * self.imageMatrix.shape[1] / pixmap_rect.width())
            img_y = int((mouse_pos.y() - pixmap_rect.y()) * self.imageMatrix.shape[0] / pixmap_rect.height())

            # Ограничиваем координаты размерами изображения
            img_x = max(0, min(img_x, self.imageMatrix.shape[1] - 1))
            img_y = max(0, min(img_y, self.imageMatrix.shape[0] - 1))

            return img_x, img_y

        return None, None

    def calculateWindowStatistics(self, center_x, center_y):
        """ Calculate avg and std for 11x11 window. """
        if self.imageMatrix is None:
            return .0, .0

        half_size = self.windowSize // 2

        # Вычисляем границы окна с проверкой выхода за границы
        x_start = max(0, center_x - half_size)
        y_start = max(0, center_y - half_size)
        x_end = min(self.imageMatrix.shape[1], center_x + half_size + 1)
        y_end = min(self.imageMatrix.shape[0], center_y + half_size + 1)

        # Извлекаем окно
        window = self.imageMatrix[y_start:y_end, x_start:x_start + (x_end - x_start)]

        if window.size == 0:
            return .0, .0

        # усреднение для вычисления статистики
        window_gray = np.mean(window, axis=2)

        window_avg = np.mean(window_gray)
        window_std = np.std(window_gray)

        return float(window_avg), float(window_std)

    def leaveEvent(self, event):
        self.mousePos = None
        self.update()
        super().leaveEvent(event)

    def paintEvent(self, event):
        super().paintEvent(event)  # draw first

        if self.mousePos and self.pixmap() and not self.pixmap().isNull():  # then draw window
            painter = QPainter(self)
            painter.setRenderHint(QPainter.RenderHint.Antialiasing)

            pen = QPen(QColor(255, 0, 0))
            pen.setWidth(1)
            painter.setPen(pen)

            rect = QRect(
                self.mousePos.x() - self.windowSize // 2,
                self.mousePos.y() - self.windowSize // 2,
                self.windowSize,
                self.windowSize
            )

            painter.drawRect(rect)
