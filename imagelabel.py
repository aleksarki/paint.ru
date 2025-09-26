from PyQt6.QtCore import QRect
from PyQt6.QtGui import QPainter, QPen, QColor
from PyQt6.QtWidgets import QLabel


class ImageLabel(QLabel):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.mousePos = None
        self.windowSize = 11
        self.setMouseTracking(True)

    def mouseMoveEvent(self, event):
        self.mousePos = event.pos()
        self.update()
        super().mouseMoveEvent(event)

    def paintEvent(self, event):
        # Сначала рисуем изображение
        super().paintEvent(event)

        # Затем рисуем рамку поверх
        if self.mousePos and self.pixmap() and not self.pixmap().isNull():
            painter = QPainter(self)
            painter.setRenderHint(QPainter.RenderHint.Antialiasing)

            pen = QPen(QColor(255, 0, 0))
            pen.setWidth(1)
            painter.setPen(pen)

            # Размер рамки
            size = 11

            rect = QRect(
                self.mousePos.x() - size // 2,
                self.mousePos.y() - size // 2,
                size,
                size
            )

            painter.drawRect(rect)
