import io

import cv2
import matplotlib.pyplot as plt
import numpy as np
from PyQt6.QtGui import QImage, QPixmap


def loadImagePixmap(path: str) -> np.ndarray:
    """ Load image file, get matrix. """
    matrix = cv2.imread(path)
    assert matrix is not None
    shape = matrix.shape
    assert (len(shape) == 3) and (shape[2] == 3)
    matrix = cv2.cvtColor(matrix, cv2.COLOR_BGR2RGB)
    return matrix


def pixmapFromMatrix(matrix: np.ndarray) -> QPixmap:
    """ Generate QPixmap from image matrix. """
    shape = matrix.shape
    height, width = shape[:2]
    if len(shape) == 2:  # grayscale
        format_ = QImage.Format.Format_Grayscale8
        bytesPerLine = width
    elif len(shape) == 3 and shape[2] == 3:  # RGB
        format_ = QImage.Format.Format_RGB888
        bytesPerLine = width * 3  # width * N channels
    else:
        raise
    image = QImage(matrix.data, width, height, bytesPerLine, format_)
    pixmap = QPixmap.fromImage(image)
    assert not pixmap.isNull()
    return pixmap


def toGrayscale(matrix: np.ndarray) -> np.ndarray:
    assert matrix is not None
    # 0.299 R + 0.587 G + 0.114 B
    grayMatrix = np.dot(matrix[..., :3], [.299, .587, .114]).astype(np.uint8)
    return grayMatrix


def toChannel(matrix: np.ndarray, channel: int) -> np.ndarray:
    assert matrix is not None
    assert len(matrix.shape) == 3
    matrix = matrix.copy()
    for i in range(matrix.shape[2]):
        if i != channel:
            matrix[..., i] *= 0
    return matrix


def getChannelHistPixmap(matrix: np.ndarray, channel: int, color: str) -> QPixmap:
    data = matrix[..., channel].flatten()
    buf = io.BytesIO()

    plt.figure(figsize=(4, 3), dpi=80)
    plt.hist(data, bins=128, range=(0, 255), color=color)
    plt.axis('off')
    plt.xlim(0, 255)
    plt.savefig(buf, format='png', dpi=80, bbox_inches='tight')
    plt.close()

    buf.seek(0)
    image = QImage()
    image.loadFromData(buf.getvalue())
    pixmap = QPixmap.fromImage(image)
    return pixmap
