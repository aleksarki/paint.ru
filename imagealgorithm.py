
import io
import cv2
import matplotlib.pyplot as plt
import numpy as np
from PyQt6.QtGui import QImage, QPixmap


def loadImageMatrix(path: str) -> np.ndarray:
    """ Load image file, get matrix. """
    matrix = cv2.imread(path)
    assert matrix is not None
    shape = matrix.shape
    assert (len(shape) == 3) and (shape[2] == 3)
    matrix = cv2.cvtColor(matrix, cv2.COLOR_BGR2RGB)
    return matrix


def saveImageMatrix(path: str, matrix: np.ndarray) -> bool:
    """ Save image matrix as png. """
    try:
        shape = matrix.shape
        assert (len(shape) == 3) and (shape[2] == 3)
        matrix_bgr = cv2.cvtColor(matrix, cv2.COLOR_RGB2BGR)
        return cv2.imwrite(path, matrix_bgr, [cv2.IMWRITE_PNG_COMPRESSION, 0])
    except:
        return False


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


def applyBrightness(matrix: np.ndarray, brightness) -> np.ndarray:
    """ Gamma-correction of brightness. """
    if brightness == 0:
        return matrix.copy()

    gamma = 1 - (brightness / 100)  # [-100; 100] -> [2; 0]

    normalized = matrix.astype(np.float32) / 255  # [0; 255] -> [0; 1]
    adjusted = np.power(normalized, gamma)  # gamma correction
    result = (adjusted * 255).astype(np.uint8)  # [0; 1] -> [0; 255]

    return result


def applyChannelAdjustment(matrix: np.ndarray, red, green, blue) -> np.ndarray:
    """ Add constants to the channels. """
    if red == 0 and green == 0 and blue == 0:
        return matrix

    result = matrix.astype(np.int16)  # prevent overflow
    for channel in range(3):
        result[:, :, channel] += (red, green, blue)[channel]  # addition of constant

    result = np.clip(result, 0, 255).astype(np.uint8)  # restrict range
    return result


def applyContrast(matrix: np.ndarray, contrast) -> np.ndarray:
    """ Change contrast. """
    if contrast == 0:
        return matrix

    gamma = contrast / 25  # [0; 100] -> [0; 4]
    normalized = matrix.astype(np.float32) / 255  # [0; 255] -> [0; 1]

    adjusted = 1 / (1 + np.exp(gamma * (.5 - normalized)))  # sigmoid

    blend = gamma / 4  # [0; 1]
    result = blend * adjusted + (1 - blend) * normalized  # blending

    result = (result * 255).astype(np.uint8)  # [0; 1] -> [0; 255]
    return result


def applyRgbNegation(matrix: np.ndarray) -> np.ndarray:
    """ Negation of all channels. """
    return 255 - matrix


def applyChannelNegation(matrix: np.ndarray, channel: int) -> np.ndarray:
    """ Negate specific channel. """
    result = matrix.copy()
    result[:, :, channel] = 255 - result[:, :, channel]
    return result


def applyChannelExchange(matrix: np.ndarray, channel1: int, channel2: int) -> np.ndarray:
    result = matrix.copy()
    result[:, :, channel1] = matrix[:, :, channel2]
    result[:, :, channel2] = matrix[:, :, channel1]
    return result


def applyMirror(matrix: np.ndarray, axis: int) -> np.ndarray:
    if axis == 0:
        return matrix[::-1]
    elif axis == 1:
        return matrix[:, ::-1]


def applyNeighbourAverage(matrix: np.ndarray) -> np.ndarray:
    result = np.zeros_like(matrix.astype(np.int16), dtype=np.uint16)
    height, width, channels = matrix.shape
    for channel in range(channels):
        for y in range(1, height - 1):
            for x in range(1, width - 1):  # besides border pixels
                print(y, x, channel)
                total = 0
                for dy in -1, 0, 1:
                    for dx in -1, 0, 1:
                        total += matrix[y + dy, x + dx, channel] // 9
                result[y, x, channel] = total
    # process borders
    result[0, :, :] = matrix[0, :, :]
    result[height - 1, :, :] = matrix[height - 1, :, :]
    result[:, 0, :] = matrix[:, 0, :]
    result[:, width - 1, :] = matrix[:, width - 1, :]
    return result.astype(np.uint8)
