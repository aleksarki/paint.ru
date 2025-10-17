# imagealgorithm.py - добавляем новые функции

import io
import cv2
import matplotlib.pyplot as plt
import numpy as np
import PIL
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
    grayMatrix = np.dot(matrix[..., :], [.2125, .7154, .0721]).astype(np.uint8)  # 0.2125 R + 0.7154 G + 0.0721 B
    return grayMatrix


def toChannel(matrix: np.ndarray, channel: int) -> np.ndarray:
    """ Return specific channel. """
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

    gamma = contrast / 1  # [0; 100] -> [0; 4]
    normalized = matrix.astype(np.float32) / 255  # [0; 255] -> [0; 1]

    adjusted = 1 / (1 + np.exp(gamma * (.5 - normalized)))  # sigmoid
    blend = gamma / 100  # [0; 1]
    result = blend * adjusted + (1 - blend) * normalized  # blending

    result = (result * 255).astype(np.uint8)  # [0; 1] -> [0; 255]
    return result


def applyRgbNegation(matrix: np.ndarray) -> np.ndarray:
    """ Negation of all channels. """
    return 255 - matrix


def applyRgbNegationPIL(matrix: np.ndarray) -> np.ndarray:
    image = PIL.Image.fromarray(matrix)
    negated = PIL.ImageOps.invert(image)
    result = np.array(negated)
    return result


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


def applyLogarithmicTransform(matrix: np.ndarray) -> np.ndarray:
    float_matrix = matrix.astype(np.float32)
    # Calculate coefficient c to map to [0; 255]
    max_vals = np.max(float_matrix, axis=(0, 1))  # for each channel
    c = 255 / np.log(1 + max_vals)
    transformed = c * np.log(1 + float_matrix)
    result = np.clip(transformed, 0, 255).astype(np.uint8)
    return result


def applyPowerTransform(matrix: np.ndarray, gamma: float) -> np.ndarray:
    float_matrix = matrix.astype(np.float32) / 255  # [0; 255] -> [0; 1]
    transformed = np.power(float_matrix, gamma)
    # Calculate coefficient c to map to [0; 255]
    max_vals = np.max(transformed, axis=(0, 1))  # for each channel
    c = 255 / np.maximum(max_vals, 1e-6)
    result = (c * transformed).astype(np.uint8)  # [0; 1] -> [0; 255]
    return result


def applyBinaryTransform(matrix: np.ndarray, threshold: int) -> np.ndarray:
    gray_matrix = toGrayscale(matrix)
    binary_matrix = np.where(gray_matrix >= threshold, 255, 0).astype(np.uint8)
    result_rgb = np.stack([binary_matrix, binary_matrix, binary_matrix], axis=2)
    return result_rgb


def applyBrightnessRangeCut(
    matrix: np.ndarray, low_threshold: int, high_threshold: int,
    constant_value: int = 0, keep_others: bool = True
) -> np.ndarray:
    """
    Cut brightness range for color image.

    Args:
        matrix: input RGB image matrix
        low_threshold: lower brightness threshold (0-255)
        high_threshold: upper brightness threshold (0-255)
        constant_value: value to set for pixels outside range when keep_others=False
        keep_others: if True - keep original values for pixels outside range
                     if False - set constant_value for pixels outside range
        use_intensity: if True - use pixel intensity (R+G+B)/3 for range check
                      if False - check each channel independently

    Returns:
        processed image matrix
    """
    result = matrix.copy()

    intensity = np.mean(matrix, axis=2).astype(np.uint8)
    range_mask = (intensity >= low_threshold) & (intensity <= high_threshold)

    result[range_mask] = 0

    if not keep_others:
        result[~range_mask] = constant_value

    return result.astype(np.uint8)


# ============================================================
# === 2. СГЛАЖИВАНИЕ (реализация без math) ==================
# ============================================================


def _pad_reflect(channel: np.ndarray, pad: int) -> np.ndarray:
    """Зеркальное дополнение границ (без сторонних библиотек)."""
    return np.pad(channel, pad, mode='reflect')


def meanFilter(matrix: np.ndarray, k: int = 3) -> np.ndarray:
    """2.1 Прямоугольный (усредняющий) фильтр."""
    if k not in (3, 5):
        raise ValueError("Размер ядра должен быть 3 или 5")

    h, w, c = matrix.shape
    pad = k // 2
    result = np.zeros_like(matrix, dtype=np.float32)

    for ch in range(c):
        padded = _pad_reflect(matrix[:, :, ch], pad)
        for y in range(h):
            for x in range(w):
                region = padded[y:y + k, x:x + k]
                result[y, x, ch] = np.mean(region)

    return np.clip(result, 0, 255).astype(np.uint8)


def medianFilter(matrix: np.ndarray, k: int = 3) -> np.ndarray:
    """2.1 Медианный фильтр."""
    if k not in (3, 5):
        raise ValueError("Размер ядра должен быть 3 или 5")

    h, w, c = matrix.shape
    pad = k // 2
    result = np.zeros_like(matrix, dtype=np.float32)

    for ch in range(c):
        padded = _pad_reflect(matrix[:, :, ch], pad)
        for y in range(h):
            for x in range(w):
                region = padded[y:y + k, x:x + k]
                result[y, x, ch] = np.median(region)

    return np.clip(result, 0, 255).astype(np.uint8)


def _gaussian_kernel(sigma: float) -> np.ndarray:
    """Построение гауссова ядра по правилу 3σ (без math)."""
    radius = int(np.ceil(3 * sigma))
    size = 2 * radius + 1
    y, x = np.mgrid[-radius:radius + 1, -radius:radius + 1]
    kernel = np.exp(-(x ** 2 + y ** 2) / (2 * sigma ** 2))
    kernel /= np.sum(kernel)
    return kernel


def gaussianFilter(matrix: np.ndarray, sigma: float) -> np.ndarray:
    """2.2 Гауссов фильтр."""
    kernel = _gaussian_kernel(sigma)
    k = kernel.shape[0]
    pad = k // 2
    h, w, c = matrix.shape
    result = np.zeros_like(matrix, dtype=np.float32)

    for ch in range(c):
        padded = _pad_reflect(matrix[:, :, ch], pad)
        for y in range(h):
            for x in range(w):
                region = padded[y:y + k, x:x + k]
                result[y, x, ch] = np.sum(region * kernel)

    return np.clip(result, 0, 255).astype(np.uint8)


def sigmaFilter(matrix: np.ndarray, k: int = 3, sigma_threshold: float = 20) -> np.ndarray:
    """
    2.3 Сигма-фильтр:
    усредняем только те пиксели в окне, которые отличаются от центрального
    не более чем на sigma_threshold.
    """
    if k not in (3, 5):
        raise ValueError("Размер ядра должен быть 3 или 5")

    h, w, c = matrix.shape
    pad = k // 2
    result = np.zeros_like(matrix, dtype=np.float32)

    for ch in range(c):
        padded = _pad_reflect(matrix[:, :, ch], pad)
        for y in range(h):
            for x in range(w):
                center = padded[y + pad, x + pad]
                region = padded[y:y + k, x:x + k]
                mask = np.abs(region - center) <= sigma_threshold
                values = region[mask]
                if values.size > 0:
                    result[y, x, ch] = np.mean(values)
                else:
                    result[y, x, ch] = center

    return np.clip(result, 0, 255).astype(np.uint8)


def absoluteDifference(matrix1: np.ndarray, matrix2: np.ndarray) -> np.ndarray:
    """2.4 Карта абсолютной разности."""
    diff = np.abs(matrix1.astype(np.float32) - matrix2.astype(np.float32))
    return np.clip(diff, 0, 255).astype(np.uint8)
