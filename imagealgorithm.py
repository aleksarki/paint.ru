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
    result = matrix.copy()

    intensity = np.mean(matrix, axis=2).astype(np.uint8)
    range_mask = (intensity >= low_threshold) & (intensity <= high_threshold)

    result[range_mask] = 0

    if not keep_others:
        result[~range_mask] = constant_value

    return result.astype(np.uint8)


# ============================================================
# === 2. СГЛАЖИВАНИЕ =========================================
# ============================================================

def _pad_reflect(channel: np.ndarray, pad: int) -> np.ndarray:
    """Зеркальное дополнение границ (без сторонних библиотек)."""
    return np.pad(channel, pad, mode='reflect')
    # np.pad() — добавляет «рамку» вокруг изображения
    # mode='reflect' — отражает края зеркально, чтобы при фильтрации не выходить за границы


def meanFilter(matrix: np.ndarray, k: int = 3) -> np.ndarray:
    """2.1 Прямоугольный (усредняющий) фильтр."""
    if k not in (3, 5):
        raise ValueError("Размер ядра должен быть 3 или 5")
    # Проверка на допустимый размер окна.

    h, w, c = matrix.shape # Извлекаем высоту (h), ширину (w) и количество каналов (c = 3 для RGB)
    pad = k // 2 # Определяем ширину рамки для отражения. Для k=3 это 1 пиксель с каждой стороны
    result = np.zeros_like(matrix, dtype=np.float32) 
    # Создаём пустое изображение (result) для результата фильтрации, тип float32, чтобы избежать потерь при усреднении

    for ch in range(c):
        padded = _pad_reflect(matrix[:, :, ch], pad)
        """
        : — взять все строки (высоту)
        : — взять все столбцы (ширину)
        ch — конкретный номер канала (0 = R, 1 = G, 2 = B)
        """
        for y in range(h):
            for x in range(w):
                region = padded[y:y + k, x:x + k]
                """
                y:y + k — диапазон строк: от y до y + k - 1
                x:x + k — диапазон столбцов: от x до x + k - 1
                """
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
    """Построение гауссова ядра по правилу 3σ."""
    radius = int(np.ceil(3 * sigma)) # np.ceil (округление вверх) делает радиус целым числом
    y, x = np.mgrid[-radius:radius + 1, -radius:radius + 1] # np.mgrid создаёт две координатные матрицы — X и Y
    kernel = np.exp(-(x ** 2 + y ** 2) / (2 * sigma ** 2)) # Формула Гаусса
    kernel /= np.sum(kernel) # Нормализация ядра
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
                result[y, x, ch] = np.sum(region * kernel) # суммирование всех произведений

    """
    np.clip обрезает значения (если получились >255 или <0)
    .astype(np.uint8) переводит обратно в формат изображения (байты 0–255)
    """

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
                center = padded[y + pad, x + pad] # Берём центр окна
                region = padded[y:y + k, x:x + k] # Берём само окно
                mask = np.abs(region - center) <= sigma_threshold 
                """
                np.abs(region - center) — модуль разницы каждого пикселя с центральным;
                <= sigma_threshold — условие “похожести”
                """
                values = region[mask] # Из окна берутся только те значения, где mask == True
                if values.size > 0:
                    result[y, x, ch] = np.mean(values)
                else:
                    result[y, x, ch] = center

    return np.clip(result, 0, 255).astype(np.uint8)


def absoluteDifference(matrix1: np.ndarray, matrix2: np.ndarray) -> np.ndarray:
    """2.4 Карта абсолютной разности."""
    diff = np.abs(matrix1.astype(np.float32) - matrix2.astype(np.float32))
    return np.clip(diff, 0, 255).astype(np.uint8)


## Резкость

def applyUnsharpMasking(matrix: np.ndarray, blur_radius: int = 5, strength: float = 1.) -> np.ndarray:
    # blur radius is odd
    if blur_radius % 2 == 0:
        blur_radius += 1
    float_matrix = matrix.astype(np.float32)
    # blurred = gaussianFilter(matrix, strength).astype(np.float32)  # works slower
    blurred = applyGaussianBlurSeparable(float_matrix, blur_radius)  # works faster
    mask = float_matrix - blurred
    sharpened = float_matrix + strength * mask
    result = np.clip(sharpened, 0, 255).astype(np.uint8)
    return result


def applyGaussianBlurSeparable(matrix: np.ndarray, kernel_size: int) -> np.ndarray:
    """ Gaussian blur using separable convolution for efficiency. """
    sigma = max(0.3 * ((kernel_size - 1) * 0.5 - 1) + 0.8, 0.1)
    kernel = np.zeros(kernel_size)
    center = kernel_size // 2

    # calculate Gaussian values
    for i in range(kernel_size):
        x = i - center
        kernel[i] = np.exp(-(x ** 2) / (2 * sigma ** 2))
    kernel /= np.sum(kernel)  # normalize
    pad_size = kernel_size // 2

    # vertical, pad top and bottom, convolution along rows
    padded = np.pad(matrix, ((pad_size, pad_size), (0, 0), (0, 0)), mode='reflect')
    result1 = np.zeros_like(matrix, dtype=np.float32)
    for i in range(matrix.shape[0]):
        for k in range(kernel_size):
            result1[i] += padded[i + k] * kernel[k]

    # horizontal, pad left and right, convolution along columns
    padded = np.pad(result1, ((0, 0), (pad_size, pad_size), (0, 0)), mode='reflect')
    result2 = np.zeros_like(result1, dtype=np.float32)
    for j in range(result1.shape[1]):
        for k in range(kernel_size):
            result2[:, j] += padded[:, j + k] * kernel[k]

    return result2

def highPassFilter(matrix: np.ndarray, method: str = "mean", c: float = 1.0, sigma: float = 1.0, k: int = 3) -> np.ndarray:
    """
    Примитивный высокочастотный фильтр:
    ВЧ = ИСХ - РАЗМ * c
    method: "mean" или "gaussian"
    """
    if method == "mean":
        blur = meanFilter(matrix, k)
    elif method == "gaussian":
        blur = gaussianFilter(matrix, sigma)
    else:
        raise ValueError("method должен быть 'mean' или 'gaussian'")

    high = matrix.astype(np.float32) - blur.astype(np.float32) * c

    # решаем проблему отрицательных значений
    min_val = high.min()
    if min_val < 0:
        high -= min_val
    high = np.clip(high, 0, 255)

    return high.astype(np.uint8)


def cornerDetectionHessian(matrix: np.ndarray, threshold: float = 1e6) -> np.ndarray:
    """
    Нахождение углов с помощью матрицы Гессе (вторые производные).
    """
    gray = np.mean(matrix, axis=2).astype(np.float32)
    h, w = gray.shape

    # --- Первые производные (операторы Собеля) ---
    Gx_kernel = np.array([[-1, 0, 1],
                          [-2, 0, 2],
                          [-1, 0, 1]], dtype=np.float32)
    Gy_kernel = np.array([[-1, -2, -1],
                          [ 0,  0,  0],
                          [ 1,  2,  1]], dtype=np.float32)

    pad = 1
    padded = np.pad(gray, pad, mode='reflect')

    Ix = np.zeros_like(gray)
    Iy = np.zeros_like(gray)

    for y in range(h):
        for x in range(w):
            region = padded[y:y+3, x:x+3]
            Ix[y, x] = np.sum(region * Gx_kernel)
            Iy[y, x] = np.sum(region * Gy_kernel)

    # --- Вторые производные ---
    Ixx = np.zeros_like(gray)
    Iyy = np.zeros_like(gray)
    Ixy = np.zeros_like(gray)

    pad = 1
    padded_Ix = np.pad(Ix, pad, mode='reflect')
    padded_Iy = np.pad(Iy, pad, mode='reflect')

    # ядро [-1, 0, 1] для приближённой второй производной
    second_kernel = np.array([[-1, 0, 1]], dtype=np.float32)

    for y in range(h):
        for x in range(w):
            # d/dx (Ix) → Ixx
            region_x = padded_Ix[y:y+3, x:x+3]
            Ixx[y, x] = np.sum(region_x * Gx_kernel)

            # d/dy (Iy) → Iyy
            region_y = padded_Iy[y:y+3, x:x+3]
            Iyy[y, x] = np.sum(region_y * Gy_kernel)

            # смешанная производная d²I/dxdy
            Ixy[y, x] = np.sum(region_x * Gy_kernel)

    # --- Детерминант Гессе ---
    detH = Ixx * Iyy - Ixy**2

    # --- Пороговая фильтрация ---
    corner_map = np.zeros_like(gray)
    corner_map[detH > threshold] = 255

    # --- Визуализация ---
    result = matrix.copy()
    result[corner_map > 0] = [255, 0, 0]  # красные точки — углы

    return result.astype(np.uint8)

