from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QDialog, QVBoxLayout, QHBoxLayout, QLabel, QDoubleSpinBox, QPushButton, QSpinBox, \
    QRadioButton, QSlider, QComboBox, QGroupBox, QScrollArea, QWidget, QGridLayout, QCheckBox
import numpy as np


class ParameterDialog(QDialog):
    """ Dialog for parameter input """
    def __init__(self, title, label, default_value=.0, min_val=.0, max_val=10., step=.1, parent=None):
        super().__init__(parent)
        self.setWindowTitle(title)
        self.setModal(True)

        layout = QVBoxLayout(self)

        # Parameter input
        param_layout = QHBoxLayout()
        param_layout.addWidget(QLabel(label))
        self.spinbox = QDoubleSpinBox()
        self.spinbox.setRange(min_val, max_val)
        self.spinbox.setValue(default_value)
        self.spinbox.setSingleStep(step)
        param_layout.addWidget(self.spinbox)
        layout.addLayout(param_layout)

        # Buttons
        button_layout = QHBoxLayout()
        self.ok_button = QPushButton("OK")
        self.cancel_button = QPushButton("Cancel")
        button_layout.addWidget(self.ok_button)
        button_layout.addWidget(self.cancel_button)
        layout.addLayout(button_layout)

        # Connections
        self.ok_button.clicked.connect(self.accept)
        self.cancel_button.clicked.connect(self.reject)

    def getValue(self):
        return self.spinbox.value()


class ThresholdDialog(QDialog):
    """ Dialog for threshold input """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Бинарное преобразование")
        self.setModal(True)

        layout = QVBoxLayout(self)

        # Threshold input
        param_layout = QHBoxLayout()
        param_layout.addWidget(QLabel("Пороговое значение (0-255):"))
        self.spinbox = QSpinBox()
        self.spinbox.setRange(0, 255)
        self.spinbox.setValue(128)
        param_layout.addWidget(self.spinbox)
        layout.addLayout(param_layout)

        # Buttons
        button_layout = QHBoxLayout()
        self.ok_button = QPushButton("OK")
        self.cancel_button = QPushButton("Cancel")
        button_layout.addWidget(self.ok_button)
        button_layout.addWidget(self.cancel_button)
        layout.addLayout(button_layout)

        # Connections
        self.ok_button.clicked.connect(self.accept)
        self.cancel_button.clicked.connect(self.reject)

    def getValue(self):
        return self.spinbox.value()


class BrightnessRangeDialog(QDialog):
    """ Dialog for brightness range cutting parameters """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Вырезание диапазона яркостей")
        self.setModal(True)

        layout = QVBoxLayout(self)

        # Lower threshold
        low_layout = QHBoxLayout()
        low_layout.addWidget(QLabel("Нижний порог (0-255):"))
        self.low_spinbox = QSpinBox()
        self.low_spinbox.setRange(0, 255)
        self.low_spinbox.setValue(150)
        low_layout.addWidget(self.low_spinbox)
        layout.addLayout(low_layout)

        # Upper threshold
        high_layout = QHBoxLayout()
        high_layout.addWidget(QLabel("Верхний порог (0-255):"))
        self.high_spinbox = QSpinBox()
        self.high_spinbox.setRange(0, 255)
        self.high_spinbox.setValue(200)
        high_layout.addWidget(self.high_spinbox)
        layout.addLayout(high_layout)

        # Constant value (only when not keeping others)
        constant_layout = QHBoxLayout()
        constant_layout.addWidget(QLabel("Константа для внешних пикселей:"))
        self.constant_spinbox = QSpinBox()
        self.constant_spinbox.setRange(0, 255)
        self.constant_spinbox.setValue(127)
        constant_layout.addWidget(self.constant_spinbox)
        layout.addLayout(constant_layout)

        # Approach selection
        approach_layout = QVBoxLayout()
        approach_layout.addWidget(QLabel("Обработка пикселей ВНЕ диапазона:"))

        self.keep_radio = QRadioButton("Сохранить исходные значения")
        self.constant_radio = QRadioButton("Привести к константе")
        self.keep_radio.setChecked(True)

        approach_layout.addWidget(self.keep_radio)
        approach_layout.addWidget(self.constant_radio)
        layout.addLayout(approach_layout)

        # Enable/disable constant value based on radio selection
        def update_constant_enabled():
            self.constant_spinbox.setEnabled(self.constant_radio.isChecked())

        self.keep_radio.toggled.connect(update_constant_enabled)
        self.constant_radio.toggled.connect(update_constant_enabled)
        update_constant_enabled()

        # Buttons
        button_layout = QHBoxLayout()
        self.ok_button = QPushButton("Применить")
        self.cancel_button = QPushButton("Отмена")
        button_layout.addWidget(self.ok_button)
        button_layout.addWidget(self.cancel_button)
        layout.addLayout(button_layout)

        # Connections
        self.ok_button.clicked.connect(self.accept)
        self.cancel_button.clicked.connect(self.reject)

    def getValues(self):
        return {
            'low_threshold': self.low_spinbox.value(),
            'high_threshold': self.high_spinbox.value(),
            'constant_value': self.constant_spinbox.value(),
            'keep_others': self.keep_radio.isChecked()
        }


class UnsharpMaskingDialog(QDialog):
    """ Dialog for unsharp masking parameters """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Нерезкое маскирование")
        self.setModal(True)

        layout = QVBoxLayout(self)

        # Blur radius
        blur_layout = QHBoxLayout()
        blur_layout.addWidget(QLabel("Радиус размытия:"))
        self.blur_spinbox = QSpinBox()
        self.blur_spinbox.setRange(1, 15)
        self.blur_spinbox.setValue(5)
        self.blur_spinbox.setSuffix(" px")
        blur_layout.addWidget(self.blur_spinbox)
        layout.addLayout(blur_layout)

        # Info label about odd numbers
        info_label = QLabel("(радиус будет преобразован в нечетное число)")
        info_label.setStyleSheet("color: gray; font-kernel_size: 10px;")
        layout.addWidget(info_label)

        # Strength
        strength_layout = QHBoxLayout()
        strength_layout.addWidget(QLabel("Сила повышения резкости:"))
        self.strength_slider = QSlider(Qt.Orientation.Horizontal)
        self.strength_slider.setRange(0, 300)  # 0.0 to 3.0 with step 0.01
        self.strength_slider.setValue(100)  # default 1.0
        strength_layout.addWidget(self.strength_slider)

        self.strength_label = QLabel("1.00")
        self.strength_label.setFixedWidth(40)
        strength_layout.addWidget(self.strength_label)
        layout.addLayout(strength_layout)

        # Buttons
        button_layout = QHBoxLayout()
        self.ok_button = QPushButton("Применить")
        self.cancel_button = QPushButton("Отмена")
        button_layout.addWidget(self.ok_button)
        button_layout.addWidget(self.cancel_button)
        layout.addLayout(button_layout)

        # Connections
        self.strength_slider.valueChanged.connect(self.update_strength_label)
        self.ok_button.clicked.connect(self.accept)
        self.cancel_button.clicked.connect(self.reject)

        self.update_strength_label()

    def update_strength_label(self):
        strength_value = self.strength_slider.value() / 100.0
        self.strength_label.setText(f"{strength_value:.2f}")

    def getValues(self):
        return {
            'blur_radius': self.blur_spinbox.value(),
            'strength': self.strength_slider.value() / 100.
        }


class ConvolutionDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Свёртка с произвольной матрицей")
        self.setModal(True)
        self.setMinimumWidth(500)

        layout = QVBoxLayout(self)

        # Размер матрицы
        size_layout = QHBoxLayout()
        size_layout.addWidget(QLabel("Размер матрицы (n×n):"))
        self.size_spinbox = QSpinBox()
        self.size_spinbox.setRange(1, 15)
        self.size_spinbox.setValue(3)
        self.size_spinbox.valueChanged.connect(self.update_matrix_inputs)
        size_layout.addWidget(self.size_spinbox)
        size_layout.addStretch()
        layout.addLayout(size_layout)

        # Стандартные матрицы
        preset_layout = QHBoxLayout()
        preset_layout.addWidget(QLabel("Стандартные матрицы:"))
        self.preset_combo = QComboBox()
        self.preset_combo.addItems([
            "Пользовательская",
            "Размытие 3×3",
            "Размытие 5×5",
            "Резкость 3×3",
            "Резкость 5×5",
            "Горизонтальный Собель",
            "Вертикальный Собель",
            "Лапласиан 3×3",
            "Лапласиан 5×5"
        ])
        self.preset_combo.currentTextChanged.connect(self.load_preset)
        preset_layout.addWidget(self.preset_combo)
        layout.addLayout(preset_layout)

        # Область для ввода матрицы
        matrix_group = QGroupBox("Матрица свёртки")
        matrix_layout = QVBoxLayout(matrix_group)

        self.scroll_area = QScrollArea()
        self.scroll_widget = QWidget()
        self.matrix_layout = QGridLayout(self.scroll_widget)
        self.scroll_area.setWidget(self.scroll_widget)
        self.scroll_area.setWidgetResizable(True)
        matrix_layout.addWidget(self.scroll_area)

        layout.addWidget(matrix_group)

        # Опции
        options_layout = QVBoxLayout()

        self.normalize_check = QCheckBox("Нормализация (деление на сумму элементов)")
        self.normalize_check.setChecked(True)
        options_layout.addWidget(self.normalize_check)

        self.add128_check = QCheckBox("Прибавить 128 (для визуализации границ)")
        options_layout.addWidget(self.add128_check)

        self.abs_check = QCheckBox("Взять модуль результата")
        options_layout.addWidget(self.abs_check)

        layout.addLayout(options_layout)

        # Кнопки
        button_layout = QHBoxLayout()
        self.ok_button = QPushButton("Применить")
        self.cancel_button = QPushButton("Отмена")
        button_layout.addWidget(self.ok_button)
        button_layout.addWidget(self.cancel_button)
        layout.addLayout(button_layout)

        # Соединения
        self.ok_button.clicked.connect(self.accept)
        self.cancel_button.clicked.connect(self.reject)

        # Инициализация матрицы
        self.matrix_inputs = []
        self.update_matrix_inputs()

    def update_matrix_inputs(self):
        """Обновить поля ввода матрицы при изменении размера"""
        # Очистить старые поля
        for i in reversed(range(self.matrix_layout.count())):
            self.matrix_layout.itemAt(i).widget().setParent(None)
        self.matrix_inputs.clear()

        size = self.size_spinbox.value()

        # Создать новые поля ввода
        for i in range(size):
            row_inputs = []
            for j in range(size):
                spinbox = QDoubleSpinBox()
                spinbox.setRange(-100, 100)
                spinbox.setValue(0.0)
                spinbox.setDecimals(3)
                spinbox.setSingleStep(0.1)
                spinbox.setFixedWidth(70)
                self.matrix_layout.addWidget(spinbox, i, j)
                row_inputs.append(spinbox)
            self.matrix_inputs.append(row_inputs)

    def load_preset(self, preset_name):
        """Загрузить стандартную матрицу"""
        if preset_name == "Пользовательская":
            return

        size = self.size_spinbox.value()
        matrix = self.get_preset_matrix(preset_name, size)

        if matrix is not None:
            for i in range(size):
                for j in range(size):
                    if i < len(matrix) and j < len(matrix[0]):
                        self.matrix_inputs[i][j].setValue(matrix[i][j])
                    else:
                        self.matrix_inputs[i][j].setValue(0.0)

    def get_preset_matrix(self, preset_name, size):
        """Получить стандартную матрицу"""
        if preset_name == "Размытие 3×3":
            return [
                [1 / 9, 1 / 9, 1 / 9],
                [1 / 9, 1 / 9, 1 / 9],
                [1 / 9, 1 / 9, 1 / 9]
            ]
        elif preset_name == "Размытие 5×5":
            return [
                [1 / 25, 1 / 25, 1 / 25, 1 / 25, 1 / 25],
                [1 / 25, 1 / 25, 1 / 25, 1 / 25, 1 / 25],
                [1 / 25, 1 / 25, 1 / 25, 1 / 25, 1 / 25],
                [1 / 25, 1 / 25, 1 / 25, 1 / 25, 1 / 25],
                [1 / 25, 1 / 25, 1 / 25, 1 / 25, 1 / 25]
            ]
        elif preset_name == "Резкость 3×3":
            return [
                [0, -1, 0],
                [-1, 5, -1],
                [0, -1, 0]
            ]
        elif preset_name == "Резкость 5×5":
            return [
                [-1, -1, -1, -1, -1],
                [-1, -1, -1, -1, -1],
                [-1, -1, 25, -1, -1],
                [-1, -1, -1, -1, -1],
                [-1, -1, -1, -1, -1]
            ]
        elif preset_name == "Горизонтальный Собель":
            return [
                [-1, 0, 1],
                [-2, 0, 2],
                [-1, 0, 1]
            ]
        elif preset_name == "Вертикальный Собель":
            return [
                [-1, -2, -1],
                [0, 0, 0],
                [1, 2, 1]
            ]
        elif preset_name == "Лапласиан 3×3":
            return [
                [0, -1, 0],
                [-1, 4, -1],
                [0, -1, 0]
            ]
        elif preset_name == "Лапласиан 5×5":
            return [
                [-1, -1, -1, -1, -1],
                [-1, -1, -1, -1, -1],
                [-1, -1, 24, -1, -1],
                [-1, -1, -1, -1, -1],
                [-1, -1, -1, -1, -1]
            ]
        return None

    def get_matrix(self):
        """Получить матрицу из полей ввода"""
        size = self.size_spinbox.value()
        matrix = []
        for i in range(size):
            row = []
            for j in range(size):
                row.append(self.matrix_inputs[i][j].value())
            matrix.append(row)
        return np.array(matrix)

    def get_options(self):
        """Получить выбранные опции"""
        return {
            'normalize': self.normalize_check.isChecked(),
            'add_128': self.add128_check.isChecked(),
            'abs_value': self.abs_check.isChecked()
        }
