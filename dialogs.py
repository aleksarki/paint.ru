from PyQt6.QtWidgets import QDialog, QVBoxLayout, QHBoxLayout, QLabel, QDoubleSpinBox, QPushButton, QSpinBox, \
    QRadioButton


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

