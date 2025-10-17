from PyQt6.QtWidgets import QDialog, QVBoxLayout, QHBoxLayout, QLabel, QDoubleSpinBox, QPushButton, QSpinBox


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
