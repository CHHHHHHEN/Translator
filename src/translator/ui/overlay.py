from __future__ import annotations

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QLabel, QVBoxLayout, QWidget


class Overlay(QWidget):
    """A frameless floating overlay used for translation results."""

    def __init__(self) -> None:
        super().__init__()
        self.setWindowFlags(Qt.WindowType.FramelessWindowHint | Qt.WindowType.WindowStaysOnTopHint)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        label = QLabel("Waiting for capture...")
        label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setFixedSize(320, 160)
        layout = QVBoxLayout(self)
        layout.addWidget(label)
