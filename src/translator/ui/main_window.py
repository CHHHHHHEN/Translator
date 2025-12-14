from __future__ import annotations

from PyQt6.QtGui import QFont
from PyQt6.QtWidgets import QLabel, QMainWindow, QVBoxLayout, QWidget


class MainWindow(QMainWindow):
    """Minimal settings window for the translator overlay."""

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Translator Overlay")
        central = QWidget()
        layout = QVBoxLayout()
        label = QLabel("Translator is ready.")
        label.setFont(QFont("Segoe UI", 10))
        layout.addWidget(label)
        layout.addStretch()
        central.setLayout(layout)
        self.setCentralWidget(central)
