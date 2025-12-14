from __future__ import annotations

from PyQt6.QtCore import Qt, QPoint
from PyQt6.QtGui import QColor, QPalette, QFont, QPainter, QBrush
from PyQt6.QtWidgets import QLabel, QVBoxLayout, QWidget


class Overlay(QWidget):
    """A frameless floating overlay used for translation results."""

    def __init__(self) -> None:
        super().__init__()
        self.setWindowFlags(
            Qt.WindowType.FramelessWindowHint | 
            Qt.WindowType.WindowStaysOnTopHint | 
            Qt.WindowType.Tool
        )
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        
        # Enable mouse tracking for dragging
        self.setMouseTracking(True)
        self._dragging = False
        self._drag_position = QPoint()
        
        self._label = QLabel("Waiting for capture...")
        self._label.setAlignment(Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignLeft)
        self._label.setWordWrap(True)
        self._label.setFont(QFont("Segoe UI", 12))
        self._label.setStyleSheet("color: white; padding: 10px;")
        
        self.setFixedSize(400, 200)
        
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self._label)

    def paintEvent(self, event) -> None:
        """Draw a semi-transparent rounded background."""
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        
        # Semi-transparent black background (Alpha 180/255)
        brush = QBrush(QColor(0, 0, 0, 180))
        painter.setBrush(brush)
        painter.setPen(Qt.PenStyle.NoPen)
        
        painter.drawRoundedRect(self.rect(), 10, 10)

    def mousePressEvent(self, event) -> None:
        if event.button() == Qt.MouseButton.LeftButton:
            self._dragging = True
            self._drag_position = event.globalPosition().toPoint() - self.frameGeometry().topLeft()
            event.accept()

    def mouseMoveEvent(self, event) -> None:
        if self._dragging and event.buttons() & Qt.MouseButton.LeftButton:
            self.move(event.globalPosition().toPoint() - self._drag_position)
            event.accept()

    def mouseReleaseEvent(self, event) -> None:
        self._dragging = False

    def update_text(self, text: str) -> None:
        self._label.setText(text)
        self.adjustSize()
        # Ensure it doesn't get too small
        if self.width() < 200:
            self.resize(200, self.height())
