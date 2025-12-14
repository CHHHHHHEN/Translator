from __future__ import annotations

from PyQt6.QtCore import Qt, QRect, QPoint, pyqtSignal
from PyQt6.QtGui import QPainter, QColor, QPen, QBrush
from PyQt6.QtWidgets import QWidget
from translator.capture.geometry import rect_to_device_pixels, virtual_desktop_geometry


class RegionSelector(QWidget):
    """Fullscreen overlay for selecting a screen region."""
    
    # Signal emits (x, y, width, height)
    region_selected = pyqtSignal(tuple)

    def __init__(self) -> None:
        super().__init__()
        self.setWindowFlags(
            Qt.WindowType.FramelessWindowHint | 
            Qt.WindowType.WindowStaysOnTopHint | 
            Qt.WindowType.Tool
        )
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setCursor(Qt.CursorShape.CrossCursor)
        
        self._start_pos: QPoint | None = None
        self._current_pos: QPoint | None = None
        self._selection_rect: QRect | None = None

    def start_selection(self) -> None:
        """Show the overlay and start selection process."""
        self.setGeometry(virtual_desktop_geometry())
        self._start_pos = None
        self._current_pos = None
        self._selection_rect = None
        self.show()
        self.activateWindow()

    def mousePressEvent(self, event) -> None:
        if event.button() == Qt.MouseButton.LeftButton:
            self._start_pos = event.pos()
            self._current_pos = event.pos()
            self.update()

    def mouseMoveEvent(self, event) -> None:
        if self._start_pos:
            self._current_pos = event.pos()
            self._update_selection_rect()
            self.update()

    def mouseReleaseEvent(self, event) -> None:
        if event.button() == Qt.MouseButton.LeftButton and self._start_pos:
            self._update_selection_rect()
            if self._selection_rect and self._selection_rect.isValid():
                # Normalize rect
                rect = self._selection_rect.normalized()
                
                self.region_selected.emit(rect_to_device_pixels(rect))
            self.close()

    def keyPressEvent(self, event) -> None:
        if event.key() == Qt.Key.Key_Escape:
            self.close()

    def _update_selection_rect(self) -> None:
        if self._start_pos and self._current_pos:
            self._selection_rect = QRect(self._start_pos, self._current_pos).normalized()

    def paintEvent(self, event) -> None:
        painter = QPainter(self)
        
        # Draw dimmed background
        painter.setBrush(QColor(0, 0, 0, 100))
        painter.setPen(Qt.PenStyle.NoPen)
        painter.drawRect(self.rect())

        if self._selection_rect:
            # Clear the selection area (make it transparent)
            painter.setCompositionMode(QPainter.CompositionMode.CompositionMode_Clear)
            painter.setBrush(Qt.BrushStyle.SolidPattern)
            painter.drawRect(self._selection_rect)
            
            # Draw border around selection
            painter.setCompositionMode(QPainter.CompositionMode.CompositionMode_SourceOver)
            painter.setPen(QPen(QColor(0, 120, 215), 2))
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.drawRect(self._selection_rect)
