from __future__ import annotations

from PyQt6.QtWidgets import QLabel, QWidget


class RegionSelector(QWidget):
    """Placeholder UI for selecting screen regions."""

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Region Selector")
        self._hint = QLabel("Drag to select area", self)
        self._hint.move(10, 10)

    def open(self) -> None:
        self.show()
