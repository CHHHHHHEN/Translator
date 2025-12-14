from __future__ import annotations

from typing import Sequence

from PyQt6.QtWidgets import QApplication

from .ui.main_window import MainWindow


class TranslatorApp:
    """Encapsulates the Qt application lifecycle."""

    def __init__(self, argv: Sequence[str] | None = None) -> None:
        self._argv = list(argv) if argv else []
        self._qt_app = QApplication(self._argv)
        self._window = MainWindow()

    def run(self) -> int:
        """Show the main window and start the Qt event loop."""
        self._window.show()
        return self._qt_app.exec()
