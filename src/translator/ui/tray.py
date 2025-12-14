from __future__ import annotations

from PyQt6.QtGui import QIcon
from PyQt6.QtWidgets import QApplication, QAction, QMenu, QSystemTrayIcon


class TrayIcon(QSystemTrayIcon):
    """Simple system tray helper with an exit action."""

    def __init__(self, parent=None) -> None:
        icon = QIcon()
        super().__init__(icon, parent)
        menu = QMenu()
        exit_action = QAction("Quit")
        exit_action.triggered.connect(QApplication.instance().quit)
        menu.addAction(exit_action)
        self.setContextMenu(menu)
        self.setToolTip("Translator Overlay")
        self.show()
