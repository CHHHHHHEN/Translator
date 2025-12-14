from __future__ import annotations

from PyQt6.QtGui import QGuiApplication


def read_clipboard() -> str:
    app = QGuiApplication.instance()
    if app is None:
        return ""
    return QGuiApplication.clipboard().text()


def write_clipboard(value: str) -> None:
    if QGuiApplication.instance() is None:
        return
    QGuiApplication.clipboard().setText(value)
