from __future__ import annotations

from PyQt6.QtCore import QRect
from PyQt6.QtWidgets import QApplication


def virtual_desktop_geometry() -> QRect:
    """Return the union of all monitor geometries for virtual desktop selection."""
    virtual_rect = QRect()
    for screen in QApplication.screens():
        virtual_rect = virtual_rect.united(screen.geometry())
    return virtual_rect


def rect_to_device_pixels(rect: QRect) -> tuple[int, int, int, int]:
    """Convert a QRect selection to device pixels using the screen DPR."""
    center = rect.center()
    for screen in QApplication.screens():
        if screen.geometry().contains(center):
            dpr = screen.devicePixelRatio()
            break
    else:
        dpr = QApplication.primaryScreen().devicePixelRatio()

    return (
        int(rect.x() * dpr),
        int(rect.y() * dpr),
        int(rect.width() * dpr),
        int(rect.height() * dpr),
    )
