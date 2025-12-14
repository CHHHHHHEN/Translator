from __future__ import annotations

from typing import Mapping

import mss


def capture_screen(region: Mapping[str, int] | None = None) -> mss.base.ScreenShot:
    """Grab the selected screen region or the primary monitor."""
    with mss.mss() as sct:
        monitor = region if region else sct.monitors[1]
        return sct.grab(monitor)
