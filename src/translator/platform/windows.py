from __future__ import annotations

import sys


def is_windows() -> bool:
    return sys.platform.startswith("win")


def register_jump_list() -> None:
    if not is_windows():
        return
    # Windows-specific extension points can be added here.
