from __future__ import annotations

from typing import Iterable


def detect_change(before: bytes | None, after: bytes, threshold: float = 0.015) -> bool:
    """Return true when the captured bytes deviate enough from the previous capture."""
    if before is None:
        return True
    length = min(len(before), len(after))
    if length == 0:
        return False
    delta = sum(abs(a - b) for a, b in zip(before[:length], after[:length]))
    max_diff = length * 255
    return (delta / max_diff) > threshold
