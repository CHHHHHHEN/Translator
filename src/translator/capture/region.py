from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Region:
    top: int
    left: int
    width: int
    height: int

    def to_dict(self) -> dict[str, int]:
        return {
            "top": self.top,
            "left": self.left,
            "width": self.width,
            "height": self.height,
        }
