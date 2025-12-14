from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Optional


@dataclass
class AppState:
    """Shared application state that services can read/write."""

    is_running: bool = False
    last_capture: Optional[datetime] = None
    errors: List[str] = field(default_factory=list)

    def record_error(self, message: str) -> None:
        self.errors.append(message)
