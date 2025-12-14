from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Optional, Tuple


@dataclass
class AppState:
    """Shared application state that services can read/write."""

    is_running: bool = False
    is_monitoring: bool = False
    
    # Region: (left, top, width, height)
    monitoring_region: Optional[Tuple[int, int, int, int]] = None
    
    last_capture_time: Optional[datetime] = None
    last_ocr_text: str = ""
    last_translated_text: str = ""
    
    errors: List[str] = field(default_factory=list)

    def record_error(self, message: str) -> None:
        self.errors.append(message)
        # Keep only last 100 errors
        if len(self.errors) > 100:
            self.errors.pop(0)

# Global state instance
state = AppState()
