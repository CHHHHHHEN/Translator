from __future__ import annotations

from ..core.state import AppState


class MonitorService:
    """Tracks capture state and notifies listeners."""

    def __init__(self, state: AppState) -> None:
        self._state = state

    def start(self) -> None:
        self._state.is_running = True

    def stop(self) -> None:
        self._state.is_running = False
