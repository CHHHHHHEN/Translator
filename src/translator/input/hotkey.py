from __future__ import annotations


class HotkeyManager:
    """Tracks global hotkeys (stub)."""

    def __init__(self) -> None:
        self._bindings: dict[str, str] = {}

    def register(self, action: str, shortcut: str) -> None:
        self._bindings[action] = shortcut

    def bindings(self) -> dict[str, str]:
        return dict(self._bindings)
