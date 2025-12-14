from __future__ import annotations

import time
from typing import Callable
from pynput import keyboard
from translator.utils.logger import get_logger

logger = get_logger(__name__)


class GlobalHotkey:
    """Listens for global hotkeys."""

    def __init__(self, on_double_space: Callable[[], None] | None = None) -> None:
        self._on_double_space = on_double_space
        self._listener: keyboard.Listener | None = None
        self._last_space_time = 0.0
        self._running = False

    def start(self) -> None:
        if self._running:
            return
        self._running = True
        self._listener = keyboard.Listener(on_release=self._on_release)
        self._listener.start()
        logger.info("Global hotkey listener started.")

    def stop(self) -> None:
        if self._listener:
            self._listener.stop()
            self._listener = None
        self._running = False
        logger.info("Global hotkey listener stopped.")

    def _on_release(self, key: keyboard.Key | keyboard.KeyCode | None) -> None:
        if key == keyboard.Key.space:
            now = time.time()
            # 300ms threshold for double tap
            if now - self._last_space_time < 0.3:
                logger.info("Double space detected!")
                if self._on_double_space:
                    # Run in a separate way or ensure it doesn't block listener?
                    # Listener is a thread. Callback should be fast or dispatch to main thread.
                    try:
                        self._on_double_space()
                    except Exception as e:
                        logger.error(f"Error in hotkey callback: {e}")
                self._last_space_time = 0.0 # Reset
            else:
                self._last_space_time = now
