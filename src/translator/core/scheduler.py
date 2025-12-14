from __future__ import annotations

import asyncio
from typing import Awaitable, Callable


class Scheduler:
    """Helper for running coroutines with debounce semantics."""

    def __init__(self) -> None:
        self._task: asyncio.Task | None = None

    def debounce(self, coro_factory: Callable[[], Awaitable[None]], delay: float) -> None:
        if self._task and not self._task.done():
            self._task.cancel()

        async def _runner() -> None:
            try:
                await asyncio.sleep(delay)
                await coro_factory()
            except asyncio.CancelledError:
                return

        self._task = asyncio.create_task(_runner())

    def shutdown(self) -> None:
        if self._task and not self._task.done():
            self._task.cancel()
