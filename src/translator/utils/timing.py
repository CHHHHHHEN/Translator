from __future__ import annotations

import contextlib
import time


def timing(name: str):
    """Context manager that logs elapsed time."""

    @contextlib.contextmanager
    def _timing():
        start = time.perf_counter()
        yield
        elapsed = time.perf_counter() - start
        print(f"[{name}] {elapsed:.3f}s")

    return _timing()
