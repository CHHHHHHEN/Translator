from __future__ import annotations

from collections import defaultdict
from typing import Callable, DefaultDict, Dict, List


EventCallback = Callable[..., None]


class EventBus:
    """Lightweight event registry for decoupling modules."""

    def __init__(self) -> None:
        self._subscribers: DefaultDict[str, List[EventCallback]] = defaultdict(list)

    def subscribe(self, event_name: str, callback: EventCallback) -> None:
        self._subscribers[event_name].append(callback)

    def emit(self, event_name: str, *args, **kwargs) -> None:
        for callback in list(self._subscribers.get(event_name, [])):
            callback(*args, **kwargs)

    def clear(self) -> None:
        self._subscribers.clear()
