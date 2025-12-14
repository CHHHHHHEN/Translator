from __future__ import annotations

from abc import ABC, abstractmethod


class TranslatorBase(ABC):
    """Abstraction for translation backends."""

    @abstractmethod
    def translate(self, text: str, target_language: str) -> str:
        raise NotImplementedError()
