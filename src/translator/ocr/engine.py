from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Iterable


class OcrEngine(ABC):
    """Defines the OCR contract used by translator services."""

    def __init__(self, languages: Iterable[str] | None = None) -> None:
        self.languages = list(languages or [])

    @abstractmethod
    def extract_text(self, image_bytes: bytes) -> str:
        raise NotImplementedError()
