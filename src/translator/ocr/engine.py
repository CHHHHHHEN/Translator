from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Iterable, Union, Any
import numpy as np


class OcrEngine(ABC):
    """Defines the OCR contract used by translator services."""

    def __init__(self, languages: Iterable[str] | None = None) -> None:
        self.languages = list(languages or ["en"])

    @abstractmethod
    def extract_text(self, image: Union[bytes, np.ndarray, Any]) -> str:
        """Extract text from an image (bytes or numpy array)."""
        raise NotImplementedError()
