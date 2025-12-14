from __future__ import annotations

from typing import Union, Any
import numpy as np

from ..ocr.paddle import PaddleOcrEngine
from ..core.config import settings


class OcrService:
    """Thin wrapper that keeps OCR engines pluggable."""

    def __init__(self) -> None:
        # Could load engine type from settings
        languages = settings.get("ocr.languages", ["en", "zh"])
        self._engine = PaddleOcrEngine(languages=languages)

    def detect_text(self, image: Union[bytes, np.ndarray, Any]) -> str:
        return self._engine.extract_text(image)
