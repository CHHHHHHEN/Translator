from __future__ import annotations

from ..ocr.paddle import PaddleOcrEngine


class OcrService:
    """Thin wrapper that keeps OCR engines pluggable."""

    def __init__(self) -> None:
        self._engine = PaddleOcrEngine()

    def detect_text(self, image_bytes: bytes) -> str:
        return self._engine.extract_text(image_bytes)
