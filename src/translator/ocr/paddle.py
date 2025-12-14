from __future__ import annotations

from .engine import OcrEngine


class PaddleOcrEngine(OcrEngine):
    """Stub for a Paddle OCR implementation."""

    def extract_text(self, image_bytes: bytes) -> str:
        return "(paddle-ocr result)"
