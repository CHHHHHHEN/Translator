from translator.ocr.paddle import PaddleOcrEngine
from translator.ocr.preprocess import preprocess


def test_preprocess_returns_same_bytes() -> None:
    payload = b"abc"
    assert preprocess(payload) is payload


def test_paddle_engine_returns_placeholder() -> None:
    engine = PaddleOcrEngine()
    assert "paddle-ocr" in engine.extract_text(b"bytes")
