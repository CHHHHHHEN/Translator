from __future__ import annotations

from ..translate.router import TranslatorRouter


class TranslateService:
    """Wraps the translator router."""

    def __init__(self) -> None:
        self._router = TranslatorRouter()

    def translate(self, text: str, target_language: str) -> str:
        return self._router.translate(text, target_language)
