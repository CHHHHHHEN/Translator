from __future__ import annotations

from .base import TranslatorBase


class LLMTranslator(TranslatorBase):
    """A mock LLM translator for rapid prototyping."""

    def translate(self, text: str, target_language: str) -> str:
        return f"[LLM {target_language}] {text}"
