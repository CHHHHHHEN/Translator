from __future__ import annotations

from .llm import LLMTranslator
from .prompt import PromptTemplate


class TranslatorRouter:
    """Chooses between translators and formats prompts."""

    def __init__(self) -> None:
        self._primary = LLMTranslator()
        self._prompt = PromptTemplate(
            template="Translate into {language}: {text}"
        )

    def translate(self, text: str, target_language: str) -> str:
        # Pass raw text to the translator. The translator should handle formatting/prompting.
        return self._primary.translate(text, target_language)
