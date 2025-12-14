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
        formatted = self._prompt.format_prompt(text, target_language)
        return self._primary.translate(formatted, target_language)
