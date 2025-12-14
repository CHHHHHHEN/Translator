from __future__ import annotations

from dataclasses import dataclass


@dataclass
class PromptTemplate:
    template: str = "Translate the following text: {text}"

    def format_prompt(self, text: str, target_language: str) -> str:
        return self.template.format(text=text, language=target_language)
