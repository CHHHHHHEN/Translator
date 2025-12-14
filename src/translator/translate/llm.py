from __future__ import annotations

try:
    import openai
    from openai import OpenAI
except ImportError:
    openai = None
    OpenAI = None

from .base import TranslatorBase
from translator.core.config import settings
from translator.utils.logger import get_logger

logger = get_logger(__name__)


class LLMTranslator(TranslatorBase):
    """LLM translator using OpenAI-compatible API."""

    def __init__(self) -> None:
        self._api_key = settings.get("translate.llm.api_key")
        self._base_url = settings.get("translate.llm.base_url")
        self._model = settings.get("translate.llm.model", "gpt-3.5-turbo")
        self._temperature = settings.get("translate.llm.temperature", 0.3)
        
        self._client = None
        if OpenAI and self._api_key:
            try:
                self._client = OpenAI(api_key=self._api_key, base_url=self._base_url)
            except Exception as e:
                logger.error(f"Failed to initialize OpenAI client: {e}")
        elif not OpenAI:
            logger.warning("OpenAI package not installed. LLM translation will not work.")

    def translate(self, text: str, target_language: str) -> str:
        if not self._client:
            if not OpenAI:
                return f"[Error: OpenAI package missing] {text}"
            return f"[Error: LLM Client not initialized. Check config.] {text}"
            
        try:
            # Construct messages
            # We assume the prompt template has already formatted the text, 
            # but LLMTranslator usually receives the raw text if Router formats it?
            # Let's check Router.
            # Router: formatted = self._prompt.format_prompt(text, target_language)
            #         return self._primary.translate(formatted, target_language)
            # So 'text' here is actually the full prompt "Translate into ...: ..."
            
            messages = [
                {"role": "system", "content": "You are a professional translator. Translate the following text accurately and concisely. You only respond with the translated text."},
                {"role": "user", "content": text}
            ]
            
            response = self._client.chat.completions.create(
                model=self._model,
                messages=messages,
                temperature=self._temperature,
            )
            
            result = response.choices[0].message.content
            return result.strip() if result else ""
            
        except Exception as e:
            logger.error(f"LLM translation failed: {e}")
            return f"[Translation Error] {text}"
