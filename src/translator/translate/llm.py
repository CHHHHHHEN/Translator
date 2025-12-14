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
        if not text or not text.strip():
            return ""

        if not self._client:
            if not OpenAI:
                return f"[Error: OpenAI package missing] {text}"
            return f"[Error: LLM Client not initialized. Check config.] {text}"
            
        try:
            # Construct a clear system prompt with the target language
            system_prompt = settings.get("translate.llm.system_prompt", 
                """
                You are a professional translator specialized in internet slang, informal speech, and real-time communication contexts (such as chats, games, forums, and social media).

                Translate the user's input into {language} with the following requirements:
                - Accurately preserve the original meaning, tone, and intent.
                - Appropriately localize internet slang, memes, abbreviations, and colloquial expressions instead of translating them literally.
                - Maintain natural fluency as used by native speakers in online environments.
                - Preserve emojis, symbols, line breaks, and basic text structure unless doing so would harm readability.
                - Normalize spacing and punctuation in the target language to improve readability.

                Output rules:
                - Output only the translated text.
                - Do not include explanations, annotations, or meta commentary.
                """
            )
            
            # Format the system prompt with the target language
            # Note: We assume the config string might have {language} placeholder.
            # If not, we append the instruction.
            if "{language}" in system_prompt:
                final_system_prompt = system_prompt.format(language=target_language)
            else:
                final_system_prompt = f"{system_prompt} Target Language: {target_language}."

            messages = [
                {"role": "system", "content": final_system_prompt},
                {"role": "user", "content": text}
            ]
            
            import time
            t0 = time.perf_counter()
            response = self._client.chat.completions.create(
                model=self._model,
                messages=messages,
                temperature=self._temperature,
                timeout=settings.get("translate.llm.timeout", 10.0)
            )
            t1 = time.perf_counter()
            logger.info(f"LLM call elapsed={t1-t0:.3f}s model={self._model}")

            result = response.choices[0].message.content
            return result.strip() if result else ""
            
        except Exception as e:
            logger.error(f"LLM translation failed: {e}")
            return f"[Translation Error] {text}"
