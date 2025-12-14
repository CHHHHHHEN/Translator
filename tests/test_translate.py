from unittest.mock import patch, MagicMock
from translator.services.translate_service import TranslateService
from translator.translate.llm import LLMTranslator


def test_translate_service_routes_to_llm() -> None:
    # Mock LLMTranslator to avoid real API call
    with patch("translator.translate.router.LLMTranslator") as MockLLM:
        MockLLM.return_value.translate.return_value = "[LLM zh] hello"
        
        service = TranslateService()
        result = service.translate("hello", "zh")
        assert "LLM" in result
        assert "zh" in result


def test_llm_translator_call() -> None:
    with patch("translator.translate.llm.OpenAI") as MockOpenAI:
        mock_client = MockOpenAI.return_value
        mock_response = MagicMock()
        mock_response.choices[0].message.content = "Translated Text"
        mock_client.chat.completions.create.return_value = mock_response
        
        translator = LLMTranslator()
        # Ensure client is set (it might not be if api_key is missing in settings)
        # We can force it
        translator._client = mock_client
        
        result = translator.translate("Hello", "fr")
        assert result == "Translated Text"
        
        mock_client.chat.completions.create.assert_called_once()
