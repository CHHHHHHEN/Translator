from translator.services.translate_service import TranslateService


def test_translate_service_routes_to_llm() -> None:
    service = TranslateService()
    result = service.translate("hello", "zh")
    assert "LLM" in result
    assert "zh" in result
