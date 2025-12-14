from translator.translate.prompt import PromptTemplate


def test_prompt_template_formats_text() -> None:
    template = PromptTemplate(template="translate {text} to {language}")
    formatted = template.format_prompt("hello world", "fr")
    assert "hello world" in formatted
    assert "fr" in formatted
