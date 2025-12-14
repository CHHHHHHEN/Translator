from __future__ import annotations


def replace_text(original: str, replacements: dict[str, str]) -> str:
    """Apply a chain of textual replacements."""
    result = original
    for needle, replacement in sorted(replacements.items(), key=lambda item: -len(item[0])):
        result = result.replace(needle, replacement)
    return result
