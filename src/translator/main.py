from __future__ import annotations

from .app import TranslatorApp


def main() -> int:
    """Application entry point for CLI scripts."""
    return TranslatorApp().run()


if __name__ == "__main__":
    raise SystemExit(main())
