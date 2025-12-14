from __future__ import annotations

from .app import TranslatorApp
from .utils.logger import configure_logging


def main() -> int:
    """Application entry point for CLI scripts."""
    configure_logging()
    return TranslatorApp().run()


if __name__ == "__main__":
    raise SystemExit(main())
