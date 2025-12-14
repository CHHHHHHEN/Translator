from __future__ import annotations

import logging
import os
from pathlib import Path

from rich.logging import RichHandler

from translator.core.config import settings


def configure_logging() -> None:
    """Configure logging based on settings."""
    log_level_str = settings.get("logging.level", "INFO").upper()
    log_level = getattr(logging, log_level_str, logging.INFO)
    
    log_path_str = settings.get("logging.path", "dist/translator.log")
    
    log_path = Path(log_path_str)
    if not log_path.is_absolute():
        log_path = Path.cwd() / log_path
        
    try:
        log_path.parent.mkdir(parents=True, exist_ok=True)
    except Exception:
        # Fallback if we can't create directory
        pass

    handlers: list[logging.Handler] = [
        RichHandler(rich_tracebacks=True, markup=True),
        logging.FileHandler(log_path, encoding="utf-8"),
    ]

    logging.basicConfig(
        level=log_level,
        format="%(message)s",
        datefmt="[%X]",
        handlers=handlers,
    )
    
    logging.getLogger().setLevel(log_level)


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)
