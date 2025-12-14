from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

import yaml


class Config:
    def __init__(self) -> None:
        self._config: dict[str, Any] = {}
        self._load()

    def _get_project_root(self) -> Path:
        """Determine the project root directory."""
        # If running as a script or frozen app, try to find the root
        # This is a heuristic: look for pyproject.toml or config folder
        current = Path(__file__).resolve().parent
        for _ in range(5):
            if (current / "pyproject.toml").exists() or (current / "config").exists():
                return current
            current = current.parent
        return Path.cwd()

    def _load(self) -> None:
        root = self._get_project_root()
        config_dir = root / "config"
        
        default_config_path = config_dir / "default.yaml"
        user_config_path = config_dir / "user.yaml"

        if default_config_path.exists():
            with open(default_config_path, "r", encoding="utf-8") as f:
                self._config = yaml.safe_load(f) or {}
        else:
            # Fallback or warning
            print(f"Warning: Default config not found at {default_config_path}")

        if user_config_path.exists():
            with open(user_config_path, "r", encoding="utf-8") as f:
                user_config = yaml.safe_load(f) or {}
                self._merge(self._config, user_config)

    def _merge(self, base: dict[str, Any], update: dict[str, Any]) -> None:
        """Recursively merge update dict into base dict."""
        for k, v in update.items():
            if k in base and isinstance(base[k], dict) and isinstance(v, dict):
                self._merge(base[k], v)
            else:
                base[k] = v

    def get(self, key: str, default: Any = None) -> Any:
        """Get a config value using dot notation (e.g. 'logging.level')."""
        keys = key.split(".")
        value = self._config
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        return value

    def __getitem__(self, key: str) -> Any:
        return self.get(key)

# Global instance
settings = Config()
