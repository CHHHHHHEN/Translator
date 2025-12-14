import os
import pytest
from pathlib import Path
from unittest.mock import patch, mock_open

from translator.core.config import Config

def test_config_defaults() -> None:
    # Test with empty config
    # Mock Path.exists to return False so it doesn't try to open files
    with patch("pathlib.Path.exists", return_value=False):
        config = Config()
        # Should handle missing files gracefully
        assert config._config == {}

def test_config_get() -> None:
    config = Config()
    config._config = {
        "logging": {
            "level": "DEBUG"
        },
        "simple": "value"
    }
    
    assert config.get("logging.level") == "DEBUG"
    assert config.get("simple") == "value"
    assert config.get("nonexistent", "default") == "default"
    assert config.get("logging.nonexistent") is None

def test_config_merge() -> None:
    config = Config()
    base = {"a": 1, "b": {"c": 2}}
    update = {"b": {"d": 3}, "e": 4}
    
    config._merge(base, update)
    
    assert base == {
        "a": 1, 
        "b": {"c": 2, "d": 3}, 
        "e": 4
    }

def test_load_config_files() -> None:
    default_yaml = """
    section:
      key: default
      other: 1
    """
    user_yaml = """
    section:
      key: user
    """
    
    with patch("translator.core.config.Config._get_project_root", return_value=Path("/mock/root")):
        with patch("pathlib.Path.exists", side_effect=lambda: True):
            with patch("builtins.open", mock_open()) as mocked_file:
                # We need to simulate different content for different files
                # This is tricky with mock_open, so let's mock yaml.safe_load instead
                with patch("yaml.safe_load", side_effect=[
                    {"section": {"key": "default", "other": 1}}, # default.yaml
                    {"section": {"key": "user"}}                 # user.yaml
                ]):
                    config = Config()
                    
                    assert config.get("section.key") == "user"
                    assert config.get("section.other") == 1
