"""
Tests for configuration management.
"""

import pytest
from unittest.mock import patch
import os

from local_llm_sdk.config import get_default_config


class TestConfigDefaults:
    """Test default configuration values."""

    def test_get_default_config_defaults(self):
        """Test default values when no environment variables are set."""
        with patch.dict(os.environ, {}, clear=True):
            config = get_default_config()

            assert config["base_url"] == "http://localhost:1234/v1"
            assert config["model"] == "auto"
            assert config["timeout"] == 300
            assert config["debug"] is False

    def test_config_base_url_env(self):
        """Test LLM_BASE_URL environment variable."""
        test_url = "http://192.168.1.100:1234/v1"
        with patch.dict(os.environ, {"LLM_BASE_URL": test_url}):
            config = get_default_config()
            assert config["base_url"] == test_url

    def test_config_model_env(self):
        """Test LLM_MODEL environment variable."""
        test_model = "my-custom-model"
        with patch.dict(os.environ, {"LLM_MODEL": test_model}):
            config = get_default_config()
            assert config["model"] == test_model

    def test_config_timeout_env(self):
        """Test LLM_TIMEOUT environment variable."""
        with patch.dict(os.environ, {"LLM_TIMEOUT": "600"}):
            config = get_default_config()
            assert config["timeout"] == 600

    def test_config_debug_env(self):
        """Test LLM_DEBUG environment variable."""
        with patch.dict(os.environ, {"LLM_DEBUG": "true"}):
            config = get_default_config()
            assert config["debug"] is True


class TestConfigEnvironmentVariables:
    """Test configuration with multiple environment variables."""

    def test_get_default_config_with_env_vars(self):
        """Test that environment variables override defaults."""
        env_vars = {
            "LLM_BASE_URL": "http://169.254.83.107:1234/v1",
            "LLM_MODEL": "llama-3-8b",
            "LLM_TIMEOUT": "600",
            "LLM_DEBUG": "true"
        }

        with patch.dict(os.environ, env_vars, clear=True):
            config = get_default_config()

            assert config["base_url"] == "http://169.254.83.107:1234/v1"
            assert config["model"] == "llama-3-8b"
            assert config["timeout"] == 600
            assert config["debug"] is True


class TestConfigTypeConversion:
    """Test type conversion for configuration values."""

    def test_config_timeout_type_conversion(self):
        """Test timeout is converted to integer."""
        with patch.dict(os.environ, {"LLM_TIMEOUT": "450"}):
            config = get_default_config()
            assert isinstance(config["timeout"], int)
            assert config["timeout"] == 450

    def test_config_debug_boolean_parsing(self):
        """Test debug is parsed as boolean with various values."""
        # Test "true"
        with patch.dict(os.environ, {"LLM_DEBUG": "true"}):
            config = get_default_config()
            assert config["debug"] is True

        # Test "1"
        with patch.dict(os.environ, {"LLM_DEBUG": "1"}):
            config = get_default_config()
            assert config["debug"] is True

        # Test "yes"
        with patch.dict(os.environ, {"LLM_DEBUG": "yes"}):
            config = get_default_config()
            assert config["debug"] is True

        # Test "TRUE" (case insensitive)
        with patch.dict(os.environ, {"LLM_DEBUG": "TRUE"}):
            config = get_default_config()
            assert config["debug"] is True

        # Test "false"
        with patch.dict(os.environ, {"LLM_DEBUG": "false"}):
            config = get_default_config()
            assert config["debug"] is False

        # Test empty string
        with patch.dict(os.environ, {"LLM_DEBUG": ""}):
            config = get_default_config()
            assert config["debug"] is False
