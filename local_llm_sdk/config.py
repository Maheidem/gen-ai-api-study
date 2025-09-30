"""
Configuration management for Local LLM SDK.

Supports environment variables for easy deployment and configuration.
"""

import os
from typing import Dict, Any


def get_default_config() -> Dict[str, Any]:
    """
    Get default configuration from environment variables.

    Environment Variables:
        LLM_BASE_URL: Base URL for the local LLM API (default: http://localhost:1234/v1)
        LLM_MODEL: Model to use (default: auto)
        LLM_TIMEOUT: Request timeout in seconds (default: 30)
        LLM_DEBUG: Enable debug logging (default: false)

    Returns:
        Dictionary with configuration values

    Example:
        >>> import os
        >>> os.environ['LLM_BASE_URL'] = 'http://192.168.1.100:1234/v1'
        >>> os.environ['LLM_MODEL'] = 'my-model'
        >>> config = get_default_config()
        >>> config['base_url']
        'http://192.168.1.100:1234/v1'
    """
    return {
        "base_url": os.getenv("LLM_BASE_URL", "http://localhost:1234/v1"),
        "model": os.getenv("LLM_MODEL", "auto"),
        "timeout": int(os.getenv("LLM_TIMEOUT", "30")),
        "debug": os.getenv("LLM_DEBUG", "false").lower() in ("true", "1", "yes"),
    }