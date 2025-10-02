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
        LLM_TIMEOUT: Request timeout in seconds (default: 300)
        LLM_DEBUG: Enable debug logging (default: false)

        # Validation settings
        LLM_ENABLE_VALIDATION: Enable response validation (default: true)
        LLM_ENABLE_LLM_JUDGE: Enable LLM judge validator (default: false)

        # Detection thresholds
        LLM_REPETITION_THRESHOLD: N-gram repetition threshold (default: 0.5)
        LLM_ENTROPY_THRESHOLD: Entropy collapse threshold (default: 0.5)

        # Recovery settings
        LLM_ENABLE_AUTO_RECOVERY: Enable automatic recovery (default: true)
        LLM_MAX_RECOVERY_ATTEMPTS: Max recovery attempts (default: 3)
        LLM_CHECKPOINT_INTERVAL: Checkpoint save interval (default: 3)

        # LLM Judge (optional)
        LLM_JUDGE_URL: URL for judge model (default: None, uses same server)
        LLM_JUDGE_MODEL: Model for judging (default: phi-2)

    Returns:
        Dictionary with configuration values

    Example:
        >>> import os
        >>> os.environ['LLM_BASE_URL'] = 'http://192.168.1.100:1234/v1'
        >>> os.environ['LLM_MODEL'] = 'my-model'
        >>> os.environ['LLM_ENABLE_VALIDATION'] = 'true'
        >>> config = get_default_config()
        >>> config['base_url']
        'http://192.168.1.100:1234/v1'
        >>> config['enable_validation']
        True
    """
    return {
        # Core settings
        "base_url": os.getenv("LLM_BASE_URL", "http://localhost:1234/v1"),
        "model": os.getenv("LLM_MODEL", "auto"),
        "timeout": int(os.getenv("LLM_TIMEOUT", "300")),
        "debug": os.getenv("LLM_DEBUG", "false").lower() in ("true", "1", "yes"),

        # Validation settings (disabled by default for backward compatibility)
        "enable_validation": os.getenv("LLM_ENABLE_VALIDATION", "false").lower() in ("true", "1", "yes"),
        "enable_llm_judge": os.getenv("LLM_ENABLE_LLM_JUDGE", "false").lower() in ("true", "1", "yes"),

        # Detection thresholds
        "repetition_threshold": float(os.getenv("LLM_REPETITION_THRESHOLD", "0.5")),
        "entropy_threshold": float(os.getenv("LLM_ENTROPY_THRESHOLD", "0.5")),

        # Recovery settings
        "enable_auto_recovery": os.getenv("LLM_ENABLE_AUTO_RECOVERY", "true").lower() in ("true", "1", "yes"),
        "max_recovery_attempts": int(os.getenv("LLM_MAX_RECOVERY_ATTEMPTS", "3")),
        "checkpoint_interval": int(os.getenv("LLM_CHECKPOINT_INTERVAL", "3")),

        # LLM Judge (optional)
        "judge_url": os.getenv("LLM_JUDGE_URL", None),
        "judge_model": os.getenv("LLM_JUDGE_MODEL", "phi-2"),
    }