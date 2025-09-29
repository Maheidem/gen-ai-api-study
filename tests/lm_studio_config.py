"""
Configuration for LM Studio integration tests.
"""

import os
from typing import Optional, List
from dataclasses import dataclass


@dataclass
class LMStudioConfig:
    """Configuration for LM Studio integration tests."""

    # Connection settings
    base_url: str = "http://169.254.83.107:1234/v1"
    timeout: int = 30
    max_retries: int = 3

    # Test settings
    default_temperature: float = 0.1  # Low temperature for consistent testing
    max_tokens: int = 500

    # Model preferences (if specific models are loaded)
    preferred_models: List[str] = None
    tool_calling_models: List[str] = None
    thinking_models: List[str] = None

    # Test content
    simple_test_prompts: List[str] = None
    tool_test_prompts: List[str] = None
    thinking_test_prompts: List[str] = None

    def __post_init__(self):
        """Set default values after initialization."""
        if self.preferred_models is None:
            self.preferred_models = [
                "llama-3.1-8b-instruct",
                "llama-3.1-70b-instruct",
                "qwen2.5-7b-instruct",
                "mistral-7b-instruct",
                "phi-3-mini-4k-instruct"
            ]

        if self.tool_calling_models is None:
            self.tool_calling_models = [
                "llama-3.1-8b-instruct",
                "llama-3.1-70b-instruct",
                "qwen2.5-7b-instruct"
            ]

        if self.thinking_models is None:
            self.thinking_models = [
                "qwq-32b-preview",
                "deepseek-r1-distill-qwen-7b",
                "marco-o1-7b"
            ]

        if self.simple_test_prompts is None:
            self.simple_test_prompts = [
                "Say 'Hello, World!' and nothing else.",
                "What is 2 + 2? Give only the number.",
                "Count to 3. Format: 1, 2, 3",
                "What color is the sky on a clear day?",
                "Translate 'hello' to Spanish."
            ]

        if self.tool_test_prompts is None:
            self.tool_test_prompts = [
                "Calculate 15 * 23 using the calculator tool.",
                "What is 156 divided by 12?",
                "Count the characters in the text 'Hello, World!'",
                "Convert 'python programming' to uppercase.",
                "Add 100 and 50, then multiply by 2."
            ]

        if self.thinking_test_prompts is None:
            self.thinking_test_prompts = [
                "Think step by step: What is the square root of 144?",
                "Explain your reasoning: Which is larger, 3/4 or 5/8?",
                "Solve this logic puzzle: If all cats are animals and some animals are pets, are all cats pets?",
                "Step through the calculation: What is 25% of 80?",
                "Think carefully: What day comes 3 days after Tuesday?"
            ]


# Global configuration instance
config = LMStudioConfig()

# Environment variable overrides
if os.getenv("LM_STUDIO_URL"):
    config.base_url = os.getenv("LM_STUDIO_URL")

if os.getenv("LM_STUDIO_TIMEOUT"):
    config.timeout = int(os.getenv("LM_STUDIO_TIMEOUT"))

if os.getenv("LM_STUDIO_MAX_TOKENS"):
    config.max_tokens = int(os.getenv("LM_STUDIO_MAX_TOKENS"))

if os.getenv("LM_STUDIO_TEMPERATURE"):
    config.default_temperature = float(os.getenv("LM_STUDIO_TEMPERATURE"))


def get_config() -> LMStudioConfig:
    """Get the current LM Studio test configuration."""
    return config


def update_config(**kwargs) -> LMStudioConfig:
    """Update configuration values."""
    for key, value in kwargs.items():
        if hasattr(config, key):
            setattr(config, key, value)
        else:
            raise ValueError(f"Unknown configuration key: {key}")
    return config