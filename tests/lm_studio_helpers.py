"""
Helper utilities for LM Studio integration tests.
"""

import time
import requests
import pytest
from typing import Optional, List, Any
from unittest.mock import Mock

from local_llm_sdk import LocalLLMClient
from local_llm_sdk.models import ModelList, ModelInfo
from .lm_studio_config import get_config


def is_lm_studio_running(base_url: str = None, timeout: int = 5) -> bool:
    """
    Check if LM Studio is running and accessible.

    Args:
        base_url: LM Studio API base URL
        timeout: Timeout in seconds

    Returns:
        True if LM Studio is accessible, False otherwise
    """
    if base_url is None:
        base_url = get_config().base_url

    try:
        response = requests.get(f"{base_url}/models", timeout=timeout)
        return response.status_code == 200
    except (requests.ConnectionError, requests.Timeout, Exception):
        return False


def wait_for_lm_studio(max_wait: int = 60, check_interval: int = 2) -> bool:
    """
    Wait for LM Studio to become available.

    Args:
        max_wait: Maximum time to wait in seconds
        check_interval: How often to check in seconds

    Returns:
        True if LM Studio became available, False if timeout
    """
    config = get_config()
    start_time = time.time()

    while time.time() - start_time < max_wait:
        if is_lm_studio_running(config.base_url, timeout=2):
            return True
        time.sleep(check_interval)

    return False


def get_available_models(client: LocalLLMClient = None) -> List[ModelInfo]:
    """
    Get list of available models from LM Studio.

    Args:
        client: LocalLLMClient instance (creates one if None)

    Returns:
        List of available models

    Raises:
        ConnectionError: If LM Studio is not accessible
    """
    if client is None:
        config = get_config()
        client = LocalLLMClient(config.base_url)

    try:
        models = client.list_models()
        return models.data
    except Exception as e:
        raise ConnectionError(f"Cannot connect to LM Studio: {e}")


def get_first_available_model(client: LocalLLMClient = None, preferred: List[str] = None) -> Optional[str]:
    """
    Get the first available model, preferring models from the preferred list.

    Args:
        client: LocalLLMClient instance
        preferred: List of preferred model names/patterns

    Returns:
        Model ID if available, None if no models found
    """
    try:
        models = get_available_models(client)

        if not models:
            return None

        # Check preferred models first
        if preferred:
            for pref in preferred:
                for model in models:
                    if pref.lower() in model.id.lower():
                        return model.id

        # Return first available model
        return models[0].id

    except Exception:
        return None


def get_tool_calling_model(client: LocalLLMClient = None) -> Optional[str]:
    """Get a model that supports tool calling."""
    config = get_config()
    return get_first_available_model(client, config.tool_calling_models)


def get_thinking_model(client: LocalLLMClient = None) -> Optional[str]:
    """Get a model that outputs thinking blocks."""
    config = get_config()
    return get_first_available_model(client, config.thinking_models)


def skip_if_no_lm_studio(func):
    """
    Decorator to skip test if LM Studio is not running.

    Usage:
        @skip_if_no_lm_studio
        def test_something():
            # Test that requires LM Studio
    """
    def wrapper(*args, **kwargs):
        if not is_lm_studio_running():
            pytest.skip("LM Studio is not running")
        return func(*args, **kwargs)
    return wrapper


def skip_if_no_model(preferred_models: List[str] = None):
    """
    Decorator to skip test if no suitable model is available.

    Args:
        preferred_models: List of preferred model names

    Usage:
        @skip_if_no_model(["llama-3.1", "qwen2.5"])
        def test_with_specific_model():
            # Test that needs specific models
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            if not is_lm_studio_running():
                pytest.skip("LM Studio is not running")

            client = LocalLLMClient(get_config().base_url)
            model = get_first_available_model(client, preferred_models)

            if not model:
                models_str = ", ".join(preferred_models) if preferred_models else "any models"
                pytest.skip(f"No suitable model available. Looking for: {models_str}")

            # Inject model into kwargs if function expects it
            import inspect
            sig = inspect.signature(func)
            if 'model_id' in sig.parameters:
                kwargs['model_id'] = model

            return func(*args, **kwargs)
        return wrapper
    return decorator


def create_test_client(model_id: str = None) -> LocalLLMClient:
    """
    Create a LocalLLMClient configured for testing.

    Args:
        model_id: Specific model to use (auto-detects if None)

    Returns:
        Configured LocalLLMClient
    """
    config = get_config()
    client = LocalLLMClient(config.base_url, model_id)

    if model_id is None:
        # Try to set a good default model
        available_model = get_first_available_model(client, config.preferred_models)
        if available_model:
            client.default_model = available_model

    return client


def measure_response_time(func, *args, **kwargs) -> tuple[Any, float]:
    """
    Measure the response time of a function call.

    Args:
        func: Function to call
        *args: Function arguments
        **kwargs: Function keyword arguments

    Returns:
        Tuple of (result, time_in_seconds)
    """
    start_time = time.time()
    result = func(*args, **kwargs)
    end_time = time.time()
    return result, end_time - start_time


def validate_response_structure(response: Any, expected_type: type = str) -> bool:
    """
    Validate that a response has the expected structure.

    Args:
        response: Response to validate
        expected_type: Expected type of response

    Returns:
        True if response is valid
    """
    if not isinstance(response, expected_type):
        return False

    if expected_type == str:
        return len(response.strip()) > 0

    return True


def extract_numbers_from_response(response: str) -> List[float]:
    """
    Extract numbers from a model response for testing calculations.

    Args:
        response: Model response text

    Returns:
        List of numbers found in the response
    """
    import re
    pattern = r'-?\d+\.?\d*'
    matches = re.findall(pattern, response)
    return [float(match) for match in matches]


def compare_tool_vs_direct_response(client: LocalLLMClient, prompt: str, expected_tool: str = None) -> dict:
    """
    Compare responses when tools are available vs disabled.

    Args:
        client: LocalLLMClient with tools registered
        prompt: Test prompt
        expected_tool: Expected tool name that should be called

    Returns:
        Dictionary with comparison results
    """
    # Get response with tools enabled
    client.last_tool_calls = []
    with_tools = client.chat(prompt, use_tools=True)
    tools_used = client.last_tool_calls.copy()

    # Get response with tools disabled
    client.last_tool_calls = []
    without_tools = client.chat(prompt, use_tools=False)

    result = {
        "prompt": prompt,
        "with_tools": with_tools,
        "without_tools": without_tools,
        "tools_used": [tc.function.name for tc in tools_used],
        "tool_called": len(tools_used) > 0,
        "expected_tool_used": expected_tool in [tc.function.name for tc in tools_used] if expected_tool else None,
        "responses_different": with_tools.strip() != without_tools.strip()
    }

    return result


class LMStudioTestBase:
    """
    Base class for LM Studio integration tests.
    Provides common setup and utilities.
    """

    @classmethod
    def setup_class(cls):
        """Set up class-level test fixtures."""
        cls.config = get_config()
        cls.client = None
        cls.available_models = []

        if is_lm_studio_running():
            cls.client = create_test_client()
            try:
                cls.available_models = get_available_models(cls.client)
            except Exception:
                cls.available_models = []

    def setup_method(self):
        """Set up method-level test fixtures."""
        if self.client:
            # Clear state
            self.client.last_tool_calls = []
            self.client.last_thinking = ""

    def get_test_model(self, preferred: List[str] = None) -> str:
        """Get a model for testing."""
        if not self.client:
            pytest.skip("LM Studio not available")

        model = get_first_available_model(self.client, preferred or self.config.preferred_models)
        if not model:
            pytest.skip("No suitable model available")

        return model

    def assert_valid_response(self, response: str, min_length: int = 1):
        """Assert that a response is valid."""
        assert isinstance(response, str), f"Expected string response, got {type(response)}"
        assert len(response.strip()) >= min_length, f"Response too short: '{response}'"

    def assert_tool_was_used(self, expected_tool: str = None):
        """Assert that tools were used in the last request."""
        assert len(self.client.last_tool_calls) > 0, "No tools were called"

        if expected_tool:
            tool_names = [tc.function.name for tc in self.client.last_tool_calls]
            assert expected_tool in tool_names, f"Expected tool '{expected_tool}' not called. Called: {tool_names}"

    def assert_thinking_present(self):
        """Assert that thinking blocks were captured."""
        assert self.client.last_thinking, "No thinking blocks captured"
        assert len(self.client.last_thinking.strip()) > 0, "Thinking content is empty"