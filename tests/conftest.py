"""
Shared pytest fixtures for local-llm-sdk tests.
"""

import pytest
from unittest.mock import Mock, patch
from typing import Dict, Any
import json
import os
from dotenv import load_dotenv

# Load .env file for consistent test configuration
load_dotenv()

from local_llm_sdk import LocalLLMClient
from local_llm_sdk.models import (
    ChatCompletion,
    ChatCompletionRequest,
    ChatMessage,
    ChatCompletionChoice,
    ToolCall,
    FunctionCall,
    ModelList,
    ModelInfo
)


@pytest.fixture
def mock_client():
    """Create a LocalLLMClient instance for testing."""
    return LocalLLMClient("http://localhost:1234/v1", "test-model")


@pytest.fixture
def sample_chat_message():
    """Create a sample ChatMessage for testing."""
    return ChatMessage(
        role="user",
        content="Hello, world!"
    )


@pytest.fixture
def sample_tool_call():
    """Create a sample ToolCall for testing."""
    return ToolCall(
        id="call_123",
        type="function",
        function=FunctionCall(
            name="test_function",
            arguments='{"arg1": "value1"}'
        )
    )


@pytest.fixture
def sample_choice(sample_chat_message):
    """Create a sample ChatCompletionChoice for testing."""
    return ChatCompletionChoice(
        index=0,
        message=sample_chat_message,
        finish_reason="stop"
    )


@pytest.fixture
def sample_chat_completion(sample_choice):
    """Create a sample ChatCompletion for testing."""
    return ChatCompletion(
        id="chatcmpl-123",
        object="chat.completion",
        created=1234567890,
        model="test-model",
        choices=[sample_choice]
    )


@pytest.fixture
def mock_response_with_content():
    """Create a mock response with content only."""
    mock_response = Mock()
    mock_response.json.return_value = {
        "id": "chatcmpl-123",
        "object": "chat.completion",
        "created": 1234567890,
        "model": "test-model",
        "choices": [{
            "index": 0,
            "message": {
                "role": "assistant",
                "content": "Test response"
            },
            "finish_reason": "stop"
        }],
        "usage": {
            "prompt_tokens": 10,
            "completion_tokens": 5,
            "total_tokens": 15
        }
    }
    mock_response.raise_for_status.return_value = None
    return mock_response


@pytest.fixture
def mock_response_with_thinking():
    """Create a mock response with thinking blocks."""
    mock_response = Mock()
    mock_response.json.return_value = {
        "id": "chatcmpl-123",
        "object": "chat.completion",
        "created": 1234567890,
        "model": "test-model",
        "choices": [{
            "index": 0,
            "message": {
                "role": "assistant",
                "content": "[THINK]This is my thinking process[/THINK]This is the final answer."
            },
            "finish_reason": "stop"
        }],
        "usage": {
            "prompt_tokens": 10,
            "completion_tokens": 5,
            "total_tokens": 15
        }
    }
    mock_response.raise_for_status.return_value = None
    return mock_response


@pytest.fixture
def mock_response_with_tool_calls():
    """Create a mock response with tool calls."""
    mock_response = Mock()
    mock_response.json.return_value = {
        "id": "chatcmpl-123",
        "object": "chat.completion",
        "created": 1234567890,
        "model": "test-model",
        "choices": [{
            "index": 0,
            "message": {
                "role": "assistant",
                "content": "I'll calculate that for you.",
                "tool_calls": [{
                    "id": "call_123",
                    "type": "function",
                    "function": {
                        "name": "math_calculator",
                        "arguments": '{"arg1": 5, "arg2": 3, "operation": "add"}'
                    }
                }]
            },
            "finish_reason": "tool_calls"
        }],
        "usage": {
            "prompt_tokens": 10,
            "completion_tokens": 5,
            "total_tokens": 15
        }
    }
    mock_response.raise_for_status.return_value = None
    return mock_response


@pytest.fixture
def mock_models_response():
    """Create a mock response for models endpoint."""
    mock_response = Mock()
    mock_response.json.return_value = {
        "object": "list",
        "data": [
            {
                "id": "test-model-1",
                "object": "model",
                "created": 1234567890,
                "owned_by": "test"
            },
            {
                "id": "test-model-2",
                "object": "model",
                "created": 1234567891,
                "owned_by": "test"
            }
        ]
    }
    mock_response.raise_for_status.return_value = None
    return mock_response


@pytest.fixture
def mock_requests_post():
    """Mock requests.post for API calls."""
    with patch('requests.post') as mock_post:
        yield mock_post


@pytest.fixture
def mock_requests_get():
    """Mock requests.get for API calls."""
    with patch('requests.get') as mock_get:
        yield mock_get


@pytest.fixture
def thinking_test_cases():
    """Provide test cases for thinking block extraction."""
    return [
        {
            "name": "simple_thinking",
            "input": "[THINK]I need to calculate 2+2[/THINK]The answer is 4.",
            "expected_clean": "The answer is 4.",
            "expected_thinking": "I need to calculate 2+2"
        },
        {
            "name": "no_thinking",
            "input": "Just a regular response.",
            "expected_clean": "Just a regular response.",
            "expected_thinking": ""
        },
        {
            "name": "multiple_thinking",
            "input": "[THINK]First thought[/THINK]Some text[THINK]Second thought[/THINK]Final answer.",
            "expected_clean": "Some textFinal answer.",
            "expected_thinking": "First thought\nSecond thought"
        },
        {
            "name": "thinking_only",
            "input": "[THINK]Just thinking, no response[/THINK]",
            "expected_clean": "",
            "expected_thinking": "Just thinking, no response"
        },
        {
            "name": "empty_thinking",
            "input": "[THINK][/THINK]Response here.",
            "expected_clean": "Response here.",
            "expected_thinking": ""
        }
    ]


@pytest.fixture
def tool_test_function():
    """Provide a simple test function for tool registration."""
    def test_add(a: float, b: float) -> dict:
        """Add two numbers together."""
        return {"result": a + b}
    return test_add


@pytest.fixture
def complex_tool_test_function():
    """Provide a complex test function with multiple parameters and types."""
    from typing import Literal

    def text_processor(
        text: str,
        operation: Literal["uppercase", "lowercase", "reverse"],
        repeat: int = 1
    ) -> dict:
        """Process text with various operations."""
        result = text

        if operation == "uppercase":
            result = result.upper()
        elif operation == "lowercase":
            result = result.lower()
        elif operation == "reverse":
            result = result[::-1]

        result = result * repeat

        return {
            "original": text,
            "operation": operation,
            "result": result,
            "length": len(result)
        }

    return text_processor


@pytest.fixture
def mock_agent_result():
    """Create a sample AgentResult for testing."""
    from local_llm_sdk.agents.models import AgentResult, AgentStatus
    return AgentResult(
        status=AgentStatus.SUCCESS,
        iterations=3,
        final_response="Task completed successfully",
        conversation=[],
        metadata={"total_tool_calls": 2}
    )


@pytest.fixture
def mock_react_agent(mock_client):
    """Create a ReACT agent with mocked client for testing."""
    from local_llm_sdk.agents import ReACT
    return ReACT(mock_client, name="TestAgent")


@pytest.fixture
def sample_agent_task():
    """Create a sample task string for agent testing."""
    return "Calculate 5 + 3 and tell me the result"


# ============================================================================
# Live LLM Testing Fixtures (uses .env configuration)
# ============================================================================

@pytest.fixture
def live_llm_client():
    """
    Create a LocalLLMClient configured from .env for live LLM testing.

    Requires:
    - LLM_BASE_URL in .env (default: http://localhost:1234/v1)
    - LLM_MODEL in .env (or auto-detection)
    - LM Studio/Ollama running at configured URL
    """
    base_url = os.getenv("LLM_BASE_URL", "http://localhost:1234/v1")
    model = os.getenv("LLM_MODEL")  # None triggers auto-detection
    timeout = int(os.getenv("LLM_TIMEOUT", "300"))

    client = LocalLLMClient(
        base_url=base_url,
        model=model,
        timeout=timeout
    )

    # Register built-in tools for agent testing
    client.register_tools_from(None)

    return client


@pytest.fixture
def live_react_agent(live_llm_client):
    """Create a ReACT agent with live LLM client for behavioral testing."""
    from local_llm_sdk.agents import ReACT
    return ReACT(live_llm_client)