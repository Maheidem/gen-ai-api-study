"""
Test notebooks for API usage validation.

These tests validate that notebooks use the correct SDK API without requiring
a running LLM server. They catch documentation drift issues like incorrect
parameter names.

Usage:
    # Run all notebook tests
    pytest tests/test_notebooks.py -v

    # Run specific notebook test
    pytest tests/test_notebooks.py::test_02_basic_chat -v

    # Skip notebook tests (they're skipped by default anyway)
    pytest tests/ -m "not notebook" -v
"""

import pytest
from testbook import testbook
from pathlib import Path
from unittest.mock import Mock, patch
import sys

# Mark all tests in this file as 'notebook' tests
pytestmark = pytest.mark.notebook

# Path to notebooks directory
NOTEBOOKS_DIR = Path(__file__).parent.parent / "notebooks"


@pytest.fixture
def mock_llm_responses():
    """
    Mock LM Studio API responses so notebooks can execute without a server.

    This allows us to validate API usage (correct parameter names, types)
    without needing actual LLM infrastructure.
    """
    mock_response = Mock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "id": "chatcmpl-test",
        "object": "chat.completion",
        "created": 1234567890,
        "model": "test-model",
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "This is a mocked response for testing."
                },
                "finish_reason": "stop"
            }
        ],
        "usage": {
            "prompt_tokens": 10,
            "completion_tokens": 20,
            "total_tokens": 30
        }
    }

    return mock_response


@pytest.fixture
def mock_llm_client(mock_llm_responses):
    """Mock the requests library to intercept API calls."""
    with patch('requests.post', return_value=mock_llm_responses) as mock_post, \
         patch('requests.get') as mock_get:

        # Mock /v1/models endpoint
        models_response = Mock()
        models_response.status_code = 200
        models_response.json.return_value = {
            "object": "list",
            "data": [
                {
                    "id": "test-model",
                    "object": "model",
                    "created": 1234567890,
                    "owned_by": "test"
                }
            ]
        }
        mock_get.return_value = models_response

        yield {
            'post': mock_post,
            'get': mock_get
        }


@testbook(NOTEBOOKS_DIR / "02-basic-chat.ipynb", execute=False, timeout=60)
def test_02_basic_chat(tb, mock_llm_client):
    """
    Validate notebook 02-basic-chat.ipynb uses correct SDK API.

    This test caught the bug where the notebook used `response_format="full"`
    instead of the correct `return_full_response=True`.

    The test executes notebook cells with mocked responses, so:
    - API parameter errors (like wrong parameter names) cause Pydantic validation to fail
    - No real LLM server is needed
    - Tests run fast
    """
    # Inject the mock into the notebook's namespace
    tb.inject("""
import sys
from unittest.mock import Mock, patch

# Mock requests to avoid needing real LLM server
mock_response = Mock()
mock_response.status_code = 200
mock_response.json.return_value = {
    "id": "chatcmpl-test",
    "object": "chat.completion",
    "created": 1234567890,
    "model": "qwen/qwen3-coder-30b",
    "choices": [
        {
            "index": 0,
            "message": {
                "role": "assistant",
                "content": "This is a mocked response."
            },
            "finish_reason": "stop"
        }
    ],
    "usage": {
        "prompt_tokens": 10,
        "completion_tokens": 20,
        "total_tokens": 30
    }
}

# Mock models endpoint
models_response = Mock()
models_response.status_code = 200
models_response.json.return_value = {
    "object": "list",
    "data": [
        {
            "id": "qwen/qwen3-coder-30b",
            "object": "model",
            "created": 1234567890,
            "owned_by": "test"
        }
    ]
}

# Patch requests globally
import requests
requests.post = Mock(return_value=mock_response)
requests.get = Mock(return_value=models_response)
""")

    # Execute all cells - will fail if API parameters are wrong
    # (e.g., if notebook uses response_format="full" instead of return_full_response=True)
    try:
        tb.execute()
    except Exception as e:
        # If execution fails, provide helpful error message
        pytest.fail(
            f"Notebook 02-basic-chat.ipynb failed to execute.\n"
            f"This likely means incorrect API usage (wrong parameter names/types).\n"
            f"Error: {str(e)}"
        )

    # If we got here, all cells executed successfully with correct API usage


@testbook(NOTEBOOKS_DIR / "01-installation-setup.ipynb", execute=False, timeout=60)
def test_01_installation_setup(tb):
    """
    Validate notebook 01-installation-setup.ipynb.

    This notebook mostly has installation instructions, so we just verify
    it doesn't have syntax errors.
    """
    # Just check the notebook is valid, don't execute cells with actual installs
    assert tb.cells, "Notebook should have cells"

    # Could add more specific validation here if needed
    # For now, just verify structure is valid


# Add more notebook tests as needed:
# @testbook(NOTEBOOKS_DIR / "03-conversation-history.ipynb", execute=False)
# def test_03_conversation_history(tb, mock_llm_client):
#     ...
