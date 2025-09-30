"""
Tests for LocalLLMClient functionality.
"""

import pytest
from unittest.mock import Mock, patch, call
import json

from local_llm_sdk import LocalLLMClient, create_chat_message
from local_llm_sdk.models import ChatCompletion, ChatMessage
from local_llm_sdk.tools import builtin


class TestLocalLLMClientInit:
    """Test LocalLLMClient initialization."""

    def test_init_with_defaults(self):
        """Test client initialization with default parameters."""
        client = LocalLLMClient()

        assert client.base_url == "http://localhost:1234/v1"
        assert client.default_model is None
        assert client.last_tool_calls == []
        assert client.last_thinking == ""
        assert len(client.tools.list_tools()) == 0

    def test_init_with_custom_params(self):
        """Test client initialization with custom parameters."""
        base_url = "http://custom:8080/v1"
        model = "custom-model"

        client = LocalLLMClient(base_url, model)

        assert client.base_url == "http://custom:8080/v1"
        assert client.default_model == "custom-model"

    def test_init_strips_trailing_slash(self):
        """Test that trailing slash is stripped from base_url."""
        client = LocalLLMClient("http://localhost:1234/v1/")
        assert client.base_url == "http://localhost:1234/v1"

    def test_endpoints_are_set_correctly(self):
        """Test that API endpoints are constructed correctly."""
        client = LocalLLMClient("http://localhost:1234/v1")

        expected_endpoints = {
            "models": "http://localhost:1234/v1/models",
            "chat": "http://localhost:1234/v1/chat/completions",
            "completions": "http://localhost:1234/v1/completions",
            "embeddings": "http://localhost:1234/v1/embeddings"
        }

        assert client.endpoints == expected_endpoints


class TestLocalLLMClientToolRegistration:
    """Test tool registration functionality."""

    def test_register_tool_decorator(self, mock_client):
        """Test registering a tool using the decorator."""
        @mock_client.register_tool("Test function")
        def test_func(x: int) -> dict:
            return {"result": x * 2}

        tools = mock_client.tools.list_tools()
        assert "test_func" in tools
        assert len(mock_client.tools.get_schemas()) == 1

    def test_register_tools_from_builtin(self, mock_client):
        """Test registering tools from builtin module."""
        mock_client.register_tools_from(builtin)

        tools = mock_client.tools.list_tools()
        # Should have at least the builtin tools
        assert len(tools) > 0
        assert "math_calculator" in tools

    def test_multiple_tool_registration(self, mock_client):
        """Test registering multiple tools."""
        @mock_client.register_tool("Function 1")
        def func1(x: int) -> dict:
            return {"result": x}

        @mock_client.register_tool("Function 2")
        def func2(y: str) -> dict:
            return {"result": y}

        tools = mock_client.tools.list_tools()
        assert "func1" in tools
        assert "func2" in tools
        assert len(tools) == 2


class TestLocalLLMClientThinkingExtraction:
    """Test thinking block extraction."""

    def test_extract_thinking_simple(self, mock_client):
        """Test basic thinking extraction."""
        content = "[THINK]I need to calculate[/THINK]The answer is 42."
        clean, thinking = mock_client._extract_thinking(content)

        assert clean == "The answer is 42."
        assert thinking == "I need to calculate"

    def test_extract_thinking_no_blocks(self, mock_client):
        """Test extraction when no thinking blocks are present."""
        content = "Just a regular response."
        clean, thinking = mock_client._extract_thinking(content)

        assert clean == "Just a regular response."
        assert thinking == ""

    def test_extract_thinking_multiple_blocks(self, mock_client):
        """Test extraction with multiple thinking blocks."""
        content = "[THINK]First[/THINK]Middle[THINK]Second[/THINK]End"
        clean, thinking = mock_client._extract_thinking(content)

        assert clean == "MiddleEnd"
        assert thinking == "First\nSecond"

    def test_extract_thinking_empty_content(self, mock_client):
        """Test extraction with empty or None content."""
        clean, thinking = mock_client._extract_thinking("")
        assert clean == ""
        assert thinking == ""

        clean, thinking = mock_client._extract_thinking(None)
        assert clean is None
        assert thinking == ""

    @pytest.mark.parametrize("test_case", [
        {
            "input": "[THINK]Simple thinking[/THINK]Response",
            "expected_clean": "Response",
            "expected_thinking": "Simple thinking"
        },
        {
            "input": "[THINK][/THINK]Empty thinking",
            "expected_clean": "Empty thinking",
            "expected_thinking": ""
        },
        {
            "input": "[THINK]Only thinking[/THINK]",
            "expected_clean": "",
            "expected_thinking": "Only thinking"
        }
    ])
    def test_extract_thinking_parametrized(self, mock_client, test_case):
        """Test thinking extraction with various inputs."""
        clean, thinking = mock_client._extract_thinking(test_case["input"])
        assert clean == test_case["expected_clean"]
        assert thinking == test_case["expected_thinking"]


class TestLocalLLMClientChat:
    """Test chat functionality."""

    def test_chat_string_input_conversion(self, mock_client, mock_requests_post, mock_response_with_content):
        """Test that string input is converted to messages."""
        mock_requests_post.return_value = mock_response_with_content

        response = mock_client.chat("Hello")

        # Check that the request was made with proper message structure
        mock_requests_post.assert_called_once()
        call_args = mock_requests_post.call_args
        request_data = call_args[1]["json"]

        assert "messages" in request_data
        assert len(request_data["messages"]) == 2  # system + user
        assert request_data["messages"][0]["role"] == "system"
        assert request_data["messages"][1]["role"] == "user"
        assert request_data["messages"][1]["content"] == "Hello"

    def test_chat_with_tools_disabled(self, mock_client, mock_requests_post, mock_response_with_content):
        """Test chat with tools disabled."""
        mock_client.register_tools_from(builtin)
        mock_requests_post.return_value = mock_response_with_content

        response = mock_client.chat("Hello", use_tools=False)

        # Check that tools were not included in request
        call_args = mock_requests_post.call_args
        request_data = call_args[1]["json"]

        assert "tools" not in request_data
        assert response == "Test response"

    def test_chat_with_tools_enabled(self, mock_client, mock_requests_post, mock_response_with_content):
        """Test chat with tools enabled."""
        mock_client.register_tools_from(builtin)
        mock_requests_post.return_value = mock_response_with_content

        response = mock_client.chat("Hello", use_tools=True)

        # Check that tools were included in request
        call_args = mock_requests_post.call_args
        request_data = call_args[1]["json"]

        assert "tools" in request_data
        assert len(request_data["tools"]) > 0
        assert request_data["tool_choice"] == "auto"

    def test_chat_clears_previous_state(self, mock_client, mock_requests_post, mock_response_with_content):
        """Test that chat clears previous tool calls and thinking."""
        mock_requests_post.return_value = mock_response_with_content

        # Set some previous state
        mock_client.last_tool_calls = ["previous_call"]
        mock_client.last_thinking = "previous thinking"

        mock_client.chat("Hello")

        # State should be cleared at start of new request
        # (will be empty since no tools/thinking in this response)
        assert mock_client.last_tool_calls == []
        assert mock_client.last_thinking == ""

    def test_chat_with_thinking_include_false(self, mock_client, mock_requests_post, mock_response_with_thinking):
        """Test chat with thinking blocks, include_thinking=False."""
        mock_requests_post.return_value = mock_response_with_thinking

        response = mock_client.chat("Hello", include_thinking=False)

        # Response should only contain clean content
        assert response == "This is the final answer."
        # But thinking should be stored in client
        assert mock_client.last_thinking == "This is my thinking process"

    def test_chat_with_thinking_include_true(self, mock_client, mock_requests_post, mock_response_with_thinking):
        """Test chat with thinking blocks, include_thinking=True."""
        mock_requests_post.return_value = mock_response_with_thinking

        response = mock_client.chat("Hello", include_thinking=True)

        # Response should include formatted thinking
        expected = "**Thinking:**\nThis is my thinking process\n\n**Response:**\nThis is the final answer."
        assert response == expected

    def test_chat_return_full_true(self, mock_client, mock_requests_post, mock_response_with_content):
        """Test chat with return_full_response=True."""
        mock_requests_post.return_value = mock_response_with_content

        response = mock_client.chat("Hello", return_full_response=True)

        # Should return ChatCompletion object
        assert isinstance(response, ChatCompletion)
        assert response.choices[0].message.content == "Test response"

    def test_chat_with_custom_parameters(self, mock_client, mock_requests_post, mock_response_with_content):
        """Test chat with custom temperature and max_tokens."""
        mock_requests_post.return_value = mock_response_with_content

        mock_client.chat("Hello", temperature=0.5, max_tokens=100)

        call_args = mock_requests_post.call_args
        request_data = call_args[1]["json"]

        assert request_data["temperature"] == 0.5
        assert request_data["max_tokens"] == 100


class TestLocalLLMClientToolChoice:
    """Test tool_choice parameter functionality."""

    def test_tool_choice_auto(self, mock_client, mock_requests_post, mock_response_with_content):
        """Test tool_choice='auto' (default behavior)."""
        mock_requests_post.return_value = mock_response_with_content
        mock_client.register_tools_from(None)

        mock_client.chat("Hello", use_tools=True, tool_choice="auto")

        call_args = mock_requests_post.call_args
        request_data = call_args[1]["json"]

        assert "tools" in request_data
        assert request_data["tool_choice"] == "auto"

    def test_tool_choice_required(self, mock_client, mock_requests_post, mock_response_with_content):
        """Test tool_choice='required' forces tool usage."""
        mock_requests_post.return_value = mock_response_with_content
        mock_client.register_tools_from(None)

        mock_client.chat("Calculate 5 * 10", use_tools=True, tool_choice="required")

        call_args = mock_requests_post.call_args
        request_data = call_args[1]["json"]

        assert "tools" in request_data
        assert request_data["tool_choice"] == "required"

    def test_tool_choice_none(self, mock_client, mock_requests_post, mock_response_with_content):
        """Test tool_choice='none' prevents tool usage."""
        mock_requests_post.return_value = mock_response_with_content
        mock_client.register_tools_from(None)

        mock_client.chat("Explain math", use_tools=True, tool_choice="none")

        call_args = mock_requests_post.call_args
        request_data = call_args[1]["json"]

        assert "tools" in request_data
        assert request_data["tool_choice"] == "none"

    def test_tool_choice_specific_function(self, mock_client, mock_requests_post, mock_response_with_content):
        """Test tool_choice with specific function."""
        mock_requests_post.return_value = mock_response_with_content
        mock_client.register_tools_from(None)

        specific_tool = {"type": "function", "function": {"name": "math_calculator"}}
        mock_client.chat("Calculate", use_tools=True, tool_choice=specific_tool)

        call_args = mock_requests_post.call_args
        request_data = call_args[1]["json"]

        assert "tools" in request_data
        assert request_data["tool_choice"] == specific_tool

    def test_tool_choice_default_is_auto(self, mock_client, mock_requests_post, mock_response_with_content):
        """Test that default tool_choice is 'auto'."""
        mock_requests_post.return_value = mock_response_with_content
        mock_client.register_tools_from(None)

        mock_client.chat("Hello", use_tools=True)

        call_args = mock_requests_post.call_args
        request_data = call_args[1]["json"]

        assert request_data["tool_choice"] == "auto"

    def test_tool_choice_without_tools(self, mock_client, mock_requests_post, mock_response_with_content):
        """Test that tool_choice is ignored when use_tools=False."""
        mock_requests_post.return_value = mock_response_with_content

        mock_client.chat("Hello", use_tools=False, tool_choice="required")

        call_args = mock_requests_post.call_args
        request_data = call_args[1]["json"]

        # Should not include tools or tool_choice when use_tools=False
        assert "tools" not in request_data
        assert "tool_choice" not in request_data


class TestLocalLLMClientChatSimple:
    """Test chat_simple functionality."""

    def test_chat_simple(self, mock_client, mock_requests_post, mock_response_with_content):
        """Test simple chat without tools."""
        mock_requests_post.return_value = mock_response_with_content

        response = mock_client.chat_simple("Hello")

        # Should return string response
        assert response == "Test response"

        # Check that tools were not used
        call_args = mock_requests_post.call_args
        request_data = call_args[1]["json"]
        assert "tools" not in request_data


class TestLocalLLMClientChatWithHistory:
    """Test chat_with_history functionality."""

    def test_chat_with_history_empty(self, mock_client, mock_requests_post, mock_response_with_content):
        """Test chat with empty history."""
        mock_requests_post.return_value = mock_response_with_content

        response, new_history = mock_client.chat_with_history("Hello", [])

        assert response == "Test response"
        assert len(new_history) == 2  # user message + assistant response
        assert new_history[0].role == "user"
        assert new_history[0].content == "Hello"
        assert new_history[1].role == "assistant"
        assert new_history[1].content == "Test response"

    def test_chat_with_history_existing(self, mock_client, mock_requests_post, mock_response_with_content):
        """Test chat with existing history."""
        mock_requests_post.return_value = mock_response_with_content

        existing_history = [
            create_chat_message("user", "Previous message"),
            create_chat_message("assistant", "Previous response")
        ]

        response, new_history = mock_client.chat_with_history("New message", existing_history)

        assert response == "Test response"
        assert len(new_history) == 4  # 2 existing + user + assistant
        assert new_history[2].content == "New message"
        assert new_history[3].content == "Test response"


class TestLocalLLMClientListModels:
    """Test list_models functionality."""

    def test_list_models_success(self, mock_client, mock_requests_get, mock_models_response):
        """Test successful model listing."""
        mock_requests_get.return_value = mock_models_response

        models = mock_client.list_models()

        mock_requests_get.assert_called_once_with(
            "http://localhost:1234/v1/models",
            timeout=5
        )

        assert len(models.data) == 2
        assert models.data[0].id == "test-model-1"
        assert models.data[1].id == "test-model-2"

    def test_list_models_network_error(self, mock_client, mock_requests_get):
        """Test list_models with network error."""
        mock_requests_get.side_effect = Exception("Network error")

        with pytest.raises(Exception, match="Network error"):
            mock_client.list_models()


class TestLocalLLMClientRepr:
    """Test string representation."""

    def test_repr_without_tools(self, mock_client):
        """Test __repr__ without registered tools."""
        repr_str = repr(mock_client)
        expected = "LocalLLMClient(base_url='http://localhost:1234/v1', model='test-model', tools=0)"
        assert repr_str == expected

    def test_repr_with_tools(self, mock_client):
        """Test __repr__ with registered tools."""
        mock_client.register_tools_from(builtin)
        repr_str = repr(mock_client)
        tools_count = len(mock_client.tools.list_tools())
        expected = f"LocalLLMClient(base_url='http://localhost:1234/v1', model='test-model', tools={tools_count})"
        assert repr_str == expected


class TestLocalLLMClientErrorHandling:
    """Test error handling in client methods."""

    def test_chat_api_error(self, mock_client, mock_requests_post):
        """Test chat with API error response."""
        mock_response = Mock()
        mock_response.raise_for_status.side_effect = Exception("API Error")
        mock_requests_post.return_value = mock_response

        with pytest.raises(Exception, match="API Error"):
            mock_client.chat("Hello")

    def test_send_request_timeout(self, mock_client, mock_requests_post):
        """Test request timeout handling."""
        mock_requests_post.side_effect = Exception("Timeout")

        with pytest.raises(Exception, match="Timeout"):
            mock_client.chat("Hello")


class TestLocalLLMClientNewFeatures:
    """Test new SDK features (timeout, conversation additions, react)."""

    def test_timeout_configuration(self):
        """Test timeout configuration."""
        # Test default timeout
        client_default = LocalLLMClient("http://localhost:1234/v1", "test-model")
        assert client_default.timeout == 300

        # Test custom timeout
        client_custom = LocalLLMClient("http://localhost:1234/v1", "test-model", timeout=600)
        assert client_custom.timeout == 600

    def test_last_conversation_additions_attribute(self, mock_client):
        """Test last_conversation_additions attribute exists and initializes empty."""
        assert hasattr(mock_client, 'last_conversation_additions')
        assert mock_client.last_conversation_additions == []

    @patch('local_llm_sdk.client.requests.post')
    def test_conversation_additions_cleared_on_chat(self, mock_post, mock_client, mock_response_with_content):
        """Test last_conversation_additions is cleared on new chat call."""
        mock_post.return_value = mock_response_with_content

        # Manually set some additions
        mock_client.last_conversation_additions = [Mock(), Mock()]

        # Call chat - should clear
        mock_client.chat("test")

        assert mock_client.last_conversation_additions == []

    def test_handle_tool_calls_return_type(self, mock_client):
        """Test _handle_tool_calls returns tuple of (response, messages)."""
        from local_llm_sdk.models import ChatCompletion, ChatCompletionChoice
        from local_llm_sdk import create_chat_message

        # Create mock response with no tool calls (simplest case)
        mock_message = create_chat_message("assistant", "Response without tools")
        mock_message.tool_calls = []  # Empty list, not None
        mock_choice = ChatCompletionChoice(
            index=0,
            message=mock_message,
            finish_reason="stop"
        )
        mock_response = Mock(spec=ChatCompletion)
        mock_response.choices = [mock_choice]

        # Create proper ChatMessage list
        messages = [create_chat_message("user", "Test message")]

        # Mock _send_request to avoid actual API calls
        with patch.object(mock_client, '_send_request', return_value=mock_response):
            result = mock_client._handle_tool_calls(mock_response, messages)

        # Should return tuple
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_react_convenience_method(self, mock_client):
        """Test client has react() convenience method for agents."""
        assert hasattr(mock_client, 'react')
        assert callable(mock_client.react)


class TestLocalLLMClientRegisterTools:
    """Test register_tools() method for registering tools from a list."""

    def test_register_single_tool_from_list(self, mock_client):
        """Test registering a single tool via list."""
        def test_func(x: int) -> dict:
            """Test function"""
            return {"result": x * 2}

        # Register the tool
        result = mock_client.register_tools([test_func])

        # Should return client for chaining
        assert result is mock_client

        # Tool should be registered
        assert "test_func" in mock_client.tools.list_tools()

        # Should have schema
        schemas = mock_client.tools.get_schemas()
        assert len(schemas) == 1
        assert schemas[0].function.name == "test_func"
        assert schemas[0].function.description == "Test function"

    def test_register_multiple_tools_from_list(self, mock_client):
        """Test registering multiple tools at once."""
        def add(a: float, b: float) -> dict:
            """Add two numbers"""
            return {"result": a + b}

        def multiply(a: float, b: float) -> dict:
            """Multiply two numbers"""
            return {"result": a * b}

        def divide(a: float, b: float) -> dict:
            """Divide two numbers"""
            return {"result": a / b if b != 0 else None}

        # Register all tools
        tools = [add, multiply, divide]
        mock_client.register_tools(tools)

        # All should be registered
        registered = mock_client.tools.list_tools()
        assert "add" in registered
        assert "multiply" in registered
        assert "divide" in registered

        # Should have 3 schemas
        schemas = mock_client.tools.get_schemas()
        assert len(schemas) == 3

    def test_registered_tools_are_executable(self, mock_client):
        """Test that tools registered via list are executable."""
        def square(x: float) -> dict:
            """Calculate square"""
            return {"result": x ** 2}

        mock_client.register_tools([square])

        # Execute the tool
        result_json = mock_client.tools.execute("square", {"x": 5})
        result = json.loads(result_json)

        assert result["success"] is True
        assert result["result"] == 25

    def test_register_tools_with_no_docstring(self, mock_client):
        """Test registering tools without docstrings."""
        def no_doc(x: int) -> dict:
            return {"result": x}

        mock_client.register_tools([no_doc])

        schemas = mock_client.tools.get_schemas()
        assert len(schemas) == 1
        # Should use function name as description
        assert schemas[0].function.description == "Function: no_doc"

    def test_register_tools_with_optional_params(self, mock_client):
        """Test registering tools with optional parameters."""
        def greet(name: str, greeting: str = "Hello") -> dict:
            """Greet someone"""
            return {"message": f"{greeting}, {name}!"}

        mock_client.register_tools([greet])

        schemas = mock_client.tools.get_schemas()
        params = schemas[0].function.parameters

        # Should have both parameters
        assert "name" in params["properties"]
        assert "greeting" in params["properties"]

        # Only name should be required
        assert "name" in params["required"]
        assert "greeting" not in params["required"]

    def test_register_empty_list(self, mock_client):
        """Test registering empty list doesn't fail."""
        result = mock_client.register_tools([])

        # Should return client (for chaining)
        assert result is mock_client

        # No tools should be added
        assert len(mock_client.tools.list_tools()) == 0

    def test_register_tools_preserves_existing(self, mock_client):
        """Test that register_tools doesn't remove existing tools."""
        def first_tool(x: int) -> dict:
            """First tool"""
            return {"result": x}

        def second_tool(y: int) -> dict:
            """Second tool"""
            return {"result": y * 2}

        # Register first tool
        mock_client.register_tools([first_tool])
        assert len(mock_client.tools.list_tools()) == 1

        # Register second tool
        mock_client.register_tools([second_tool])

        # Both should exist
        registered = mock_client.tools.list_tools()
        assert len(registered) == 2
        assert "first_tool" in registered
        assert "second_tool" in registered

    def test_register_tools_method_chaining(self, mock_client):
        """Test that register_tools supports method chaining."""
        def tool1(x: int) -> dict:
            return {"result": x}

        def tool2(x: int) -> dict:
            return {"result": x * 2}

        # Should support chaining
        result = (mock_client
                  .register_tools([tool1])
                  .register_tools([tool2]))

        assert result is mock_client
        assert len(mock_client.tools.list_tools()) == 2