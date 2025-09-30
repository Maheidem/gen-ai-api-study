"""
Tests for conversation state management in LocalLLMClient.

Tests the last_conversation_additions attribute and conversation state tracking
that enables ReACT agents and other multi-turn interactions to maintain proper
OpenAI-formatted message history including tool results.
"""

import pytest
from unittest.mock import Mock, patch
import json

from local_llm_sdk import LocalLLMClient, create_chat_message
from local_llm_sdk.models import ChatCompletion, ChatMessage


class TestConversationStateInitialization:
    """Test conversation state initialization."""

    def test_last_conversation_additions_initialization(self, mock_client):
        """Test that last_conversation_additions starts empty."""
        assert hasattr(mock_client, 'last_conversation_additions')
        assert mock_client.last_conversation_additions == []
        assert isinstance(mock_client.last_conversation_additions, list)

    def test_last_conversation_additions_cleared_on_chat(self, mock_client, mock_requests_post, mock_response_with_content):
        """Test that last_conversation_additions is cleared on new chat() call."""
        mock_requests_post.return_value = mock_response_with_content

        # Set some previous state
        mock_client.last_conversation_additions = [
            create_chat_message("assistant", "Previous message")
        ]

        # New chat should clear it
        mock_client.chat("Hello", use_tools=False)

        # Should be empty (no tools were used in this simple response)
        assert mock_client.last_conversation_additions == []


class TestHandleToolCallsReturnType:
    """Test _handle_tool_calls return type."""

    def test_handle_tool_calls_returns_tuple(self, mock_client, mock_response_with_tool_calls):
        """Test that _handle_tool_calls returns tuple of (response, messages)."""
        # Register a test tool
        @mock_client.register_tool("Add numbers")
        def math_calculator(arg1: float, arg2: float, operation: str) -> dict:
            return {"result": 8}

        # Create mock for second request (after tool execution)
        mock_final_response = Mock()
        mock_final_response.json.return_value = {
            "id": "chatcmpl-456",
            "object": "chat.completion",
            "created": 1234567891,
            "model": "test-model",
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "The result is 8."
                },
                "finish_reason": "stop"
            }],
            "usage": {
                "prompt_tokens": 20,
                "completion_tokens": 10,
                "total_tokens": 30
            }
        }
        mock_final_response.raise_for_status.return_value = None

        with patch('requests.post', return_value=mock_final_response):
            original_messages = [create_chat_message("user", "What is 5+3?")]

            # Parse the first response to get ChatCompletion
            from local_llm_sdk.models import ChatCompletion
            response = ChatCompletion.model_validate(mock_response_with_tool_calls.json())

            # Call _handle_tool_calls
            result = mock_client._handle_tool_calls(response, original_messages)

            # Should return tuple
            assert isinstance(result, tuple)
            assert len(result) == 2

            final_response, full_messages = result

            # First element should be ChatCompletion
            assert isinstance(final_response, ChatCompletion)

            # Second element should be list of messages
            assert isinstance(full_messages, list)
            assert all(isinstance(msg, ChatMessage) for msg in full_messages)


class TestConversationAdditionsPopulated:
    """Test that conversation additions are populated correctly."""

    def test_conversation_additions_populated(self, mock_client, mock_requests_post, mock_response_with_tool_calls):
        """Test that last_conversation_additions is populated after tool execution."""
        # Register a test tool
        @mock_client.register_tool("Calculator")
        def math_calculator(arg1: float, arg2: float, operation: str) -> dict:
            return {"result": 8}

        # Mock sequence: first response with tool calls, then final response
        mock_final_response = Mock()
        mock_final_response.json.return_value = {
            "id": "chatcmpl-456",
            "object": "chat.completion",
            "created": 1234567891,
            "model": "test-model",
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "The result is 8."
                },
                "finish_reason": "stop"
            }],
            "usage": {
                "prompt_tokens": 10,
                "completion_tokens": 5,
                "total_tokens": 15
            }
        }
        mock_final_response.raise_for_status.return_value = None

        mock_requests_post.side_effect = [
            mock_response_with_tool_calls,  # First: tool calls
            mock_final_response             # Second: final response
        ]

        # Make the chat call
        response = mock_client.chat("What is 5+3?", use_tools=True)

        # last_conversation_additions should now be populated
        assert len(mock_client.last_conversation_additions) > 0


class TestConversationAdditionsStructure:
    """Test the structure of conversation additions."""

    def test_conversation_additions_structure(self, mock_client, mock_requests_post, mock_response_with_tool_calls):
        """Test structure: [assistant_with_tools, tool_result, assistant_final]."""
        # Register a test tool
        @mock_client.register_tool("Calculator")
        def math_calculator(arg1: float, arg2: float, operation: str) -> dict:
            return {"result": 8}

        # Mock final response
        mock_final_response = Mock()
        mock_final_response.json.return_value = {
            "id": "chatcmpl-456",
            "object": "chat.completion",
            "created": 1234567891,
            "model": "test-model",
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "The result is 8."
                },
                "finish_reason": "stop"
            }],
            "usage": {
                "prompt_tokens": 10,
                "completion_tokens": 5,
                "total_tokens": 15
            }
        }
        mock_final_response.raise_for_status.return_value = None

        mock_requests_post.side_effect = [
            mock_response_with_tool_calls,  # First: tool calls
            mock_final_response             # Second: final response
        ]

        # Make the chat call
        response = mock_client.chat("What is 5+3?", use_tools=True)

        additions = mock_client.last_conversation_additions

        # Should have at least 2 messages: assistant_with_tools + tool_result
        # (final assistant is only added if it's different from the one we return)
        assert len(additions) >= 2

        # First message: assistant with tool calls
        assert additions[0].role == "assistant"
        assert hasattr(additions[0], 'tool_calls')
        assert additions[0].tool_calls is not None
        assert len(additions[0].tool_calls) > 0

        # Second message: tool result
        assert additions[1].role == "tool"
        assert hasattr(additions[1], 'tool_call_id')
        assert additions[1].tool_call_id == "call_123"


class TestConversationAdditionsMultipleTools:
    """Test conversation additions with multiple tool calls."""

    def test_conversation_additions_multiple_tools(self, mock_client, mock_requests_post):
        """Test conversation additions with multiple tool calls."""
        # Register test tools
        @mock_client.register_tool("Calculator")
        def math_calculator(arg1: float, arg2: float, operation: str) -> dict:
            return {"result": arg1 + arg2 if operation == "add" else arg1 - arg2}

        @mock_client.register_tool("Text")
        def text_transformer(text: str, operation: str) -> dict:
            return {"result": text.upper()}

        # Mock response with multiple tool calls
        mock_multi_tool_response = Mock()
        mock_multi_tool_response.json.return_value = {
            "id": "chatcmpl-123",
            "object": "chat.completion",
            "created": 1234567890,
            "model": "test-model",
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "I'll use multiple tools.",
                    "tool_calls": [
                        {
                            "id": "call_1",
                            "type": "function",
                            "function": {
                                "name": "math_calculator",
                                "arguments": '{"arg1": 5, "arg2": 3, "operation": "add"}'
                            }
                        },
                        {
                            "id": "call_2",
                            "type": "function",
                            "function": {
                                "name": "text_transformer",
                                "arguments": '{"text": "hello", "operation": "uppercase"}'
                            }
                        }
                    ]
                },
                "finish_reason": "tool_calls"
            }],
            "usage": {
                "prompt_tokens": 10,
                "completion_tokens": 5,
                "total_tokens": 15
            }
        }
        mock_multi_tool_response.raise_for_status.return_value = None

        # Mock final response
        mock_final_response = Mock()
        mock_final_response.json.return_value = {
            "id": "chatcmpl-456",
            "object": "chat.completion",
            "created": 1234567891,
            "model": "test-model",
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "Results: 8 and HELLO"
                },
                "finish_reason": "stop"
            }],
            "usage": {
                "prompt_tokens": 10,
                "completion_tokens": 5,
                "total_tokens": 15
            }
        }
        mock_final_response.raise_for_status.return_value = None

        mock_requests_post.side_effect = [
            mock_multi_tool_response,  # First: multiple tool calls
            mock_final_response        # Second: final response
        ]

        # Make the chat call
        response = mock_client.chat("Do both", use_tools=True)

        additions = mock_client.last_conversation_additions

        # Should have: 1 assistant message + 2 tool results = at least 3 messages
        assert len(additions) >= 3

        # Count tool result messages
        tool_messages = [msg for msg in additions if msg.role == "tool"]
        assert len(tool_messages) == 2


class TestConversationAdditionsNoTools:
    """Test conversation additions when no tools are used."""

    def test_conversation_additions_no_tools(self, mock_client, mock_requests_post, mock_response_with_content):
        """Test that additions remain empty when no tools are used."""
        mock_requests_post.return_value = mock_response_with_content

        # Make the chat call without tools
        response = mock_client.chat("Hello", use_tools=False)

        # Should be empty since no tools were executed
        assert mock_client.last_conversation_additions == []


class TestConversationStateInReactAgent:
    """Test that ReACT agent uses conversation additions correctly."""

    def test_conversation_state_in_react_agent(self, mock_client, mock_requests_post):
        """Test that ReACT agent extends messages with last_conversation_additions."""
        # Register a tool
        @mock_client.register_tool("Calculator")
        def math_calculator(arg1: float, arg2: float, operation: str) -> dict:
            return {"result": 8}

        # Mock tool call response
        mock_tool_response = Mock()
        mock_tool_response.json.return_value = {
            "id": "chatcmpl-1",
            "object": "chat.completion",
            "created": 1234567890,
            "model": "test-model",
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "Using calculator.",
                    "tool_calls": [{
                        "id": "call_1",
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
        mock_tool_response.raise_for_status.return_value = None

        # Mock final response
        mock_final_response = Mock()
        mock_final_response.json.return_value = {
            "id": "chatcmpl-2",
            "object": "chat.completion",
            "created": 1234567891,
            "model": "test-model",
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "Result is 8. TASK_COMPLETE"
                },
                "finish_reason": "stop"
            }],
            "usage": {
                "prompt_tokens": 10,
                "completion_tokens": 5,
                "total_tokens": 15
            }
        }
        mock_final_response.raise_for_status.return_value = None

        mock_requests_post.side_effect = [
            mock_tool_response,   # First iteration: tool call
            mock_final_response,  # After tool execution
            mock_final_response   # Second iteration (will complete)
        ]

        # Run ReACT agent
        from local_llm_sdk.agents import ReACT
        agent = ReACT(mock_client)
        result = agent.run(task="Calculate 5+3", max_iterations=2, verbose=False)

        # Agent should have completed successfully
        from local_llm_sdk.agents.models import AgentStatus
        assert result.status == AgentStatus.SUCCESS

        # Conversation should include tool result messages
        tool_messages = [msg for msg in result.conversation if msg.role == "tool"]
        assert len(tool_messages) > 0


class TestToolResultMessages:
    """Test tool result message format."""

    def test_tool_results_in_messages(self, mock_client, mock_requests_post, mock_response_with_tool_calls):
        """Test that tool result messages have correct format."""
        # Register a tool
        @mock_client.register_tool("Calculator")
        def math_calculator(arg1: float, arg2: float, operation: str) -> dict:
            return {"result": 8, "operation": operation}

        # Mock final response
        mock_final_response = Mock()
        mock_final_response.json.return_value = {
            "id": "chatcmpl-456",
            "object": "chat.completion",
            "created": 1234567891,
            "model": "test-model",
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "The result is 8."
                },
                "finish_reason": "stop"
            }],
            "usage": {
                "prompt_tokens": 10,
                "completion_tokens": 5,
                "total_tokens": 15
            }
        }
        mock_final_response.raise_for_status.return_value = None

        mock_requests_post.side_effect = [
            mock_response_with_tool_calls,  # First: tool calls
            mock_final_response             # Second: final response
        ]

        # Make the chat call
        response = mock_client.chat("What is 5+3?", use_tools=True)

        additions = mock_client.last_conversation_additions

        # Find tool message
        tool_messages = [msg for msg in additions if msg.role == "tool"]
        assert len(tool_messages) > 0

        tool_msg = tool_messages[0]

        # Should have tool_call_id
        assert hasattr(tool_msg, 'tool_call_id')
        assert tool_msg.tool_call_id == "call_123"

        # Should have content with tool result
        assert tool_msg.content is not None

        # Content should be JSON string
        result_data = json.loads(tool_msg.content)
        assert "result" in result_data


class TestConversationContinuity:
    """Test that conversation messages maintain proper OpenAI format."""

    def test_conversation_continuity(self, mock_client, mock_requests_post, mock_response_with_tool_calls):
        """Test that messages maintain proper OpenAI format for continuation."""
        # Register a tool
        @mock_client.register_tool("Calculator")
        def math_calculator(arg1: float, arg2: float, operation: str) -> dict:
            return {"result": 8}

        # Mock final response
        mock_final_response = Mock()
        mock_final_response.json.return_value = {
            "id": "chatcmpl-456",
            "object": "chat.completion",
            "created": 1234567891,
            "model": "test-model",
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "The result is 8."
                },
                "finish_reason": "stop"
            }],
            "usage": {
                "prompt_tokens": 10,
                "completion_tokens": 5,
                "total_tokens": 15
            }
        }
        mock_final_response.raise_for_status.return_value = None

        mock_requests_post.side_effect = [
            mock_response_with_tool_calls,  # First: tool calls
            mock_final_response             # Second: final response
        ]

        # Start with initial messages
        messages = [
            create_chat_message("system", "You are a helpful assistant."),
            create_chat_message("user", "What is 5+3?")
        ]

        # Make the chat call
        response = mock_client.chat(messages, use_tools=True, return_full=True)

        # Build complete conversation
        complete_conversation = messages + mock_client.last_conversation_additions

        # Validate message sequence
        roles = [msg.role for msg in complete_conversation]

        # Should start with system and user
        assert roles[0] == "system"
        assert roles[1] == "user"

        # After user, should have assistant with tool calls
        assert roles[2] == "assistant"

        # Then tool result
        assert roles[3] == "tool"

        # All messages should be valid ChatMessage objects
        assert all(isinstance(msg, ChatMessage) for msg in complete_conversation)

        # All messages should have required fields
        for msg in complete_conversation:
            assert hasattr(msg, 'role')
            assert msg.role in ["system", "user", "assistant", "tool"]
