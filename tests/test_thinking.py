"""
Tests specifically for thinking blocks extraction functionality.
"""

import pytest
from unittest.mock import Mock, patch

from local_llm_sdk import LocalLLMClient
from local_llm_sdk.models import ChatCompletion, ChatCompletionChoice, ChatMessage


class TestThinkingBlockExtraction:
    """Test the core thinking block extraction logic."""

    def test_extract_simple_thinking_block(self, mock_client):
        """Test extraction of a simple thinking block."""
        content = "[THINK]I need to solve this step by step[/THINK]The answer is 42."
        clean, thinking = mock_client._extract_thinking(content)

        assert clean == "The answer is 42."
        assert thinking == "I need to solve this step by step"

    def test_extract_no_thinking_blocks(self, mock_client):
        """Test content with no thinking blocks."""
        content = "This is just a normal response without thinking."
        clean, thinking = mock_client._extract_thinking(content)

        assert clean == "This is just a normal response without thinking."
        assert thinking == ""

    def test_extract_multiple_thinking_blocks(self, mock_client):
        """Test extraction of multiple thinking blocks."""
        content = "[THINK]First thought[/THINK]Some text[THINK]Second thought[/THINK]Final result."
        clean, thinking = mock_client._extract_thinking(content)

        assert clean == "Some textFinal result."
        assert thinking == "First thought\nSecond thought"

    def test_extract_consecutive_thinking_blocks(self, mock_client):
        """Test consecutive thinking blocks with no content between them."""
        content = "[THINK]Step 1[/THINK][THINK]Step 2[/THINK][THINK]Step 3[/THINK]Conclusion."
        clean, thinking = mock_client._extract_thinking(content)

        assert clean == "Conclusion."
        assert thinking == "Step 1\nStep 2\nStep 3"

    def test_extract_thinking_only_content(self, mock_client):
        """Test content that is only thinking blocks."""
        content = "[THINK]Just thinking, no response[/THINK]"
        clean, thinking = mock_client._extract_thinking(content)

        assert clean == ""
        assert thinking == "Just thinking, no response"

    def test_extract_empty_thinking_block(self, mock_client):
        """Test empty thinking blocks."""
        content = "[THINK][/THINK]Response here."
        clean, thinking = mock_client._extract_thinking(content)

        assert clean == "Response here."
        assert thinking == ""

    def test_extract_multiline_thinking(self, mock_client):
        """Test thinking blocks with multiline content."""
        content = "[THINK]Line 1\nLine 2\nLine 3[/THINK]Final answer."
        clean, thinking = mock_client._extract_thinking(content)

        assert clean == "Final answer."
        assert thinking == "Line 1\nLine 2\nLine 3"

    def test_extract_thinking_with_special_characters(self, mock_client):
        """Test thinking blocks with special characters."""
        content = "[THINK]Math: 2+2=4, symbols: @#$%[/THINK]Result: 4"
        clean, thinking = mock_client._extract_thinking(content)

        assert clean == "Result: 4"
        assert thinking == "Math: 2+2=4, symbols: @#$%"

    def test_extract_thinking_case_sensitivity(self, mock_client):
        """Test that thinking extraction is case sensitive."""
        content = "[think]lowercase[/think][THINK]uppercase[/THINK]Result"
        clean, thinking = mock_client._extract_thinking(content)

        # Only uppercase tags should be processed
        assert clean == "[think]lowercase[/think]Result"
        assert thinking == "uppercase"


class TestThinkingBlockEdgeCases:
    """Test edge cases and malformed thinking blocks."""

    def test_extract_malformed_opening_tag_only(self, mock_client):
        """Test malformed thinking block with only opening tag."""
        content = "[THINK]Incomplete thinking block without closing..."
        clean, thinking = mock_client._extract_thinking(content)

        # Should not extract anything if not properly closed
        assert clean == "[THINK]Incomplete thinking block without closing..."
        assert thinking == ""

    def test_extract_malformed_closing_tag_only(self, mock_client):
        """Test malformed thinking block with only closing tag."""
        content = "Some content [/THINK] more content"
        clean, thinking = mock_client._extract_thinking(content)

        # Should not extract anything if no opening tag
        assert clean == "Some content [/THINK] more content"
        assert thinking == ""

    def test_extract_nested_brackets_in_thinking(self, mock_client):
        """Test thinking blocks containing nested brackets."""
        content = "[THINK]I need to [calculate] this value[/THINK]Answer"
        clean, thinking = mock_client._extract_thinking(content)

        assert clean == "Answer"
        assert thinking == "I need to [calculate] this value"

    def test_extract_thinking_tags_in_content(self, mock_client):
        """Test content that mentions thinking tags but doesn't use them."""
        content = "The model should use [THINK] tags to show reasoning, like [THINK]example[/THINK]."
        clean, thinking = mock_client._extract_thinking(content)

        # Should extract the actual thinking block
        assert clean == "The model should use [THINK] tags to show reasoning, like ."
        assert thinking == "example"

    def test_extract_empty_string(self, mock_client):
        """Test extraction with empty string."""
        clean, thinking = mock_client._extract_thinking("")
        assert clean == ""
        assert thinking == ""

    def test_extract_none_input(self, mock_client):
        """Test extraction with None input."""
        clean, thinking = mock_client._extract_thinking(None)
        assert clean is None
        assert thinking == ""

    def test_extract_whitespace_only_thinking(self, mock_client):
        """Test thinking block with only whitespace."""
        content = "[THINK]   \n\t  [/THINK]Response"
        clean, thinking = mock_client._extract_thinking(content)

        assert clean == "Response"
        assert thinking == ""  # Whitespace should be stripped


class TestThinkingStateManagement:
    """Test thinking state management in LocalLLMClient."""

    def test_thinking_state_cleared_on_new_chat(self, mock_client, mock_requests_post, mock_response_with_content):
        """Test that thinking state is cleared when starting new chat."""
        mock_requests_post.return_value = mock_response_with_content

        # Set some previous thinking state
        mock_client.last_thinking = "Previous thinking"

        # Make a new chat request
        mock_client.chat("Hello")

        # Thinking should be cleared (since response has no thinking)
        assert mock_client.last_thinking == ""

    def test_thinking_accumulated_from_first_response(self, mock_client, mock_requests_post):
        """Test thinking is captured from first response in tool calling."""
        # Mock first response with thinking and tool calls
        first_response = Mock()
        first_response.json.return_value = {
            "id": "chatcmpl-123",
            "object": "chat.completion",
            "created": 1234567890,
            "model": "test-model",
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "[THINK]I need to use a tool[/THINK]I'll calculate this.",
                    "tool_calls": [{
                        "id": "call_123",
                        "type": "function",
                        "function": {"name": "test_tool", "arguments": "{}"}
                    }]
                },
                "finish_reason": "tool_calls"
            }]
        }

        # Mock final response after tool execution
        final_response = Mock()
        final_response.json.return_value = {
            "id": "chatcmpl-124",
            "object": "chat.completion",
            "created": 1234567891,
            "model": "test-model",
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "[THINK]The tool returned the result[/THINK]The answer is 42."
                },
                "finish_reason": "stop"
            }]
        }

        # Mock tool execution
        with patch.object(mock_client.tools, 'execute', return_value='{"result": 42}'):
            mock_requests_post.side_effect = [first_response, final_response]

            response = mock_client.chat("Calculate something")

            # Should accumulate thinking from both responses
            expected_thinking = "I need to use a tool\n\nThe tool returned the result"
            assert mock_client.last_thinking == expected_thinking

    def test_thinking_only_from_final_response(self, mock_client, mock_requests_post, mock_response_with_thinking):
        """Test thinking captured when only final response has thinking."""
        mock_requests_post.return_value = mock_response_with_thinking

        mock_client.chat("Hello")

        assert mock_client.last_thinking == "This is my thinking process"

    def test_thinking_persistence_across_state_checks(self, mock_client):
        """Test that thinking persists until explicitly cleared."""
        # Simulate thinking being set
        mock_client.last_thinking = "Persistent thinking"

        # Check multiple times
        assert mock_client.last_thinking == "Persistent thinking"
        assert mock_client.last_thinking == "Persistent thinking"

        # Clear it
        mock_client.last_thinking = ""
        assert mock_client.last_thinking == ""


class TestThinkingIncludeParameter:
    """Test the include_thinking parameter in chat methods."""

    def test_include_thinking_false_default(self, mock_client, mock_requests_post, mock_response_with_thinking):
        """Test that thinking is excluded by default."""
        mock_requests_post.return_value = mock_response_with_thinking

        response = mock_client.chat("Hello")

        # Response should not include thinking
        assert response == "This is the final answer."
        # But thinking should be stored
        assert mock_client.last_thinking == "This is my thinking process"

    def test_include_thinking_true(self, mock_client, mock_requests_post, mock_response_with_thinking):
        """Test including thinking in response."""
        mock_requests_post.return_value = mock_response_with_thinking

        response = mock_client.chat("Hello", include_thinking=True)

        # Response should include formatted thinking
        expected = "**Thinking:**\nThis is my thinking process\n\n**Response:**\nThis is the final answer."
        assert response == expected

    def test_include_thinking_true_thinking_only(self, mock_client, mock_requests_post):
        """Test include_thinking=True when response is only thinking."""
        thinking_only_response = Mock()
        thinking_only_response.json.return_value = {
            "id": "chatcmpl-123",
            "object": "chat.completion",
            "created": 1234567890,
            "model": "test-model",
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "[THINK]Just thinking, no final answer[/THINK]"
                },
                "finish_reason": "stop"
            }]
        }

        mock_requests_post.return_value = thinking_only_response

        response = mock_client.chat("Hello", include_thinking=True)

        # Should format thinking-only response correctly
        expected = "**Thinking:**\nJust thinking, no final answer"
        assert response == expected

    def test_include_thinking_true_no_thinking(self, mock_client, mock_requests_post, mock_response_with_content):
        """Test include_thinking=True when there's no thinking."""
        mock_requests_post.return_value = mock_response_with_content

        response = mock_client.chat("Hello", include_thinking=True)

        # Should return normal response when no thinking
        assert response == "Test response"

    def test_include_thinking_with_return_full(self, mock_client, mock_requests_post, mock_response_with_thinking):
        """Test include_thinking with return_full=True."""
        mock_requests_post.return_value = mock_response_with_thinking

        response = mock_client.chat("Hello", include_thinking=True, return_full=True)

        # Should return ChatCompletion object with thinking included
        assert isinstance(response, ChatCompletion)
        expected_content = "**Thinking:**\nThis is my thinking process\n\n**Response:**\nThis is the final answer."
        assert response.choices[0].message.content == expected_content


class TestThinkingParametrizedTests:
    """Parametrized tests for various thinking scenarios."""

    @pytest.mark.parametrize("test_case", [
        {
            "name": "simple_thinking",
            "input": "[THINK]Simple reasoning[/THINK]Answer",
            "expected_clean": "Answer",
            "expected_thinking": "Simple reasoning"
        },
        {
            "name": "no_thinking",
            "input": "Direct answer",
            "expected_clean": "Direct answer",
            "expected_thinking": ""
        },
        {
            "name": "multiple_blocks",
            "input": "[THINK]Step 1[/THINK]Interim[THINK]Step 2[/THINK]Final",
            "expected_clean": "InterimFinal",
            "expected_thinking": "Step 1\nStep 2"
        },
        {
            "name": "thinking_only",
            "input": "[THINK]Only reasoning[/THINK]",
            "expected_clean": "",
            "expected_thinking": "Only reasoning"
        },
        {
            "name": "empty_thinking",
            "input": "[THINK][/THINK]Result",
            "expected_clean": "Result",
            "expected_thinking": ""
        }
    ])
    def test_thinking_extraction_cases(self, mock_client, test_case):
        """Test various thinking extraction scenarios."""
        clean, thinking = mock_client._extract_thinking(test_case["input"])

        assert clean == test_case["expected_clean"], f"Failed for case: {test_case['name']}"
        assert thinking == test_case["expected_thinking"], f"Failed for case: {test_case['name']}"

    @pytest.mark.parametrize("include_thinking,expected_format", [
        (False, "clean_only"),
        (True, "formatted_with_thinking")
    ])
    def test_include_thinking_parameter_variations(self, mock_client, mock_requests_post, include_thinking, expected_format):
        """Test include_thinking parameter with different values."""
        thinking_response = Mock()
        thinking_response.json.return_value = {
            "id": "chatcmpl-123",
            "object": "chat.completion",
            "created": 1234567890,
            "model": "test-model",
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "[THINK]My reasoning[/THINK]My answer"
                },
                "finish_reason": "stop"
            }]
        }

        mock_requests_post.return_value = thinking_response

        response = mock_client.chat("Test", include_thinking=include_thinking)

        if expected_format == "clean_only":
            assert response == "My answer"
        else:  # formatted_with_thinking
            assert response == "**Thinking:**\nMy reasoning\n\n**Response:**\nMy answer"


class TestThinkingIntegration:
    """Integration tests for thinking with other features."""

    def test_thinking_with_tool_calls_integration(self, mock_client, mock_requests_post):
        """Test thinking blocks work correctly with tool calling."""
        # Register a simple tool
        @mock_client.register_tool("Test tool")
        def test_tool() -> dict:
            return {"result": "tool_result"}

        # Mock responses
        first_response = Mock()
        first_response.json.return_value = {
            "id": "chatcmpl-123",
            "object": "chat.completion",
            "created": 1234567890,
            "model": "test-model",
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "[THINK]I should use the test tool[/THINK]Let me use a tool.",
                    "tool_calls": [{
                        "id": "call_123",
                        "type": "function",
                        "function": {"name": "test_tool", "arguments": "{}"}
                    }]
                },
                "finish_reason": "tool_calls"
            }]
        }

        final_response = Mock()
        final_response.json.return_value = {
            "id": "chatcmpl-124",
            "object": "chat.completion",
            "created": 1234567891,
            "model": "test-model",
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "[THINK]The tool worked perfectly[/THINK]The result is tool_result."
                },
                "finish_reason": "stop"
            }]
        }

        mock_requests_post.side_effect = [first_response, final_response]

        response = mock_client.chat("Use the tool")

        # Should have both tool calls and accumulated thinking
        assert len(mock_client.last_tool_calls) == 1
        assert mock_client.last_tool_calls[0].function.name == "test_tool"

        expected_thinking = "I should use the test tool\n\nThe tool worked perfectly"
        assert mock_client.last_thinking == expected_thinking

        assert response == "The result is tool_result."

    def test_thinking_with_chat_history(self, mock_client, mock_requests_post, mock_response_with_thinking):
        """Test thinking works with chat history."""
        mock_requests_post.return_value = mock_response_with_thinking

        history = []
        response, new_history = mock_client.chat_with_history("Hello", history, include_thinking=True)

        # Should include thinking in response
        expected = "**Thinking:**\nThis is my thinking process\n\n**Response:**\nThis is the final answer."
        assert response == expected

        # History should contain the clean message
        assert len(new_history) == 2
        assert new_history[1].content == expected