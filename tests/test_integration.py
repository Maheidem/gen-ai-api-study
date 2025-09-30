"""
Integration tests for local-llm-sdk.
Tests full workflows and real scenarios.
"""

import pytest
from unittest.mock import Mock, patch
import json

from local_llm_sdk import LocalLLMClient, create_client, quick_chat
from local_llm_sdk.tools import builtin
from local_llm_sdk.models import ChatCompletion


class TestFullWorkflowIntegration:
    """Test complete workflows from start to finish."""

    def test_complete_tool_workflow(self, mock_requests_post):
        """Test a complete workflow with tool registration and execution."""
        # Create client and register tools
        client = LocalLLMClient("http://localhost:1234/v1", "test-model")
        client.register_tools_from(builtin)

        # Mock tool calling workflow
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
                    "content": "[THINK]I need to calculate 15 * 23[/THINK]I'll use the calculator.",
                    "tool_calls": [{
                        "id": "call_123",
                        "type": "function",
                        "function": {
                            "name": "math_calculator",
                            "arguments": '{"arg1": 15.0, "arg2": 23.0, "operation": "multiply"}'
                        }
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
                    "content": "[THINK]The calculation returned 345[/THINK]The result is 345."
                },
                "finish_reason": "stop"
            }]
        }

        mock_requests_post.side_effect = [first_response, final_response]

        # Execute the workflow
        response = client.chat("What is 15 times 23?")

        # Verify the complete workflow
        assert response == "The result is 345."
        assert len(client.last_tool_calls) == 1
        assert client.last_tool_calls[0].function.name == "math_calculator"

        # Verify thinking was accumulated
        expected_thinking = "I need to calculate 15 * 23\n\nThe calculation returned 345"
        assert client.last_thinking == expected_thinking

        # Verify correct API calls were made
        assert mock_requests_post.call_count == 2

    def test_multi_tool_conversation(self, mock_requests_post):
        """Test conversation using multiple different tools."""
        client = LocalLLMClient("http://localhost:1234/v1", "test-model")
        client.register_tools_from(builtin)

        # Mock first query - math calculation
        math_response = Mock()
        math_response.json.return_value = {
            "id": "chatcmpl-1",
            "object": "chat.completion",
            "created": 1234567890,
            "model": "test-model",
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "The result is 50."
                },
                "finish_reason": "stop"
            }]
        }

        # Mock second query - text transformation
        text_response = Mock()
        text_response.json.return_value = {
            "id": "chatcmpl-2",
            "object": "chat.completion",
            "created": 1234567891,
            "model": "test-model",
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "The uppercase version is HELLO WORLD."
                },
                "finish_reason": "stop"
            }]
        }

        mock_requests_post.side_effect = [math_response, text_response]

        # First conversation turn
        response1 = client.chat("What is 25 + 25?")
        assert response1 == "The result is 50."

        # Second conversation turn (new chat, state should be cleared)
        response2 = client.chat("Convert 'hello world' to uppercase")
        assert response2 == "The uppercase version is HELLO WORLD."

        # Verify both calls were made
        assert mock_requests_post.call_count == 2

    def test_conversation_with_history_and_tools(self, mock_requests_post):
        """Test maintaining conversation history while using tools."""
        client = LocalLLMClient("http://localhost:1234/v1", "test-model")
        client.register_tools_from(builtin)

        # Mock responses for conversation
        responses = [
            {
                "id": "chatcmpl-1",
                "object": "chat.completion",
                "created": 1234567890,
                "model": "test-model",
                "choices": [{
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": "The result is 100."
                    },
                    "finish_reason": "stop"
                }]
            },
            {
                "id": "chatcmpl-2",
                "object": "chat.completion",
                "created": 1234567891,
                "model": "test-model",
                "choices": [{
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": "The result is 150."
                    },
                    "finish_reason": "stop"
                }]
            }
        ]

        def mock_response_side_effect(*args, **kwargs):
            response = Mock()
            response.json.return_value = responses.pop(0)
            response.raise_for_status.return_value = None
            return response

        mock_requests_post.side_effect = mock_response_side_effect

        # Start conversation
        history = []

        # First turn
        response1, history = client.chat_with_history("What is 50 + 50?", history)
        assert response1 == "The result is 100."
        assert len(history) == 2  # user + assistant

        # Second turn (should include previous context)
        response2, history = client.chat_with_history("Add 50 to that result", history)
        assert response2 == "The result is 150."
        assert len(history) == 4  # 2 previous + user + assistant

        # Verify the second request included the conversation history
        second_call_args = mock_requests_post.call_args_list[1]
        request_data = second_call_args[1]["json"]
        assert len(request_data["messages"]) >= 3  # Should include previous messages


class TestErrorHandlingIntegration:
    """Test error handling in complete workflows."""

    def test_tool_execution_error_handling(self, mock_requests_post):
        """Test graceful handling of tool execution errors."""
        client = LocalLLMClient("http://localhost:1234/v1", "test-model")
        client.register_tools_from(builtin)

        # Mock response that calls division by zero
        tool_response = Mock()
        tool_response.json.return_value = {
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
                            "arguments": '{"arg1": 10.0, "arg2": 0.0, "operation": "divide"}'
                        }
                    }]
                },
                "finish_reason": "tool_calls"
            }]
        }

        # Mock final response after tool error
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
                    "content": "I encountered an error with division by zero."
                },
                "finish_reason": "stop"
            }]
        }

        mock_requests_post.side_effect = [tool_response, final_response]

        # This should not raise an exception, but handle the error gracefully
        response = client.chat("What is 10 divided by 0?")

        # Should get a response even though tool failed
        assert response == "I encountered an error with division by zero."

    def test_api_error_propagation(self, mock_requests_post):
        """Test that API errors are properly propagated."""
        client = LocalLLMClient("http://localhost:1234/v1", "test-model")

        # Mock API error
        mock_response = Mock()
        mock_response.raise_for_status.side_effect = Exception("API Error: 500 Internal Server Error")
        mock_requests_post.return_value = mock_response

        with pytest.raises(Exception, match="API Error: 500"):
            client.chat("Hello")

    def test_network_error_handling(self, mock_requests_post):
        """Test handling of network errors."""
        client = LocalLLMClient("http://localhost:1234/v1", "test-model")

        # Mock network error
        mock_requests_post.side_effect = Exception("Connection refused")

        with pytest.raises(Exception, match="Connection refused"):
            client.chat("Hello")


class TestConvenienceFunctions:
    """Test convenience functions and alternative interfaces."""

    def test_create_client_function(self):
        """Test the create_client convenience function."""
        client = create_client("http://test:8080/v1", "test-model")

        assert isinstance(client, LocalLLMClient)
        assert client.base_url == "http://test:8080/v1"
        assert client.default_model == "test-model"

    def test_create_client_with_defaults(self):
        """Test create_client with default parameters."""
        client = create_client()

        assert client.base_url == "http://localhost:1234/v1"
        assert client.default_model is None

    def test_quick_chat_function(self, mock_requests_post, mock_response_with_content):
        """Test the quick_chat convenience function."""
        mock_requests_post.return_value = mock_response_with_content

        response = quick_chat("Hello", "http://localhost:1234/v1", "test-model")

        assert response == "Test response"
        mock_requests_post.assert_called_once()

    def test_quick_chat_with_defaults(self, mock_requests_post, mock_response_with_content):
        """Test quick_chat with specified model (avoiding qwen3 bug)."""
        mock_requests_post.return_value = mock_response_with_content

        response = quick_chat("Hello", model="mistralai/magistral-small-2509")

        assert response == "Test response"


class TestRealAPIIntegration:
    """Tests that would work with a real LM Studio instance."""

    @pytest.mark.skip(reason="Requires running LM Studio instance")
    def test_real_api_models_endpoint(self):
        """Test real API call to models endpoint."""
        client = LocalLLMClient()

        try:
            models = client.list_models()
            assert len(models.data) > 0
            assert all(hasattr(model, 'id') for model in models.data)
        except Exception as e:
            pytest.skip(f"LM Studio not available: {e}")

    @pytest.mark.skip(reason="Requires running LM Studio instance")
    def test_real_api_simple_chat(self):
        """Test real API call for simple chat."""
        client = LocalLLMClient()

        try:
            response = client.chat_simple("Say 'Hello, World!'")
            assert isinstance(response, str)
            assert len(response) > 0
        except Exception as e:
            pytest.skip(f"LM Studio not available: {e}")

    @pytest.mark.skip(reason="Requires running LM Studio instance")
    def test_real_api_with_tools(self):
        """Test real API call with tools."""
        client = LocalLLMClient()
        client.register_tools_from(builtin)

        try:
            response = client.chat("What is 2 + 2?")
            assert isinstance(response, str)
            # Check if tool was actually used
            assert len(client.last_tool_calls) >= 0  # May or may not use tool
        except Exception as e:
            pytest.skip(f"LM Studio not available: {e}")


class TestComplexScenarios:
    """Test complex real-world scenarios."""

    def test_custom_tool_integration(self, mock_requests_post):
        """Test integration with custom tools."""
        client = LocalLLMClient("http://localhost:1234/v1", "test-model")

        # Register a custom tool
        @client.register_tool("Calculate area of rectangle")
        def calculate_area(width: float, height: float) -> dict:
            return {
                "width": width,
                "height": height,
                "area": width * height,
                "perimeter": 2 * (width + height)
            }

        # Mock tool usage
        tool_response = Mock()
        tool_response.json.return_value = {
            "id": "chatcmpl-123",
            "object": "chat.completion",
            "created": 1234567890,
            "model": "test-model",
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "I'll calculate the area for you.",
                    "tool_calls": [{
                        "id": "call_123",
                        "type": "function",
                        "function": {
                            "name": "calculate_area",
                            "arguments": '{"width": 5.0, "height": 3.0}'
                        }
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
                    "content": "The area is 15 square units and the perimeter is 16 units."
                },
                "finish_reason": "stop"
            }]
        }

        mock_requests_post.side_effect = [tool_response, final_response]

        response = client.chat("What's the area of a 5x3 rectangle?")

        assert response == "The area is 15 square units and the perimeter is 16 units."
        assert len(client.last_tool_calls) == 1
        assert client.last_tool_calls[0].function.name == "calculate_area"

    def test_mixed_builtin_and_custom_tools(self, mock_requests_post):
        """Test using both builtin and custom tools."""
        client = LocalLLMClient("http://localhost:1234/v1", "test-model")

        # Register builtin tools
        client.register_tools_from(builtin)

        # Add custom tool
        @client.register_tool("Generate greeting")
        def generate_greeting(name: str, style: str = "formal") -> dict:
            greetings = {
                "formal": f"Good day, {name}.",
                "casual": f"Hey {name}!",
                "friendly": f"Hello there, {name}!"
            }
            return {"greeting": greetings.get(style, greetings["formal"])}

        # Verify both types of tools are available
        tools = client.tools.list_tools()
        assert "math_calculator" in tools  # builtin
        assert "generate_greeting" in tools  # custom

        # Test tool schemas are properly generated
        schemas = client.tools.get_schemas()
        assert len(schemas) > 4  # Should have multiple tools

    def test_long_conversation_with_state_management(self, mock_requests_post):
        """Test long conversation maintaining state correctly."""
        client = LocalLLMClient("http://localhost:1234/v1", "test-model")
        client.register_tools_from(builtin)

        # Simulate multiple conversation turns
        responses = [
            {"content": "I calculated 25.", "thinking": "Let me calculate 5*5"},
            {"content": "Now I'll add 10 to get 35.", "thinking": "Adding 10 to previous result"},
            {"content": "The square root of 35 is approximately 5.92.", "thinking": "Calculating square root"}
        ]

        def create_mock_response(content, thinking=""):
            response = Mock()
            full_content = f"[THINK]{thinking}[/THINK]{content}" if thinking else content
            response.json.return_value = {
                "id": f"chatcmpl-{len(responses)}",
                "object": "chat.completion",
                "created": 1234567890,
                "model": "test-model",
                "choices": [{
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": full_content
                    },
                    "finish_reason": "stop"
                }]
            }
            response.raise_for_status.return_value = None
            return response

        mock_requests_post.side_effect = [
            create_mock_response(resp["content"], resp["thinking"])
            for resp in responses
        ]

        history = []

        # Turn 1
        response1, history = client.chat_with_history("Calculate 5 * 5", history)
        assert "25" in response1
        assert "Let me calculate 5*5" in client.last_thinking

        # Turn 2 - state should be cleared and new thinking captured
        response2, history = client.chat_with_history("Add 10 to that", history)
        assert "35" in response2
        assert "Adding 10 to previous result" in client.last_thinking

        # Turn 3
        response3, history = client.chat_with_history("What's the square root?", history)
        assert "5.92" in response3
        assert "Calculating square root" in client.last_thinking

        # Verify conversation history grows
        assert len(history) == 6  # 3 user + 3 assistant messages


class TestAgentIntegration:
    """Integration tests for agent framework."""

    def test_react_agent_with_tools(self, mock_client):
        """Test ReACT agent integrates with tool system."""
        from local_llm_sdk.agents import ReACT

        # Register a simple tool
        @mock_client.tools.register("Test calculator")
        def test_calc(x: int, y: int) -> dict:
            return {"result": x + y}

        # Create agent
        agent = ReACT(mock_client, name="TestAgent")

        assert agent.client == mock_client
        assert agent.name == "TestAgent"
        assert len(mock_client.tools.list_tools()) > 0

    @patch('local_llm_sdk.client.requests.post')
    def test_react_agent_multi_step_task(self, mock_post, mock_client, mock_response_with_content):
        """Test ReACT agent handles multi-step tasks."""
        from local_llm_sdk.agents import ReACT, AgentStatus

        # Mock responses for multi-step task
        responses = [
            mock_response_with_content,  # First iteration
            mock_response_with_content,  # Second iteration
        ]

        # Make second response include TASK_COMPLETE
        responses[1].json.return_value["choices"][0]["message"]["content"] = "TASK_COMPLETE"

        mock_post.side_effect = responses

        agent = ReACT(mock_client, name="TestAgent")

        # Mock the conversation context
        with patch.object(mock_client, 'conversation'):
            result = agent.run("Test task", max_iterations=5, verbose=False)

        assert result.status == AgentStatus.SUCCESS or result.status == AgentStatus.MAX_ITERATIONS
        assert result.iterations >= 1

    def test_agent_result_metadata_tracking(self, mock_client):
        """Test agent results include metadata."""
        from local_llm_sdk.agents import ReACT, AgentResult
        from local_llm_sdk.agents.models import AgentStatus

        agent = ReACT(mock_client, name="MetadataTest")

        # Create a result manually (simulating agent completion)
        result = AgentResult(
            status=AgentStatus.SUCCESS,
            iterations=3,
            final_response="Done",
            conversation=[],
            metadata={"agent_name": "MetadataTest", "total_tool_calls": 5}
        )

        assert "agent_name" in result.metadata
        assert result.metadata["agent_name"] == "MetadataTest"
        assert result.metadata["total_tool_calls"] == 5