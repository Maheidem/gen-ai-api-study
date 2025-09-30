"""
Tests for agent framework: BaseAgent, ReACT, AgentResult, AgentStatus.
"""

import pytest
from unittest.mock import Mock, patch, call
from typing import List

from local_llm_sdk import LocalLLMClient, create_chat_message
from local_llm_sdk.agents import BaseAgent, ReACT, AgentResult, AgentStatus
from local_llm_sdk.models import ChatCompletion, ChatMessage, ChatCompletionChoice


class TestAgentResult:
    """Test AgentResult dataclass functionality."""

    def test_agent_result_creation(self):
        """Test creating AgentResult with different statuses."""
        # Success result
        success_result = AgentResult(
            status=AgentStatus.SUCCESS,
            iterations=5,
            final_response="Task completed successfully"
        )
        assert success_result.status == AgentStatus.SUCCESS
        assert success_result.iterations == 5
        assert success_result.final_response == "Task completed successfully"
        assert success_result.conversation == []
        assert success_result.error is None
        assert success_result.metadata == {}

        # Error result
        error_result = AgentResult(
            status=AgentStatus.ERROR,
            iterations=2,
            final_response="",
            error="Something went wrong"
        )
        assert error_result.status == AgentStatus.ERROR
        assert error_result.error == "Something went wrong"

        # Max iterations result
        max_iter_result = AgentResult(
            status=AgentStatus.MAX_ITERATIONS,
            iterations=15,
            final_response="Partial completion",
            metadata={"max_iterations_reached": True}
        )
        assert max_iter_result.status == AgentStatus.MAX_ITERATIONS
        assert max_iter_result.metadata["max_iterations_reached"] is True

    def test_agent_result_success_property(self):
        """Test success property returns correct boolean."""
        success_result = AgentResult(
            status=AgentStatus.SUCCESS,
            iterations=3,
            final_response="Done"
        )
        assert success_result.success is True

        # Non-success statuses
        for status in [AgentStatus.ERROR, AgentStatus.FAILED,
                      AgentStatus.MAX_ITERATIONS, AgentStatus.STOPPED]:
            result = AgentResult(status=status, iterations=1, final_response="")
            assert result.success is False


class TestBaseAgent:
    """Test BaseAgent abstract class."""

    def test_base_agent_abstract(self, mock_client):
        """Test BaseAgent cannot be instantiated directly."""
        with pytest.raises(TypeError, match="Can't instantiate abstract class"):
            BaseAgent(mock_client)

    def test_base_agent_sanitize_task_name(self, mock_client):
        """Test task name sanitization for tracing."""
        # Create a concrete implementation for testing
        class ConcreteAgent(BaseAgent):
            def _execute(self, task: str, **kwargs) -> AgentResult:
                return AgentResult(
                    status=AgentStatus.SUCCESS,
                    iterations=1,
                    final_response="Done"
                )

        agent = ConcreteAgent(mock_client)

        # Test basic sanitization
        assert agent._sanitize_task_name("Simple task") == "Simple_task"

        # Test with special characters
        assert agent._sanitize_task_name("Task with @#$% chars!") == "Task_with__chars"

        # Test with length limit (default 30)
        # Note: dots are stripped during sanitization, only alphanumeric kept
        long_task = "A" * 50
        result = agent._sanitize_task_name(long_task)
        assert len(result) == 30  # Truncated to max_length, dots removed
        assert result == "A" * 30

        # Test with custom max_length
        result = agent._sanitize_task_name("Short task name", max_length=10)
        # "Short task" -> "Short_task" (11 chars) but first truncated to 10 then "..." added -> sanitized
        assert len(result) <= 13  # At most 10 + "..." but dots get stripped

        # Test with multiline (takes first line only)
        multiline = "First line\nSecond line\nThird line"
        assert agent._sanitize_task_name(multiline) == "First_line"

        # Test empty fallback
        assert agent._sanitize_task_name("") == "unnamed_task"
        assert agent._sanitize_task_name("   ") == "unnamed_task"


class TestReACTAgent:
    """Test ReACT agent implementation."""

    def test_react_agent_initialization(self, mock_client):
        """Test ReACT agent initialization with and without custom prompt."""
        # Default initialization
        agent = ReACT(mock_client)
        assert agent.name == "ReACT"
        assert agent.client == mock_client
        assert agent.system_prompt == ReACT.DEFAULT_SYSTEM_PROMPT

        # Custom name and prompt
        custom_prompt = "Custom system prompt for testing"
        agent = ReACT(mock_client, name="CustomReACT", system_prompt=custom_prompt)
        assert agent.name == "CustomReACT"
        assert agent.system_prompt == custom_prompt

    def test_react_agent_default_prompt(self):
        """Verify DEFAULT_SYSTEM_PROMPT structure contains key elements."""
        prompt = ReACT.DEFAULT_SYSTEM_PROMPT

        # Check for key components
        assert "execute_python" in prompt
        assert "filesystem_operation" in prompt
        assert "TASK_COMPLETE" in prompt
        assert "tool" in prompt.lower()
        assert isinstance(prompt, str)
        assert len(prompt) > 100  # Substantial prompt

    def test_react_agent_simple_task(self, mock_client):
        """Test simple task completion with mocked responses."""
        # Mock the client.chat method to return a completion response
        mock_response = Mock(spec=ChatCompletion)
        mock_message = Mock(spec=ChatMessage)
        mock_message.content = "I've completed the task. TASK_COMPLETE"
        mock_message.tool_calls = None
        mock_choice = Mock(spec=ChatCompletionChoice)
        mock_choice.message = mock_message
        mock_response.choices = [mock_choice]

        mock_client.chat = Mock(return_value=mock_response)
        mock_client.last_conversation_additions = None

        agent = ReACT(mock_client)
        result = agent.run("Simple test task", max_iterations=5, verbose=False)

        # Verify result
        assert result.status == AgentStatus.SUCCESS
        assert result.iterations == 1
        assert "TASK_COMPLETE" in result.final_response
        assert result.success is True
        assert "agent_name" in result.metadata
        assert result.metadata["agent_name"] == "ReACT"

    def test_react_agent_max_iterations(self, mock_client):
        """Test max_iterations limit is enforced."""
        # Mock responses that never complete
        mock_response = Mock(spec=ChatCompletion)
        mock_message = Mock(spec=ChatMessage)
        mock_message.content = "Still working on it..."
        mock_message.tool_calls = None
        mock_choice = Mock(spec=ChatCompletionChoice)
        mock_choice.message = mock_message
        mock_response.choices = [mock_choice]

        mock_client.chat = Mock(return_value=mock_response)
        mock_client.last_conversation_additions = None

        agent = ReACT(mock_client)
        result = agent.run("Never-ending task", max_iterations=3, verbose=False)

        # Should hit max iterations
        assert result.status == AgentStatus.MAX_ITERATIONS
        assert result.iterations == 3
        assert not result.success
        assert result.metadata.get("max_iterations_reached") is True

        # Verify chat was called 3 times
        assert mock_client.chat.call_count == 3

    def test_react_agent_stop_condition(self, mock_client):
        """Test custom stop conditions work correctly."""
        # Mock response
        mock_response = Mock(spec=ChatCompletion)
        mock_message = Mock(spec=ChatMessage)
        mock_message.content = "The answer is 42. CUSTOM_STOP"
        mock_message.tool_calls = None
        mock_choice = Mock(spec=ChatCompletionChoice)
        mock_choice.message = mock_message
        mock_response.choices = [mock_choice]

        mock_client.chat = Mock(return_value=mock_response)
        mock_client.last_conversation_additions = None

        # Custom stop condition
        def custom_stop(content: str) -> bool:
            return "CUSTOM_STOP" in content

        agent = ReACT(mock_client)
        result = agent.run(
            "Task with custom stop",
            max_iterations=10,
            stop_condition=custom_stop,
            verbose=False
        )

        assert result.status == AgentStatus.SUCCESS
        assert result.iterations == 1
        assert "CUSTOM_STOP" in result.final_response

    def test_react_agent_should_stop(self, mock_client):
        """Test _should_stop method logic."""
        agent = ReACT(mock_client)

        # Test default TASK_COMPLETE detection
        assert agent._should_stop("Here is the result. TASK_COMPLETE") is True
        assert agent._should_stop("task_complete in lowercase") is True
        assert agent._should_stop("Still working...") is False

        # Test with custom stop condition
        custom_stop = lambda content: "DONE" in content
        assert agent._should_stop("DONE with task", custom_stop) is True
        assert agent._should_stop("Not finished yet", custom_stop) is False

        # Custom condition takes precedence but both work
        assert agent._should_stop("DONE and TASK_COMPLETE", custom_stop) is True

    def test_react_agent_tool_call_counting(self, mock_client):
        """Test _count_tool_calls method."""
        agent = ReACT(mock_client)

        # Create messages with tool calls
        msg1 = create_chat_message("user", "Calculate 5 + 3")

        msg2 = Mock(spec=ChatMessage)
        msg2.role = "assistant"
        msg2.content = "I'll calculate that"
        msg2.tool_calls = [
            Mock(id="call_1", function=Mock(name="math_calculator")),
            Mock(id="call_2", function=Mock(name="text_transformer"))
        ]

        msg3 = create_chat_message("tool", "Result: 8")

        msg4 = Mock(spec=ChatMessage)
        msg4.role = "assistant"
        msg4.content = "Final answer"
        msg4.tool_calls = None

        messages = [msg1, msg2, msg3, msg4]
        count = agent._count_tool_calls(messages)

        assert count == 2  # msg2 had 2 tool calls

    def test_react_agent_error_handling(self, mock_client):
        """Test agent error handling during execution."""
        # Mock client.chat to raise an exception
        mock_client.chat = Mock(side_effect=Exception("API Error"))

        agent = ReACT(mock_client)
        result = agent.run("Task that will fail", verbose=False)

        # Should return ERROR status, not raise exception
        assert result.status == AgentStatus.ERROR
        assert result.error == "API Error"
        assert not result.success
        assert "error_type" in result.metadata
        assert result.metadata["error_type"] == "Exception"

    def test_react_agent_metadata(self, mock_client):
        """Test metadata is properly populated in AgentResult."""
        # Mock successful completion
        mock_response = Mock(spec=ChatCompletion)
        mock_message = Mock(spec=ChatMessage)
        mock_message.content = "All done! TASK_COMPLETE"
        mock_message.tool_calls = None
        mock_choice = Mock(spec=ChatCompletionChoice)
        mock_choice.message = mock_message
        mock_response.choices = [mock_choice]

        mock_client.chat = Mock(return_value=mock_response)
        mock_client.last_conversation_additions = None

        agent = ReACT(mock_client, name="TestAgent")
        result = agent.run("Metadata test task", verbose=False)

        # Check metadata
        assert "agent_name" in result.metadata
        assert "agent_class" in result.metadata
        assert result.metadata["agent_name"] == "TestAgent"
        assert result.metadata["agent_class"] == "ReACT"
        assert "total_tool_calls" in result.metadata
        assert "final_iteration" in result.metadata

    def test_react_agent_conversation_context(self, mock_client):
        """Test conversation wrapping and history management."""
        mock_response = Mock(spec=ChatCompletion)
        mock_message = Mock(spec=ChatMessage)
        mock_message.content = "Done. TASK_COMPLETE"
        mock_message.tool_calls = None
        mock_choice = Mock(spec=ChatCompletionChoice)
        mock_choice.message = mock_message
        mock_response.choices = [mock_choice]

        mock_client.chat = Mock(return_value=mock_response)
        mock_client.last_conversation_additions = None

        # Mock conversation context
        with patch.object(mock_client, 'conversation') as mock_conv:
            mock_conv.return_value.__enter__ = Mock(return_value=None)
            mock_conv.return_value.__exit__ = Mock(return_value=None)

            agent = ReACT(mock_client)
            result = agent.run("Test conversation context", verbose=False)

            # Verify conversation context was used
            mock_conv.assert_called_once()
            call_args = mock_conv.call_args[0][0]
            assert "ReACT" in call_args
            assert "Test_conversation_context" in call_args

    @patch('builtins.print')
    def test_react_agent_verbose_output(self, mock_print, mock_client):
        """Test verbose mode produces expected output."""
        mock_response = Mock(spec=ChatCompletion)
        mock_message = Mock(spec=ChatMessage)
        mock_message.content = "Task completed. TASK_COMPLETE"
        mock_message.tool_calls = None
        mock_choice = Mock(spec=ChatCompletionChoice)
        mock_choice.message = mock_message
        mock_response.choices = [mock_choice]

        mock_client.chat = Mock(return_value=mock_response)
        mock_client.last_conversation_additions = None

        agent = ReACT(mock_client)
        result = agent.run("Verbose test task", max_iterations=5, verbose=True)

        # Verify print was called (verbose output)
        assert mock_print.call_count > 0

        # Check for expected verbose output elements
        print_outputs = [str(call) for call in mock_print.call_args_list]
        all_output = ''.join(print_outputs)

        assert "Starting task" in all_output or any("Starting" in str(c) for c in mock_print.call_args_list)
        assert result.success is True
