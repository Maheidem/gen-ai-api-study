"""
Tests for Pydantic models in local-llm-sdk.
"""

import pytest
from pydantic import ValidationError
import json

from local_llm_sdk.models import (
    ChatMessage,
    ChatCompletion,
    ChatCompletionRequest,
    ChatCompletionChoice,
    Tool,
    ToolCall,
    FunctionCall,
    Function,
    ModelList,
    ModelInfo,
    EmbeddingsRequest,
    EmbeddingsResponse,
    EmbeddingData,
    CompletionUsage,
    create_chat_message,
    create_chat_completion_request
)


class TestChatMessage:
    """Test ChatMessage model."""

    def test_valid_chat_message(self):
        """Test creating a valid ChatMessage."""
        message = ChatMessage(role="user", content="Hello")
        assert message.role == "user"
        assert message.content == "Hello"
        assert message.name is None
        assert message.tool_call_id is None

    def test_chat_message_with_all_fields(self):
        """Test ChatMessage with all optional fields."""
        message = ChatMessage(
            role="assistant",
            content="Response",
            name="assistant_name",
            tool_call_id="call_123"
        )
        assert message.role == "assistant"
        assert message.content == "Response"
        assert message.name == "assistant_name"
        assert message.tool_call_id == "call_123"

    def test_chat_message_serialization(self):
        """Test ChatMessage serialization."""
        message = ChatMessage(role="user", content="Hello")
        data = message.model_dump(exclude_none=True)

        expected = {"role": "user", "content": "Hello"}
        assert data == expected

    def test_chat_message_deserialization(self):
        """Test ChatMessage deserialization."""
        data = {"role": "assistant", "content": "Hi there"}
        message = ChatMessage.model_validate(data)

        assert message.role == "assistant"
        assert message.content == "Hi there"

    @pytest.mark.parametrize("invalid_role", ["invalid", "", None])
    def test_chat_message_invalid_role(self, invalid_role):
        """Test ChatMessage with invalid role."""
        with pytest.raises(ValidationError):
            ChatMessage(role=invalid_role, content="Hello")


class TestFunctionCall:
    """Test FunctionCall model."""

    def test_valid_function_call(self):
        """Test creating a valid FunctionCall."""
        func_call = FunctionCall(name="test_func", arguments='{"x": 1}')
        assert func_call.name == "test_func"
        assert func_call.arguments == '{"x": 1}'

    def test_function_call_serialization(self):
        """Test FunctionCall serialization."""
        func_call = FunctionCall(name="add", arguments='{"a": 1, "b": 2}')
        data = func_call.model_dump()

        expected = {"name": "add", "arguments": '{"a": 1, "b": 2}'}
        assert data == expected

    def test_function_call_empty_name(self):
        """Test FunctionCall with empty name."""
        with pytest.raises(ValidationError):
            FunctionCall(name="", arguments="{}")


class TestToolCall:
    """Test ToolCall model."""

    def test_valid_tool_call(self):
        """Test creating a valid ToolCall."""
        func_call = FunctionCall(name="test", arguments="{}")
        tool_call = ToolCall(id="call_123", type="function", function=func_call)

        assert tool_call.id == "call_123"
        assert tool_call.type == "function"
        assert tool_call.function.name == "test"

    def test_tool_call_serialization(self):
        """Test ToolCall serialization."""
        func_call = FunctionCall(name="add", arguments='{"a": 1, "b": 2}')
        tool_call = ToolCall(id="call_123", type="function", function=func_call)

        data = tool_call.model_dump()
        expected = {
            "id": "call_123",
            "type": "function",
            "function": {"name": "add", "arguments": '{"a": 1, "b": 2}'}
        }
        assert data == expected


class TestFunction:
    """Test Function model."""

    def test_valid_function(self):
        """Test creating a valid Function."""
        function = Function(
            name="add",
            description="Add two numbers",
            parameters={
                "type": "object",
                "properties": {
                    "a": {"type": "number"},
                    "b": {"type": "number"}
                },
                "required": ["a", "b"]
            }
        )

        assert function.name == "add"
        assert function.description == "Add two numbers"
        assert "properties" in function.parameters

    def test_function_minimal(self):
        """Test Function with minimal required fields."""
        function = Function(name="test", parameters={})
        assert function.name == "test"
        assert function.description is None
        assert function.parameters == {}


class TestTool:
    """Test Tool model."""

    def test_valid_tool(self):
        """Test creating a valid Tool."""
        function = Function(name="test", parameters={})
        tool = Tool(type="function", function=function)

        assert tool.type == "function"
        assert tool.function.name == "test"

    def test_tool_serialization(self):
        """Test Tool serialization."""
        function = Function(
            name="calculator",
            description="Calculate numbers",
            parameters={"type": "object"}
        )
        tool = Tool(type="function", function=function)

        data = tool.model_dump(exclude_none=True)
        expected = {
            "type": "function",
            "function": {
                "name": "calculator",
                "description": "Calculate numbers",
                "parameters": {"type": "object"}
            }
        }
        assert data == expected


class TestChatCompletionChoice:
    """Test ChatCompletionChoice model."""

    def test_valid_choice(self):
        """Test creating a valid ChatCompletionChoice."""
        message = ChatMessage(role="assistant", content="Hello")
        choice = ChatCompletionChoice(index=0, message=message, finish_reason="stop")

        assert choice.index == 0
        assert choice.message.content == "Hello"
        assert choice.finish_reason == "stop"

    def test_choice_with_tool_calls(self):
        """Test ChatCompletionChoice with tool calls in message."""
        func_call = FunctionCall(name="test", arguments="{}")
        tool_call = ToolCall(id="call_123", type="function", function=func_call)

        message = ChatMessage(
            role="assistant",
            content="Using tool",
            tool_calls=[tool_call]
        )
        choice = ChatCompletionChoice(index=0, message=message, finish_reason="tool_calls")

        assert len(choice.message.tool_calls) == 1
        assert choice.finish_reason == "tool_calls"


class TestChatCompletion:
    """Test ChatCompletion model."""

    def test_valid_chat_completion(self):
        """Test creating a valid ChatCompletion."""
        message = ChatMessage(role="assistant", content="Response")
        choice = ChatCompletionChoice(index=0, message=message, finish_reason="stop")

        completion = ChatCompletion(
            id="chatcmpl-123",
            object="chat.completion",
            created=1234567890,
            model="test-model",
            choices=[choice]
        )

        assert completion.id == "chatcmpl-123"
        assert completion.model == "test-model"
        assert len(completion.choices) == 1

    def test_chat_completion_with_usage(self):
        """Test ChatCompletion with usage information."""
        message = ChatMessage(role="assistant", content="Response")
        choice = ChatCompletionChoice(index=0, message=message, finish_reason="stop")
        usage = CompletionUsage(prompt_tokens=10, completion_tokens=5, total_tokens=15)

        completion = ChatCompletion(
            id="chatcmpl-123",
            object="chat.completion",
            created=1234567890,
            model="test-model",
            choices=[choice],
            usage=usage
        )

        assert completion.usage.total_tokens == 15

    def test_chat_completion_from_api_response(self):
        """Test creating ChatCompletion from API response data."""
        api_data = {
            "id": "chatcmpl-abc123",
            "object": "chat.completion",
            "created": 1234567890,
            "model": "gpt-3.5-turbo",
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "Hello! How can I help you today?"
                },
                "finish_reason": "stop"
            }],
            "usage": {
                "prompt_tokens": 10,
                "completion_tokens": 9,
                "total_tokens": 19
            }
        }

        completion = ChatCompletion.model_validate(api_data)
        assert completion.id == "chatcmpl-abc123"
        assert completion.choices[0].message.content == "Hello! How can I help you today?"
        assert completion.usage.total_tokens == 19


class TestChatCompletionRequest:
    """Test ChatCompletionRequest model."""

    def test_valid_request(self):
        """Test creating a valid ChatCompletionRequest."""
        messages = [ChatMessage(role="user", content="Hello")]
        request = ChatCompletionRequest(model="test-model", messages=messages)

        assert request.model == "test-model"
        assert len(request.messages) == 1
        assert request.temperature is None  # Should use default

    def test_request_with_optional_params(self):
        """Test ChatCompletionRequest with optional parameters."""
        messages = [ChatMessage(role="user", content="Hello")]
        request = ChatCompletionRequest(
            model="test-model",
            messages=messages,
            temperature=0.7,
            max_tokens=100,
            top_p=0.9,
            frequency_penalty=0.1,
            presence_penalty=0.2,
            stream=False
        )

        assert request.temperature == 0.7
        assert request.max_tokens == 100
        assert request.top_p == 0.9
        assert request.frequency_penalty == 0.1
        assert request.presence_penalty == 0.2
        assert request.stream is False

    def test_request_serialization_excludes_none(self):
        """Test that request serialization excludes None values."""
        messages = [ChatMessage(role="user", content="Hello")]
        request = ChatCompletionRequest(model="test-model", messages=messages)

        data = request.model_dump(exclude_none=True)

        # Should only contain model and messages
        assert "model" in data
        assert "messages" in data
        assert "temperature" not in data
        assert "max_tokens" not in data


class TestModelInfo:
    """Test ModelInfo model."""

    def test_valid_model(self):
        """Test creating a valid ModelInfo."""
        model = ModelInfo(id="test-model", object="model", created=1234567890, owned_by="test")

        assert model.id == "test-model"
        assert model.object == "model"
        assert model.created == 1234567890
        assert model.owned_by == "test"


class TestModelList:
    """Test ModelList model."""

    def test_valid_model_list(self):
        """Test creating a valid ModelList."""
        models = [
            ModelInfo(id="model-1", object="model", created=1234567890, owned_by="test"),
            ModelInfo(id="model-2", object="model", created=1234567891, owned_by="test")
        ]
        model_list = ModelList(object="list", data=models)

        assert model_list.object == "list"
        assert len(model_list.data) == 2
        assert model_list.data[0].id == "model-1"

    def test_model_list_from_api_response(self):
        """Test creating ModelList from API response."""
        api_data = {
            "object": "list",
            "data": [
                {"id": "model-1", "object": "model", "created": 1234567890, "owned_by": "test"},
                {"id": "model-2", "object": "model", "created": 1234567891, "owned_by": "test"}
            ]
        }

        model_list = ModelList.model_validate(api_data)
        assert len(model_list.data) == 2
        assert model_list.data[1].id == "model-2"


class TestEmbeddings:
    """Test embeddings-related models."""

    def test_embedding_request(self):
        """Test EmbeddingsRequest model."""
        request = EmbeddingsRequest(input="Hello world", model="text-embedding-ada-002")

        assert request.input == "Hello world"
        assert request.model == "text-embedding-ada-002"

    def test_embedding_request_with_list_input(self):
        """Test EmbeddingsRequest with list input."""
        inputs = ["Hello", "World"]
        request = EmbeddingsRequest(input=inputs, model="test-model")

        assert request.input == inputs
        assert len(request.input) == 2

    def test_embedding_model(self):
        """Test EmbeddingData model."""
        embedding = EmbeddingData(
            object="embedding",
            embedding=[0.1, 0.2, 0.3],
            index=0
        )

        assert embedding.object == "embedding"
        assert len(embedding.embedding) == 3
        assert embedding.index == 0

    def test_embeddings_response(self):
        """Test EmbeddingsResponse model."""
        embeddings = [
            EmbeddingData(object="embedding", embedding=[0.1, 0.2], index=0),
            EmbeddingData(object="embedding", embedding=[0.3, 0.4], index=1)
        ]
        usage = CompletionUsage(prompt_tokens=5, total_tokens=5)

        response = EmbeddingsResponse(
            object="list",
            data=embeddings,
            model="test-model",
            usage=usage
        )

        assert response.object == "list"
        assert len(response.data) == 2
        assert response.model == "test-model"
        assert response.usage.total_tokens == 5


class TestCompletionUsage:
    """Test CompletionUsage model."""

    def test_valid_usage(self):
        """Test creating a valid CompletionUsage."""
        usage = CompletionUsage(prompt_tokens=10, completion_tokens=5, total_tokens=15)

        assert usage.prompt_tokens == 10
        assert usage.completion_tokens == 5
        assert usage.total_tokens == 15

    def test_usage_optional_completion_tokens(self):
        """Test CompletionUsage with optional completion_tokens."""
        usage = CompletionUsage(prompt_tokens=10, total_tokens=10)

        assert usage.prompt_tokens == 10
        assert usage.completion_tokens is None
        assert usage.total_tokens == 10


class TestHelperFunctions:
    """Test helper functions."""

    def test_create_chat_message(self):
        """Test create_chat_message helper."""
        message = create_chat_message("user", "Hello")

        assert isinstance(message, ChatMessage)
        assert message.role == "user"
        assert message.content == "Hello"

    def test_create_chat_message_with_optional_fields(self):
        """Test create_chat_message with optional fields."""
        message = create_chat_message("assistant", "Hi", name="bot")

        assert message.role == "assistant"
        assert message.content == "Hi"
        assert message.name == "bot"

    def test_create_chat_completion_request(self):
        """Test create_chat_completion_request helper."""
        messages = [create_chat_message("user", "Hello")]
        request = create_chat_completion_request("test-model", messages)

        assert isinstance(request, ChatCompletionRequest)
        assert request.model == "test-model"
        assert len(request.messages) == 1

    def test_create_chat_completion_request_with_params(self):
        """Test create_chat_completion_request with parameters."""
        messages = [create_chat_message("user", "Hello")]
        request = create_chat_completion_request(
            "test-model",
            messages,
            temperature=0.5,
            max_tokens=50
        )

        assert request.temperature == 0.5
        assert request.max_tokens == 50


class TestModelValidation:
    """Test model validation edge cases."""

    def test_negative_token_counts(self):
        """Test that negative token counts are invalid."""
        with pytest.raises(ValidationError):
            CompletionUsage(prompt_tokens=-1, total_tokens=10)

    def test_empty_message_content(self):
        """Test handling of empty message content."""
        # Empty content should be allowed for some message types
        message = ChatMessage(role="assistant", content="")
        assert message.content == ""

    def test_invalid_temperature_range(self):
        """Test that temperature outside valid range raises error."""
        messages = [ChatMessage(role="user", content="Hello")]

        # Temperature should be between 0 and 2
        with pytest.raises(ValidationError):
            ChatCompletionRequest(
                model="test",
                messages=messages,
                temperature=-0.1
            )

        with pytest.raises(ValidationError):
            ChatCompletionRequest(
                model="test",
                messages=messages,
                temperature=2.1
            )