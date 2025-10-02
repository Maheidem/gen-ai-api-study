"""
Tests for response validation pipeline.
"""

import pytest
import json
from local_llm_sdk.models import ChatCompletion, ChatCompletionChoice, ChatMessage, ToolCall, FunctionCall
from local_llm_sdk.utils.validators import (
    FastValidator,
    StructuralValidator,
    SemanticValidator,
    LLMJudge,
    ValidationPipeline,
    ValidationResult
)


class TestFastValidator:
    """Tests for FastValidator."""

    def test_detects_xml_drift(self):
        """Should detect XML-like tool call format."""
        # Create response with XML in content
        message = ChatMessage(role="assistant", content="<tool_call><function=test></function></tool_call>")
        response = ChatCompletion(
            id="test",
            choices=[ChatCompletionChoice(index=0, message=message, finish_reason="stop")],
            created=123,
            model="test",
            object="chat.completion"
        )

        validator = FastValidator()
        result = validator.validate(response)

        assert not result.is_valid
        assert result.error_type == "XML_DRIFT"

    def test_detects_invalid_json_in_tool_calls(self):
        """Should detect malformed JSON in tool_calls."""
        # Create tool call with invalid JSON
        tool_call = ToolCall(
            id="call_1",
            type="function",
            function=FunctionCall(name="test_func", arguments="{invalid json}")
        )
        message = ChatMessage(role="assistant", content="", tool_calls=[tool_call])
        response = ChatCompletion(
            id="test",
            choices=[ChatCompletionChoice(index=0, message=message, finish_reason="stop")],
            created=123,
            model="test",
            object="chat.completion"
        )

        validator = FastValidator()
        result = validator.validate(response)

        assert not result.is_valid
        assert result.error_type == "INVALID_JSON"

    def test_passes_valid_response(self):
        """Should pass valid responses."""
        tool_call = ToolCall(
            id="call_1",
            type="function",
            function=FunctionCall(name="test_func", arguments='{"param": "value"}')
        )
        message = ChatMessage(role="assistant", content="", tool_calls=[tool_call])
        response = ChatCompletion(
            id="test",
            choices=[ChatCompletionChoice(index=0, message=message, finish_reason="stop")],
            created=123,
            model="test",
            object="chat.completion"
        )

        validator = FastValidator()
        result = validator.validate(response)

        assert result.is_valid


class TestStructuralValidator:
    """Tests for StructuralValidator."""

    def test_validates_against_schema(self):
        """Should validate tool calls against schemas."""
        schemas = [{
            "type": "function",
            "function": {
                "name": "test_func",
                "parameters": {
                    "type": "object",
                    "properties": {"param": {"type": "string"}},
                    "required": ["param"]
                }
            }
        }]

        # Valid tool call
        tool_call = ToolCall(
            id="call_1",
            type="function",
            function=FunctionCall(name="test_func", arguments='{"param": "value"}')
        )
        message = ChatMessage(role="assistant", content="", tool_calls=[tool_call])
        response = ChatCompletion(
            id="test",
            choices=[ChatCompletionChoice(index=0, message=message, finish_reason="stop")],
            created=123,
            model="test",
            object="chat.completion"
        )

        validator = StructuralValidator(schemas)
        result = validator.validate(response)

        assert result.is_valid

    def test_detects_unknown_function(self):
        """Should detect calls to unknown functions."""
        schemas = [{
            "type": "function",
            "function": {
                "name": "known_func",
                "parameters": {"type": "object"}
            }
        }]

        # Call to unknown function
        tool_call = ToolCall(
            id="call_1",
            type="function",
            function=FunctionCall(name="unknown_func", arguments='{}')
        )
        message = ChatMessage(role="assistant", content="", tool_calls=[tool_call])
        response = ChatCompletion(
            id="test",
            choices=[ChatCompletionChoice(index=0, message=message, finish_reason="stop")],
            created=123,
            model="test",
            object="chat.completion"
        )

        validator = StructuralValidator(schemas)
        result = validator.validate(response)

        assert not result.is_valid
        assert result.error_type == "UNKNOWN_FUNCTION"

    def test_detects_missing_required_field(self):
        """Should detect missing required fields."""
        schemas = [{
            "type": "function",
            "function": {
                "name": "test_func",
                "parameters": {
                    "type": "object",
                    "properties": {"required_param": {"type": "string"}},
                    "required": ["required_param"]
                }
            }
        }]

        # Missing required field
        tool_call = ToolCall(
            id="call_1",
            type="function",
            function=FunctionCall(name="test_func", arguments='{}')  # Empty args
        )
        message = ChatMessage(role="assistant", content="", tool_calls=[tool_call])
        response = ChatCompletion(
            id="test",
            choices=[ChatCompletionChoice(index=0, message=message, finish_reason="stop")],
            created=123,
            model="test",
            object="chat.completion"
        )

        validator = StructuralValidator(schemas)
        result = validator.validate(response)

        assert not result.is_valid
        assert result.error_type == "MISSING_FIELD"

    def test_passes_with_no_tool_calls(self):
        """Should pass responses without tool calls."""
        message = ChatMessage(role="assistant", content="Just a text response")
        response = ChatCompletion(
            id="test",
            choices=[ChatCompletionChoice(index=0, message=message, finish_reason="stop")],
            created=123,
            model="test",
            object="chat.completion"
        )

        validator = StructuralValidator([])
        result = validator.validate(response)

        assert result.is_valid


class TestSemanticValidator:
    """Tests for SemanticValidator."""

    def test_detects_repetition(self):
        """Should detect repetitive content."""
        message = ChatMessage(role="assistant", content="the the the the the the the")
        response = ChatCompletion(
            id="test",
            choices=[ChatCompletionChoice(index=0, message=message, finish_reason="stop")],
            created=123,
            model="test",
            object="chat.completion"
        )

        validator = SemanticValidator()
        result = validator.validate(response)

        assert not result.is_valid
        assert "REPETITION" in result.error_type or "ENTROPY" in result.error_type or "DIVERSITY" in result.error_type

    def test_passes_normal_content(self):
        """Should pass normal diverse content."""
        message = ChatMessage(role="assistant", content="This is a normal response with good diversity")
        response = ChatCompletion(
            id="test",
            choices=[ChatCompletionChoice(index=0, message=message, finish_reason="stop")],
            created=123,
            model="test",
            object="chat.completion"
        )

        validator = SemanticValidator()
        result = validator.validate(response)

        assert result.is_valid

    def test_allows_empty_with_tool_calls(self):
        """Should allow empty content if tool calls present."""
        tool_call = ToolCall(
            id="call_1",
            type="function",
            function=FunctionCall(name="test", arguments='{}')
        )
        message = ChatMessage(role="assistant", content="", tool_calls=[tool_call])
        response = ChatCompletion(
            id="test",
            choices=[ChatCompletionChoice(index=0, message=message, finish_reason="stop")],
            created=123,
            model="test",
            object="chat.completion"
        )

        validator = SemanticValidator()
        result = validator.validate(response)

        assert result.is_valid

    def test_fails_empty_without_tool_calls(self):
        """Should fail if content empty and no tool calls."""
        message = ChatMessage(role="assistant", content="")
        response = ChatCompletion(
            id="test",
            choices=[ChatCompletionChoice(index=0, message=message, finish_reason="stop")],
            created=123,
            model="test",
            object="chat.completion"
        )

        validator = SemanticValidator()
        result = validator.validate(response)

        assert not result.is_valid
        assert result.error_type == "EMPTY_RESPONSE"


class TestLLMJudge:
    """Tests for LLMJudge."""

    def test_passes_when_disabled(self):
        """Should pass through when disabled."""
        message = ChatMessage(role="assistant", content="any content")
        response = ChatCompletion(
            id="test",
            choices=[ChatCompletionChoice(index=0, message=message, finish_reason="stop")],
            created=123,
            model="test",
            object="chat.completion"
        )

        judge = LLMJudge(judge_client=None, enabled=False)
        result = judge.validate(response)

        assert result.is_valid

    def test_handles_judge_errors_gracefully(self):
        """Should pass through if judge fails."""
        # Mock client that raises error
        class BrokenClient:
            def chat(self, *args, **kwargs):
                raise Exception("Judge error")

        message = ChatMessage(role="assistant", content="test")
        response = ChatCompletion(
            id="test",
            choices=[ChatCompletionChoice(index=0, message=message, finish_reason="stop")],
            created=123,
            model="test",
            object="chat.completion"
        )

        judge = LLMJudge(judge_client=BrokenClient(), enabled=True)
        result = judge.validate(response)

        # Should pass through on error
        assert result.is_valid


class TestValidationPipeline:
    """Tests for ValidationPipeline."""

    def test_fails_fast_on_xml_drift(self):
        """Should stop at first validator failure."""
        message = ChatMessage(role="assistant", content="<tool_call>xml</tool_call>")
        response = ChatCompletion(
            id="test",
            choices=[ChatCompletionChoice(index=0, message=message, finish_reason="stop")],
            created=123,
            model="test",
            object="chat.completion"
        )

        pipeline = ValidationPipeline()
        is_valid, error = pipeline.validate_all(response)

        assert not is_valid
        assert error == "XML_DRIFT"

    def test_passes_all_stages(self):
        """Should pass through all validators."""
        tool_call = ToolCall(
            id="call_1",
            type="function",
            function=FunctionCall(name="test", arguments='{"param": "value"}')
        )
        message = ChatMessage(role="assistant", content="Normal text", tool_calls=[tool_call])
        response = ChatCompletion(
            id="test",
            choices=[ChatCompletionChoice(index=0, message=message, finish_reason="stop")],
            created=123,
            model="test",
            object="chat.completion"
        )

        pipeline = ValidationPipeline()
        is_valid, error = pipeline.validate_all(response)

        assert is_valid
        assert error == ""

    def test_uses_judge_for_qwen_model(self):
        """Should trigger judge for known problematic models."""
        pipeline = ValidationPipeline(enable_judge=True)

        # Should use judge for qwen
        assert pipeline.should_use_judge(None, "qwen/qwen3-coder-30b")

        # Should not use judge for other models if not configured
        pipeline.enable_judge = False
        assert not pipeline.should_use_judge(None, "mistralai/magistral")
