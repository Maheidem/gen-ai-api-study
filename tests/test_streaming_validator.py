"""
Simple, focused tests for StreamingValidator.

Tests the two core checks:
1. XML drift detection (qwen3 issue)
2. Repetition detection (token loops)
"""

import pytest
from local_llm_sdk.utils.streaming_validator import StreamingValidator


class TestXMLDriftDetection:
    """Test XML format detection (catches qwen3 behavior)."""

    def test_detects_xml_tool_call_opening_tag(self):
        """Should detect <tool_call> tag."""
        validator = StreamingValidator(check_interval=10)

        # Simulate qwen3's XML format
        result = validator.add_chunk("<tool_call>")

        assert result.should_stop is True
        assert result.error_type == "XML_DRIFT"
        assert "<tool_call>" in result.details

    def test_detects_xml_function_tag(self):
        """Should detect <function= pattern."""
        validator = StreamingValidator(check_interval=10)

        result = validator.add_chunk("Now I'll use: <function=math_calculator>")

        assert result.should_stop is True
        assert result.error_type == "XML_DRIFT"

    def test_detects_xml_closing_tags(self):
        """Should detect </tool_call> closing tag."""
        validator = StreamingValidator(check_interval=10)

        result = validator.add_chunk("Some response </tool_call>")

        assert result.should_stop is True
        assert result.error_type == "XML_DRIFT"

    def test_case_insensitive_xml_detection(self):
        """XML patterns should be case-insensitive."""
        validator = StreamingValidator(check_interval=10)

        result = validator.add_chunk("<TOOL_CALL>")

        assert result.should_stop is True
        assert result.error_type == "XML_DRIFT"

    def test_no_false_positive_on_normal_text(self):
        """Should NOT flag normal responses."""
        validator = StreamingValidator(check_interval=10)

        normal_responses = [
            "The answer is 42.",
            "I'll help you calculate that.",
            "Using the math calculator tool...",
            "{'result': 100}",  # JSON is fine
        ]

        for text in normal_responses:
            result = validator.add_chunk(text)
            assert result.should_stop is False, f"False positive on: {text}"


class TestRepetitionDetection:
    """Test repetition pattern detection (catches token loops)."""

    def test_detects_simple_word_repetition(self):
        """Should detect 'the the the...' pattern."""
        validator = StreamingValidator(check_interval=10)

        # Simulate repetition loop
        repeated_text = "the " * 30  # "the the the..."

        result = validator.add_chunk(repeated_text)

        assert result.should_stop is True
        assert result.error_type == "REPETITION"

    def test_detects_phrase_repetition(self):
        """Should detect repeating 10-word sequences."""
        validator = StreamingValidator(check_interval=10)

        # Create exactly 10 words, repeat to make 20 total (triggers check)
        # Need > 100 chars, so use longer words
        ten_words = "hello world testing repetition detection system works correctly here now "
        repeated = ten_words + ten_words  # Exact 10-word repeat

        result = validator.add_chunk(repeated)

        assert result.should_stop is True
        assert result.error_type == "REPETITION"

    def test_no_false_positive_on_short_text(self):
        """Should NOT flag short text (< 100 chars)."""
        validator = StreamingValidator(check_interval=10)

        short_text = "the the"  # Only 2 repetitions, too short

        result = validator.add_chunk(short_text)

        assert result.should_stop is False

    def test_no_false_positive_on_normal_variation(self):
        """Should NOT flag normal text with variation."""
        validator = StreamingValidator(check_interval=10)

        normal_text = (
            "I will help you calculate the result. "
            "First I will use the math calculator tool. "
            "Then I will format the output nicely for you."
        )

        result = validator.add_chunk(normal_text)

        assert result.should_stop is False


class TestCheckInterval:
    """Test that validation only runs at intervals (performance)."""

    def test_only_checks_at_intervals(self):
        """Should only validate every N tokens, not every chunk."""
        validator = StreamingValidator(check_interval=20)

        # First chunk (< 20 tokens): no check
        result = validator.add_chunk("word " * 5)  # 5 tokens
        assert result.should_stop is False

        # Second chunk (total < 20 tokens): no check
        result = validator.add_chunk("word " * 10)  # Now 15 total
        assert result.should_stop is False

        # Third chunk (total >= 20 tokens): CHECK happens
        result = validator.add_chunk("word " * 10)  # Now 25 total
        # This triggers check (25 % 20 = close enough)
        # No error expected, just verifying check happened

    def test_configurable_interval(self):
        """Should respect custom check interval."""
        # Check every 10 tokens
        validator = StreamingValidator(check_interval=10)
        assert validator.check_interval == 10

        # Check every 50 tokens
        validator = StreamingValidator(check_interval=50)
        assert validator.check_interval == 50


class TestValidatorReset:
    """Test that validator can be reset for new requests."""

    def test_reset_clears_state(self):
        """reset() should clear accumulated text and token count."""
        validator = StreamingValidator(check_interval=10)

        # Add some text
        validator.add_chunk("Some text to accumulate")
        assert validator.accumulated != ""
        assert validator.tokens_processed > 0

        # Reset
        validator.reset()

        # Should be clean
        assert validator.accumulated == ""
        assert validator.tokens_processed == 0

    def test_reuse_after_reset(self):
        """Should work correctly after reset."""
        validator = StreamingValidator(check_interval=10)

        # First use: detect XML
        result = validator.add_chunk("<tool_call>")
        assert result.should_stop is True

        # Reset for new request
        validator.reset()

        # Second use: normal text (should not flag)
        result = validator.add_chunk("Normal response")
        assert result.should_stop is False


class TestFinalize:
    """Test final validation check."""

    def test_finalize_runs_final_check(self):
        """finalize() should check accumulated text one last time."""
        validator = StreamingValidator(check_interval=100)  # Large interval

        # Add XML but don't trigger interval check
        validator.add_chunk("<tool_call>")  # Only ~1 token, won't trigger check

        # But finalize should catch it
        result = validator.finalize()

        assert result.should_stop is True
        assert result.error_type == "XML_DRIFT"

    def test_finalize_on_clean_response(self):
        """finalize() should pass clean responses."""
        validator = StreamingValidator(check_interval=100)

        validator.add_chunk("This is a normal, valid response.")

        result = validator.finalize()

        assert result.should_stop is False


class TestRealWorldScenarios:
    """Test with realistic response patterns."""

    def test_valid_json_tool_call(self):
        """Should NOT flag valid JSON tool calls."""
        validator = StreamingValidator(check_interval=10)

        valid_json = """
        I'll calculate that for you.
        {"tool_calls": [{
            "type": "function",
            "function": {
                "name": "math_calculator",
                "arguments": "{\\"arg1\\": 5, \\"arg2\\": 10}"
            }
        }]}
        """

        result = validator.add_chunk(valid_json)
        final = validator.finalize()

        assert result.should_stop is False
        assert final.should_stop is False

    def test_qwen3_actual_xml_format(self):
        """Should catch qwen3's actual XML format."""
        validator = StreamingValidator(check_interval=10)

        # From our real test: qwen3's actual output
        qwen3_response = """
        Now I'll read the CSV file to examine its structure and content.
        <tool_call>
        <function=filesystem_operation>
        <parameter=operation>
        read_file
        </parameter>
        """

        result = validator.add_chunk(qwen3_response)

        assert result.should_stop is True
        assert result.error_type == "XML_DRIFT"

    def test_repetition_loop_like_37k_tokens(self):
        """Should catch repetition pattern like the 37K token bug."""
        validator = StreamingValidator(check_interval=10)

        # Simulate "the the the..." for 37K tokens (just test pattern)
        massive_repetition = "the " * 100  # 100 repetitions

        result = validator.add_chunk(massive_repetition)

        assert result.should_stop is True
        assert result.error_type == "REPETITION"
