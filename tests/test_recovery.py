"""
Tests for recovery strategies.
"""

import pytest
from local_llm_sdk.models import ChatMessage, ChatCompletion, ChatCompletionChoice, create_chat_message
from local_llm_sdk.utils.recovery import (
    CorrectionPromptStrategy,
    HistorySanitizationStrategy,
    TemperatureOverrideStrategy,
    CheckpointRollbackStrategy,
    GracefulAbortStrategy,
    RecoveryManager
)


class MockClient:
    """Mock client for testing recovery."""

    def __init__(self, should_succeed=True):
        self.should_succeed = should_succeed
        self.call_count = 0
        self.last_temperature = None

    def chat(self, messages, use_tools=True, return_full_response=True, **kwargs):
        """Mock chat method."""
        self.call_count += 1
        self.last_temperature = kwargs.get('temperature', 0.7)

        if self.should_succeed:
            # Return valid response
            message = ChatMessage(role="assistant", content="Recovered response")
            return ChatCompletion(
                id="test",
                choices=[ChatCompletionChoice(index=0, message=message, finish_reason="stop")],
                created=123,
                model="test",
                object="chat.completion"
            )
        else:
            raise Exception("Mock failure")


class TestCorrectionPromptStrategy:
    """Tests for CorrectionPromptStrategy."""

    def test_adds_correction_prompt(self):
        """Should add correction prompt and retry."""
        client = MockClient(should_succeed=True)
        strategy = CorrectionPromptStrategy(client)

        invalid_response = ChatCompletion(
            id="test",
            choices=[ChatCompletionChoice(
                index=0,
                message=ChatMessage(role="assistant", content="<tool_call>xml</tool_call>"),
                finish_reason="stop"
            )],
            created=123,
            model="test",
            object="chat.completion"
        )

        messages = [create_chat_message("user", "test")]

        success, result = strategy.attempt_recovery(invalid_response, messages, "XML_DRIFT")

        assert success
        assert client.call_count == 1
        assert client.last_temperature == 0.3  # Should use lower temperature

    def test_handles_recovery_failure(self):
        """Should handle when recovery fails."""
        client = MockClient(should_succeed=False)
        strategy = CorrectionPromptStrategy(client)

        invalid_response = ChatCompletion(
            id="test",
            choices=[ChatCompletionChoice(
                index=0,
                message=ChatMessage(role="assistant", content="bad"),
                finish_reason="stop"
            )],
            created=123,
            model="test",
            object="chat.completion"
        )

        messages = [create_chat_message("user", "test")]

        success, result = strategy.attempt_recovery(invalid_response, messages, "XML_DRIFT")

        assert not success
        assert isinstance(result, str)  # Error message


class TestHistorySanitizationStrategy:
    """Tests for HistorySanitizationStrategy."""

    def test_retries_with_clean_history(self):
        """Should retry without including invalid response."""
        client = MockClient(should_succeed=True)
        strategy = HistorySanitizationStrategy(client)

        invalid_response = ChatCompletion(
            id="test",
            choices=[ChatCompletionChoice(
                index=0,
                message=ChatMessage(role="assistant", content="bad"),
                finish_reason="stop"
            )],
            created=123,
            model="test",
            object="chat.completion"
        )

        messages = [create_chat_message("user", "test")]

        success, result = strategy.attempt_recovery(invalid_response, messages, "INVALID_JSON")

        assert success
        assert client.call_count == 1
        assert client.last_temperature == 0.2  # Even lower temp


class TestTemperatureOverrideStrategy:
    """Tests for TemperatureOverrideStrategy."""

    def test_uses_low_temperature(self):
        """Should retry with very low temperature."""
        client = MockClient(should_succeed=True)
        strategy = TemperatureOverrideStrategy(client)

        invalid_response = ChatCompletion(
            id="test",
            choices=[ChatCompletionChoice(
                index=0,
                message=ChatMessage(role="assistant", content="bad"),
                finish_reason="stop"
            )],
            created=123,
            model="test",
            object="chat.completion"
        )

        messages = [create_chat_message("user", "test")]

        success, result = strategy.attempt_recovery(invalid_response, messages, "NGRAM_REPETITION")

        assert success
        assert client.last_temperature == 0.1  # Near-deterministic


class TestCheckpointRollbackStrategy:
    """Tests for CheckpointRollbackStrategy."""

    def test_rolls_back_to_checkpoint(self):
        """Should rollback to last good state."""
        client = MockClient(should_succeed=True)

        # Create checkpoint
        checkpoint = [create_chat_message("user", "original")]
        strategy = CheckpointRollbackStrategy(client, checkpoints=[checkpoint])

        invalid_response = ChatCompletion(
            id="test",
            choices=[ChatCompletionChoice(
                index=0,
                message=ChatMessage(role="assistant", content="bad"),
                finish_reason="stop"
            )],
            created=123,
            model="test",
            object="chat.completion"
        )

        # Current corrupted messages
        messages = [
            create_chat_message("user", "original"),
            create_chat_message("assistant", "corrupted")
        ]

        success, result = strategy.attempt_recovery(invalid_response, messages, "XML_DRIFT")

        assert success

    def test_fails_without_checkpoints(self):
        """Should fail if no checkpoints available."""
        client = MockClient(should_succeed=True)
        strategy = CheckpointRollbackStrategy(client, checkpoints=[])

        invalid_response = ChatCompletion(
            id="test",
            choices=[ChatCompletionChoice(
                index=0,
                message=ChatMessage(role="assistant", content="bad"),
                finish_reason="stop"
            )],
            created=123,
            model="test",
            object="chat.completion"
        )

        messages = [create_chat_message("user", "test")]

        success, result = strategy.attempt_recovery(invalid_response, messages, "XML_DRIFT")

        assert not success


class TestGracefulAbortStrategy:
    """Tests for GracefulAbortStrategy."""

    def test_always_fails_with_message(self):
        """Should always fail but provide error message."""
        client = MockClient()
        strategy = GracefulAbortStrategy(client)

        invalid_response = ChatCompletion(
            id="test",
            choices=[ChatCompletionChoice(
                index=0,
                message=ChatMessage(role="assistant", content="bad"),
                finish_reason="stop"
            )],
            created=123,
            model="test",
            object="chat.completion"
        )

        messages = [create_chat_message("user", "test")]

        success, result = strategy.attempt_recovery(invalid_response, messages, "UNKNOWN_ERROR")

        assert not success
        assert isinstance(result, str)
        assert "Unable to recover" in result


class TestRecoveryManager:
    """Tests for RecoveryManager."""

    def test_tries_strategies_in_order(self):
        """Should try strategies until one succeeds."""
        client = MockClient(should_succeed=True)
        manager = RecoveryManager(client, max_attempts=3)

        invalid_response = ChatCompletion(
            id="test",
            choices=[ChatCompletionChoice(
                index=0,
                message=ChatMessage(role="assistant", content="bad"),
                finish_reason="stop"
            )],
            created=123,
            model="test",
            object="chat.completion"
        )

        messages = [create_chat_message("user", "test")]

        success, result = manager.recover(invalid_response, messages, "XML_DRIFT")

        assert success
        # First strategy should succeed
        assert client.call_count == 1

    def test_respects_max_attempts(self):
        """Should stop after max attempts."""
        client = MockClient(should_succeed=False)
        manager = RecoveryManager(client, max_attempts=2)

        invalid_response = ChatCompletion(
            id="test",
            choices=[ChatCompletionChoice(
                index=0,
                message=ChatMessage(role="assistant", content="bad"),
                finish_reason="stop"
            )],
            created=123,
            model="test",
            object="chat.completion"
        )

        messages = [create_chat_message("user", "test")]

        success, result = manager.recover(invalid_response, messages, "XML_DRIFT")

        assert not success
        # Should try max_attempts times
        assert client.call_count == 2

    def test_saves_checkpoints(self):
        """Should save and manage checkpoints."""
        client = MockClient()
        manager = RecoveryManager(client)

        messages1 = [create_chat_message("user", "msg1")]
        messages2 = [create_chat_message("user", "msg2")]

        manager.save_checkpoint(messages1)
        assert len(manager.checkpoints) == 1

        manager.save_checkpoint(messages2)
        assert len(manager.checkpoints) == 2

    def test_limits_checkpoint_count(self):
        """Should keep only last 5 checkpoints."""
        client = MockClient()
        manager = RecoveryManager(client)

        # Save 10 checkpoints
        for i in range(10):
            messages = [create_chat_message("user", f"msg{i}")]
            manager.save_checkpoint(messages)

        # Should only keep last 5
        assert len(manager.checkpoints) == 5
