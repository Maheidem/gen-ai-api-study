"""
Integration tests for validation system in client.py.

Tests the ACTUAL validation flow: client.chat() → validate → alert → recover → error
"""

import pytest
import os
from local_llm_sdk import LocalLLMClient
from local_llm_sdk.models import ChatMessage, ChatCompletion, ChatCompletionChoice


class TestValidationIntegration:
    """Test validation catches drift in client.chat()."""

    def test_validation_catches_xml_drift(self, monkeypatch, capsys):
        """Validation should catch XML drift and alert user."""
        # Enable validation
        monkeypatch.setenv("LLM_ENABLE_VALIDATION", "true")
        monkeypatch.setenv("LLM_ENABLE_AUTO_RECOVERY", "false")  # Disable recovery to test detection

        # Mock response with XML drift
        def mock_post(*args, **kwargs):
            class MockResponse:
                def raise_for_status(self):
                    pass

                def json(self):
                    return {
                        "id": "test",
                        "object": "chat.completion",
                        "created": 123,
                        "model": "test",
                        "choices": [{
                            "index": 0,
                            "message": {
                                "role": "assistant",
                                "content": "<tool_call><function=test></function></tool_call>"  # XML DRIFT!
                            },
                            "finish_reason": "stop"
                        }]
                    }
            return MockResponse()

        import requests
        monkeypatch.setattr(requests, "post", mock_post)

        # Create client with validation enabled
        client = LocalLLMClient("http://test:1234/v1", "test-model")

        # Should raise error and print alert
        with pytest.raises(ValueError, match="XML_DRIFT"):
            client.chat("test message")

        # Check alerts were printed
        captured = capsys.readouterr()
        assert "🚨 ALERT" in captured.out
        assert "XML_DRIFT" in captured.out

    def test_validation_catches_repetition(self, monkeypatch, capsys):
        """Validation should catch repetitive responses."""
        monkeypatch.setenv("LLM_ENABLE_VALIDATION", "true")
        monkeypatch.setenv("LLM_ENABLE_AUTO_RECOVERY", "false")

        def mock_post(*args, **kwargs):
            class MockResponse:
                def raise_for_status(self):
                    pass

                def json(self):
                    return {
                        "id": "test",
                        "object": "chat.completion",
                        "created": 123,
                        "model": "test",
                        "choices": [{
                            "index": 0,
                            "message": {
                                "role": "assistant",
                                "content": "the the the the the the the the the the"  # REPETITION!
                            },
                            "finish_reason": "stop"
                        }]
                    }
            return MockResponse()

        import requests
        monkeypatch.setattr(requests, "post", mock_post)

        client = LocalLLMClient("http://test:1234/v1", "test-model")

        with pytest.raises(ValueError, match="REPETITION|ENTROPY|DIVERSITY"):
            client.chat("test")

        captured = capsys.readouterr()
        assert "🚨 ALERT" in captured.out

    def test_validation_disabled_by_default(self, monkeypatch):
        """Validation should be OFF by default for backward compatibility."""
        # Don't set LLM_ENABLE_VALIDATION

        def mock_post(*args, **kwargs):
            class MockResponse:
                def raise_for_status(self):
                    pass

                def json(self):
                    return {
                        "id": "test",
                        "object": "chat.completion",
                        "created": 123,
                        "model": "test",
                        "choices": [{
                            "index": 0,
                            "message": {
                                "role": "assistant",
                                "content": "<tool_call>XML</tool_call>"  # Would fail validation
                            },
                            "finish_reason": "stop"
                        }]
                    }
            return MockResponse()

        import requests
        monkeypatch.setattr(requests, "post", mock_post)

        client = LocalLLMClient("http://test:1234/v1", "test-model")

        # Should NOT raise error (validation disabled)
        response = client.chat("test")
        assert response  # Just shouldn't crash

    def test_recovery_attempts_on_validation_failure(self, monkeypatch, capsys):
        """Recovery should be attempted when validation fails."""
        monkeypatch.setenv("LLM_ENABLE_VALIDATION", "true")
        monkeypatch.setenv("LLM_ENABLE_AUTO_RECOVERY", "true")

        call_count = [0]

        def mock_post(*args, **kwargs):
            call_count[0] += 1

            class MockResponse:
                def raise_for_status(self):
                    pass

                def json(self):
                    if call_count[0] == 1:
                        # First call: return XML drift
                        return {
                            "id": "test",
                            "object": "chat.completion",
                            "created": 123,
                            "model": "test",
                            "choices": [{
                                "index": 0,
                                "message": {
                                    "role": "assistant",
                                    "content": "<tool_call>drift</tool_call>"
                                },
                                "finish_reason": "stop"
                            }]
                        }
                    else:
                        # Recovery attempt: return valid response
                        return {
                            "id": "test",
                            "object": "chat.completion",
                            "created": 123,
                            "model": "test",
                            "choices": [{
                                "index": 0,
                                "message": {
                                    "role": "assistant",
                                    "content": "Valid response"
                                },
                                "finish_reason": "stop"
                            }]
                        }
            return MockResponse()

        import requests
        monkeypatch.setattr(requests, "post", mock_post)

        client = LocalLLMClient("http://test:1234/v1", "test-model")

        # Should succeed via recovery
        response = client.chat("test")
        assert "Valid response" in str(response)

        # Check recovery was attempted
        captured = capsys.readouterr()
        assert "🔧 Attempting recovery" in captured.out
        assert "✅ Recovery successful" in captured.out

    def test_max_tokens_set_when_validation_enabled(self, monkeypatch):
        """Should set max_tokens=2048 when validation enabled to prevent runaway."""
        monkeypatch.setenv("LLM_ENABLE_VALIDATION", "true")

        request_data = {}

        def mock_post(*args, **kwargs):
            request_data.update(kwargs.get('json', {}))

            class MockResponse:
                def raise_for_status(self):
                    pass

                def json(self):
                    return {
                        "id": "test",
                        "object": "chat.completion",
                        "created": 123,
                        "model": "test",
                        "choices": [{
                            "index": 0,
                            "message": {
                                "role": "assistant",
                                "content": "Normal response"
                            },
                            "finish_reason": "stop"
                        }]
                    }
            return MockResponse()

        import requests
        monkeypatch.setattr(requests, "post", mock_post)

        client = LocalLLMClient("http://test:1234/v1", "test-model")
        client.chat("test")

        # Check max_tokens was set to 2048
        assert request_data.get("max_tokens") == 2048
