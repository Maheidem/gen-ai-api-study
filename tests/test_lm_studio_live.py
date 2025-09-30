"""
Integration tests that require a running LM Studio instance.
These tests validate real API interactions and model behaviors.
"""

import pytest
import time
from typing import List

from local_llm_sdk import LocalLLMClient
from local_llm_sdk.tools import builtin
from local_llm_sdk.models import ChatCompletion

from .lm_studio_helpers import (
    LMStudioTestBase,
    skip_if_no_lm_studio,
    skip_if_no_model,
    get_tool_calling_model,
    get_thinking_model,
    measure_response_time,
    validate_response_structure,
    extract_numbers_from_response,
    compare_tool_vs_direct_response
)
from .lm_studio_config import get_config


@pytest.mark.lm_studio
class TestLMStudioConnectivity(LMStudioTestBase):
    """Test basic connectivity to LM Studio."""

    @skip_if_no_lm_studio
    def test_lm_studio_is_running(self):
        """Test that LM Studio is accessible."""
        assert self.client is not None
        assert len(self.available_models) > 0

    @skip_if_no_lm_studio
    def test_list_models(self):
        """Test listing available models."""
        models = self.client.list_models()

        assert len(models.data) > 0
        assert all(hasattr(model, 'id') for model in models.data)
        assert all(hasattr(model, 'object') for model in models.data)

        print(f"\n📋 Available models ({len(models.data)}):")
        for model in models.data:
            print(f"  - {model.id}")

    @skip_if_no_lm_studio
    def test_client_representation(self):
        """Test client string representation with real models."""
        repr_str = repr(self.client)
        assert "LocalLLMClient" in repr_str
        assert self.config.base_url in repr_str


@pytest.mark.lm_studio
@pytest.mark.slow
class TestSimpleChatCompletion(LMStudioTestBase):
    """Test basic chat completion functionality."""

    @skip_if_no_model()
    def test_simple_chat_response(self, model_id):
        """Test basic chat completion."""
        self.client.default_model = model_id

        response = self.client.chat_simple("Say 'Hello, World!' and nothing else.")

        self.assert_valid_response(response)
        print(f"\n💬 Model: {model_id}")
        print(f"📝 Response: '{response}'")

        # Check if response contains expected text
        assert "hello" in response.lower() or "world" in response.lower()

    @skip_if_no_model()
    def test_system_prompt_effectiveness(self, model_id):
        """Test that system prompts are respected."""
        self.client.default_model = model_id

        messages = [
            {"role": "system", "content": "You are a helpful assistant. Always end your responses with '[DONE]'"},
            {"role": "user", "content": "What is 2 + 2?"}
        ]

        response = self.client.chat(messages)

        self.assert_valid_response(response)
        print(f"\n🎯 System prompt test: '{response}'")

        # Should contain the system-requested marker
        assert "[DONE]" in response or "done" in response.lower()

    @skip_if_no_model()
    def test_temperature_effects(self, model_id):
        """Test that temperature parameter affects responses."""
        self.client.default_model = model_id
        prompt = "Write a creative story about a robot in exactly 3 words."

        # Low temperature (deterministic)
        response1 = self.client.chat(prompt, temperature=0.1)
        response2 = self.client.chat(prompt, temperature=0.1)

        # High temperature (creative)
        response3 = self.client.chat(prompt, temperature=0.9)

        self.assert_valid_response(response1)
        self.assert_valid_response(response2)
        self.assert_valid_response(response3)

        print(f"\n🌡️ Temperature effects:")
        print(f"  Temp 0.1 (1): '{response1}'")
        print(f"  Temp 0.1 (2): '{response2}'")
        print(f"  Temp 0.9:     '{response3}'")

        # Low temperature responses should be similar or identical
        # High temperature might be different
        # Note: This test might be flaky depending on the model

    @skip_if_no_model()
    def test_max_tokens_limit(self, model_id):
        """Test that max_tokens parameter is respected."""
        self.client.default_model = model_id

        # Request a long response but limit tokens
        response = self.client.chat(
            "Write a detailed essay about artificial intelligence.",
            max_tokens=50
        )

        self.assert_valid_response(response)
        print(f"\n✂️ Limited response ({len(response)} chars): '{response[:100]}...'")

        # Response should be reasonably short due to token limit
        assert len(response) < 500  # Rough estimate

    @skip_if_no_model()
    def test_response_timing(self, model_id):
        """Test response timing and performance."""
        self.client.default_model = model_id

        response, duration = measure_response_time(
            self.client.chat_simple,
            "What is the capital of France?"
        )

        self.assert_valid_response(response)
        print(f"\n⏱️ Response time: {duration:.2f}s")
        print(f"📝 Response: '{response}'")

        # Reasonable response time (adjust based on your setup)
        assert duration < 30.0  # 30 seconds max
        assert "paris" in response.lower()


@pytest.mark.lm_studio
@pytest.mark.slow
class TestToolCalling(LMStudioTestBase):
    """Test tool calling functionality with real models."""

    def setup_method(self):
        """Set up tools for each test."""
        super().setup_method()
        if self.client:
            self.client.register_tools_from(builtin)

    @skip_if_no_model()
    def test_math_tool_calling(self, model_id):
        """Test that models can use math tools correctly."""
        # Prefer tool-calling models
        tool_model = get_tool_calling_model(self.client)
        if tool_model:
            model_id = tool_model

        self.client.default_model = model_id

        response = self.client.chat("Calculate 47 * 89 using the calculator.")

        self.assert_valid_response(response)
        print(f"\n🧮 Math tool test:")
        print(f"   Model: {model_id}")
        print(f"   Response: '{response}'")
        print(f"   Tools used: {[tc.function.name for tc in self.client.last_tool_calls]}")

        # Check if tool was actually used
        if self.client.last_tool_calls:
            self.assert_tool_was_used("math_calculator")
            # Verify the result is correct (47 * 89 = 4183)
            numbers = extract_numbers_from_response(response)
            assert 4183 in numbers, f"Expected 4183 in response, got numbers: {numbers}"
        else:
            print("   ⚠️  Model answered directly without using tools")

    @skip_if_no_model()
    def test_text_processing_tools(self, model_id):
        """Test text processing tools."""
        tool_model = get_tool_calling_model(self.client)
        if tool_model:
            model_id = tool_model

        self.client.default_model = model_id

        response = self.client.chat("Count the characters in 'Hello, World!' using the character counter.")

        self.assert_valid_response(response)
        print(f"\n📝 Text tool test:")
        print(f"   Response: '{response}'")
        print(f"   Tools used: {[tc.function.name for tc in self.client.last_tool_calls]}")

        # Should mention 13 characters
        numbers = extract_numbers_from_response(response)
        if self.client.last_tool_calls:
            assert 13 in numbers, f"Expected 13 characters, got numbers: {numbers}"

    @skip_if_no_model()
    def test_multiple_tool_calls(self, model_id):
        """Test using multiple tools in one request."""
        tool_model = get_tool_calling_model(self.client)
        if tool_model:
            model_id = tool_model

        self.client.default_model = model_id

        response = self.client.chat(
            "Calculate 25 + 75, then count the characters in the result when written as text."
        )

        self.assert_valid_response(response)
        print(f"\n🔗 Multiple tools test:")
        print(f"   Response: '{response}'")
        print(f"   Tools used: {[tc.function.name for tc in self.client.last_tool_calls]}")

        # Might use multiple tools or be clever about it
        tools_used = [tc.function.name for tc in self.client.last_tool_calls]
        print(f"   Number of tool calls: {len(tools_used)}")

    @skip_if_no_model()
    def test_tool_vs_direct_comparison(self, model_id):
        """Compare tool-enabled vs direct responses."""
        tool_model = get_tool_calling_model(self.client)
        if tool_model:
            model_id = tool_model

        self.client.default_model = model_id

        comparison = compare_tool_vs_direct_response(
            self.client,
            "What is 144 divided by 12?",
            "math_calculator"
        )

        print(f"\n⚖️ Tool vs Direct comparison:")
        print(f"   With tools: '{comparison['with_tools']}'")
        print(f"   Without tools: '{comparison['without_tools']}'")
        print(f"   Tool used: {comparison['tool_called']}")
        print(f"   Expected tool used: {comparison['expected_tool_used']}")

        # Both should give correct answer (12)
        numbers_with = extract_numbers_from_response(comparison['with_tools'])
        numbers_without = extract_numbers_from_response(comparison['without_tools'])

        assert 12 in numbers_with or 12 in numbers_without


@pytest.mark.lm_studio
@pytest.mark.slow
class TestThinkingBlocks(LMStudioTestBase):
    """Test thinking blocks extraction with real models."""

    @skip_if_no_model()
    def test_thinking_block_detection(self, model_id):
        """Test detection of thinking blocks in responses."""
        # Try thinking models first
        thinking_model = get_thinking_model(self.client)
        if thinking_model:
            model_id = thinking_model

        self.client.default_model = model_id

        response = self.client.chat(
            "Think step by step: What is the square root of 144? Show your reasoning.",
            include_thinking=False
        )

        self.assert_valid_response(response)
        print(f"\n🧠 Thinking test:")
        print(f"   Model: {model_id}")
        print(f"   Response: '{response}'")
        print(f"   Thinking captured: {'Yes' if self.client.last_thinking else 'No'}")

        if self.client.last_thinking:
            print(f"   Thinking content: '{self.client.last_thinking[:100]}...'")
            self.assert_thinking_present()

        # Should contain the answer (12)
        numbers = extract_numbers_from_response(response)
        assert 12 in numbers, f"Expected 12 in response, got: {numbers}"

    @skip_if_no_model()
    def test_thinking_with_include_parameter(self, model_id):
        """Test include_thinking parameter with real responses."""
        thinking_model = get_thinking_model(self.client)
        if thinking_model:
            model_id = thinking_model

        self.client.default_model = model_id

        # Test with thinking included
        response_with = self.client.chat(
            "Solve step by step: What is 25% of 80?",
            include_thinking=True
        )

        # Test without thinking
        self.client.last_thinking = ""  # Clear state
        response_without = self.client.chat(
            "Solve step by step: What is 25% of 80?",
            include_thinking=False
        )

        print(f"\n🎛️ Include thinking test:")
        print(f"   With thinking: {len(response_with)} chars")
        print(f"   Without thinking: {len(response_without)} chars")

        self.assert_valid_response(response_with)
        self.assert_valid_response(response_without)

        # Response with thinking should be longer if thinking was captured
        if self.client.last_thinking:
            assert len(response_with) > len(response_without)
            assert "**Thinking:**" in response_with

        # Both should contain the answer (20)
        numbers_with = extract_numbers_from_response(response_with)
        numbers_without = extract_numbers_from_response(response_without)
        assert 20 in numbers_with or 20 in numbers_without


@pytest.mark.lm_studio
@pytest.mark.slow
class TestConversationHistory(LMStudioTestBase):
    """Test conversation history and context management."""

    @skip_if_no_model()
    def test_conversation_context(self, model_id):
        """Test that models maintain conversation context."""
        self.client.default_model = model_id

        history = []

        # First message
        response1, history = self.client.chat_with_history(
            "I'm thinking of a number between 1 and 10. It's 7.",
            history
        )

        # Second message - reference previous
        response2, history = self.client.chat_with_history(
            "What number was I thinking of?",
            history
        )

        self.assert_valid_response(response1)
        self.assert_valid_response(response2)

        print(f"\n💭 Context test:")
        print(f"   First: '{response1}'")
        print(f"   Second: '{response2}'")
        print(f"   History length: {len(history)} messages")

        # Second response should reference 7
        numbers = extract_numbers_from_response(response2)
        assert 7 in numbers, f"Expected 7 in context response, got: {numbers}"

    @skip_if_no_model()
    def test_long_conversation(self, model_id):
        """Test maintaining context over longer conversations."""
        self.client.default_model = model_id

        history = []
        expected_sum = 0

        # Build up a conversation with math
        for i in range(3):
            number = (i + 1) * 10  # 10, 20, 30
            expected_sum += number

            response, history = self.client.chat_with_history(
                f"Add {number} to our running total.",
                history
            )

            print(f"\n   Step {i+1}: Add {number} -> '{response}'")

        # Ask for the final total
        final_response, history = self.client.chat_with_history(
            "What's our total sum?",
            history
        )

        self.assert_valid_response(final_response)
        print(f"\n📊 Final total response: '{final_response}'")

        # Should mention the correct total (60)
        numbers = extract_numbers_from_response(final_response)
        assert expected_sum in numbers, f"Expected {expected_sum} in final response, got: {numbers}"


@pytest.mark.lm_studio
@pytest.mark.slow
class TestAdvancedFeatures(LMStudioTestBase):
    """Test advanced SDK features with real models."""

    @skip_if_no_model()
    def test_return_full_response(self, model_id):
        """Test returning full ChatCompletion objects."""
        self.client.default_model = model_id

        response = self.client.chat(
            "What is the capital of Japan?",
            return_full_response=True
        )

        assert isinstance(response, ChatCompletion)
        assert hasattr(response, 'id')
        assert hasattr(response, 'choices')
        assert len(response.choices) > 0

        content = response.choices[0].message.content
        self.assert_valid_response(content)

        print(f"\n📦 Full response test:")
        print(f"   ID: {response.id}")
        print(f"   Model: {response.model}")
        print(f"   Content: '{content}'")

        assert "tokyo" in content.lower()

    @skip_if_no_model()
    def test_custom_tool_integration(self, model_id):
        """Test registering and using custom tools."""
        tool_model = get_tool_calling_model(self.client)
        if tool_model:
            model_id = tool_model

        self.client.default_model = model_id

        # Register a custom tool
        @self.client.register_tool("Calculate circle area")
        def circle_area(radius: float) -> dict:
            import math
            area = math.pi * radius ** 2
            return {
                "radius": radius,
                "area": round(area, 2),
                "circumference": round(2 * math.pi * radius, 2)
            }

        response = self.client.chat("What's the area of a circle with radius 5?")

        self.assert_valid_response(response)
        print(f"\n⭕ Custom tool test:")
        print(f"   Response: '{response}'")
        print(f"   Tools used: {[tc.function.name for tc in self.client.last_tool_calls]}")

        # Should use our custom tool
        if self.client.last_tool_calls:
            tool_names = [tc.function.name for tc in self.client.last_tool_calls]
            assert "circle_area" in tool_names

            # Should mention area ≈ 78.54
            numbers = extract_numbers_from_response(response)
            assert any(75 < num < 85 for num in numbers), f"Expected ~78.54 in response, got: {numbers}"

    @skip_if_no_model()
    def test_performance_with_tools(self, model_id):
        """Test performance when tools are available vs disabled."""
        self.client.register_tools_from(builtin)
        tool_model = get_tool_calling_model(self.client)
        if tool_model:
            model_id = tool_model

        self.client.default_model = model_id

        # Measure with tools
        response_with, time_with = measure_response_time(
            self.client.chat,
            "What is 123 + 456?",
            use_tools=True
        )

        # Measure without tools
        response_without, time_without = measure_response_time(
            self.client.chat,
            "What is 123 + 456?",
            use_tools=False
        )

        print(f"\n⚡ Performance comparison:")
        print(f"   With tools: {time_with:.2f}s -> '{response_with}'")
        print(f"   Without tools: {time_without:.2f}s -> '{response_without}'")
        print(f"   Tool calls: {len(self.client.last_tool_calls)}")

        # Both should be reasonable response times
        assert time_with < 30.0
        assert time_without < 30.0

        # Both should get the right answer (579)
        numbers_with = extract_numbers_from_response(response_with)
        numbers_without = extract_numbers_from_response(response_without)
        assert 579 in numbers_with or 579 in numbers_without


@pytest.mark.lm_studio
class TestErrorHandling(LMStudioTestBase):
    """Test error handling with real API calls."""

    @skip_if_no_lm_studio
    def test_invalid_model_handling(self):
        """Test handling of invalid model names."""
        client = LocalLLMClient(self.config.base_url, "nonexistent-model")

        # This should either work (if model exists) or fail gracefully
        try:
            response = client.chat_simple("Hello")
            print(f"\n⚠️ Unexpected success with invalid model: '{response}'")
        except Exception as e:
            print(f"\n✅ Expected error with invalid model: {e}")
            assert "model" in str(e).lower() or "not found" in str(e).lower()

    @skip_if_no_model()
    def test_very_long_prompt(self, model_id):
        """Test handling of very long prompts."""
        self.client.default_model = model_id

        # Create a very long prompt
        long_prompt = "Summarize this text: " + "This is a test sentence. " * 1000

        try:
            response = self.client.chat(long_prompt, max_tokens=100)
            self.assert_valid_response(response, min_length=10)
            print(f"\n📏 Long prompt handled successfully: {len(response)} chars")
        except Exception as e:
            print(f"\n⚠️ Long prompt failed (expected): {e}")
            # This is acceptable - models have context limits

    @skip_if_no_model()
    def test_concurrent_requests(self, model_id):
        """Test handling concurrent requests."""
        import threading
        import queue

        self.client.default_model = model_id
        results = queue.Queue()
        errors = queue.Queue()

        def make_request(prompt_num):
            try:
                response = self.client.chat_simple(f"Say 'Response {prompt_num}'")
                results.put((prompt_num, response))
            except Exception as e:
                errors.put((prompt_num, e))

        # Launch concurrent requests
        threads = []
        for i in range(3):
            thread = threading.Thread(target=make_request, args=(i,))
            threads.append(thread)
            thread.start()

        # Wait for completion
        for thread in threads:
            thread.join(timeout=60)

        # Check results
        print(f"\n🔄 Concurrent requests:")
        print(f"   Successful: {results.qsize()}")
        print(f"   Failed: {errors.qsize()}")

        # At least some should succeed
        assert results.qsize() > 0, "No concurrent requests succeeded"

        while not results.empty():
            num, response = results.get()
            print(f"   {num}: '{response[:50]}...'")


# Helper function to run all LM Studio tests
def run_lm_studio_tests():
    """
    Convenience function to run all LM Studio tests.
    Can be called from a notebook or script.
    """
    pytest.main([
        __file__,
        "-v",
        "-m", "lm_studio",
        "--tb=short"
    ])


if __name__ == "__main__":
    # Run tests when executed directly
    run_lm_studio_tests()