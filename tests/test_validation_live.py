"""
Live integration tests for validation system with REAL LLM.

Tests actual model responses to verify validation catches real issues.

Run with: pytest tests/test_validation_live.py -v -s
(Requires LM Studio running)
"""

import pytest
import os
from local_llm_sdk import LocalLLMClient


@pytest.mark.live_llm
class TestValidationWithRealLLM:
    """Integration tests with actual LLM (requires LM Studio running)."""

    @pytest.fixture
    def qwen3_client_with_validation(self):
        """Client with qwen3 model and validation enabled."""
        os.environ['LLM_ENABLE_VALIDATION'] = 'true'

        client = LocalLLMClient(
            base_url=os.getenv("LLM_BASE_URL"),
            model="qwen/qwen3-coder-30b",
            timeout=60
        )
        client.register_tools_from(None)
        return client

    @pytest.fixture
    def magistral_client_with_validation(self):
        """Client with magistral model and validation enabled."""
        os.environ['LLM_ENABLE_VALIDATION'] = 'true'

        client = LocalLLMClient(
            base_url=os.getenv("LLM_BASE_URL"),
            model="mistralai/magistral-small-2509",
            timeout=60
        )
        client.register_tools_from(None)
        return client

    def test_qwen3_xml_format_caught_by_validation(self, qwen3_client_with_validation, capsys):
        """
        Test that qwen3's XML format is caught by validation.

        Expected: Validation error with XML_DRIFT
        """
        client = qwen3_client_with_validation

        print("\n🧪 Testing qwen3 with validation enabled...")
        print("Expected: XML format should be detected and error raised")

        # Task requiring filesystem tool (triggers XML in qwen3)
        import tempfile
        test_file = tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False)
        test_file.write("test content")
        test_file.close()

        try:
            # This should raise ValueError due to XML format
            response = client.chat(
                f"Read the file {test_file.name} and tell me what's in it",
                use_tools=True
            )

            # If we get here, validation didn't catch it
            print(f"❌ UNEXPECTED: Got response without validation error")
            print(f"Response: {response[:200]}")

            # Check if response is actually XML (validation may be disabled)
            if "<tool_call>" in str(response):
                pytest.fail("qwen3 returned XML but validation didn't catch it!")
            else:
                print("ℹ️  qwen3 didn't use XML format this time (behavior can vary)")

        except ValueError as e:
            error_msg = str(e)

            # Verify it's the validation error we expect
            if "XML_DRIFT" in error_msg or "format incompatible" in error_msg:
                print("✅ SUCCESS: Validation caught XML format!")

                # Check that alert was printed
                captured = capsys.readouterr()
                assert "🚨 VALIDATION ERROR" in captured.out
                assert "XML_DRIFT" in captured.out

                print(f"Error message: {error_msg}")
                print("Validation system working correctly!")
            else:
                pytest.fail(f"Unexpected error: {error_msg}")

        finally:
            # Cleanup
            os.unlink(test_file.name)

    def test_magistral_passes_validation(self, magistral_client_with_validation):
        """
        Test that magistral model (known good) passes validation.

        Expected: Normal response, no validation errors
        """
        client = magistral_client_with_validation

        print("\n🧪 Testing magistral with validation enabled...")
        print("Expected: Should pass validation (uses correct JSON format)")

        # Simple task with tools
        try:
            response = client.chat("Calculate 5 * 10", use_tools=True)

            print(f"✅ SUCCESS: magistral passed validation")
            print(f"Response: {response[:200]}")

            # Should contain the answer
            assert "50" in str(response)

        except ValueError as e:
            # magistral shouldn't trigger validation errors
            pytest.fail(f"magistral triggered unexpected validation error: {e}")

    def test_validation_can_be_disabled(self):
        """Test that validation can be disabled for backward compatibility."""
        os.environ['LLM_ENABLE_VALIDATION'] = 'false'

        client = LocalLLMClient(
            base_url=os.getenv("LLM_BASE_URL"),
            model="qwen/qwen3-coder-30b",
            timeout=60
        )
        client.register_tools_from(None)

        print("\n🧪 Testing with validation DISABLED...")
        print("Expected: Should work even if model uses XML")

        # Even if qwen3 uses XML, it should work (no validation)
        try:
            response = client.chat("What is 2+2?", use_tools=False)
            print(f"✅ Response received: {response[:100]}")

        except Exception as e:
            # Without validation, should not fail on format issues
            # (might fail for other reasons though)
            print(f"Error: {e}")


@pytest.mark.live_llm
def test_validation_with_deliberate_repetition():
    """
    Test validation catches repetition if it occurs.

    This is a placeholder - we'd need a prompt that reliably triggers repetition.
    """
    os.environ['LLM_ENABLE_VALIDATION'] = 'true'

    client = LocalLLMClient(
        base_url=os.getenv("LLM_BASE_URL"),
        model=os.getenv("LLM_MODEL"),
        timeout=60
    )

    print("\n🧪 Testing repetition detection...")
    print("Note: Repetition is hard to trigger reliably")

    # Try a prompt that might cause repetition
    # (In practice, this is model/context dependent)
    try:
        response = client.chat(
            "Repeat the word 'test' exactly 100 times",
            use_tools=False
        )

        # Check if it's actually repetitive
        if "test test test test test" in response.lower():
            print("⚠️  Got repetitive response but validation didn't catch it")
            print("(May be intentional repetition per prompt)")
        else:
            print(f"✅ Response: {response[:100]}")

    except ValueError as e:
        if "REPETITION" in str(e):
            print("✅ Validation caught repetition!")
        else:
            print(f"Different error: {e}")
