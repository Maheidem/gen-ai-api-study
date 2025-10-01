"""
Behavioral tests for agent framework with real LLM calls.

These tests validate that the LLM:
1. Uses tools correctly (not all at once)
2. Produces clean final responses (no markers)
3. Follows multi-step reasoning patterns
4. Handles errors gracefully

IMPORTANT: These tests require a running LM Studio instance on http://169.254.83.107:1234

Run with: pytest tests/test_agents_behavioral.py -m live_llm -v

To run without these tests (default):
    pytest tests/  # Automatically skips live_llm tests

To run only behavioral tests:
    pytest tests/ -m "live_llm and behavioral" -v
"""

import pytest
from typing import List
from local_llm_sdk import create_client_with_tools
from local_llm_sdk.agents import ReACT, AgentResult, AgentStatus
from local_llm_sdk.models import ChatMessage


@pytest.mark.live_llm
@pytest.mark.behavioral
class TestReACTBehavior:
    """
    Behavioral tests for ReACT agent implementation.

    These tests use property-based assertions rather than exact matching:
    - ✅ assert iterations > 1 (robust)
    - ❌ assert iterations == 3 (brittle)

    This acknowledges LLM non-determinism while maintaining quality bar.
    """

    @pytest.fixture
    def client(self, live_llm_client):
        """Use centralized live LLM client from conftest (configured via .env)."""
        return live_llm_client

    @pytest.fixture
    def agent(self, live_react_agent):
        """Use centralized live ReACT agent from conftest (configured via .env)."""
        return live_react_agent

    @pytest.mark.xfail(
        reason="Non-deterministic LLM behavior - use test_multi_step_statistical instead",
        strict=False
    )
    def test_multi_step_iteration_pattern(self, agent):
        """
        Verify agent uses multiple iterations for multi-tool tasks.

        This test caught the bug where agent crammed all tools into iteration 1.

        Property: For tasks requiring 3+ tools, iterations should be > 1

        NOTE: Marked xfail due to non-deterministic behavior. Use the statistical
        version (test_multi_step_statistical) for reliable validation.
        """
        # Task requiring multiple tools in sequence
        task = "Calculate 5 factorial, convert the result to uppercase text, then count characters"

        # Execute
        result = agent.run(task, max_iterations=15, verbose=False)

        # Property-based assertions
        assert result.success, f"Task failed: {result.error}"
        assert result.iterations > 1, \
            f"Expected multi-step execution, got {result.iterations} iteration(s). " \
            f"Agent should not cram all tools into one iteration."
        assert result.metadata["total_tool_calls"] >= 3, \
            f"Expected 3+ tool calls, got {result.metadata['total_tool_calls']}"

        # Verify tools used in sequence (not all at once)
        tool_iterations = []
        for i, msg in enumerate(result.conversation):
            if hasattr(msg, 'tool_calls') and msg.tool_calls:
                tool_iterations.append(i)

        assert len(set(tool_iterations)) > 1, \
            "All tools used in same message (cramming detected). " \
            f"Tool calls at message indices: {tool_iterations}"

    @pytest.mark.slow
    def test_multi_step_statistical(self, agent):
        """
        Statistical test: Verify multi-step behavior across multiple runs.

        This test acknowledges LLM non-determinism by running the task 5 times
        and requiring ≥80% success rate (both task completion AND tool distribution).

        CRITICAL FINDING: Temperature significantly affects tool usage:
        - temp=0.0: 0% success (model never uses tools - gets stuck!)
        - temp=0.7: 60-70% success (default, high variance)
        - temp=1.0: 100% success (high randomness unlocks tool use)

        This test uses temperature=1.0 for reliable tool usage behavior.
        The 80% threshold accounts for remaining LLM variance.

        This is the RECOMMENDED test for validating multi-step behavior.
        """
        task = "Calculate 5 factorial, convert the result to uppercase text, then count characters"

        successes = 0
        runs = 5

        for run_num in range(runs):
            # Use temperature=1.0 for better tool-use consistency
            # Empirical testing shows temp=0 causes model to never use tools (0% success)
            # while temp=1.0 achieves 100% success rate on this task
            result = agent.run(task, max_iterations=15, verbose=False, temperature=1.0)

            # Check if task succeeded
            if not result.success:
                continue

            # Check if tools were distributed (not crammed)
            tool_iterations = []
            for i, msg in enumerate(result.conversation):
                if hasattr(msg, 'tool_calls') and msg.tool_calls:
                    tool_iterations.append(i)

            # Both conditions must be met
            if result.success and len(set(tool_iterations)) > 1:
                successes += 1

        success_rate = successes / runs
        assert success_rate >= 0.8, \
            f"Success rate {success_rate:.0%} below 80% threshold ({successes}/{runs} runs passed). " \
            f"With temperature=1.0, this task typically achieves 80-100% success. " \
            f"Below 80% indicates behavioral regression or model issues."

    def test_final_response_quality(self, agent):
        """
        Verify final response contains answer, not completion markers.

        This test caught the bug where "TASK_COMPLETE" appeared in final_response.

        Property: final_response should not contain internal markers
        """
        task = "What is 42 + 58?"

        # Execute
        result = agent.run(task, max_iterations=10, verbose=False)

        # Property-based assertions
        assert result.success, f"Task failed: {result.error}"

        # Final response should not contain completion markers
        assert "TASK_COMPLETE" not in result.final_response, \
            "Final response contains TASK_COMPLETE marker (should be stripped)"
        assert "task_complete" not in result.final_response.lower(), \
            "Final response contains task_complete marker in any case (should be stripped)"

        # Final response should contain the actual answer
        assert "100" in result.final_response, \
            f"Final response missing expected answer. Got: {result.final_response}"

    def test_tool_usage_sequence(self, agent):
        """
        Verify tools are used in logical sequence for multi-step tasks.

        Property: Tools should be called in separate iterations, showing step-by-step reasoning
        """
        task = "Create a file at /tmp/test_behavioral.txt with content 'Hello, World!', then read it back"

        # Execute
        result = agent.run(task, max_iterations=15, verbose=False)

        # Property-based assertions
        assert result.success, f"Task failed: {result.error}"
        assert result.iterations >= 2, \
            f"Expected at least 2 iterations for write+read, got {result.iterations}"

        # Verify filesystem_operation tool was used at least twice (write + read)
        filesystem_calls = 0
        for msg in result.conversation:
            if hasattr(msg, 'tool_calls') and msg.tool_calls:
                for tc in msg.tool_calls:
                    if tc.function.name == "filesystem_operation":
                        filesystem_calls += 1

        assert filesystem_calls >= 2, \
            f"Expected at least 2 filesystem operations (write+read), got {filesystem_calls}"

        # Final response should mention the content
        assert "Hello, World!" in result.final_response or "Hello" in result.final_response, \
            f"Final response should mention the file content. Got: {result.final_response}"

    def test_single_tool_task_efficiency(self, agent):
        """
        Verify simple single-tool tasks complete efficiently.

        Property: Tasks requiring only 1 tool should complete in 1-2 iterations
        """
        task = "What is 12 multiplied by 8?"

        # Execute
        result = agent.run(task, max_iterations=10, verbose=False)

        # Property-based assertions
        assert result.success, f"Task failed: {result.error}"
        assert result.iterations <= 3, \
            f"Simple task took {result.iterations} iterations (expected ≤3 for efficiency)"
        assert result.metadata["total_tool_calls"] >= 1, \
            f"Expected at least 1 tool call, got {result.metadata['total_tool_calls']}"

        # Final response should contain the answer
        assert "96" in result.final_response, \
            f"Final response missing expected answer. Got: {result.final_response}"

    def test_error_handling_behavior(self, agent):
        """
        Verify agent handles tool errors gracefully.

        Property: Agent should catch tool errors and report them clearly
        """
        # Task that will cause an error (reading non-existent file)
        task = "Read the contents of /tmp/this_file_definitely_does_not_exist_12345.txt"

        # Execute
        result = agent.run(task, max_iterations=5, verbose=False)

        # The agent might succeed (by acknowledging the file doesn't exist)
        # or fail (if it can't handle the error). Either is acceptable,
        # but we check that it doesn't crash
        assert result.status in [AgentStatus.SUCCESS, AgentStatus.ERROR, AgentStatus.MAX_ITERATIONS], \
            f"Unexpected status: {result.status}"

        # If it succeeded, the response should acknowledge the issue
        if result.success:
            response_lower = result.final_response.lower()
            assert any(word in response_lower for word in ["not", "exist", "found", "error"]), \
                f"Expected acknowledgment of missing file. Got: {result.final_response}"

    @pytest.mark.slow
    def test_consistency_across_runs(self, agent):
        """
        Verify agent behavior is consistent across multiple runs.

        Property: Success rate should be ≥80% for the same task

        This is a statistical test acknowledging LLM non-determinism.
        """
        task = "Calculate the square root of 144"
        num_runs = 5
        successes = 0

        for i in range(num_runs):
            result = agent.run(task, max_iterations=10, verbose=False)
            if result.success and "12" in result.final_response:
                successes += 1

        success_rate = successes / num_runs
        assert success_rate >= 0.8, \
            f"Success rate {success_rate:.1%} below 80% threshold ({successes}/{num_runs} succeeded)"

    def test_python_execution_result_capture(self, agent):
        """
        Verify execute_python tool captures results correctly.

        Property: Variable results should be captured even without print statements
        """
        task = "Use Python to calculate the factorial of 10"

        # Execute with temperature=1.0 for reliable tool usage
        result = agent.run(task, max_iterations=10, verbose=False, temperature=1.0)

        # Property-based assertions
        assert result.success, f"Task failed: {result.error}"

        # Final response should contain the factorial result
        assert "3628800" in result.final_response, \
            f"Final response missing factorial result. Got: {result.final_response}"

        # Verify execute_python was actually called
        python_calls = 0
        for msg in result.conversation:
            if hasattr(msg, 'tool_calls') and msg.tool_calls:
                for tc in msg.tool_calls:
                    if tc.function.name == "execute_python":
                        python_calls += 1

        assert python_calls >= 1, \
            f"Expected at least 1 execute_python call, got {python_calls}"


@pytest.mark.live_llm
@pytest.mark.behavioral
class TestReACTEdgeCases:
    """
    Edge case behavioral tests for ReACT agent.

    These test boundary conditions and unusual inputs.
    """

    @pytest.fixture
    def client(self, live_llm_client):
        """Use centralized live LLM client from conftest (configured via .env)."""
        return live_llm_client

    @pytest.fixture
    def agent(self, live_react_agent):
        """Use centralized live ReACT agent from conftest (configured via .env)."""
        return live_react_agent

    def test_max_iterations_graceful_handling(self, agent):
        """
        Verify agent handles max_iterations limit gracefully.

        Property: Agent should return MAX_ITERATIONS status, not crash
        """
        # Task with artificially low iteration limit
        task = "Calculate 5 factorial, convert to uppercase, count characters, reverse it, count vowels"

        # Execute with low limit
        result = agent.run(task, max_iterations=2, verbose=False)

        # Should hit max iterations gracefully
        assert result.status == AgentStatus.MAX_ITERATIONS, \
            f"Expected MAX_ITERATIONS status, got {result.status}"
        assert result.iterations == 2, \
            f"Expected exactly 2 iterations, got {result.iterations}"
        assert not result.success, \
            "Task should not be marked as successful when max iterations hit"
        assert result.metadata.get("max_iterations_reached") is True, \
            "Metadata should indicate max_iterations_reached"

    def test_empty_task_handling(self, agent):
        """
        Verify agent handles edge case of empty or trivial task.

        Property: Agent should handle gracefully, not crash
        """
        task = "Say hello"

        # Execute
        result = agent.run(task, max_iterations=5, verbose=False)

        # Should complete successfully
        assert result.status in [AgentStatus.SUCCESS, AgentStatus.MAX_ITERATIONS], \
            f"Agent should handle simple task gracefully, got status: {result.status}"

        if result.success:
            assert len(result.final_response) > 0, \
                "Final response should not be empty"
