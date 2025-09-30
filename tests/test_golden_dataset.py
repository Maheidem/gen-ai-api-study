"""
Golden dataset regression tests for agent framework.

These tests run known-good task examples to detect behavioral regressions.
Success rates are tracked over time to monitor LLM behavior stability.

IMPORTANT: These tests require a running LM Studio instance on http://169.254.83.107:1234

Run with: pytest tests/test_golden_dataset.py -m golden -v

Success rate thresholds:
- single_tool_tasks: ≥90%
- multi_tool_tasks: ≥80%
- complex_tasks: ≥70%
- edge_cases: ≥80%
"""

import json
import pytest
from pathlib import Path
from typing import Dict, Any, List
from local_llm_sdk import create_client_with_tools
from local_llm_sdk.agents import ReACT, AgentResult, AgentStatus


def load_golden_dataset() -> List[Dict[str, Any]]:
    """
    Load golden dataset from JSON file.

    Returns:
        List of task dictionaries with metadata
    """
    dataset_path = Path(__file__).parent / "golden_dataset.json"
    with open(dataset_path, 'r') as f:
        data = json.load(f)

    # Flatten all task categories into a single list
    all_tasks = []

    for category_name, tasks in data.get("react_agent", {}).items():
        if isinstance(tasks, list):
            for task in tasks:
                task_with_meta = task.copy()
                task_with_meta["category"] = category_name
                all_tasks.append(task_with_meta)

    return all_tasks


def validate_task_result(result: AgentResult, expected_properties: Dict[str, Any]) -> List[str]:
    """
    Validate an agent result against expected properties.

    Args:
        result: The agent execution result
        expected_properties: Dictionary of expected properties

    Returns:
        List of validation error messages (empty if all passed)
    """
    errors = []

    # Check success
    if not result.success:
        errors.append(f"Task failed with status {result.status}: {result.error}")
        return errors  # Early return on failure

    # Check iteration bounds
    if "min_iterations" in expected_properties:
        min_iter = expected_properties["min_iterations"]
        if result.iterations < min_iter:
            errors.append(
                f"Iterations {result.iterations} below minimum {min_iter}"
            )

    if "max_iterations" in expected_properties:
        max_iter = expected_properties["max_iterations"]
        if result.iterations > max_iter:
            errors.append(
                f"Iterations {result.iterations} exceeds maximum {max_iter}"
            )

    # Check tool call count
    if "min_tool_calls" in expected_properties:
        min_calls = expected_properties["min_tool_calls"]
        actual_calls = result.metadata.get("total_tool_calls", 0)
        if actual_calls < min_calls:
            errors.append(
                f"Tool calls {actual_calls} below minimum {min_calls}"
            )

    # Check tools used
    if "tools_used" in expected_properties:
        expected_tools = set(expected_properties["tools_used"])
        actual_tools = set()

        for msg in result.conversation:
            if hasattr(msg, 'tool_calls') and msg.tool_calls:
                for tc in msg.tool_calls:
                    actual_tools.add(tc.function.name)

        missing_tools = expected_tools - actual_tools
        if missing_tools:
            errors.append(
                f"Expected tools not used: {missing_tools}"
            )

    # Check final response content
    if "final_response_contains" in expected_properties:
        for expected_text in expected_properties["final_response_contains"]:
            if expected_text and expected_text not in result.final_response:
                errors.append(
                    f"Final response missing expected text: '{expected_text}'"
                )

    # Check final response exclusions
    if "final_response_excludes" in expected_properties:
        for excluded_text in expected_properties["final_response_excludes"]:
            if excluded_text and excluded_text.lower() in result.final_response.lower():
                errors.append(
                    f"Final response contains excluded text: '{excluded_text}'"
                )

    return errors


@pytest.fixture
def client():
    """Create a LocalLLMClient with tools for testing."""
    from local_llm_sdk import LocalLLMClient
    client = LocalLLMClient(
        base_url="http://169.254.83.107:1234/v1",
        timeout=300
    )
    # Register built-in tools
    client.register_tools_from(None)
    return client


@pytest.fixture
def agent(client):
    """Create a ReACT agent for testing."""
    return ReACT(client)


@pytest.mark.live_llm
@pytest.mark.golden
@pytest.mark.parametrize("task_data", load_golden_dataset(), ids=lambda t: t.get("id", "unknown"))
def test_golden_task(agent, task_data):
    """
    Execute a golden dataset task and validate against expected properties.

    This is a property-based regression test. We don't check exact outputs,
    but verify that behavioral properties still hold.
    """
    task = task_data["task"]
    expected_properties = task_data.get("expected_properties", {})
    category = task_data.get("category", "unknown")
    difficulty = task_data.get("difficulty", "medium")

    # Adjust max_iterations based on difficulty
    max_iterations_map = {
        "easy": 10,
        "medium": 15,
        "hard": 20
    }
    max_iterations = max_iterations_map.get(difficulty, 15)

    # Execute task
    result = agent.run(task, max_iterations=max_iterations, verbose=False)

    # Validate result
    errors = validate_task_result(result, expected_properties)

    # Build detailed failure message
    if errors:
        failure_msg = f"\n{'='*80}\n"
        failure_msg += f"Golden Task Failed: {task_data.get('id', 'unknown')}\n"
        failure_msg += f"Category: {category}\n"
        failure_msg += f"Difficulty: {difficulty}\n"
        failure_msg += f"Task: {task}\n"
        failure_msg += f"\nValidation Errors:\n"
        for error in errors:
            failure_msg += f"  - {error}\n"
        failure_msg += f"\nResult Details:\n"
        failure_msg += f"  Status: {result.status}\n"
        failure_msg += f"  Iterations: {result.iterations}\n"
        failure_msg += f"  Tool Calls: {result.metadata.get('total_tool_calls', 0)}\n"
        failure_msg += f"  Final Response: {result.final_response[:200]}...\n"
        failure_msg += f"{'='*80}\n"

        pytest.fail(failure_msg)


@pytest.mark.live_llm
@pytest.mark.golden
@pytest.mark.slow
class TestGoldenDatasetSuccessRate:
    """
    Test success rates for each category of golden tasks.

    These tests run multiple tasks and calculate success rates to
    detect behavioral regressions at the category level.
    """

    @pytest.fixture
    def agent(self, client):
        """Create a ReACT agent for testing."""
        return ReACT(client)

    def _run_category_tasks(self, agent, category_name: str) -> tuple[int, int]:
        """
        Run all tasks in a category and return (successes, total).

        Args:
            agent: ReACT agent instance
            category_name: Name of category to test

        Returns:
            Tuple of (successes, total_tasks)
        """
        dataset = load_golden_dataset()
        category_tasks = [t for t in dataset if t.get("category") == category_name]

        successes = 0
        total = len(category_tasks)

        for task_data in category_tasks:
            task = task_data["task"]
            expected_properties = task_data.get("expected_properties", {})

            # Adjust max_iterations based on difficulty
            difficulty = task_data.get("difficulty", "medium")
            max_iterations_map = {"easy": 10, "medium": 15, "hard": 20}
            max_iterations = max_iterations_map.get(difficulty, 15)

            # Execute
            result = agent.run(task, max_iterations=max_iterations, verbose=False)

            # Validate
            errors = validate_task_result(result, expected_properties)

            if not errors:
                successes += 1

        return successes, total

    def test_single_tool_tasks_success_rate(self, agent):
        """
        Verify success rate ≥90% for single-tool tasks.

        Single-tool tasks should be highly reliable.
        """
        successes, total = self._run_category_tasks(agent, "single_tool_tasks")
        success_rate = successes / total if total > 0 else 0

        assert success_rate >= 0.9, \
            f"Single-tool tasks success rate {success_rate:.1%} below 90% threshold " \
            f"({successes}/{total} succeeded)"

    def test_multi_tool_tasks_success_rate(self, agent):
        """
        Verify success rate ≥80% for multi-tool tasks.

        Multi-tool tasks are more complex, slightly lower threshold acceptable.
        """
        successes, total = self._run_category_tasks(agent, "multi_tool_tasks")
        success_rate = successes / total if total > 0 else 0

        assert success_rate >= 0.8, \
            f"Multi-tool tasks success rate {success_rate:.1%} below 80% threshold " \
            f"({successes}/{total} succeeded)"

    def test_complex_tasks_success_rate(self, agent):
        """
        Verify success rate ≥70% for complex tasks.

        Complex tasks may have multiple valid approaches, lower threshold acceptable.
        """
        successes, total = self._run_category_tasks(agent, "complex_tasks")

        # Skip if no complex tasks
        if total == 0:
            pytest.skip("No complex tasks in golden dataset")

        success_rate = successes / total
        assert success_rate >= 0.7, \
            f"Complex tasks success rate {success_rate:.1%} below 70% threshold " \
            f"({successes}/{total} succeeded)"

    def test_edge_cases_success_rate(self, agent):
        """
        Verify success rate ≥80% for edge case tasks.

        Edge cases test boundary conditions and should be fairly reliable.
        """
        successes, total = self._run_category_tasks(agent, "edge_cases")

        # Skip if no edge case tasks
        if total == 0:
            pytest.skip("No edge case tasks in golden dataset")

        success_rate = successes / total
        assert success_rate >= 0.8, \
            f"Edge case tasks success rate {success_rate:.1%} below 80% threshold " \
            f"({successes}/{total} succeeded)"
