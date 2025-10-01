# Testing Guide

Complete guide to writing and running tests for the Local LLM SDK.

## Table of Contents
1. [Testing Philosophy](#testing-philosophy)
2. [Running Tests](#running-tests)
3. [Writing Unit Tests](#writing-unit-tests)
4. [Writing Behavioral Tests](#writing-behavioral-tests)
5. [Golden Dataset](#golden-dataset)
6. [Test Organization](#test-organization)
7. [Mocking Patterns](#mocking-patterns)
8. [Coverage Requirements](#coverage-requirements)
9. [CI/CD Integration](#cicd-integration)
10. [Troubleshooting](#troubleshooting)

---

## Testing Philosophy

### Tiered Testing Architecture

The SDK uses a three-tier testing approach:

```
┌─────────────────────────────────────────────────────────────┐
│ Tier 1: Unit Tests (~213 tests) - Fast, Mocked             │
│ - Validates code logic                                      │
│ - Run on every commit (<10s)                                │
│ - Command: pytest tests/ -v                                 │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│ Tier 2: Behavioral Tests (~20 tests) - Real LLM            │
│ - Validates LLM behavior patterns                           │
│ - Property-based assertions                                 │
│ - Run nightly or on demand                                  │
│ - Command: pytest tests/ -m "live_llm and behavioral" -v    │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│ Tier 3: Golden Dataset (~16 tests) - Regression Detection  │
│ - Known-good task examples                                  │
│ - Success rate tracking (≥90% threshold)                    │
│ - Run weekly or before releases                             │
│ - Command: pytest tests/ -m "live_llm and golden" -v        │
└─────────────────────────────────────────────────────────────┘
```

### Why Three Tiers?

**The Testing Gap We Discovered:**

Traditional unit tests with mocks validate **code correctness** but miss **LLM behavior issues**:

- ✅ Tests verified: Code handles "TASK_COMPLETE" correctly
- ❌ Tests missed: LLM shouldn't put "TASK_COMPLETE" in final response
- ✅ Tests verified: Agent supports multiple iterations
- ❌ Tests missed: LLM actually uses multiple iterations instead of cramming into one

**Solution**: Behavioral testing with real LLM interactions validates patterns mocks can't catch.

---

## Running Tests

### Unit Tests (Fast, Always Run)

```bash
# Run all unit tests (skips live_llm by default)
pytest tests/ -v

# Run specific test file
pytest tests/test_client.py -v

# Run specific test
pytest tests/test_client.py::test_client_initialization -v

# Run with coverage
pytest tests/ --cov=local_llm_sdk --cov-report=html

# Stop on first failure
pytest tests/ -x

# Run tests matching pattern
pytest tests/ -k "test_agent" -v
```

### Behavioral Tests (Requires LM Studio)

**Prerequisites**: LM Studio running on http://169.254.83.107:1234 with model loaded

```bash
# All behavioral tests
pytest tests/ -m "live_llm and behavioral" -v

# Golden dataset only
pytest tests/ -m "live_llm and golden" -v

# Specific behavioral test
pytest tests/test_agents_behavioral.py::TestReACTBehavior::test_multi_step_iteration_pattern -v

# Run single golden task
pytest tests/test_golden_dataset.py::test_golden_task[factorial_uppercase_count] -v

# Success rate tests (slow - runs each task 10 times)
pytest tests/test_golden_dataset.py::TestGoldenDatasetSuccessRate -v
```

### Test Selection

```bash
# Run tests WITHOUT golden dataset
pytest tests/ -m "live_llm and behavioral and not golden" -v

# Run only slow tests
pytest tests/ -m "slow" -v

# Run everything including live_llm (override default skip)
pytest tests/ --run-live-llm -v

# Verbose output with print statements
pytest tests/test_client.py -v -s
```

---

## Writing Unit Tests

### Basic Test Structure

```python
import pytest
from local_llm_sdk import LocalLLMClient
from local_llm_sdk.models import ChatMessage, ChatCompletion

def test_client_initialization():
    """Test LocalLLMClient initializes with correct defaults."""
    client = LocalLLMClient(
        base_url="http://localhost:1234/v1",
        model="test-model"
    )
    
    assert client.base_url == "http://localhost:1234/v1"
    assert client.model == "test-model"
    assert client.timeout == 300  # Default timeout
    assert len(client.conversation) == 0
```

### Using Fixtures

```python
# In conftest.py
import pytest
from unittest.mock import Mock

@pytest.fixture
def mock_client():
    """Create a LocalLLMClient instance for testing."""
    return LocalLLMClient("http://localhost:1234/v1", "test-model")

@pytest.fixture
def mock_response():
    """Create a mock ChatCompletion response."""
    return ChatCompletion(
        id="test-123",
        model="test-model",
        choices=[...],
        usage={"total_tokens": 100}
    )

# In test file
def test_with_fixtures(mock_client, mock_response):
    """Test using shared fixtures."""
    # Use mock_client and mock_response
    ...
```

### Mocking HTTP Requests

```python
from unittest.mock import patch, Mock
import pytest

def test_chat_completion_success(mock_client):
    """Test successful chat completion."""
    # Mock the HTTP POST request
    with patch('requests.post') as mock_post:
        mock_post.return_value = Mock(
            status_code=200,
            json=lambda: {
                "id": "chatcmpl-123",
                "model": "test-model",
                "choices": [{
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": "Hello, World!"
                    },
                    "finish_reason": "stop"
                }],
                "usage": {
                    "prompt_tokens": 10,
                    "completion_tokens": 5,
                    "total_tokens": 15
                }
            }
        )
        
        response = mock_client.chat("Hello")
        
        assert response == "Hello, World!"
        mock_post.assert_called_once()
```

### Testing Error Handling

```python
import requests
import pytest

def test_connection_error(mock_client):
    """Test handling of connection errors."""
    with patch('requests.post') as mock_post:
        mock_post.side_effect = requests.exceptions.ConnectionError()
        
        with pytest.raises(requests.exceptions.ConnectionError):
            mock_client.chat("Hello")

def test_timeout_error(mock_client):
    """Test handling of timeout errors."""
    with patch('requests.post') as mock_post:
        mock_post.side_effect = requests.exceptions.Timeout()
        
        with pytest.raises(requests.exceptions.Timeout):
            mock_client.chat("Hello", timeout=1)
```

### Testing Tools

```python
from local_llm_sdk.tools import tool, ToolRegistry

def test_tool_registration():
    """Test tool registration works correctly."""
    registry = ToolRegistry()
    
    @tool("Test tool description")
    def test_tool(x: int) -> dict:
        return {"result": x * 2}
    
    # Verify tool is registered
    assert "test_tool" in registry.get_tool_names()
    
    # Verify tool execution
    result = registry.execute("test_tool", {"x": 5})
    assert result == {"result": 10}

def test_tool_schema_generation():
    """Test automatic schema generation from function signature."""
    registry = ToolRegistry()
    
    @tool("Schema test")
    def schema_test(name: str, age: int, active: bool = True) -> dict:
        return {"name": name, "age": age, "active": active}
    
    definition = registry.get("schema_test")
    params = definition.function.parameters
    
    # Verify schema structure
    assert params["type"] == "object"
    assert "name" in params["properties"]
    assert "age" in params["properties"]
    assert params["properties"]["name"]["type"] == "string"
    assert params["properties"]["age"]["type"] == "integer"
    assert "active" in params["properties"]  # Optional param included
```

---

## Writing Behavioral Tests

### Key Principle: Test Properties, Not Exact Outputs

**DON'T** ❌ (Brittle - fails with LLM variance)
```python
assert result.iterations == 3
assert result.final_response == "The answer is 100"
```

**DO** ✅ (Robust - tests behavioral properties)
```python
assert result.iterations > 1, "Expected multi-step execution"
assert "TASK_COMPLETE" not in result.final_response
assert "100" in result.final_response
assert result.metadata["total_tool_calls"] >= 2
```

### Example Behavioral Test

```python
import pytest
from local_llm_sdk import create_client_with_tools
from local_llm_sdk.agents import ReACT

@pytest.mark.live_llm
@pytest.mark.behavioral
def test_multi_step_iteration_pattern():
    """
    Verify agent uses multiple iterations for multi-tool tasks.

    This test caught the bug where agent crammed all tools into iteration 1.

    Property: For tasks requiring 3+ tools, iterations should be > 1
    """
    client = create_client_with_tools()
    agent = ReACT(client)
    
    task = "Calculate 5 factorial, convert to uppercase, count characters"
    result = agent.run(task, max_iterations=15, verbose=False)

    # Property-based assertions
    assert result.success, f"Task failed: {result.error}"
    assert result.iterations > 1, \
        f"Expected multi-step execution, got {result.iterations} iteration(s)"
    assert result.metadata["total_tool_calls"] >= 3

    # Verify tools used in sequence (not all at once)
    tool_iterations = [i for i, msg in enumerate(result.conversation)
                       if hasattr(msg, 'tool_calls') and msg.tool_calls]
    assert len(set(tool_iterations)) > 1, \
        "All tools used in same iteration (cramming detected)"

    # Final response quality
    assert "TASK_COMPLETE" not in result.final_response
    assert "120" in result.final_response
```

### Behavioral Test Best Practices

**1. Use Property-Based Assertions**
```python
# ✅ Good: Tests invariants
assert result.iterations > 1
assert result.metadata["total_tool_calls"] >= expected_minimum
assert "expected_value" in result.final_response

# ❌ Bad: Tests exact values
assert result.iterations == 3
assert result.final_response == "Exact text"
```

**2. Include Descriptive Failure Messages**
```python
# ✅ Good: Provides context on failure
assert result.success, \
    f"Task failed with error: {result.error}\n" \
    f"Iterations: {result.iterations}\n" \
    f"Tool calls: {result.metadata['total_tool_calls']}"

# ❌ Bad: No context
assert result.success
```

**3. Test Patterns That Caught Real Bugs**
```python
@pytest.mark.live_llm
@pytest.mark.behavioral
def test_task_complete_not_in_response():
    """
    Verify TASK_COMPLETE is stripped from final response.
    
    Bug: Agent was including "TASK_COMPLETE" in user-facing response.
    """
    result = agent.run("Simple task")
    assert "TASK_COMPLETE" not in result.final_response
```

### Setting Up Behavioral Test Fixtures

```python
# In conftest.py
import pytest
from local_llm_sdk import create_client_with_tools
from local_llm_sdk.agents import ReACT

@pytest.fixture(scope="module")
def live_client():
    """Create client connected to real LM Studio."""
    return create_client_with_tools(
        base_url="http://169.254.83.107:1234/v1",
        model="mistralai/magistral-small-2509"
    )

@pytest.fixture
def react_agent(live_client):
    """Create ReACT agent with live client."""
    return ReACT(live_client)
```

---

## Golden Dataset

### What is the Golden Dataset?

A collection of 16 known-good tasks with expected properties, used for regression detection.

**Location**: `tests/golden_dataset.json`

### Structure

```json
{
  "single_tool_tasks": [
    {
      "id": "simple_calculation",
      "task": "Calculate 2 + 2 using the calculator tool",
      "expected_properties": {
        "min_iterations": 1,
        "max_iterations": 3,
        "min_tool_calls": 1,
        "tools_used": ["math_calculator"],
        "final_response_contains": ["4"],
        "final_response_excludes": ["TASK_COMPLETE"]
      },
      "difficulty": "easy",
      "tags": ["math", "single_tool"]
    }
  ],
  "multi_tool_tasks": [...],
  "complex_tasks": [...],
  "edge_cases": [...]
}
```

### Success Rate Thresholds

| Category | Threshold | Rationale |
|----------|-----------|-----------|
| single_tool_tasks | ≥90% | Simple tasks should be highly reliable |
| multi_tool_tasks | ≥80% | More complex, slight variance acceptable |
| complex_tasks | ≥70% | Multiple valid approaches possible |
| edge_cases | ≥80% | Boundary handling should be consistent |

### Running Golden Dataset Tests

```bash
# Run all golden tasks once
pytest tests/test_golden_dataset.py::test_golden_task -v

# Run specific golden task
pytest tests/test_golden_dataset.py::test_golden_task[simple_calculation] -v

# Run success rate tests (10 runs per task)
pytest tests/test_golden_dataset.py::TestGoldenDatasetSuccessRate -v
```

### Adding to Golden Dataset

When you discover a bug that unit tests missed:

1. **Create entry in golden_dataset.json**:
```json
{
  "id": "new_task",
  "task": "Task description that exposed the bug",
  "expected_properties": {
    "min_iterations": 2,
    "max_iterations": 10,
    "min_tool_calls": 3,
    "tools_used": ["tool1", "tool2"],
    "final_response_contains": ["expected_value"],
    "final_response_excludes": ["TASK_COMPLETE"]
  },
  "difficulty": "medium",
  "tags": ["relevant", "tags"]
}
```

2. **Run to verify it catches the issue**:
```bash
pytest tests/test_golden_dataset.py::test_golden_task[new_task] -v
```

---

## Test Organization

### File Structure

```
tests/
├── conftest.py                   # Shared fixtures
├── pytest.ini                    # Pytest configuration (markers, etc.)
├── golden_dataset.json           # Known-good tasks
│
├── test_client.py                # LocalLLMClient tests (~49 tests)
├── test_models.py                # Pydantic model tests (~40 tests)
├── test_tools.py                 # Tool system tests (~30 tests)
├── test_agents.py                # Agent framework tests (~15 tests)
├── test_config.py                # Configuration tests (~8 tests)
├── test_conversation_state.py    # Conversation state tests (~10 tests)
├── test_integration.py           # End-to-end tests (~19 tests)
│
├── test_agents_behavioral.py     # Behavioral tests (~20 tests)
├── test_golden_dataset.py        # Golden dataset runner
│
└── test_lm_studio_live.py        # Live server tests (optional)
```

### Test Naming Conventions

```python
# Unit tests: test_<component>_<behavior>
def test_client_initialization():
    ...

def test_tool_registration():
    ...

def test_agent_max_iterations():
    ...

# Behavioral tests: test_<pattern>_<property>
@pytest.mark.live_llm
def test_multi_step_iteration_pattern():
    ...

@pytest.mark.live_llm
def test_task_complete_removal():
    ...
```

---

## Mocking Patterns

### Pattern 1: Mock HTTP Responses

```python
from unittest.mock import patch, Mock

def test_chat_with_mock():
    """Test chat using mocked HTTP response."""
    with patch('requests.post') as mock_post:
        mock_post.return_value = Mock(
            status_code=200,
            json=lambda: {
                "id": "test-123",
                "model": "test-model",
                "choices": [{
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": "Mocked response"
                    },
                    "finish_reason": "stop"
                }]
            }
        )
        
        client = LocalLLMClient("http://test", "test-model")
        response = client.chat("Hello")
        
        assert response == "Mocked response"
```

### Pattern 2: Mock Tools

```python
from unittest.mock import Mock

def test_agent_with_mock_tools(monkeypatch):
    """Test agent using mocked tool execution."""
    mock_tool_result = {"result": 42}
    
    # Mock tool execution
    def mock_execute(tool_name, arguments):
        return mock_tool_result
    
    monkeypatch.setattr(
        "local_llm_sdk.tools.registry.ToolRegistry.execute",
        mock_execute
    )
    
    # Agent will use mocked tools
    result = agent.run("Calculate something")
    # ... assertions
```

### Pattern 3: Parametrized Tests

```python
import pytest

@pytest.mark.parametrize("input,expected", [
    ("2 + 2", 4),
    ("10 * 5", 50),
    ("100 / 4", 25),
])
def test_math_calculator_operations(input, expected):
    """Test calculator with multiple inputs."""
    result = math_calculator(expression=input, precision=0)
    assert result["result"] == expected
```

---

## Coverage Requirements

### Running Coverage

```bash
# Generate coverage report
pytest tests/ --cov=local_llm_sdk --cov-report=html

# View in browser
open htmlcov/index.html

# Show coverage in terminal
pytest tests/ --cov=local_llm_sdk --cov-report=term

# Fail if coverage below threshold
pytest tests/ --cov=local_llm_sdk --cov-fail-under=80
```

### Coverage Targets

| Component | Target | Current |
|-----------|--------|---------|
| client.py | ≥85% | ~88% |
| models.py | ≥90% | ~92% |
| tools/ | ≥85% | ~86% |
| agents/ | ≥80% | ~82% |
| Overall | ≥85% | ~87% |

### Coverage Best Practices

```python
# ✅ Good: Test all branches
def test_with_branches(client):
    # Test success path
    result = client.chat("Hello")
    assert result
    
    # Test error path
    with pytest.raises(ValueError):
        client.chat("")  # Empty message

# ❌ Bad: Only test happy path
def test_only_success(client):
    result = client.chat("Hello")
    assert result
```

---

## Running Tests Locally

### All Tests Must Run Locally

**Why**: Tests require LM Studio running locally and cannot be automated in CI/CD.

**Unit Tests** (Fast, ~2 minutes):
```bash
pytest tests/ -v  # Skips live_llm by default
```

**Behavioral Tests** (Requires LM Studio, ~15-30 minutes):
```bash
# Start LM Studio on http://169.254.83.107:1234 first
pytest tests/ -m "live_llm and behavioral" -v
pytest tests/ -m "live_llm and golden" -v
```

**Test Matrix** (Manual):
```bash
# Test across Python versions if needed
pyenv local 3.9 && pytest tests/ -v
pyenv local 3.10 && pytest tests/ -v
pyenv local 3.11 && pytest tests/ -v
pyenv local 3.12 && pytest tests/ -v
```

---

## Troubleshooting

### "Tests are being skipped"

**Problem**: `live_llm` tests skipped by default

**Solution**: By design - keeps unit test runs fast
```bash
# To run them explicitly:
pytest tests/ -m "live_llm" -v
```

### "LM Studio connection refused"

**Problem**: Can't connect to LM Studio

**Solutions**:
```bash
# Check LM Studio is running and model is loaded:
curl http://169.254.83.107:1234/v1/models

# Check logs:
ls -lt /mnt/c/Users/mahei/.cache/lm-studio/server-logs/$(date +%Y-%m)/ | head
```

### "Behavioral tests timeout"

**Problem**: Tests taking too long

**Solutions**:
```python
# Increase timeout in test:
@pytest.mark.timeout(600)  # 10 minutes
def test_complex_task():
    ...

# Or globally in pytest.ini:
[pytest]
timeout = 300
```

### "Success rate below threshold"

**Problem**: Golden dataset success rate dropping

**Steps**:
1. Run multiple times to confirm (not just variance)
2. Check LM Studio logs for errors
3. Review recent prompt/code changes
4. Consider if model changed
5. If persistent: investigate behavioral regression

### "Import errors in tests"

**Problem**: Can't import SDK modules

**Solution**:
```bash
# Install SDK in development mode
pip install -e .

# Verify installation
python -c "import local_llm_sdk; print(local_llm_sdk.__version__)"
```

---

## Summary

**Testing Workflow**:
1. Write unit tests first (TDD)
2. Run full regression before commit
3. Add behavioral tests for LLM behavior
4. Update golden dataset for common patterns
5. Monitor success rates over time

**Quick Commands**:
```bash
# Development: Fast unit tests
pytest tests/ -v

# Before commit: Full regression
pytest tests/ -v --cov=local_llm_sdk

# Weekly: Behavioral + golden dataset
pytest tests/ -m "live_llm" -v

# Release: Success rate validation
pytest tests/test_golden_dataset.py::TestGoldenDatasetSuccessRate -v
```

This tiered approach ensures both code correctness and LLM behavior reliability.
