# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Quick Reference

**Setup:**
```bash
pip install -e .                    # Install in development mode
pip install -r requirements.txt     # Install dependencies
```

**Testing:**
```bash
pytest tests/ -v                    # Run all tests (fast, mocked)
pytest tests/test_client.py -v      # Run specific test file
pytest tests/test_agents.py::TestReACTAgent::test_react_agent_simple_task -v  # Run single test
pytest tests/ --cov=local_llm_sdk --cov-report=html  # With coverage
```

**Behavioral Tests (requires LM Studio running):**
```bash
pytest tests/ -m "live_llm and behavioral" -v  # Real LLM behavior tests
pytest tests/ -m "live_llm and golden" -v      # Golden dataset regression
```

**Code Quality:**
```bash
black local_llm_sdk/          # Format code
isort local_llm_sdk/          # Sort imports
pytest tests/ -v              # Verify tests pass
```

**Environment Variables:**
```bash
export LLM_BASE_URL="http://169.254.83.107:1234/v1"  # LM Studio URL
export LLM_MODEL="your-model"                         # Model name
export LLM_TIMEOUT="300"                              # Request timeout (seconds)
export LLM_DEBUG="true"                               # Enable debug logging
```

## Project Overview

This is a Generative AI API study repository that has evolved into **Local LLM SDK** - a type-safe Python SDK for interacting with local LLM APIs that implement the OpenAI specification. The project provides a clean, extensible interface for working with LM Studio, Ollama, and other OpenAI-compatible servers.

**Key Technologies:**
- **Python 3.12.11** with Pydantic v2 for type safety
- **OpenAI API compatibility** (works with LM Studio, Ollama, LocalAI)
- **MLflow integration** (optional) for tracing and observability
- **Pytest** with 231 unit tests + 38 live LLM tests (behavioral + golden + notebooks)

## Study Approach

The repository follows a structured research methodology:
1. **API Documentation**: Comprehensive reference documents for both OpenAI and LM Studio APIs
2. **Practical Testing**: Jupyter notebooks with hands-on API interaction examples
3. **Comparative Analysis**: Side-by-side feature comparison and compatibility assessment
4. **Code Examples**: Pydantic models and type-safe API client implementations

## Development Environment

### Python Setup
- **Python Version**: 3.12.11
- **Key Libraries**:
  - `pydantic`: Type-safe data models for API responses
  - `requests`: HTTP client for API interactions
  - `openai`: Official OpenAI SDK
  - `ipykernel`: Jupyter notebook support

### LM Studio Configuration
- **Base URL**: `http://169.254.83.107:1234/v1` (local network)
- **Authentication**: Fixed key "lm-studio"
- **Available Models**: Check via `/v1/models` endpoint
- **Server Logs**: `/mnt/c/Users/mahei/.cache/lm-studio/server-logs` (organized by year-month)

#### Debugging with LM Studio Logs

**✅ RECOMMENDED: Use LM Studio CLI (cross-platform)**
```bash
# Stream live logs (works on Mac, Windows, Linux)
lms log stream

# Filter for specific events
lms log stream | grep -E "POST|error|stream"

# Use during tests to see real-time activity
lms log stream &  # Run in background
pytest tests/test_validation_live.py -v
```

**Alternative: Direct file access (OS-specific)**
- **Mac**: `~/.lmstudio/logs/`
- **Windows/WSL**: `/mnt/c/Users/mahei/.cache/lm-studio/server-logs/YYYY-MM/`

```bash
# Mac
tail -f ~/.lmstudio/logs/main.log

# Windows/WSL
tail -f /mnt/c/Users/mahei/.cache/lm-studio/server-logs/$(date +%Y-%m)/*.log
```

## Project Structure

```
gen-ai-api-study/
├── local_llm_sdk/              # Main Python package
│   ├── __init__.py            # Package exports and convenience functions
│   ├── client.py              # LocalLLMClient implementation
│   ├── config.py              # Configuration management (env vars)
│   ├── models.py              # Pydantic models (OpenAI spec)
│   ├── agents/                # Agent framework (ReACT, BaseAgent)
│   │   ├── base.py           # BaseAgent with MLflow tracing
│   │   ├── react.py          # ReACT agent implementation
│   │   └── models.py         # AgentResult, AgentStatus
│   ├── tools/                 # Tool system
│   │   ├── registry.py       # Tool registry and decorator
│   │   └── builtin.py        # Unified bash tool (full terminal capabilities)
│   └── utils/                 # Utility functions
├── tests/                      # Comprehensive test suite (269 tests)
│   ├── test_client.py         # Client functionality
│   ├── test_agents.py         # Agent framework tests
│   ├── test_agents_behavioral.py  # Behavioral tests (live LLM)
│   ├── test_golden_dataset.py     # Regression tests with success rates
│   ├── test_notebooks.py      # Notebook execution tests (11 notebooks, 10 passing)
│   └── golden_dataset.json        # 16 known-good task examples
├── notebooks/                  # Educational Jupyter notebooks (11 total)
│   ├── 01-installation-setup.ipynb
│   ├── 02-basic-chat.ipynb
│   └── 07-react-agents.ipynb  # ReACT agent tutorial
├── .documentation/            # Research documentation
├── setup.py                   # Package configuration
└── requirements.txt           # Dependencies (pydantic, requests, mlflow)
```

## Common Development Tasks

### Configuration via Environment Variables

The SDK uses `local_llm_sdk/config.py` for configuration management:

```bash
# Set environment variables (optional - defaults provided)
export LLM_BASE_URL="http://169.254.83.107:1234/v1"  # Default: http://localhost:1234/v1
export LLM_MODEL="mistralai/magistral-small-2509"    # Default: auto
export LLM_TIMEOUT="300"                              # Default: 300 seconds
export LLM_DEBUG="true"                               # Default: false

# Streaming and Validation (NEWLY IMPLEMENTED!)
export LLM_STREAM="true"                              # Enable streaming with SSE parsing (default: false)
export LLM_ENABLE_VALIDATION="true"                   # Enable response validation (default: false)
export LLM_VALIDATION_CHECK_INTERVAL="20"             # Check every N tokens (default: 20)
```

Or use defaults in code:
```python
from local_llm_sdk import create_client_with_tools

# Uses environment variables or defaults
client = create_client_with_tools()
```

### Using the SDK

**Basic Chat:**
```python
from local_llm_sdk import LocalLLMClient

client = LocalLLMClient(
    base_url="http://169.254.83.107:1234/v1",
    model="your-model"
)
response = client.chat("Hello!")
```

**With Tools:**
```python
from local_llm_sdk import create_client_with_tools

client = create_client_with_tools()
# LLM will use bash tool for calculations: bash(command="python -c 'print(42 * 17)'")
response = client.chat("Calculate 42 * 17", use_tools=True)
client.print_tool_calls()  # See which tools were used
```

**ReACT Agent (Multi-Step Tasks):**
```python
from local_llm_sdk.agents import ReACT

agent = ReACT(client)
result = agent.run("Calculate 5 factorial, convert to uppercase, count chars", max_iterations=15)
print(result.final_response)
print(f"Iterations: {result.iterations}, Tools: {result.metadata['total_tool_calls']}")
```

**Streaming with Validation (Early Termination):**
```python
import os

# Enable streaming and validation
os.environ['LLM_STREAM'] = 'true'
os.environ['LLM_ENABLE_VALIDATION'] = 'true'

from local_llm_sdk import LocalLLMClient

client = LocalLLMClient()

try:
    # This will use streaming with validation
    response = client.chat("Repeat the word 'test' exactly 100 times")
except ValueError as e:
    # Validation caught REPETITION during streaming!
    # Early termination after ~20 chunks (instead of 100+)
    print(f"Caught error: {e}")
    # Output: "🚨 VALIDATION ERROR: REPETITION"
    #         "💡 TIP: This is EARLY TERMINATION during streaming (stopped after 21 chunks)"
```

**Key Benefits:**
- ✅ Validation runs **DURING** generation (not after)
- ✅ Stops bad responses **immediately** (saves 70-80% time)
- ✅ Works with Server-Sent Events (SSE) format
- ✅ Detects: XML drift (if API conversion fails) and repetition loops

**Note on XML_DRIFT Detection:**
- Checks for XML patterns in the **API response** (what SDK receives)
- LM Studio converts qwen3's XML → JSON automatically, so SDK gets proper format
- Validator would only trigger if LM Studio's conversion fails (extremely rare)
- Primarily useful for detecting malformed responses or API-level issues

### Adding New Tools
```python
from local_llm_sdk import tool

@tool("Description of your tool")
def your_tool(param: str) -> dict:
    """
    Detailed description for LLM.

    Args:
        param: Description of parameter
    """
    return {"result": param.upper()}

# Register with client
client.register_tool("Your tool")(your_tool)
```

### Running Notebooks
```bash
cd notebooks/
jupyter notebook  # Start with 01-installation-setup.ipynb
```

## Code Architecture

### Package Architecture

The SDK follows a **layered architecture** with clear separation of concerns:

**Layer 1: Core Client (`client.py`)**
- `LocalLLMClient` - Main entry point with chat methods
- Automatic tool handling and conversation state management
- MLflow tracing integration (optional)
- 300s default timeout for local models

**Layer 2: Type System (`models.py`)**
- Pydantic v2 models following OpenAI API specification
- Strict validation for all request/response types
- `ChatMessage`, `ChatCompletion`, `Tool`, `ToolCall`, etc.

**Layer 3: Tool System (`tools/`)**
- `ToolRegistry` - Schema generation and tool execution
- `@tool` decorator for simple registration
- Unified `bash` tool - Full terminal capabilities (Python, files, git, text processing)

**Layer 4: Agent Framework (`agents/`)**
- `BaseAgent` - Abstract base with automatic tracing
- `ReACT` - Reasoning + Acting pattern for multi-step tasks
- `AgentResult`, `AgentStatus` - Result types

**Layer 5: Configuration (`config.py`)**
- Environment variable support
- Default configuration values
- Type-safe config loading

### Key Architectural Patterns

**1. Conversation State Management**
- All messages preserved in conversation history
- Tool results automatically added as `tool` role messages
- `client.last_conversation_additions` tracks new messages since last call
- Enables debugging and MLflow tracing

**2. Tool Execution Flow**
```
Client.chat(use_tools=True)
  → Add tool schemas to request
  → LLM returns tool_calls
  → Execute via ToolRegistry
  → Add results as messages
  → Send back to LLM
  → Repeat until no tool_calls
```

**3. Agent Pattern**
```
Agent.run(task)
  → BaseAgent wraps with tracing
  → Subclass implements _execute()
  → Conversation context preserved
  → Returns AgentResult with metadata
```

**4. MLflow Integration (Optional)**
- Graceful degradation if not installed
- Hierarchical traces: agent → iterations → tool calls
- Automatic span creation with proper parent-child relationships
- See `agents/base.py:14-19` for fallback handling

**MLflow Tracking URI Setup:**
- **Critical:** Tracking URI must point to where MLflow UI is serving from
- By default, uses `file://./mlruns` (current directory)
- **For notebooks:** Set tracking URI to project root to match MLflow UI location

```python
import mlflow

# Set to project root (where MLflow UI serves from)
mlflow.set_tracking_uri("file:///path/to/gen-ai-api-study/mlruns")

# Verify
print(f"Tracking URI: {mlflow.get_tracking_uri()}")
```

**Start MLflow UI:**
```bash
# Run from project root
cd /path/to/gen-ai-api-study
mlflow ui --port 5000
```

**Verification:**
- Run traced code (client.chat, agent.run, etc.)
- Check http://127.0.0.1:5000 for traces
- If traces missing: check tracking URI matches MLflow UI directory

### Critical Implementation Details

**Conversation State Tracking (local_llm_sdk/client.py:458-510)**
- `_handle_tool_calls()` returns tuple: `(content, new_messages)`
- New messages include: assistant message + tool result messages
- This fixes the "empty response bug" where tool results disappeared
- `last_conversation_additions` populated for tracing

**Tool Schema Generation (local_llm_sdk/tools/registry.py:40-95)**
- Automatic schema generation from Python type hints
- Converts Python types → JSON Schema types
- Handles required vs optional parameters
- Returns OpenAI-compatible `Tool` objects

**ReACT Agent Loop (local_llm_sdk/agents/react.py:99-148)**
- Optimized system prompt for clear instructions
- Iteration loop with stop condition checking
- Tool call counting and metadata tracking
- Final response extraction (strips "TASK_COMPLETE")

### Documentation Standards
- All research includes source citations with timestamps
- Reliability ratings (1-5 stars) for each source
- Code examples in Python, TypeScript, and cURL
- Performance benchmarks and cost analysis

## Testing Requirements

**CRITICAL: All SDK code changes MUST include tests and pass full regression.**

### Test-Driven Development Workflow

When modifying SDK code (`local_llm_sdk/`), follow this mandatory pattern:

1. **Identify the Change**
   - Document what's being fixed/added
   - Note expected behavior changes

2. **Update/Add Tests**
   ```bash
   # Example: Adding new feature to client.py
   # → Add tests to tests/test_client.py
   # → Update related integration tests
   ```

3. **Run Full Regression**
   ```bash
   # MUST run before committing
   pytest tests/ -v --tb=short

   # Should see: "X passed, Y skipped, 0 failed"
   ```

4. **Verify No Regressions**
   - All existing tests must still pass
   - New functionality must have test coverage
   - Test count should increase (new tests added)

### Testing Standards

**Required Test Coverage:**
- ✅ **Unit Tests**: Every public method/function
- ✅ **Integration Tests**: Multi-component workflows
- ✅ **Edge Cases**: Error conditions, empty inputs, boundary values
- ✅ **Regression Tests**: Previously fixed bugs stay fixed

**Test Organization:**
```
tests/
├── test_client.py          # LocalLLMClient tests
├── test_models.py          # Pydantic model validation
├── test_tools.py           # Tool system tests
├── test_agents.py          # Agent framework tests
├── test_config.py          # Configuration tests
├── test_integration.py     # End-to-end workflows
└── test_lm_studio_live.py  # Live server tests (optional)
```

**Example Test Update Pattern:**
```python
# Changed: ReACT agent now strips "TASK_COMPLETE" from responses

# Added test:
def test_react_agent_extract_final_answer(self, mock_client):
    """Test _extract_final_answer removes TASK_COMPLETE correctly."""
    agent = ReACT(mock_client)

    result = agent._extract_final_answer("Answer: 42. TASK_COMPLETE")
    assert result == "Answer: 42."
    assert "TASK_COMPLETE" not in result

# Updated existing test:
def test_react_agent_simple_task(self, mock_client):
    # ...
    # Changed: TASK_COMPLETE should now be removed
    assert "TASK_COMPLETE" not in result.final_response
    assert "completed the task" in result.final_response
```

### Running Tests

```bash
# Full regression (required before commit)
pytest tests/ -v

# Specific test file
pytest tests/test_agents.py -v

# Single test
pytest tests/test_agents.py::TestReACTAgent::test_react_agent_simple_task -v

# With coverage report
pytest tests/ --cov=local_llm_sdk --cov-report=html

# Watch mode (during development)
pytest tests/ --watch
```

### Test Quality Checklist

Before committing SDK changes:

- [ ] All tests pass (`pytest tests/`)
- [ ] New functionality has tests
- [ ] Edge cases are tested
- [ ] Error conditions are tested
- [ ] No test count decrease (unless removing obsolete tests)
- [ ] Test execution time reasonable (<60s for full suite)
- [ ] Tests are deterministic (no flaky tests)

### When Tests Fail

**DO NOT:**
- ❌ Commit with failing tests
- ❌ Skip tests to make them pass
- ❌ Comment out failing assertions

**DO:**
- ✅ Fix the code to make tests pass
- ✅ Update tests if behavior intentionally changed
- ✅ Add tests for newly discovered edge cases
- ✅ Document breaking changes in commit message

### Example: Complete Change Workflow

```bash
# 1. Make code changes
vim local_llm_sdk/agents/react.py

# 2. Update/add tests
vim tests/test_agents.py

# 3. Run tests locally
pytest tests/test_agents.py -v

# 4. Run full regression
pytest tests/ -v

# 5. Verify count increased
# Before: 227 tests
# After: 228 tests ✓

# 6. Commit with confidence
git add local_llm_sdk/agents/react.py tests/test_agents.py
git commit -m "feat: improve ReACT agent final response extraction

- Strip TASK_COMPLETE from agent responses
- Add _extract_final_answer() method
- Update system prompt for clearer instructions
- Add test coverage for new functionality

Tests: 228 passed (was 227)"
```

This ensures production-quality code with confidence that nothing broke.

## LLM Behavioral Testing

### The Testing Gap We Discovered

Traditional unit tests with mocks validate **code correctness** but miss **LLM behavior issues**:

- ✅ Tests verified: Code handles "TASK_COMPLETE" correctly
- ❌ Tests missed: LLM shouldn't put "TASK_COMPLETE" in final response
- ✅ Tests verified: Agent supports multiple iterations
- ❌ Tests missed: LLM actually uses multiple iterations instead of cramming into one

**Solution**: Tiered testing architecture with behavioral validation.

### Tiered Testing Architecture

```
┌─────────────────────────────────────────────────────────────┐
│ Tier 1: Unit Tests (231 tests) - Fast, Mocked              │
│ - Validates code logic                                      │
│ - Run on every commit (<10s)                                │
│ - Command: pytest tests/ -v                                 │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│ Tier 2: Behavioral Tests (9 tests) - Real LLM              │
│ - Validates LLM behavior patterns                           │
│ - Property-based assertions                                 │
│ - Run nightly or on demand                                  │
│ - Command: pytest tests/ -m "live_llm and behavioral" -v    │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│ Tier 3: Golden Dataset (17 tests) - Regression Detection   │
│ - Known-good task examples                                  │
│ - Success rate tracking (≥90% threshold)                    │
│ - Run weekly or before releases                             │
│ - Command: pytest tests/ -m "live_llm and golden" -v        │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│ Tier 4: Notebook Tests (11 tests, 10 passing) - Real LLM   │
│ - End-to-end validation of educational notebooks            │
│ - Ensures documentation matches reality                     │
│ - Extended timeouts (up to 20 minutes for complex ones)     │
│ - Command: pytest tests/test_notebooks.py -v                │
└─────────────────────────────────────────────────────────────┘
```

### Running Behavioral Tests

**Prerequisites**: LM Studio running on http://169.254.83.107:1234 with model loaded

```bash
# All behavioral tests
pytest tests/ -m "live_llm and behavioral" -v

# Golden dataset regression tests
pytest tests/ -m "live_llm and golden" -v

# Notebook tests (all 11 notebooks)
pytest tests/test_notebooks.py -v

# Specific notebook test
pytest tests/test_notebooks.py::test_07_react_agents -v

# Specific behavioral test
pytest tests/test_agents_behavioral.py::TestReACTBehavior::test_multi_step_iteration_pattern -v

# Run single golden task
pytest tests/test_golden_dataset.py::test_golden_task[factorial_uppercase_count] -v

# Success rate tests (slow)
pytest tests/test_golden_dataset.py::TestGoldenDatasetSuccessRate -v
```

### Notebook Testing

**Current Status**: 91% pass rate (10/11 notebooks passing)

The project includes comprehensive automated testing of all 11 educational notebooks with **REAL LLM execution**:

**Why Test Notebooks?**
- Validates examples work exactly as users will experience
- Catches SDK API changes that break documentation
- Ensures `.env` configuration works correctly
- Tests real integration with LM Studio

**Timeout Strategy:**
- Basic notebooks: 180s (simple chat)
- Tool notebooks: 240s (tool execution overhead)
- Agent notebooks: 300s (multi-step iterations)
- Production patterns: 600s (complex error handling)
- Mini projects: 900-1200s (complex multi-agent tasks)

**Recent Fixes (October 2025):**
1. **Notebook 07** (react-agents): Added missing `_execute()` method, fixed ChatMessage access
2. **Notebook 09** (production-patterns): Removed invalid SDK params, defined local exceptions
3. **Notebook 10** (code-helper): Extended timeout to 15 minutes
4. **Notebook 11** (data-analyzer): Extended timeout to 20 minutes, ready for testing

See `tests/NOTEBOOK_TESTING_GUIDE.md` for complete details.

**Note**: By default, `live_llm` tests are skipped (configured in pytest.ini). This keeps unit test runs fast.

### Writing Behavioral Tests

**Key Principle**: Use property-based assertions, not exact matching

**DON'T** ❌
```python
# Brittle - fails with LLM variance
assert result.iterations == 3
assert result.final_response == "The answer is 100"
```

**DO** ✅
```python
# Robust - tests behavioral properties
assert result.iterations > 1, "Expected multi-step execution"
assert "TASK_COMPLETE" not in result.final_response
assert "100" in result.final_response
assert result.metadata["total_tool_calls"] >= 2
```

### Example Behavioral Test

```python
@pytest.mark.live_llm
@pytest.mark.behavioral
def test_multi_step_iteration_pattern(agent):
    """
    Verify agent uses multiple iterations for multi-tool tasks.

    This test caught the bug where agent crammed all tools into iteration 1.

    Property: For tasks requiring 3+ tools, iterations should be > 1
    """
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

### Golden Dataset Structure

The golden dataset (`tests/golden_dataset.json`) contains 16 known-good tasks:

**Categories:**
- **single_tool_tasks** (5 tasks): Simple, 1 tool, ≥90% success expected
- **multi_tool_tasks** (4 tasks): Multi-step, 2+ tools, ≥80% success expected
- **complex_tasks** (2 tasks): Advanced, 3+ steps, ≥70% success expected
- **edge_cases** (2 tasks): Boundary conditions, ≥80% success expected

**Example Entry:**
```json
{
  "id": "factorial_uppercase_count",
  "task": "Calculate 5 factorial, convert to uppercase, count characters",
  "expected_properties": {
    "min_iterations": 2,
    "max_iterations": 10,
    "min_tool_calls": 3,
    "tools_used": ["bash"],
    "final_response_contains": ["120", "3"],
    "final_response_excludes": ["TASK_COMPLETE"]
  },
  "difficulty": "medium",
  "tags": ["math", "text_processing", "multi_step"]
}
```

### Success Rate Thresholds

Behavioral tests acknowledge LLM non-determinism through statistical thresholds:

| Category | Threshold | Rationale |
|----------|-----------|-----------|
| single_tool_tasks | ≥90% | Simple tasks should be highly reliable |
| multi_tool_tasks | ≥80% | More complex, slight variance acceptable |
| complex_tasks | ≥70% | Multiple valid approaches possible |
| edge_cases | ≥80% | Boundary handling should be consistent |

**Interpreting Results:**
- **≥90%**: Behavior is stable, no action needed
- **70-89%**: Investigate edge cases, consider prompt tuning
- **<70%**: Behavioral regression detected, requires investigation

### Adding New Behavioral Tests

When you discover a bug that unit tests missed:

1. **Write a behavioral test that would catch it:**
   ```python
   @pytest.mark.live_llm
   @pytest.mark.behavioral
   def test_new_behavior(agent):
       """Describe what behavior you're validating."""
       result = agent.run("task that exposed the bug")
       assert expected_property, "Why this matters"
   ```

2. **Add to golden dataset if it's a common pattern:**
   ```json
   {
     "id": "descriptive_id",
     "task": "Task description",
     "expected_properties": { ... },
     "tags": ["relevant", "tags"]
   }
   ```

3. **Run to verify it catches the issue:**
   ```bash
   pytest tests/test_agents_behavioral.py::test_new_behavior -v
   ```

### Running Tests Locally

**All tests must be run locally** as they require or benefit from LM Studio:

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

### Troubleshooting

**"Tests are being skipped"**
```bash
# By design - live_llm tests skipped by default
# To run them explicitly:
pytest tests/ -m "live_llm" -v
```

**"LM Studio connection refused"**
```bash
# Check LM Studio is running and model is loaded:
curl http://169.254.83.107:1234/v1/models
```

**"Behavioral tests timeout"**
```python
# Increase timeout in test or globally in pytest.ini:
@pytest.mark.timeout(600)  # 10 minutes
def test_complex_task():
    ...
```

**"Success rate below threshold"**
1. Run multiple times to confirm (not just variance)
2. Check LM Studio logs for errors
3. Review recent prompt/code changes
4. Consider if model changed
5. If persistent: investigate behavioral regression

### Best Practices

**Testing LLM Behavior:**
- ✅ Test invariants, not exact outputs
- ✅ Use statistical thresholds for success rates
- ✅ Write descriptive failure messages with context
- ✅ Test properties that caught real bugs
- ✅ Include both happy and edge cases

**Maintaining Golden Dataset:**
- ✅ Keep tasks realistic (mirror real usage)
- ✅ Update when fixing behavioral bugs
- ✅ Remove obsolete tasks that no longer apply
- ✅ Review quarterly for relevance
- ✅ Document expected properties clearly

**Interpreting Failures:**
- ✅ Single failure → investigate specific test
- ✅ Category failure → check prompt/model changes
- ✅ All failures → LLM server issue
- ✅ Gradual degradation → behavioral drift (prompt tuning needed)

### Files Reference

```
tests/
├── pytest.ini                    # Markers config (live_llm, behavioral, golden, notebook)
├── test_agents_behavioral.py     # 9 behavioral tests with real LLM
├── test_golden_dataset.py        # Golden dataset runner + success rate tests
├── test_notebooks.py             # 11 notebook execution tests (10 passing)
├── golden_dataset.json           # 16 known-good tasks with properties
├── NOTEBOOK_TESTING_GUIDE.md     # Complete notebook testing documentation
└── conftest.py                   # Shared fixtures
```

## API Compatibility Notes

### Fully Supported in LM Studio
- Chat completions (`/v1/chat/completions`)
- Streaming responses (SSE format)
- Embeddings (`/v1/embeddings`)
- Model listing (`/v1/models`)

### Partially Supported
- Function calling (~60% compatibility, model-dependent)
- JSON mode (~50% compatibility)

### Not Supported in LM Studio
- Assistants API
- Fine-tuning endpoints
- Image generation (DALL-E)
- Audio transcription/generation
- Content moderation

## Research Methodology

When adding new API comparisons:
1. Test both platforms with identical requests
2. Document response format differences
3. Note performance characteristics
4. Calculate cost implications
5. Update compatibility matrix
6. Include migration code examples

## Development Best Practices

### Before Making Changes

1. **Read existing tests first** - Understand expected behavior
2. **Check CLAUDE.md files** - Each directory has context (local_llm_sdk/, tests/, agents/, tools/)
3. **Run tests locally** - Ensure clean baseline (`pytest tests/ -v`)
4. **Check conversation state** - Many bugs relate to message handling

### When Adding Features

1. **Write tests first** (TDD) - Define expected behavior
2. **Update tests for existing functionality** - Ensure no regressions
3. **Add behavioral tests if LLM behavior matters** - Use property-based assertions
4. **Update golden dataset** - If pattern is common
5. **Verify test count increases** - New features need new tests

### Common Pitfalls to Avoid

**❌ Don't:**
- Skip running full test suite before committing
- Mock away the thing you're trying to test
- Test exact LLM outputs (use properties instead)
- Forget to update `last_conversation_additions` when adding messages
- Assume tool results are automatically preserved (must add as messages)
- Use mocked tests to validate LLM behavior (use behavioral tests)

**✅ Do:**
- Run `pytest tests/ -v` before every commit
- Test invariants and properties, not exact strings
- Preserve full conversation context (including tool results)
- Return tuples from `_handle_tool_calls()`: `(content, new_messages)`
- Use behavioral tests (`@pytest.mark.live_llm`) for LLM behavior
- Check LM Studio logs when live tests fail: `/mnt/c/Users/mahei/.cache/lm-studio/server-logs/`

### Architecture Guidelines

**When extending the client:**
- Preserve conversation state (all messages)
- Track new messages in `last_conversation_additions`
- Use MLflow tracing for observability
- Default timeout: 300s (local models are slower)

**When adding tools:**
- Use `@tool` decorator with clear description
- Return dict (JSON-serializable)
- Use type hints for automatic schema generation
- Handle errors gracefully (return error dict, don't raise)

**When creating agents:**
- Inherit from `BaseAgent`
- Implement `_execute()` with task logic
- Use `@mlflow.trace` for spans
- Return `AgentResult` with status and metadata

### Model Compatibility Notes

**LM Studio:**
- Fully supports chat completions, streaming, embeddings
- Function calling: ~60% compatibility (model-dependent)
- JSON mode: ~50% compatibility
- Context windows vary by model (check `/v1/models`)
- Requires significant RAM (8-64GB depending on model)

**Known Issues:**
- **qwen3 model**: ✅ **Works with SDK!** Model outputs XML format (`<tool_call>`) but LM Studio API automatically converts to JSON, so SDK receives correct OpenAI format. Validated with 642+ requests, 0 errors.
- **Reasoning models**: May skip tools with `tool_choice="auto"` - use `tool_choice="required"`
- **Streaming**: Implementation varies by model
- **Timeout**: Local models slower than cloud APIs (use 300s timeout)

**qwen3 Technical Details:**
- **Raw model output**: Uses XML format for tool calls (trained on XML schema)
- **LM Studio API layer**: Automatically converts XML → OpenAI JSON format
- **What SDK receives**: Proper JSON `tool_calls` structure (never sees XML)
- **Validation**: XML_DRIFT validator checks SDK input (JSON), not raw model output
- **Result**: Zero compatibility issues, works perfectly with all SDK features

### Debugging Techniques

**Client Issues:**
```python
# Enable debug logging
import os
os.environ['LLM_DEBUG'] = 'true'

# Check tool calls
client.chat("Calculate 5!", use_tools=True)
client.print_tool_calls(detailed=True)

# Inspect conversation
for msg in client.conversation:
    print(f"{msg.role}: {msg.content}")
```

**Agent Issues:**
```python
# Run with verbose output
result = agent.run("task", max_iterations=15, verbose=True)

# Check metadata
print(f"Iterations: {result.iterations}")
print(f"Tool calls: {result.metadata['total_tool_calls']}")
print(f"Status: {result.status}")
```

**Test Issues:**
```bash
# Run with verbose output and stop on first failure
pytest tests/test_agents.py -v -x

# Run specific test with full traceback
pytest tests/test_client.py::test_client_chat_with_tools -v --tb=long

# Run behavioral test with LM Studio
pytest tests/test_agents_behavioral.py::TestReACTBehavior::test_multi_step_iteration_pattern -v -s
```

**LM Studio Issues:**
```bash
# Stream logs in real-time (RECOMMENDED)
lms log stream | grep -E "error|POST|stream"

# Test connection
curl http://169.254.83.107:1234/v1/models

# Check if streaming is enabled
lms log stream | grep "stream.*true"
```

**MLflow Tracing Issues:**
```python
# Check current tracking URI
import mlflow
print(f"Tracking URI: {mlflow.get_tracking_uri()}")

# List experiments and traces
from mlflow.tracking import MlflowClient
client = MlflowClient()

print("\nExperiments:")
for exp in client.search_experiments():
    print(f"  - {exp.name} (ID: {exp.experiment_id})")

# If traces missing: tracking URI mismatch
# Solution: Set to match where MLflow UI is serving from
mlflow.set_tracking_uri("file:///path/to/project/mlruns")

# Verify traces are being created
import os
print(f"\nmlruns directory exists: {os.path.exists('mlruns')}")
print(f"Recent traces: {len(os.listdir('mlruns/0/traces'))} in default experiment")
```