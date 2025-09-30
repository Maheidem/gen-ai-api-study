# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a Generative AI API study repository that has evolved into **Local LLM SDK** - a type-safe Python SDK for interacting with local LLM APIs that implement the OpenAI specification. The project provides a clean, extensible interface for working with LM Studio, Ollama, and other OpenAI-compatible servers.

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
```bash
# Check latest logs
ls -lt /mnt/c/Users/mahei/.cache/lm-studio/server-logs/$(date +%Y-%m)/ | head

# Search for errors
grep -r "error" /mnt/c/Users/mahei/.cache/lm-studio/server-logs/$(date +%Y-%m)/

# Follow live logs
tail -f /mnt/c/Users/mahei/.cache/lm-studio/server-logs/$(date +%Y-%m)/*.log
```

## Project Structure

```
gen-ai-api-study/
├── local_llm_sdk/              # Main Python package
│   ├── __init__.py            # Package exports
│   ├── client.py              # LocalLLMClient implementation
│   ├── models.py              # Pydantic models (OpenAI spec)
│   ├── tools/                 # Tool system
│   │   ├── __init__.py
│   │   ├── registry.py        # Tool registry and decorator
│   │   └── builtin.py         # Built-in tools
│   └── utils/                 # Utility functions
├── notebooks/                  # Jupyter notebooks
│   ├── api-hello-world-local.ipynb
│   ├── tool-use-math-calculator.ipynb
│   └── tool-use-simplified.ipynb
├── .documentation/            # Research documentation
├── setup.py                   # Package configuration
├── requirements.txt           # Dependencies
└── README.md                  # Project documentation
```

## Common Development Tasks

### Installing the Package
```bash
# Install in development mode
pip install -e .

# Install dependencies
pip install -r requirements.txt
```

### Using the SDK
```python
from local_llm_sdk import LocalLLMClient, create_client

# Create client
client = LocalLLMClient(
    base_url="http://169.254.83.107:1234/v1",
    model="your-model"
)

# Simple chat
response = client.chat("Hello!")
```

### Running Notebooks
```bash
# Navigate to notebooks directory
cd notebooks/

# Start Jupyter
jupyter notebook
```

### Adding New Tools
```python
from local_llm_sdk import tool

@tool("Description of your tool")
def your_tool(param: str) -> dict:
    return {"result": param.upper()}
```

## Code Architecture

### Package Architecture
- **`local_llm_sdk/`**: Main package with clean separation of concerns
  - `client.py`: Main LocalLLMClient class
  - `models.py`: Pydantic models following OpenAI spec
  - `tools/`: Tool registration and execution system
- **`notebooks/`**: Interactive examples and tutorials
- **`.documentation/`**: Research documents with citations

### Key Components
1. **LocalLLMClient**: Type-safe client with automatic tool handling
2. **Tool System**: Decorator-based tool registration
3. **Pydantic Models**: Full OpenAI API specification coverage

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
│ Tier 1: Unit Tests (213 tests) - Fast, Mocked              │
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

### Running Behavioral Tests

**Prerequisites**: LM Studio running on http://169.254.83.107:1234 with model loaded

```bash
# All behavioral tests
pytest tests/ -m "live_llm and behavioral" -v

# Golden dataset regression tests
pytest tests/ -m "live_llm and golden" -v

# Specific behavioral test
pytest tests/test_agents_behavioral.py::TestReACTBehavior::test_multi_step_iteration_pattern -v

# Run single golden task
pytest tests/test_golden_dataset.py::test_golden_task[factorial_uppercase_count] -v

# Success rate tests (slow)
pytest tests/test_golden_dataset.py::TestGoldenDatasetSuccessRate -v
```

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
    "tools_used": ["math_calculator", "text_transformer", "char_counter"],
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

### CI/CD Integration

**Unit Tests** (GitHub Actions: `test-unit.yml`):
- Trigger: Every push/PR
- Duration: ~2 minutes
- Tests: All except `live_llm` (default)

**Behavioral Tests** (GitHub Actions: `test-behavioral.yml`):
- Trigger: Nightly at 2:00 AM UTC + manual
- Duration: ~15-30 minutes
- Tests: Behavioral + golden dataset
- **Status**: 🚧 Requires LLM server setup (see `.github/workflows/README.md`)

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
├── pytest.ini                    # Markers config (live_llm, behavioral, golden)
├── test_agents_behavioral.py     # ~20 behavioral tests with real LLM
├── test_golden_dataset.py        # Golden dataset runner + success rate tests
├── golden_dataset.json           # 16 known-good tasks with properties
└── conftest.py                   # Shared fixtures

.github/workflows/
├── test-unit.yml                 # Fast unit tests on every commit
├── test-behavioral.yml           # Nightly behavioral tests (needs LLM setup)
└── README.md                     # Workflow documentation
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

## Tips for Future Development

### When Testing New Endpoints
- Always use Pydantic models for response validation
- Test with multiple models when available
- Document timeout requirements (LM Studio can be slower)
- Include error handling examples

### When Writing Documentation
- Follow the existing citation format in `.documentation/`
- Include practical use cases
- Provide cost-benefit analysis
- Note hardware requirements for local deployment

### Common Pitfalls to Avoid
- LM Studio models may have different context windows than OpenAI
- Streaming implementation varies by model
- Function calling syntax differs between models
- Local models require significant RAM (8-64GB depending on model size)