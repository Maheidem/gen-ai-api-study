# Local LLM SDK

Type-safe Python SDK for OpenAI-compatible local LLM APIs (LM Studio, Ollama, LocalAI).

## Tech Stack
- Python 3.12, Pydantic v2, pytest (200+ tests)
- Optional: MLflow for tracing

## Essential Commands

```bash
# Install
pip install -e .

# Test (fast, mocked)
pytest tests/ -v

# Test (live LLM - requires LM Studio)
pytest tests/ -m "live_llm and behavioral" -v

# Code quality
black local_llm_sdk/ && isort local_llm_sdk/
```

## Environment Variables

```bash
export LLM_BASE_URL="http://169.254.83.107:1234/v1"  # LM Studio URL
export LLM_MODEL="your-model"                         # Model name
export LLM_TIMEOUT="300"                              # Timeout (seconds)
export LLM_DEBUG="true"                               # Debug logging
```

## Quick Usage

```python
from local_llm_sdk import LocalLLMClient, create_client_with_tools
from local_llm_sdk.agents import ReACT

# Basic chat
client = LocalLLMClient()
response = client.chat("Hello!")

# With tools
client = create_client_with_tools()
response = client.chat("Calculate 42 * 17", use_tools=True)

# ReACT agent (multi-step tasks)
agent = ReACT(client)
result = agent.run("Calculate 5 factorial", max_iterations=15)
```

## Project Structure

```
local_llm_sdk/           # Main package
  client.py              # LocalLLMClient
  models.py              # Pydantic models (OpenAI spec)
  agents/                # ReACT, BaseAgent
  tools/                 # Tool registry, bash tool
tests/                   # 200+ tests (unit + live)
notebooks/               # 11 educational notebooks
```

## Testing Rules

**CRITICAL**: All SDK changes require tests and full regression.

```bash
# Before EVERY commit
pytest tests/ -v --tb=short
# Must see: "X passed, 0 failed"
```

Test markers: `live_llm`, `behavioral`, `golden`, `notebook`

## Things NOT To Do

- Never skip full test suite before committing
- Never commit with failing tests - fix the code or update tests intentionally
- Never test exact LLM outputs - use property-based assertions instead
- Never mock the thing you're testing
- Don't assume tool results auto-persist - must add as messages

## Debugging

```bash
# LM Studio logs (recommended)
lms log stream | grep -E "error|POST"

# Test connection
curl http://169.254.83.107:1234/v1/models
```

## Documentation

| Topic | Location |
|-------|----------|
| Installation/Setup | `docs/getting-started/` |
| Architecture | `docs/architecture/overview.md` |
| Testing Guide | `docs/contributing/testing.md` |
| Notebook Testing | `tests/NOTEBOOK_TESTING_GUIDE.md` |
| Model Compatibility | `.documentation/model-compatibility-guide.md` |
| API Research | `.documentation/` |

Component-specific guidance in `local_llm_sdk/CLAUDE.md`, `tests/CLAUDE.md`, etc.
