# Development Guide

Complete guide to setting up and contributing to the Local LLM SDK.

## Table of Contents
1. [Setting Up Development Environment](#setting-up-development-environment)
2. [Project Structure](#project-structure)
3. [Development Workflow](#development-workflow)
4. [Code Style and Formatting](#code-style-and-formatting)
5. [Running Tests During Development](#running-tests-during-development)
6. [Debugging Tips](#debugging-tips)
7. [Git Workflow](#git-workflow)
8. [Common Development Tasks](#common-development-tasks)
9. [Contributing Guidelines](#contributing-guidelines)

---

## Setting Up Development Environment

### Prerequisites

- **Python 3.8+** (3.12.11 recommended)
- **Git** for version control
- **LM Studio** (optional, for behavioral testing)

### Initial Setup

1. **Clone the repository**:
   ```bash
   git clone https://github.com/Maheidem/gen-ai-api-study.git
   cd gen-ai-api-study
   ```

2. **Create a virtual environment**:
   ```bash
   # Using venv
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate

   # Or using conda
   conda create -n local-llm-sdk python=3.12
   conda activate local-llm-sdk
   ```

3. **Install the package in development mode**:
   ```bash
   pip install -e .
   ```

   This installs the package in "editable" mode, so code changes are immediately reflected without reinstalling.

4. **Install development dependencies**:
   ```bash
   pip install -r requirements.txt

   # Or install with dev extras
   pip install -e ".[dev]"
   ```

5. **Verify installation**:
   ```bash
   # Check package imports
   python -c "import local_llm_sdk; print(local_llm_sdk.__version__)"

   # Run quick test
   pytest tests/test_client.py::test_client_initialization -v
   ```

### Development Tools

Install recommended development tools:

```bash
# Code formatting
pip install black>=22.0.0
pip install isort>=5.0.0

# Type checking
pip install mypy>=0.950

# Testing
pip install pytest>=7.0.0
pip install pytest-cov>=3.0.0

# Notebook support
pip install jupyter>=1.0.0
pip install ipykernel>=6.0.0

# MLflow (optional, for tracing)
pip install mlflow>=3.0.0

# Notebook testing
pip install testbook>=0.4.2
```

### IDE Setup

#### VS Code

Recommended extensions:
- Python (Microsoft)
- Pylance (Microsoft)
- Jupyter (Microsoft)
- Black Formatter
- isort

`.vscode/settings.json`:
```json
{
  "python.linting.enabled": true,
  "python.linting.pylintEnabled": false,
  "python.linting.flake8Enabled": true,
  "python.formatting.provider": "black",
  "editor.formatOnSave": true,
  "python.testing.pytestEnabled": true,
  "python.testing.pytestArgs": [
    "tests",
    "-v"
  ]
}
```

#### PyCharm

1. Set Python interpreter to your virtual environment
2. Enable pytest as test runner: Preferences → Tools → Python Integrated Tools
3. Install Black plugin: Preferences → Plugins → Marketplace
4. Configure Black: Preferences → Tools → Black → Enable "On save"

### Environment Variables

Set up configuration for local development:

```bash
# LM Studio configuration
export LLM_BASE_URL="http://169.254.83.107:1234/v1"  # Or http://localhost:1234/v1
export LLM_MODEL="mistralai/magistral-small-2509"    # Your preferred model
export LLM_TIMEOUT="300"                              # 5 minutes
export LLM_DEBUG="true"                               # Enable debug logging

# Optional: MLflow tracking
export MLFLOW_TRACKING_URI="http://localhost:5000"
```

Add to `.bashrc`, `.zshrc`, or create a `.env` file:

```bash
# .env (add to .gitignore!)
LLM_BASE_URL=http://169.254.83.107:1234/v1
LLM_MODEL=mistralai/magistral-small-2509
LLM_TIMEOUT=300
LLM_DEBUG=true
```

### LM Studio Setup (Optional)

For behavioral testing, you'll need LM Studio running:

1. **Download and install**: [LM Studio](https://lmstudio.ai/)
2. **Load a model**: Download a compatible model (e.g., Mistral, Llama)
3. **Start server**: Enable "Local Server" on port 1234
4. **Verify connection**:
   ```bash
   curl http://localhost:1234/v1/models
   ```

**Server Logs Location**:
- Windows: `C:\Users\{username}\.cache\lm-studio\server-logs\`
- macOS: `~/Library/Application Support/LM Studio/server-logs/`
- Linux: `~/.cache/lm-studio/server-logs/`

---

## Project Structure

### Directory Layout

```
gen-ai-api-study/
├── local_llm_sdk/              # Main Python package
│   ├── __init__.py             # Package exports and convenience functions
│   ├── client.py               # LocalLLMClient implementation (500+ lines)
│   ├── config.py               # Configuration management (env vars)
│   ├── models.py               # Pydantic models (OpenAI spec)
│   │
│   ├── agents/                 # Agent framework
│   │   ├── __init__.py        # Agent exports
│   │   ├── base.py            # BaseAgent with MLflow tracing
│   │   ├── react.py           # ReACT agent implementation
│   │   └── models.py          # AgentResult, AgentStatus
│   │
│   ├── tools/                  # Tool system
│   │   ├── __init__.py        # Tool exports
│   │   ├── registry.py        # ToolRegistry and @tool decorator
│   │   └── builtin.py         # Built-in tools (math, text, file ops)
│   │
│   └── utils/                  # Utility functions
│       └── __init__.py
│
├── tests/                      # Comprehensive test suite (~213 tests)
│   ├── conftest.py            # Shared fixtures
│   ├── pytest.ini             # Pytest configuration
│   ├── golden_dataset.json    # Known-good tasks for regression
│   │
│   ├── test_client.py         # LocalLLMClient tests (~49 tests)
│   ├── test_models.py         # Pydantic model tests (~40 tests)
│   ├── test_tools.py          # Tool system tests (~30 tests)
│   ├── test_agents.py         # Agent framework tests (~15 tests)
│   ├── test_config.py         # Configuration tests (~8 tests)
│   ├── test_conversation_state.py  # Conversation state tests (~10 tests)
│   ├── test_integration.py    # End-to-end tests (~19 tests)
│   │
│   ├── test_agents_behavioral.py  # Behavioral tests (~20 tests)
│   ├── test_golden_dataset.py     # Golden dataset runner
│   ├── test_notebooks.py          # Notebook validation tests
│   └── test_lm_studio_live.py     # Live server tests (optional)
│
├── notebooks/                  # Educational Jupyter notebooks
│   ├── 01-installation-setup.ipynb
│   ├── 02-basic-chat.ipynb
│   ├── 03-streaming.ipynb
│   ├── 04-function-calling.ipynb
│   ├── 05-tools.ipynb
│   ├── 06-tool-registry.ipynb
│   ├── 07-react-agents.ipynb
│   └── ...
│
├── docs/                       # Documentation
│   ├── README.md              # Documentation index
│   ├── getting-started/       # Tutorials and quickstart
│   ├── guides/                # How-to guides
│   ├── api-reference/         # API documentation
│   ├── architecture/          # Architecture docs
│   └── contributing/          # This file and testing guide
│
├── .documentation/            # Research documentation (API specs)
├── .github/workflows/         # CI/CD workflows
│
├── setup.py                   # Package configuration
├── requirements.txt           # Dependencies
├── pytest.ini                 # Pytest configuration
├── CLAUDE.md                  # Project guidance for Claude Code
└── README.md                  # Project overview
```

### Package Architecture

The SDK follows a **layered architecture**:

**Layer 1: Core Client (`client.py`)**
- `LocalLLMClient` - Main entry point
- Automatic tool handling
- Conversation state management
- MLflow tracing integration (optional)
- 300s default timeout for local models

**Layer 2: Type System (`models.py`)**
- Pydantic v2 models (OpenAI API specification)
- Strict validation for all types
- Models: `ChatMessage`, `ChatCompletion`, `Tool`, `ToolCall`, etc.

**Layer 3: Tool System (`tools/`)**
- `ToolRegistry` - Schema generation and execution
- `@tool` decorator for registration
- Built-in tools: Python execution, file ops, math, text processing

**Layer 4: Agent Framework (`agents/`)**
- `BaseAgent` - Abstract base with automatic tracing
- `ReACT` - Reasoning + Acting pattern for multi-step tasks
- `AgentResult`, `AgentStatus` - Result types

**Layer 5: Configuration (`config.py`)**
- Environment variable support
- Default configuration values
- Type-safe config loading

### Key Files to Know

**Core SDK**:
- `local_llm_sdk/client.py` - Most development happens here
- `local_llm_sdk/models.py` - Type definitions
- `local_llm_sdk/tools/registry.py` - Tool system core
- `local_llm_sdk/agents/react.py` - Agent implementation

**Testing**:
- `tests/conftest.py` - Shared fixtures
- `tests/test_client.py` - Most comprehensive test file
- `tests/golden_dataset.json` - Regression detection

**Documentation**:
- `CLAUDE.md` - Project context (read this first!)
- `docs/contributing/testing.md` - Testing guide
- `README.md` - User-facing overview

---

## Development Workflow

### Test-Driven Development (TDD)

We follow a strict TDD workflow for all SDK changes:

1. **Write tests first** (defines expected behavior)
2. **Run tests** (they should fail)
3. **Write minimal code** (make tests pass)
4. **Refactor** (improve code quality)
5. **Run full regression** (ensure nothing broke)

### Example TDD Workflow

**Scenario**: Add a new method to reset conversation history

1. **Write the test**:
   ```python
   # In tests/test_client.py
   def test_reset_conversation(mock_client):
       """Test reset_conversation() clears history."""
       # Add some messages
       mock_client.conversation.append(
           ChatMessage(role="user", content="Hello")
       )
       assert len(mock_client.conversation) == 1

       # Reset
       mock_client.reset_conversation()

       # Verify cleared
       assert len(mock_client.conversation) == 0
       assert len(mock_client.last_conversation_additions) == 0
   ```

2. **Run test** (should fail):
   ```bash
   pytest tests/test_client.py::test_reset_conversation -v
   # AttributeError: 'LocalLLMClient' object has no attribute 'reset_conversation'
   ```

3. **Implement feature**:
   ```python
   # In local_llm_sdk/client.py
   def reset_conversation(self) -> None:
       """Reset conversation history."""
       self.conversation.clear()
       self.last_conversation_additions.clear()
   ```

4. **Run test again** (should pass):
   ```bash
   pytest tests/test_client.py::test_reset_conversation -v
   # PASSED
   ```

5. **Run full regression**:
   ```bash
   pytest tests/ -v
   # Should see: X passed, Y skipped, 0 failed
   ```

### Development Cycle

```
┌──────────────────────────────────────────────────────────┐
│ 1. Read existing code and tests                          │
│    - Understand current behavior                         │
│    - Check related functionality                         │
└──────────────────────────────────────────────────────────┘
                          ↓
┌──────────────────────────────────────────────────────────┐
│ 2. Write/update tests                                    │
│    - Unit tests for new functionality                    │
│    - Update tests for changed behavior                   │
│    - Add edge case tests                                 │
└──────────────────────────────────────────────────────────┘
                          ↓
┌──────────────────────────────────────────────────────────┐
│ 3. Implement changes                                     │
│    - Write minimal code to make tests pass               │
│    - Follow architecture guidelines                      │
│    - Add type hints and docstrings                       │
└──────────────────────────────────────────────────────────┘
                          ↓
┌──────────────────────────────────────────────────────────┐
│ 4. Run tests locally                                     │
│    pytest tests/ -v                                      │
│    - All tests must pass                                 │
│    - Test count should increase (new tests)              │
└──────────────────────────────────────────────────────────┘
                          ↓
┌──────────────────────────────────────────────────────────┐
│ 5. Format and lint                                       │
│    black local_llm_sdk/ && isort local_llm_sdk/          │
│    - Consistent code style                               │
└──────────────────────────────────────────────────────────┘
                          ↓
┌──────────────────────────────────────────────────────────┐
│ 6. Commit changes                                        │
│    - Descriptive commit message                          │
│    - Reference test count in commit message              │
└──────────────────────────────────────────────────────────┘
```

### Before Committing Checklist

- [ ] All tests pass (`pytest tests/ -v`)
- [ ] New functionality has tests
- [ ] Code is formatted (`black` and `isort`)
- [ ] Type hints added
- [ ] Docstrings updated
- [ ] Test count increased (or same if refactoring)
- [ ] No regressions in existing functionality

---

## Code Style and Formatting

### Black (Code Formatter)

We use **Black** for consistent code formatting:

```bash
# Format all SDK code
black local_llm_sdk/

# Format specific file
black local_llm_sdk/client.py

# Check without formatting (dry run)
black local_llm_sdk/ --check

# Show diff without applying
black local_llm_sdk/ --diff
```

**Configuration** (in `pyproject.toml` or command line):
```toml
[tool.black]
line-length = 88
target-version = ['py38', 'py39', 'py310', 'py311', 'py312']
include = '\.pyi?$'
extend-exclude = '''
/(
  # directories
  \.eggs
  | \.git
  | \.venv
  | build
  | dist
)/
'''
```

### isort (Import Sorting)

We use **isort** for consistent import ordering:

```bash
# Sort imports in all SDK code
isort local_llm_sdk/

# Sort specific file
isort local_llm_sdk/client.py

# Check without sorting (dry run)
isort local_llm_sdk/ --check-only

# Show diff without applying
isort local_llm_sdk/ --diff
```

**Configuration** (compatible with Black):
```toml
[tool.isort]
profile = "black"
line_length = 88
multi_line_output = 3
include_trailing_comma = true
force_grid_wrap = 0
use_parentheses = true
ensure_newline_before_comments = true
```

### Combined Format Command

Format everything at once:

```bash
# Format and sort
black local_llm_sdk/ && isort local_llm_sdk/

# Or create an alias
alias fmt="black local_llm_sdk/ tests/ && isort local_llm_sdk/ tests/"
```

### Type Hints

Use type hints for all functions:

```python
from typing import List, Dict, Optional, Union
from local_llm_sdk.models import ChatMessage, ChatCompletion

def chat(
    self,
    message: str,
    use_tools: bool = False,
    max_tool_iterations: int = 5,
    timeout: Optional[int] = None
) -> str:
    """
    Send a chat message to the LLM.

    Args:
        message: User message to send
        use_tools: Whether to enable tool calling
        max_tool_iterations: Maximum tool call iterations
        timeout: Request timeout in seconds

    Returns:
        Assistant's response text

    Raises:
        requests.exceptions.Timeout: If request times out
        requests.exceptions.RequestException: For other HTTP errors
    """
    # Implementation...
```

### Docstring Style

We use **Google-style docstrings**:

```python
def example_function(param1: str, param2: int = 0) -> Dict[str, Any]:
    """
    Brief one-line description.

    Longer description if needed. Can span multiple paragraphs
    and include examples.

    Args:
        param1: Description of first parameter
        param2: Description of second parameter (default: 0)

    Returns:
        Description of return value

    Raises:
        ValueError: When param2 is negative

    Example:
        >>> result = example_function("test", 42)
        >>> print(result)
        {'status': 'success', 'value': 42}
    """
    if param2 < 0:
        raise ValueError("param2 must be non-negative")
    return {"status": "success", "value": param2}
```

### Code Style Guidelines

**Naming Conventions**:
- **Classes**: `PascalCase` (e.g., `LocalLLMClient`, `ReACT`)
- **Functions/methods**: `snake_case` (e.g., `chat`, `handle_tool_calls`)
- **Constants**: `UPPER_SNAKE_CASE` (e.g., `DEFAULT_TIMEOUT`)
- **Private methods**: `_leading_underscore` (e.g., `_handle_tool_calls`)

**Line Length**:
- Maximum 88 characters (Black default)
- Break long lines naturally at commas, operators

**Imports**:
- Standard library first
- Third-party libraries second
- Local imports third
- Alphabetically sorted within each group

**Example**:
```python
# Standard library
import json
import logging
from typing import Dict, List, Optional

# Third-party
import requests
from pydantic import BaseModel, Field

# Local
from local_llm_sdk.models import ChatMessage
from local_llm_sdk.tools import ToolRegistry
```

---

## Running Tests During Development

See [Testing Guide](testing.md) for comprehensive testing documentation.

### Quick Test Commands

```bash
# Fast unit tests (run frequently during development)
pytest tests/ -v

# Specific test file
pytest tests/test_client.py -v

# Single test
pytest tests/test_client.py::test_client_initialization -v

# Stop on first failure
pytest tests/ -x

# Run last failed tests
pytest tests/ --lf

# Watch mode (requires pytest-watch)
ptw tests/ -- -v
```

### Test While Coding

Run tests continuously while developing:

```bash
# Install pytest-watch
pip install pytest-watch

# Watch for changes and run tests
ptw tests/ -- -v

# Watch specific file
ptw tests/test_client.py -- -v -x
```

### Coverage During Development

Check coverage for your changes:

```bash
# Generate coverage report
pytest tests/ --cov=local_llm_sdk --cov-report=term

# HTML report (more detailed)
pytest tests/ --cov=local_llm_sdk --cov-report=html
open htmlcov/index.html

# Coverage for specific module
pytest tests/test_client.py --cov=local_llm_sdk.client --cov-report=term
```

### Behavioral Tests

Test with real LLM (requires LM Studio running):

```bash
# All behavioral tests
pytest tests/ -m "live_llm and behavioral" -v

# Specific behavioral test
pytest tests/test_agents_behavioral.py::TestReACTBehavior::test_multi_step_iteration_pattern -v

# Golden dataset
pytest tests/ -m "live_llm and golden" -v
```

### Pre-Commit Test Script

Create a pre-commit check script:

```bash
#!/bin/bash
# pre-commit.sh

echo "Running pre-commit checks..."

# Format code
echo "1. Formatting code with Black..."
black local_llm_sdk/ tests/
if [ $? -ne 0 ]; then
    echo "Black formatting failed"
    exit 1
fi

# Sort imports
echo "2. Sorting imports with isort..."
isort local_llm_sdk/ tests/
if [ $? -ne 0 ]; then
    echo "isort failed"
    exit 1
fi

# Run tests
echo "3. Running tests..."
pytest tests/ -v --tb=short
if [ $? -ne 0 ]; then
    echo "Tests failed"
    exit 1
fi

echo "All checks passed!"
```

Make it executable:
```bash
chmod +x pre-commit.sh
```

Use it:
```bash
./pre-commit.sh && git commit -m "Your message"
```

---

## Debugging Tips

### Client Debugging

**Enable debug logging**:
```python
import os
import logging

# Enable debug mode
os.environ['LLM_DEBUG'] = 'true'

# Configure logging
logging.basicConfig(level=logging.DEBUG)

from local_llm_sdk import create_client_with_tools

client = create_client_with_tools()
response = client.chat("Calculate 5!", use_tools=True)
```

**Inspect conversation state**:
```python
# View all messages
for msg in client.conversation:
    print(f"{msg.role}: {msg.content}")
    if hasattr(msg, 'tool_calls') and msg.tool_calls:
        print(f"  Tool calls: {msg.tool_calls}")

# View last additions
print(f"Last additions: {len(client.last_conversation_additions)}")
for msg in client.last_conversation_additions:
    print(f"  {msg.role}: {msg.content[:50]}...")
```

**Print tool calls**:
```python
client.chat("Calculate 42 * 17", use_tools=True)
client.print_tool_calls(detailed=True)
```

### Agent Debugging

**Run with verbose output**:
```python
from local_llm_sdk.agents import ReACT

agent = ReACT(client)
result = agent.run(
    "Calculate 5 factorial, convert to uppercase",
    max_iterations=15,
    verbose=True  # Prints each iteration
)

# Check metadata
print(f"Iterations: {result.iterations}")
print(f"Tool calls: {result.metadata['total_tool_calls']}")
print(f"Status: {result.status}")
```

**Inspect iteration history**:
```python
# View each iteration's messages
for i, msg in enumerate(result.conversation):
    print(f"\nMessage {i} ({msg.role}):")
    print(f"  Content: {msg.content[:100]}...")
    if hasattr(msg, 'tool_calls') and msg.tool_calls:
        for tc in msg.tool_calls:
            print(f"  Tool: {tc.function.name}")
            print(f"  Args: {tc.function.arguments}")
```

### Test Debugging

**Run with verbose output**:
```bash
# Show print statements
pytest tests/test_client.py -v -s

# Full traceback
pytest tests/test_client.py -v --tb=long

# Stop at first failure with debugger
pytest tests/test_client.py -v -x --pdb
```

**Use pytest's built-in debugger**:
```python
def test_something(mock_client):
    """Test with debugger."""
    result = mock_client.chat("Hello")

    # Drop into debugger
    import pdb; pdb.set_trace()

    assert result == "Expected"
```

Run and interact:
```bash
pytest tests/test_client.py::test_something -v -s
# Drops into pdb when breakpoint hits
# Commands: n (next), s (step), c (continue), p variable (print)
```

### LM Studio Debugging

**Check server logs**:
```bash
# macOS/Linux
ls -lt /mnt/c/Users/mahei/.cache/lm-studio/server-logs/$(date +%Y-%m)/ | head

# Search for errors
grep -r "error" /mnt/c/Users/mahei/.cache/lm-studio/server-logs/$(date +%Y-%m)/

# Follow live logs
tail -f /mnt/c/Users/mahei/.cache/lm-studio/server-logs/$(date +%Y-%m)/*.log
```

**Test connection**:
```bash
# Check models
curl http://localhost:1234/v1/models

# Test chat completion
curl http://localhost:1234/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "local-model",
    "messages": [{"role": "user", "content": "Hello!"}]
  }'
```

### Common Issues

**Import errors**:
```bash
# Reinstall in dev mode
pip install -e .

# Verify installation
python -c "import local_llm_sdk; print(local_llm_sdk.__version__)"
```

**Test failures**:
```bash
# Run single test to isolate
pytest tests/test_client.py::test_specific_function -v -s

# Check fixtures
pytest --fixtures tests/test_client.py
```

**LM Studio connection refused**:
```bash
# Verify server is running
curl http://localhost:1234/v1/models

# Check firewall settings
# On Windows: Check Windows Defender Firewall
# On macOS: System Preferences → Security & Privacy → Firewall
```

---

## Git Workflow

### Branching Strategy

We use a simplified **feature branch workflow**:

```
main (production-ready code)
  ↓
feature/your-feature-name (your work)
  ↓
PR → Review → Merge
```

### Creating a Feature Branch

```bash
# Update main
git checkout main
git pull origin main

# Create feature branch
git checkout -b feature/add-conversation-reset

# Work on your feature...
# (write tests, implement code, format, test)

# Stage changes
git add local_llm_sdk/client.py tests/test_client.py

# Commit with descriptive message
git commit -m "feat: add conversation reset functionality

- Add reset_conversation() method to LocalLLMClient
- Clears conversation history and last_conversation_additions
- Add comprehensive test coverage

Tests: 214 passed (was 213)"

# Push to remote
git push origin feature/add-conversation-reset
```

### Commit Message Format

Follow **Conventional Commits** format:

```
<type>(<scope>): <subject>

<body>

<footer>
```

**Types**:
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation changes
- `test`: Adding or updating tests
- `refactor`: Code refactoring (no behavior change)
- `style`: Code style changes (formatting, etc.)
- `perf`: Performance improvements
- `chore`: Maintenance tasks

**Examples**:

```bash
# Feature
git commit -m "feat(client): add conversation reset functionality"

# Bug fix
git commit -m "fix(tools): handle empty string arguments correctly"

# Documentation
git commit -m "docs: update testing guide with behavioral test examples"

# Test
git commit -m "test(agents): add edge case tests for max iterations"

# Refactor
git commit -m "refactor(client): simplify tool execution flow"
```

**Good commit message**:
```
feat(agents): improve ReACT agent final response extraction

- Strip "TASK_COMPLETE" marker from final responses
- Add _extract_final_answer() method for clean output
- Update system prompt with clearer instructions
- Add comprehensive test coverage for edge cases

This fixes the issue where users saw internal markers in responses.

Tests: 228 passed (was 227)
```

**Bad commit message**:
```
fix stuff
```

### Pull Request Process

1. **Create PR on GitHub**:
   - Go to repository on GitHub
   - Click "New Pull Request"
   - Select your feature branch
   - Fill in PR template

2. **PR Description Template**:
   ```markdown
   ## Summary
   Brief description of changes (1-2 sentences)

   ## Changes
   - Bullet point list of specific changes
   - What was added/modified/removed

   ## Testing
   - [ ] All tests pass locally
   - [ ] Added tests for new functionality
   - [ ] Behavioral tests pass (if applicable)

   ## Test Results
   ```
   Tests: 214 passed, 16 skipped, 0 failed
   Coverage: 87%
   ```

   ## Breaking Changes
   None / List any breaking changes
   ```

3. **Code Review**:
   - Address reviewer feedback
   - Make changes in your feature branch
   - Push updates (PR updates automatically)

4. **Merge**:
   - Once approved, merge PR
   - Delete feature branch after merge

### Keeping Your Branch Updated

```bash
# Update your branch with latest main
git checkout main
git pull origin main

git checkout feature/your-feature
git merge main

# Or use rebase (cleaner history)
git checkout feature/your-feature
git rebase main
```

### Git Aliases (Optional)

Add to `~/.gitconfig`:

```ini
[alias]
    st = status
    co = checkout
    br = branch
    ci = commit
    unstage = reset HEAD --
    last = log -1 HEAD
    visual = log --graph --oneline --all

    # Commit with test count
    cit = "!f() { \
        count=$(pytest tests/ -q --collect-only | tail -1 | awk '{print $1}'); \
        git commit -m \"$1\n\nTests: $count passed\"; \
    }; f"
```

Usage:
```bash
git st              # git status
git co main         # git checkout main
git br              # git branch
git visual          # pretty log
```

---

## Common Development Tasks

### Adding a New Tool

1. **Write the tool function**:
   ```python
   # In local_llm_sdk/tools/builtin.py

   @tool("Reverse a string")
   def string_reverser(text: str) -> dict:
       """
       Reverse the characters in a string.

       Args:
           text: The string to reverse

       Returns:
           dict with 'result' key containing reversed string
       """
       return {"result": text[::-1]}
   ```

2. **Register the tool**:
   ```python
   # In local_llm_sdk/__init__.py or client initialization
   from local_llm_sdk.tools.builtin import string_reverser

   # Auto-registered via @tool decorator
   ```

3. **Write tests**:
   ```python
   # In tests/test_tools.py

   def test_string_reverser():
       """Test string reverser tool."""
       result = string_reverser(text="hello")
       assert result == {"result": "olleh"}

   def test_string_reverser_empty():
       """Test string reverser with empty string."""
       result = string_reverser(text="")
       assert result == {"result": ""}
   ```

4. **Test with client**:
   ```python
   from local_llm_sdk import create_client_with_tools

   client = create_client_with_tools()
   response = client.chat("Reverse the word 'Python'", use_tools=True)
   print(response)  # Should use string_reverser tool
   ```

### Adding a New Agent

1. **Create agent class**:
   ```python
   # In local_llm_sdk/agents/custom.py

   from local_llm_sdk.agents.base import BaseAgent
   from local_llm_sdk.agents.models import AgentResult, AgentStatus

   class CustomAgent(BaseAgent):
       """Custom agent for specific task."""

       def _execute(self, task: str, **kwargs) -> AgentResult:
           """
           Execute task using custom logic.

           Args:
               task: Task description
               **kwargs: Additional parameters

           Returns:
               AgentResult with final response and metadata
           """
           # Your agent logic here
           response = self.client.chat(task, use_tools=True)

           return AgentResult(
               success=True,
               final_response=response,
               iterations=1,
               status=AgentStatus.COMPLETED,
               conversation=self.client.conversation,
               metadata={"custom_key": "value"}
           )
   ```

2. **Add tests**:
   ```python
   # In tests/test_agents.py

   from local_llm_sdk.agents.custom import CustomAgent

   def test_custom_agent(mock_client):
       """Test custom agent execution."""
       agent = CustomAgent(mock_client)
       result = agent.run("Test task")

       assert result.success
       assert result.final_response
       assert result.iterations == 1
   ```

3. **Export in `__init__.py`**:
   ```python
   # In local_llm_sdk/agents/__init__.py

   from local_llm_sdk.agents.custom import CustomAgent

   __all__ = ["BaseAgent", "ReACT", "CustomAgent", ...]
   ```

### Adding a New Model/Type

1. **Define Pydantic model**:
   ```python
   # In local_llm_sdk/models.py

   from pydantic import BaseModel, Field
   from typing import Optional, List

   class CustomModel(BaseModel):
       """Custom model for new feature."""

       id: str = Field(..., description="Unique identifier")
       name: str = Field(..., description="Model name")
       parameters: Optional[List[str]] = Field(None, description="Parameters")

       model_config = {
           "json_schema_extra": {
               "examples": [
                   {
                       "id": "model-123",
                       "name": "custom-model",
                       "parameters": ["param1", "param2"]
                   }
               ]
           }
       }
   ```

2. **Add validation tests**:
   ```python
   # In tests/test_models.py

   from local_llm_sdk.models import CustomModel
   from pydantic import ValidationError
   import pytest

   def test_custom_model_valid():
       """Test CustomModel with valid data."""
       model = CustomModel(
           id="model-123",
           name="test-model",
           parameters=["p1", "p2"]
       )
       assert model.id == "model-123"
       assert model.name == "test-model"

   def test_custom_model_invalid():
       """Test CustomModel with invalid data."""
       with pytest.raises(ValidationError):
           CustomModel(id="model-123")  # Missing required 'name'
   ```

### Updating Documentation

1. **Update docstrings**:
   ```python
   def new_method(self, param: str) -> str:
       """
       New method description.

       Args:
           param: Parameter description

       Returns:
           Return value description

       Example:
           >>> client.new_method("test")
           "result"
       """
       pass
   ```

2. **Update README/docs**:
   ```markdown
   # In docs/api-reference/client.md

   ## LocalLLMClient.new_method()

   New method description...

   **Parameters:**
   - `param` (str): Description

   **Returns:**
   - str: Description

   **Example:**
   \```python
   client = LocalLLMClient(...)
   result = client.new_method("test")
   \```
   ```

3. **Update CLAUDE.md** (if adding major feature):
   ```markdown
   # In CLAUDE.md

   ## Common Development Tasks

   ### Using New Feature
   \```python
   # Example code...
   \```
   ```

### Releasing a New Version

1. **Update version**:
   ```python
   # In setup.py
   setup(
       name="local-llm-sdk",
       version="0.2.0",  # Increment version
       ...
   )
   ```

2. **Update CHANGELOG**:
   ```markdown
   # CHANGELOG.md

   ## [0.2.0] - 2024-10-01

   ### Added
   - New conversation reset functionality
   - Additional behavioral tests

   ### Changed
   - Improved ReACT agent prompt

   ### Fixed
   - Tool result preservation bug
   ```

3. **Tag release**:
   ```bash
   git tag -a v0.2.0 -m "Release version 0.2.0"
   git push origin v0.2.0
   ```

---

## Contributing Guidelines

### Code of Conduct

- Be respectful and constructive
- Welcome newcomers and help them get started
- Focus on what is best for the community
- Show empathy towards other contributors

### How to Contribute

1. **Find something to work on**:
   - Check GitHub Issues for open issues
   - Look for issues tagged `good first issue`
   - Propose new features in Discussions

2. **Discuss first** (for large changes):
   - Open an issue to discuss your idea
   - Get feedback before starting work
   - Ensure alignment with project goals

3. **Follow development workflow**:
   - Fork repository (external contributors)
   - Create feature branch
   - Write tests first (TDD)
   - Implement feature
   - Format and lint code
   - Run full test suite
   - Submit PR

4. **PR requirements**:
   - All tests pass
   - Code coverage maintained (≥85%)
   - Code formatted (Black + isort)
   - Descriptive commit messages
   - Updated documentation
   - Added examples (if applicable)

### What to Contribute

**Welcome contributions**:
- Bug fixes
- New tools
- Agent improvements
- Documentation improvements
- Test coverage improvements
- Performance optimizations
- New examples/notebooks

**Discuss before contributing**:
- Major architectural changes
- Breaking API changes
- New dependencies
- Large refactorings

### Review Process

1. **Automated checks** (GitHub Actions):
   - Unit tests must pass
   - Code formatting verified
   - No security vulnerabilities

2. **Manual review**:
   - Code quality and style
   - Test coverage
   - Documentation completeness
   - Architecture consistency

3. **Feedback**:
   - Address review comments
   - Make requested changes
   - Update PR description if needed

4. **Approval and merge**:
   - At least 1 approval required
   - All checks passing
   - Merge by maintainer

### Getting Help

- **Documentation**: Start with `CLAUDE.md` and `docs/`
- **Issues**: Open GitHub issue for bugs or questions
- **Discussions**: Use GitHub Discussions for ideas/help
- **Testing**: See [Testing Guide](testing.md)

### Recognition

Contributors are recognized in:
- `CONTRIBUTORS.md` file
- Release notes
- GitHub contributors page

---

## Summary

**Quick Setup**:
```bash
git clone https://github.com/Maheidem/gen-ai-api-study.git
cd gen-ai-api-study
python -m venv venv && source venv/bin/activate
pip install -e ".[dev]"
pytest tests/ -v
```

**Development Cycle**:
1. Write tests → 2. Implement → 3. Format → 4. Test → 5. Commit

**Key Commands**:
```bash
pytest tests/ -v                          # Run tests
black local_llm_sdk/ && isort local_llm_sdk/  # Format
pytest tests/ --cov=local_llm_sdk         # Coverage
```

**Resources**:
- [Testing Guide](testing.md) - Comprehensive testing documentation
- `CLAUDE.md` - Project context and architecture
- `docs/` - User-facing documentation
- GitHub Issues - Bug reports and feature requests

Happy coding!
