# Local LLM SDK

A type-safe Python SDK for interacting with local LLM APIs that implement the OpenAI specification.

## Features

- 🔒 **Type-Safe**: Full Pydantic model validation
- 🛠️ **Tool Support**: Simple decorator-based tool/function calling
- 🚀 **Easy to Use**: Clean, intuitive API
- 🔌 **OpenAI Compatible**: Works with LM Studio, Ollama, and other OpenAI-compatible servers
- 📦 **Extensible**: Easy to add new tools and capabilities

## Installation

### From Source
```bash
git clone https://github.com/Maheidem/gen-ai-api-study.git
cd gen-ai-api-study
pip install -e .
```

### Install Dependencies
```bash
pip install -r requirements.txt
```

## Quick Start

### 1. Configure Your LLM Server

Create a `.env` file in the project root:

```bash
LLM_BASE_URL=http://localhost:1234/v1
LLM_MODEL=your-model-name
```

### 2. Use the SDK

```python
from local_llm_sdk import LocalLLMClient
from dotenv import load_dotenv
import os

# Load configuration
load_dotenv()

# Create client
client = LocalLLMClient(
    base_url=os.getenv("LLM_BASE_URL"),
    model=os.getenv("LLM_MODEL")
)

# Simple chat
response = client.chat("What is the capital of France?")
print(response)
```

**Or use hardcoded values** (not recommended for production):

```python
client = LocalLLMClient(
    base_url="http://localhost:1234/v1",
    model="your-model-name"
)
```

## Tool Usage

### Using Built-in Tools

```python
from local_llm_sdk import LocalLLMClient
from local_llm_sdk.tools import builtin

# Create client with tools
client = LocalLLMClient()
client.register_tools_from(builtin)

# Tools are automatically used when needed
response = client.chat("What is 42 times 17?")
print(response)  # Will use the math calculator tool
```

### Creating Custom Tools

```python
from local_llm_sdk import tool

@tool("Get current weather")
def get_weather(city: str, units: str = "celsius") -> dict:
    # Your implementation here
    return {"temp": 22, "condition": "sunny"}

# Register with client
client.register_tool("Weather tool")(get_weather)
```

## Advanced Usage

### Type-Safe Responses

```python
from local_llm_sdk import create_chat_message, ChatCompletion

# Create messages with type safety
messages = [
    create_chat_message("system", "You are helpful"),
    create_chat_message("user", "Hello!")
]

# Get full ChatCompletion object
response: ChatCompletion = client.chat(messages)
print(f"Tokens used: {response.usage.total_tokens}")
```

### Conversation History

```python
history = []

# First message
response, history = client.chat_with_history("My name is Alice.", history)
print(response)  # "Hello Alice! How can I assist you today?"

# Continue conversation - the LLM remembers!
response, history = client.chat_with_history("What's my name?", history)
print(response)  # "Your name is Alice!"

# Inspect history (ChatMessage objects)
for msg in history:
    print(f"{msg.role}: {msg.content}")
```

Learn more in `notebooks/03-conversation-history.ipynb`

### Debugging Tool Calls

```python
# See which tools were called and their results
response = client.chat("What is 127 * 893?")
client.print_tool_calls()

# Output:
# 🔧 Tool Execution Summary (1 call):
# ======================================================================
#   [1] math_calculator(arg1=127, arg2=893, operation=multiply) → result=113411
# ======================================================================

# Use detailed=True for full JSON output
client.print_tool_calls(detailed=True)
```

### Controlling Tool Usage

```python
# Let model decide (default)
response = client.chat("Calculate 25 * 16", tool_choice="auto")

# Force tool usage (bypasses reasoning for models like Magistral)
response = client.chat("Calculate 25 * 16", tool_choice="required")

# Prevent tool usage
response = client.chat("Explain how to multiply", tool_choice="none")
```

**Note:** Reasoning models (Magistral, etc.) may skip tools for simple calculations when using `tool_choice="auto"`. Use `tool_choice="required"` to guarantee tool execution.

## Configuration

### Environment Variables

The recommended way to configure the SDK is using a `.env` file:

```bash
# .env file
LLM_BASE_URL=http://localhost:1234/v1
LLM_MODEL=your-model-name
LLM_TIMEOUT=300
LLM_DEBUG=false
```

**Available Variables:**

| Variable | Default | Description |
|----------|---------|-------------|
| `LLM_BASE_URL` | `http://localhost:1234/v1` | Base URL for your LLM server |
| `LLM_MODEL` | `auto` | Model to use (or "auto" for auto-detection) |
| `LLM_TIMEOUT` | `300` | Request timeout in seconds |
| `LLM_DEBUG` | `false` | Enable debug logging |

**Using Environment Variables:**

```python
from local_llm_sdk import LocalLLMClient
from dotenv import load_dotenv
import os

load_dotenv()

client = LocalLLMClient(
    base_url=os.getenv("LLM_BASE_URL"),
    model=os.getenv("LLM_MODEL"),
    timeout=int(os.getenv("LLM_TIMEOUT", "300"))
)
```

### Multi-Environment Setup

Create different `.env` files for different environments:

```bash
.env.local       # Your local machine
.env.production  # Production settings
.env.development # Development settings
```

Load specific environment:

```python
from dotenv import load_dotenv

# Load specific env file
load_dotenv(".env.production")
```

**Important:** Add `.env` to `.gitignore` to avoid committing secrets. Provide a `.env.example` for documentation.

## Project Structure

```
local_llm_sdk/
├── __init__.py         # Package exports
├── client.py           # Main client implementation
├── models.py           # Pydantic models (OpenAI spec)
├── tools/
│   ├── registry.py     # Tool registration system
│   └── builtin.py      # Built-in tools
└── utils/              # Utility functions
```

## 📚 Educational Notebooks

The `notebooks/` directory contains a **progressive learning path** from beginner to advanced (total: ~3 hours):

### Prerequisites

All notebooks now use `.env` configuration for portability:

1. **Create `.env` file** in project root:
   ```bash
   LLM_BASE_URL=http://localhost:1234/v1
   LLM_MODEL=your-model-name
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt  # Includes python-dotenv
   ```

### Level 1: Foundations (30 min)
- `01-installation-setup.ipynb` - Install SDK, configure .env, connect to LM Studio
- `02-basic-chat.ipynb` - Simple chat, system prompts, temperature control
- `03-conversation-history.ipynb` - Multi-turn conversations with context

### Level 2: Core Features (45 min)
- `04-tool-calling-basics.ipynb` - Using built-in tools (math, text, etc.)
- `05-custom-tools.ipynb` - Creating your own tools with @tool decorator
- `06-filesystem-code-execution.ipynb` - File I/O and code execution tools

### Level 3: Advanced (60 min)
- `07-react-agents.ipynb` - ReACT pattern for multi-step tasks
- `08-mlflow-observability.ipynb` - Tracing and debugging with MLflow
- `09-production-patterns.ipynb` - Error handling, retries, .env configuration patterns

### Level 4: Projects (60 min)
- `10-mini-project-code-helper.ipynb` - Code review assistant agent
- `11-mini-project-data-analyzer.ipynb` - Data analysis pipeline

**Start with `01-installation-setup.ipynb` if you're new to the SDK!**

**Note:** All notebooks load configuration from `.env` - update once, works everywhere!

## ReACT Agents

The SDK includes a powerful **ReACT (Reasoning, Action, Observation)** pattern for building autonomous agents that can solve complex, multi-step tasks. Learn how to build ReACT agents in `notebooks/07-react-agents.ipynb`.

**Key Capabilities:**
- 🧠 **Multi-step reasoning** - Break down complex tasks into steps
- 🛠️ **Tool execution** - Python code execution, file operations, calculations
- 🔄 **Iterative problem solving** - Observe results and adjust approach
- 📊 **Self-correction** - Detect and fix errors autonomously

**Example use cases covered in notebook 07:**
- Data analysis pipelines
- Code generation and testing
- File processing workflows
- Research and information gathering

See `notebooks/07-react-agents.ipynb` for complete tutorials and examples.

## Supported Servers

- **LM Studio** - Fully supported
- **Ollama** - OpenAI compatibility mode
- **LocalAI** - OpenAI endpoints
- **Text Generation WebUI** - With OpenAI extension
- Any OpenAI-compatible API server

## Development

### Setup Development Environment

```bash
# Clone the repository
git clone https://github.com/Maheidem/gen-ai-api-study.git
cd gen-ai-api-study

# Install in development mode
pip install -e .
pip install -r requirements.txt

# Create .env file for testing
cp .env.example .env
# Edit .env with your LM Studio settings
```

### Run Tests

**All tests must be run locally** as they require LM Studio:

```bash
# Run unit tests (fast, ~2 min)
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=local_llm_sdk --cov-report=html

# Run behavioral tests (requires LM Studio running)
pytest tests/ -m "live_llm and behavioral" -v

# Run golden dataset tests
pytest tests/ -m "live_llm and golden" -v
```

**Note:** Tests require LM Studio running on configured URL. See `docs/contributing/testing.md` for details.

### Code Formatting

```bash
black local_llm_sdk/
isort local_llm_sdk/
```

### Testing Philosophy

- **Unit tests** (213+ tests): Fast, mocked tests for code correctness
- **Behavioral tests** (~20 tests): Real LLM validation with property-based assertions
- **Golden dataset** (16 tasks): Regression tests with success rate tracking

All tests run locally - no CI/CD due to LM Studio dependency.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

MIT License - see LICENSE file for details.

## Acknowledgments

- Built for compatibility with OpenAI API specification
- Inspired by the need for type-safe local LLM interactions
- Thanks to the open-source LLM community