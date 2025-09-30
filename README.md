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

```python
from local_llm_sdk import LocalLLMClient

# Create client
client = LocalLLMClient(
    base_url="http://localhost:1234/v1",
    model="your-model-name"
)

# Simple chat
response = client.chat("What is the capital of France?")
print(response)
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

### Level 1: Foundations (30 min)
- `01-installation-setup.ipynb` - Install SDK, connect to LM Studio, verify setup
- `02-basic-chat.ipynb` - Simple chat, system prompts, temperature control
- `03-conversation-history.ipynb` - Multi-turn conversations with context

### Level 2: Core Features (45 min)
- `04-tool-calling-basics.ipynb` - Using built-in tools (math, text, etc.)
- `05-custom-tools.ipynb` - Creating your own tools with @tool decorator
- `06-filesystem-code-execution.ipynb` - File I/O and code execution tools

### Level 3: Advanced (60 min)
- `07-react-agents.ipynb` - ReACT pattern for multi-step tasks
- `08-mlflow-observability.ipynb` - Tracing and debugging with MLflow
- `09-production-patterns.ipynb` - Error handling, retries, configuration

### Level 4: Projects (60 min)
- `10-mini-project-code-helper.ipynb` - Code review assistant agent
- `11-mini-project-data-analyzer.ipynb` - Data analysis pipeline

**Start with `01-installation-setup.ipynb` if you're new to the SDK!**

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

### Run Tests
```bash
pytest tests/
```

### Code Formatting
```bash
black local_llm_sdk/
isort local_llm_sdk/
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

MIT License - see LICENSE file for details.

## Acknowledgments

- Built for compatibility with OpenAI API specification
- Inspired by the need for type-safe local LLM interactions
- Thanks to the open-source LLM community