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
response, history = client.chat_with_history("Hello!", history)

# Continue conversation
response, history = client.chat_with_history("What's your name?", history)
```

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

## Notebooks

See the `notebooks/` directory for interactive examples:
- `api-hello-world-local.ipynb` - Basic API usage
- `react-agent-flow.ipynb` - **NEW!** ReACT agent with code execution and filesystem tools

## ReACT Agent (NEW!)

The SDK now includes a powerful **ReACT (Reasoning, Action, Observation)** agent that can solve complex, multi-step tasks autonomously:

```python
from local_llm_sdk import LocalLLMClient

# Create client with built-in tools (including Python execution & filesystem)
client = LocalLLMClient(base_url="http://localhost:1234/v1")
client.register_tools_from(None)

# Create ReACT agent
from react_agent_flow import ReACTAgent  # From notebook
agent = ReACTAgent(client, max_iterations=10)

# Give it a complex task
task = """
Implement a sorting algorithm, test it, and benchmark
against Python's built-in sort with different array sizes.
Save all results to organized files.
"""

conversation = agent.think_and_act(task)
```

**New Tools Available:**
- `execute_python` - Safe Python code execution with timeout
- `filesystem_operation` - Create dirs, read/write files, list contents

See `REACT_GUIDE.md` for detailed documentation and examples.

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