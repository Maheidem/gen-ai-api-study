# Quick Start

Get up and running with Local LLM SDK in 5 minutes.

## Prerequisites

- Python 3.8+ installed
- Local LLM server running (LM Studio or Ollama)
- SDK installed (`pip install -e .`)

If you haven't installed yet, see the [Installation Guide](installation.md).

## Your First Chat

### 1. Import and Create Client

```python
from local_llm_sdk import LocalLLMClient

client = LocalLLMClient(
    base_url="http://localhost:1234/v1",
    model="auto"  # Auto-detect first available model
)
```

### 2. Send a Message

```python
response = client.chat("What is the capital of France?")
print(response)
# Output: "The capital of France is Paris."
```

**That's it!** You've just completed your first LLM interaction.

## Simple Examples

### Basic Chat

```python
from local_llm_sdk import LocalLLMClient

client = LocalLLMClient()  # Uses defaults from environment variables

# Ask a question
answer = client.chat("Explain recursion in one sentence")
print(answer)
```

### System Prompts

```python
from local_llm_sdk import create_chat_message

# Create messages with system prompt
messages = [
    create_chat_message("system", "You are a helpful Python tutor."),
    create_chat_message("user", "What is a list comprehension?")
]

response = client.chat(messages)
print(response)
```

### With Tools (Function Calling)

```python
from local_llm_sdk import create_client_with_tools

# Create client with built-in tools pre-loaded
client = create_client_with_tools()

# Ask something that requires calculation
response = client.chat("What is 127 times 893?", use_tools=True)
print(response)
# The LLM will automatically use the math_calculator tool

# See which tools were called
client.print_tool_calls()
# Output:
# 🔧 Tool Execution Summary (1 call):
# ======================================================================
#   [1] math_calculator(arg1=127, arg2=893, operation=multiply) → result=113411
# ======================================================================
```

### Multi-Turn Conversation

```python
from local_llm_sdk import LocalLLMClient

client = LocalLLMClient()
history = []

# First message
response, history = client.chat_with_history("My name is Alice", history)
print(response)
# Output: "Hello Alice! How can I help you today?"

# Continue conversation - the LLM remembers!
response, history = client.chat_with_history("What's my name?", history)
print(response)
# Output: "Your name is Alice."

# Inspect conversation history
for msg in history:
    print(f"{msg.role}: {msg.content}")
```

## Quick Examples by Use Case

### Use Case 1: Q&A Assistant

```python
from local_llm_sdk import quick_chat

# One-liner for simple questions
answer = quick_chat("What is machine learning?")
print(answer)
```

### Use Case 2: Code Helper

```python
from local_llm_sdk import create_client_with_tools

client = create_client_with_tools()

# Ask for code help
response = client.chat(
    "Write a Python function to calculate fibonacci numbers",
    use_tools=True
)
print(response)
```

### Use Case 3: Data Analysis

```python
from local_llm_sdk.agents import ReACT

client = create_client_with_tools()
agent = ReACT(client)

# Complex multi-step task
result = agent.run(
    "Calculate the factorial of 5, convert the result to a string, "
    "then count the number of characters",
    max_iterations=15
)

print(result.final_response)
print(f"Completed in {result.iterations} iterations")
```

## Next Steps

### Tutorials
- **[Basic Usage](basic-usage.md)** - Learn core concepts
- **[Tool Calling Guide](../guides/tool-calling.md)** - Master function calling
- **[ReACT Agents Guide](../guides/react-agents.md)** - Build autonomous agents

### Interactive Learning
- **Notebooks**: Check out `/notebooks` for 11 progressive tutorials
  - Start with `01-installation-setup.ipynb`
  - Progress through to `11-mini-project-data-analyzer.ipynb`

### API Reference
- **[Client API](../api-reference/client.md)** - All client methods
- **[Models Reference](../api-reference/models.md)** - Type system
- **[Tools Reference](../api-reference/tools.md)** - Built-in tools

## Common Patterns

### Pattern 1: Streaming Responses (Coming Soon)

```python
# Streaming is on the roadmap
for chunk in client.chat_stream("Tell me a story"):
    print(chunk, end="", flush=True)
```

### Pattern 2: Temperature Control

```python
from local_llm_sdk import create_chat_completion_request

request = create_chat_completion_request(
    messages=[{"role": "user", "content": "Be creative!"}],
    temperature=0.9,  # Higher = more creative (0.0 - 1.0)
    max_tokens=500
)

response = client.chat(request)
```

### Pattern 3: Custom Tools

```python
from local_llm_sdk import tool

@tool("Get current weather for a city")
def get_weather(city: str) -> dict:
    """
    Fetch weather data for a city.

    Args:
        city: Name of the city
    """
    # Your implementation here
    return {"city": city, "temp": 72, "condition": "sunny"}

# Register with client
client.register_tool("Weather tool")(get_weather)

# Use it
response = client.chat("What's the weather in Paris?", use_tools=True)
```

## Troubleshooting Quick Fixes

### Issue: "Connection refused"
```python
# Check if server is running
import requests
try:
    r = requests.get("http://localhost:1234/v1/models")
    print("✓ Server is running")
    print(r.json())
except:
    print("✗ Server not running - start LM Studio or Ollama")
```

### Issue: "No model loaded"
```python
# Use auto-detect
client = LocalLLMClient(model="auto")

# Or list available models
models = client.list_models()
print("Available models:", [m.id for m in models.data])
```

### Issue: "Response too slow"
```python
# Increase timeout
client = LocalLLMClient(timeout=600)  # 10 minutes

# Or use a faster model
client = LocalLLMClient(model="mistral-7b")  # Smaller/faster
```

## Ready to Go Deeper?

- **[Configuration Guide](configuration.md)** - Environment variables and advanced settings
- **[Architecture Overview](../architecture/overview.md)** - How it all works
- **[Contributing Guide](../contributing/development.md)** - Help improve the SDK

## Need Help?

- **Examples**: Check `/notebooks` directory for 11 interactive tutorials
- **API Docs**: See `/docs/api-reference` for detailed documentation
- **Issues**: [GitHub Issues](https://github.com/Maheidem/gen-ai-api-study/issues)
