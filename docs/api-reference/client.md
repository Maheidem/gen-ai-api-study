# LocalLLMClient API Reference

**Module:** `local_llm_sdk.client`

**Import:** `from local_llm_sdk import LocalLLMClient`

## Overview

`LocalLLMClient` is a type-safe Python client for interacting with local LLM servers that implement the OpenAI API specification (LM Studio, Ollama, LocalAI, etc.). It provides automatic tool handling, conversation state management, and optional MLflow tracing for observability.

### Key Features

- **Type-safe API**: Full Pydantic validation for all requests and responses
- **Automatic tool execution**: Built-in support for function calling with automatic iteration
- **Conversation state management**: Tool results and message history preserved automatically
- **MLflow tracing**: Optional hierarchical tracing for debugging and observability
- **Thinking block extraction**: Automatic extraction of reasoning blocks from reasoning models
- **Flexible configuration**: Environment variables, defaults, or explicit parameters
- **Production-ready**: Automatic retries, timeout handling, error propagation

### Quick Start

```python
from local_llm_sdk import LocalLLMClient

# Basic usage
client = LocalLLMClient(
    base_url="http://localhost:1234/v1",
    model="mistralai/magistral-small-2509"
)

response = client.chat("Hello, how are you?")
print(response)

# With tools
client.register_tools_from(None)  # Load built-in tools
response = client.chat("Calculate 42 * 17", use_tools=True)
print(response)
```

---

## Constructor

### `LocalLLMClient(base_url=None, model=None, timeout=None)`

Initialize a new Local LLM client instance.

#### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `base_url` | `str` | Environment variable `LLM_BASE_URL` or `"http://localhost:1234/v1"` | Base URL for the local LLM API endpoint |
| `model` | `str` | Environment variable `LLM_MODEL` or `"auto"` | Model identifier to use. Use `"auto"` to automatically detect the first available model |
| `timeout` | `int` | Environment variable `LLM_TIMEOUT` or `300` | Request timeout in seconds. Local models may need 120-600s for complex tasks |

#### Returns

`LocalLLMClient` instance

#### Raises

- `ValueError`: If auto-detection fails and no model is specified
- `ConnectionError`: If unable to connect to the LLM server

#### Example

```python
# Explicit configuration
client = LocalLLMClient(
    base_url="http://169.254.83.107:1234/v1",
    model="mistralai/magistral-small-2509",
    timeout=600  # 10 minutes for complex tasks
)

# Auto-detection
client = LocalLLMClient()  # Uses environment variables or defaults
# Output: ✓ Auto-detected model: mistralai/magistral-small-2509

# Environment variables (alternative)
# export LLM_BASE_URL="http://localhost:1234/v1"
# export LLM_MODEL="auto"
# export LLM_TIMEOUT="300"
client = LocalLLMClient()  # Uses environment configuration
```

#### Configuration Priority

1. **Explicit parameters** (highest priority)
2. **Environment variables** (`LLM_BASE_URL`, `LLM_MODEL`, `LLM_TIMEOUT`)
3. **Default values** (lowest priority)

---

## Core Methods

### `chat(messages, model=None, temperature=0.7, max_tokens=None, use_tools=True, tool_choice="auto", stream=False, return_full_response=False, include_thinking=False, **kwargs)`

Send a chat completion request with automatic tool handling.

#### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `messages` | `str` or `List[ChatMessage]` | Required | Either a simple string query or list of structured chat messages |
| `model` | `str` | Client default | Model to use (overrides client default) |
| `temperature` | `float` | `0.7` | Sampling temperature (0.0-2.0). Lower = more deterministic, higher = more creative |
| `max_tokens` | `int` | `None` | Maximum tokens to generate (None = model default) |
| `use_tools` | `bool` | `True` | Whether to include registered tools in the request |
| `tool_choice` | `str` or `dict` | `"auto"` | Tool usage control: `"auto"` (model decides), `"required"` (force tool use), `"none"` (no tools), or specific tool dict |
| `stream` | `bool` | `False` | Whether to stream the response (SSE format) |
| `return_full_response` | `bool` | `False` | Force returning full `ChatCompletion` object instead of string |
| `include_thinking` | `bool` | `False` | Include thinking blocks from reasoning models in response |
| `**kwargs` | `Any` | - | Additional parameters passed to `ChatCompletionRequest`. When `messages` is a string, accepts `system` kwarg for custom system prompt |

#### Returns

- **Simple mode** (when `messages` is a string or 1-2 messages): `str` - Just the response content
- **Detailed mode** (when `messages` is a long list or `return_full_response=True`): `ChatCompletion` - Full response object

#### Special Kwargs (String Input Only)

| Kwarg | Type | Default | Description |
|-------|------|---------|-------------|
| `system` | `str` | `"You are a helpful assistant with access to tools."` | Custom system prompt when using string input |

#### Example

```python
# Simple string input (returns string)
response = client.chat("What is 2+2?")
print(response)  # "2 + 2 equals 4."

# With custom system prompt
response = client.chat(
    "Translate 'hello' to Spanish",
    system="You are a professional translator."
)

# With tools
client.register_tools_from(None)
response = client.chat("Calculate 5 factorial", use_tools=True)
# Output: "The factorial of 5 is 120."

# Force tool usage (useful for reasoning models)
response = client.chat(
    "What's 2+2?",
    use_tools=True,
    tool_choice="required"  # Force immediate tool use
)

# Structured message input (returns ChatCompletion for >2 messages)
from local_llm_sdk import create_chat_message

messages = [
    create_chat_message("system", "You are a helpful assistant."),
    create_chat_message("user", "Hello!"),
    create_chat_message("assistant", "Hi! How can I help?"),
    create_chat_message("user", "What's the weather?")
]
response = client.chat(messages)
print(response.choices[0].message.content)

# Get full response object
response = client.chat("Hello", return_full_response=True)
print(f"Model: {response.model}")
print(f"Tokens: {response.usage.total_tokens}")

# Include thinking from reasoning models
response = client.chat(
    "Explain quantum computing",
    include_thinking=True
)
# Output includes: "**Thinking:**\n...\n\n**Response:**\n..."
```

#### Tool Choice Options

| Value | Behavior |
|-------|----------|
| `"auto"` | Model decides whether to use tools based on context |
| `"required"` | Model must use at least one tool before responding |
| `"none"` | Tools are not available for this request |
| `{"type": "function", "function": {"name": "tool_name"}}` | Force specific tool usage |

#### Note on Reasoning Models

For reasoning models (like Magistral), use `tool_choice="required"` to force immediate tool usage and bypass internal reasoning for simple problems:

```python
# Without "required" - model may solve mentally (no tool call)
response = client.chat("What's 2+2?", use_tools=True, tool_choice="auto")

# With "required" - forces tool usage
response = client.chat("What's 2+2?", use_tools=True, tool_choice="required")
```

---

### `list_models()`

Get list of available models from the LLM server.

#### Returns

`ModelList` - List of available models with metadata

#### Example

```python
models = client.list_models()
print(f"Available models: {len(models.data)}")

for model in models.data:
    print(f"  - {model.id}")

# Output:
# Available models: 2
#   - mistralai/magistral-small-2509
#   - llama-3.2-1b-instruct
```

---

### `embeddings(input, model=None)`

Generate embeddings for text.

#### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `input` | `str` or `List[str]` | Required | Text or list of texts to embed |
| `model` | `str` | Client default | Embedding model to use |

#### Returns

`EmbeddingsResponse` - Response with embedding vectors

#### Example

```python
# Single text
response = client.embeddings("Hello world")
vector = response.data[0].embedding
print(f"Embedding dimensions: {len(vector)}")

# Multiple texts
texts = ["Hello", "World", "AI"]
response = client.embeddings(texts)

for i, embedding_data in enumerate(response.data):
    print(f"Text {i}: {len(embedding_data.embedding)} dimensions")
```

---

### `chat_simple(query)`

Simple chat without tool support or conversation history.

#### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `query` | `str` | Required | User query |

#### Returns

`str` - Model's response

#### Example

```python
response = client.chat_simple("What is Python?")
print(response)
# No tools are used, even if registered
```

---

### `chat_with_history(query, history, **kwargs)`

Chat with explicit conversation history management.

#### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `query` | `str` | Required | New user query |
| `history` | `List[ChatMessage]` | Required | Previous conversation messages |
| `**kwargs` | `Any` | - | Additional parameters passed to `chat()` |

#### Returns

`tuple[str, List[ChatMessage]]` - Tuple of (response_text, updated_history)

#### Example

```python
from local_llm_sdk import create_chat_message

# Initialize conversation
history = [
    create_chat_message("system", "You are a helpful assistant.")
]

# First turn
response, history = client.chat_with_history("Hello!", history)
print(response)  # "Hi! How can I help you today?"

# Second turn (history preserved)
response, history = client.chat_with_history("What's the weather?", history)
print(response)

# History now contains: system, user1, assistant1, user2, assistant2
print(f"Conversation length: {len(history)} messages")
```

---

### `react(task, max_iterations=15, stop_condition=None, temperature=0.7, verbose=True, **kwargs)`

Run a ReACT (Reasoning + Acting) agent for complex multi-step tasks.

This is a convenience method that creates and runs a ReACT agent, which uses tools iteratively to accomplish complex tasks through reasoning and acting cycles.

#### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `task` | `str` | Required | Task description/prompt |
| `max_iterations` | `int` | `15` | Maximum number of reasoning-acting iterations |
| `stop_condition` | `callable` | `None` | Optional function that returns `True` when task is complete |
| `temperature` | `float` | `0.7` | Sampling temperature |
| `verbose` | `bool` | `True` | Whether to print progress during execution |
| `**kwargs` | `Any` | - | Additional arguments passed to the agent |

#### Returns

`AgentResult` - Result object with execution details

#### Example

```python
# Simple task
result = client.react("Calculate 5 factorial")

if result.success:
    print(f"Completed in {result.iterations} iterations")
    print(result.final_response)

# Complex multi-step task
result = client.react(
    task="Calculate 5 factorial, convert to uppercase string, count characters",
    max_iterations=15,
    verbose=True
)

# Check execution details
print(f"Status: {result.status}")
print(f"Iterations: {result.iterations}")
print(f"Tool calls: {result.metadata['total_tool_calls']}")
print(f"Response: {result.final_response}")

# Custom stop condition
def is_complete(conversation):
    """Stop when we see a specific keyword."""
    last_msg = conversation[-1].content if conversation else ""
    return "COMPLETE" in last_msg

result = client.react(
    task="Implement bubble sort",
    stop_condition=is_complete,
    max_iterations=20
)
```

---

## Tool Management

### `register_tool(description="")`

Decorator to register a tool function with the client.

#### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `description` | `str` | `""` | Human-readable description for the LLM |

#### Returns

Decorator function

#### Example

```python
@client.register_tool("Add two numbers together")
def add(a: float, b: float) -> dict:
    """
    Add two numbers.

    Args:
        a: First number
        b: Second number
    """
    return {"result": a + b}

# Use the tool
response = client.chat("What is 5 + 10?", use_tools=True)
# Model will call add(5, 10) and return result
```

---

### `register_tools(tools)`

Register multiple tool functions at once.

#### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `tools` | `List[callable]` | Required | List of functions to register as tools |

#### Returns

`LocalLLMClient` - Self (for chaining)

#### Example

```python
def add(a: float, b: float) -> dict:
    """Add two numbers"""
    return {"result": a + b}

def multiply(a: float, b: float) -> dict:
    """Multiply two numbers"""
    return {"result": a * b}

client.register_tools([add, multiply])

# Use the tools
response = client.chat("Calculate (5 + 3) * 2", use_tools=True)
```

---

### `register_tools_from(module)`

Import tools from a module or registry.

#### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `module` | `Module` or `ToolRegistry` or `None` | Required | Module with tools, registry, or `None` for built-in tools |

#### Returns

`LocalLLMClient` - Self (for chaining)

#### Example

```python
# Load built-in tools
client.register_tools_from(None)

# Load from custom module
import my_tools
client.register_tools_from(my_tools)

# Load from another client's registry
other_client = LocalLLMClient()
# ... register tools on other_client ...
client.register_tools_from(other_client.tools)
```

---

### `print_tool_calls(detailed=False)`

Print summary of tool calls from the last chat request.

#### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `detailed` | `bool` | `False` | Show full arguments and results (True) or compact summary (False) |

#### Example

```python
response = client.chat("Calculate 5 * 10", use_tools=True)
client.print_tool_calls()

# Output (compact):
# 🔧 Tool Execution Summary (1 call):
# ======================================================================
#   [1] math_calculator(arg1=5, arg2=10, operation=multiply) → result=50
# ======================================================================

client.print_tool_calls(detailed=True)

# Output (detailed):
# 🔧 Tool Execution Summary (1 call):
# ======================================================================
#
# [1] math_calculator
#     Arguments: {
#       "arg1": 5,
#       "arg2": 10,
#       "operation": "multiply"
#     }
#     Result: {
#       "result": 50
#     }
# ======================================================================
```

---

## Context Managers

### `conversation(name="conversation")`

Context manager for multi-turn conversations with unified tracing.

Groups multiple `chat()` calls under a single parent trace in MLflow for better observability. Useful for agent loops, chat sessions, or any multi-turn interaction.

#### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `name` | `str` | `"conversation"` | Name for the conversation trace |

#### Example

```python
with client.conversation("react-agent-task"):
    for i in range(10):
        response = client.chat(messages, use_tools=True)
        messages.append(response.choices[0].message)

        if "COMPLETE" in response.choices[0].message.content:
            break

# Creates MLflow hierarchy:
# conversation
# ├─ chat (iteration 1)
# │  ├─ send_request
# │  └─ handle_tool_calls
# ├─ chat (iteration 2)
# ...
```

---

## Properties

### `base_url` (read-only)

Base URL of the LLM API endpoint.

**Type:** `str`

**Example:** `"http://localhost:1234/v1"`

---

### `default_model` (read-only)

Default model identifier used for requests.

**Type:** `str` or `None`

**Example:** `"mistralai/magistral-small-2509"`

---

### `timeout` (read-only)

Request timeout in seconds.

**Type:** `int`

**Example:** `300`

---

### `last_tool_calls` (read-only)

List of tool calls from the most recent chat request.

**Type:** `List[ToolCall]`

**Example:**

```python
response = client.chat("Calculate 5!", use_tools=True)

for tool_call in client.last_tool_calls:
    print(f"Tool: {tool_call.function.name}")
    print(f"Args: {tool_call.function.arguments}")
```

---

### `last_thinking` (read-only)

Thinking blocks extracted from the most recent chat request (reasoning models).

**Type:** `str`

**Example:**

```python
response = client.chat("Explain quantum computing")

if client.last_thinking:
    print("Model's reasoning:")
    print(client.last_thinking)
```

---

### `last_conversation_additions` (read-only)

Messages added during the last chat request (tool calls, tool results, final response).

**Type:** `List[ChatMessage]`

**Example:**

```python
response = client.chat("Calculate 5 * 10", use_tools=True)

print(f"Added {len(client.last_conversation_additions)} messages:")
for msg in client.last_conversation_additions:
    print(f"  - {msg.role}: {msg.content[:50]}...")

# Output:
# Added 3 messages:
#   - assistant: (tool call)
#   - tool: {"result": 50}
#   - assistant: The result of 5 * 10 is 50.
```

---

### `tools` (read-only)

Tool registry instance managing registered tools.

**Type:** `ToolRegistry`

**Example:**

```python
print(f"Registered tools: {client.tools.list_tools()}")
# Output: ['math_calculator', 'python_exec', 'text_transformer', ...]
```

---

### `endpoints` (read-only)

Dictionary of API endpoint URLs.

**Type:** `Dict[str, str]`

**Keys:** `"models"`, `"chat"`, `"completions"`, `"embeddings"`

**Example:**

```python
print(client.endpoints["chat"])
# Output: "http://localhost:1234/v1/chat/completions"
```

---

## Error Handling

### Common Exceptions

| Exception | When Raised | Handling |
|-----------|-------------|----------|
| `ValueError` | No model specified or found | Provide explicit `model` parameter or ensure LLM server has models loaded |
| `ConnectionError` | Cannot connect to LLM server | Verify server is running and `base_url` is correct |
| `Timeout` | Request exceeds timeout | Increase `timeout` parameter (default 300s may be insufficient) |
| `requests.HTTPError` | HTTP error from server (4xx/5xx) | Check server logs, model availability, or request parameters |
| `pydantic.ValidationError` | Invalid response format | May indicate server incompatibility or malformed response |

### Error Handling Patterns

```python
from requests.exceptions import ConnectionError, Timeout
from pydantic import ValidationError

try:
    response = client.chat("Hello", timeout=60)

except ValueError as e:
    print(f"Configuration error: {e}")
    # Fix: Provide model parameter or load model in LM Studio

except ConnectionError:
    print("Cannot connect to LLM server")
    # Fix: Start LM Studio or check base_url

except Timeout:
    print("Request timed out")
    # Fix: Increase timeout or simplify query

except ValidationError as e:
    print(f"Invalid response format: {e}")
    # Fix: Check server compatibility or update SDK

except Exception as e:
    print(f"Unexpected error: {e}")
    # Fix: Check server logs
```

---

## Performance Tips

### 1. Timeout Configuration

Local models are slower than cloud APIs. Adjust timeout based on task complexity:

```python
# Simple queries
client = LocalLLMClient(timeout=60)  # 1 minute

# Complex tasks with tools
client = LocalLLMClient(timeout=300)  # 5 minutes (default)

# Very complex agent tasks
client = LocalLLMClient(timeout=600)  # 10 minutes
```

### 2. Tool Choice Optimization

Use `tool_choice` strategically:

```python
# For reasoning models: force tools for simple math
response = client.chat(
    "What's 2+2?",
    use_tools=True,
    tool_choice="required"  # Faster - skips reasoning
)

# For complex tasks: let model decide
response = client.chat(
    "Analyze this data and compute statistics",
    use_tools=True,
    tool_choice="auto"  # Model chooses when to use tools
)
```

### 3. Reuse Client Instances

Creating a client has minimal overhead, but reusing is more efficient:

```python
# Good - reuse client
client = LocalLLMClient()
client.register_tools_from(None)

for query in queries:
    response = client.chat(query, use_tools=True)

# Avoid - creating client per query
for query in queries:
    client = LocalLLMClient()
    client.register_tools_from(None)
    response = client.chat(query, use_tools=True)
```

### 4. Temperature Tuning

Lower temperature for deterministic tasks, higher for creative tasks:

```python
# Deterministic (code, math, factual)
response = client.chat("Calculate 5!", temperature=0.0)

# Balanced (default)
response = client.chat("Explain AI", temperature=0.7)

# Creative (writing, brainstorming)
response = client.chat("Write a poem", temperature=1.2)
```

### 5. Conversation State Management

The client preserves conversation state automatically when tools are used. Access via `last_conversation_additions`:

```python
# First request
response = client.chat("Calculate 5!", use_tools=True)

# Access full conversation state
for msg in client.last_conversation_additions:
    print(f"{msg.role}: {msg.content}")

# Output:
# assistant: (tool call: math_calculator)
# tool: {"result": 120}
# assistant: The factorial of 5 is 120.
```

---

## Advanced Usage

### Custom System Prompts

```python
# Per-request system prompt
response = client.chat(
    "Translate 'hello' to French",
    system="You are a professional French translator with 20 years experience."
)

# Structured messages for fine control
messages = [
    create_chat_message("system", "Custom system instructions here"),
    create_chat_message("user", "User query")
]
response = client.chat(messages)
```

### MLflow Tracing Integration

```python
import mlflow

# Enable MLflow tracking (if installed)
mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("llm-experiments")

# All chat() calls automatically traced
with mlflow.start_run():
    response = client.chat("Calculate 5!", use_tools=True)

# View traces in MLflow UI:
# - Hierarchical spans (chat → tool calls)
# - Input/output capture
# - Timing metrics
```

### Thinking Block Extraction

Reasoning models often output thinking in `[THINK]...[/THINK]` blocks. The client automatically extracts and separates this:

```python
response = client.chat("Solve this complex problem", include_thinking=False)
print(response)  # Clean response without thinking

# Access thinking separately
print(client.last_thinking)  # Model's reasoning process

# Or include in response
response = client.chat("Solve this problem", include_thinking=True)
# Output: "**Thinking:**\n...\n\n**Response:**\n..."
```

### Text-Based Tool Calls

Some models output tool calls as text rather than structured JSON. The client automatically parses these:

```python
# Model outputs: "I'll use [TOOL_CALLS]calculator[ARGS]{\"a\": 5, \"b\": 10}"
response = client.chat("Calculate 5 + 10", use_tools=True)

# Client automatically:
# 1. Detects text-based tool call
# 2. Parses into ToolCall object
# 3. Executes tool
# 4. Returns result

# Works transparently with print_tool_calls()
client.print_tool_calls()
```

### Connection Retry Logic

The client automatically retries on connection failures with exponential backoff:

```python
# Automatically retries 3 times (1s, 2s, 4s delays)
try:
    response = client.chat("Hello")
except ConnectionError:
    print("Failed after 3 retry attempts")

# Output during retries:
# ⚠ Connection failed, retrying in 1s... (attempt 1/3)
# ⚠ Connection failed, retrying in 2s... (attempt 2/3)
# ✗ Connection failed after 3 attempts
```

---

## Complete Examples

### Example 1: Simple Chat

```python
from local_llm_sdk import LocalLLMClient

client = LocalLLMClient()
response = client.chat("What is Python?")
print(response)
```

### Example 2: Chat with Tools

```python
from local_llm_sdk import LocalLLMClient

client = LocalLLMClient()
client.register_tools_from(None)  # Load built-in tools

response = client.chat("Calculate 42 * 17", use_tools=True)
print(response)

client.print_tool_calls()
```

### Example 3: Custom Tools

```python
from local_llm_sdk import LocalLLMClient

client = LocalLLMClient()

@client.register_tool("Get current weather for a city")
def get_weather(city: str) -> dict:
    """
    Get weather information.

    Args:
        city: City name
    """
    # In production, call real API
    return {
        "city": city,
        "temperature": 72,
        "condition": "sunny"
    }

response = client.chat("What's the weather in San Francisco?", use_tools=True)
print(response)
```

### Example 4: Multi-turn Conversation

```python
from local_llm_sdk import LocalLLMClient, create_chat_message

client = LocalLLMClient()

history = [create_chat_message("system", "You are a helpful assistant.")]

response, history = client.chat_with_history("Hello!", history)
print(response)

response, history = client.chat_with_history("What's 2+2?", history)
print(response)

print(f"Conversation has {len(history)} messages")
```

### Example 5: ReACT Agent

```python
from local_llm_sdk import LocalLLMClient

client = LocalLLMClient()
client.register_tools_from(None)

result = client.react(
    task="Calculate 5 factorial, convert to uppercase, count characters",
    max_iterations=15,
    verbose=True
)

if result.success:
    print(f"Completed in {result.iterations} iterations")
    print(f"Tool calls: {result.metadata['total_tool_calls']}")
    print(f"Result: {result.final_response}")
```

### Example 6: Embeddings

```python
from local_llm_sdk import LocalLLMClient

client = LocalLLMClient(model="text-embedding-model")

texts = [
    "The quick brown fox",
    "jumps over the lazy dog"
]

response = client.embeddings(texts)

for i, data in enumerate(response.data):
    print(f"Text {i}: {len(data.embedding)} dimensions")
```

---

## See Also

- [Tool System Reference](tools.md) - Detailed tool registration and execution
- [Agent Framework Reference](agents.md) - ReACT and other agent patterns
- [Models Reference](models.md) - Pydantic models and type system
- [Configuration Guide](../guides/configuration.md) - Environment variables and advanced config
