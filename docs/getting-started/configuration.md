# Configuration

Learn how to configure the Local LLM SDK using environment variables, code, or configuration files.

## Configuration Methods

The SDK supports three configuration methods (in order of precedence):

1. **Code parameters** (highest priority)
2. **Environment variables**
3. **Default values** (lowest priority)

## Environment Variables

### Core Variables

```bash
# LLM Server Configuration
export LLM_BASE_URL="http://localhost:1234/v1"    # Server URL
export LLM_MODEL="auto"                            # Model name
export LLM_TIMEOUT="300"                           # Request timeout (seconds)
export LLM_DEBUG="false"                           # Enable debug logging

# Streaming and Validation (Advanced)
export LLM_STREAM="false"                          # Enable streaming responses (SSE format)
export LLM_ENABLE_VALIDATION="false"               # Enable early termination validation
export LLM_VALIDATION_CHECK_INTERVAL="20"          # Validation frequency (tokens)

# Tool Execution
export LLM_MAX_TOOL_ITERATIONS="10"                # Maximum tool execution loops
```

### Example Configurations

**LM Studio (Local Network)**
```bash
export LLM_BASE_URL="http://169.254.83.107:1234/v1"
export LLM_MODEL="mistralai/magistral-small-2509"
export LLM_TIMEOUT="300"
```

**Ollama (Local)**
```bash
export LLM_BASE_URL="http://localhost:11434/v1"
export LLM_MODEL="mistral"
export LLM_TIMEOUT="300"
```

**LocalAI (Custom Port)**
```bash
export LLM_BASE_URL="http://localhost:8080/v1"
export LLM_MODEL="gpt-3.5-turbo"  # LocalAI model alias
export LLM_TIMEOUT="180"
```

## Code Configuration

### Basic Client Configuration

```python
from local_llm_sdk import LocalLLMClient

client = LocalLLMClient(
    base_url="http://localhost:1234/v1",
    model="mistralai/magistral-small-2509",
    timeout=300,
    api_key="lm-studio"  # Optional, default for LM Studio
)
```

### Using Default Configuration

```python
from local_llm_sdk.config import get_default_config

# Load from environment variables
config = get_default_config()
print(config)
# {
#     'base_url': 'http://localhost:1234/v1',
#     'model': 'auto',
#     'timeout': 300,
#     'debug': False
# }

# Create client from config
client = LocalLLMClient(**config)
```

### Convenience Functions

```python
from local_llm_sdk import create_client, create_client_with_tools

# Simple client (no tools)
client = create_client()

# Client with built-in tools pre-loaded
client = create_client_with_tools(
    base_url="http://localhost:1234/v1",
    model="auto"
)
```

## Streaming and Validation

### Overview

The SDK supports **streaming responses** with **real-time validation** for early termination of problematic generations. This is an advanced feature that can save significant time when models produce bad outputs.

### Streaming (LLM_STREAM)

**Description**: Enable Server-Sent Events (SSE) streaming for incremental response generation

**Default**: `false` (disabled for backward compatibility)

**When to Enable**:
- You want to see responses as they're generated
- You're using validation for early termination
- You need to process partial responses

**Example**:
```python
import os
os.environ['LLM_STREAM'] = 'true'

from local_llm_sdk import LocalLLMClient
client = LocalLLMClient()

# Responses are now streamed chunk-by-chunk
response = client.chat("Tell me a story")
```

**Note**: Streaming is automatically used when `LLM_ENABLE_VALIDATION=true` regardless of this setting.

### Validation (LLM_ENABLE_VALIDATION)

**Description**: Enable real-time validation during generation with early termination

**Default**: `false` (disabled for backward compatibility)

**What it Does**:
- Detects **XML_DRIFT**: Model outputs XML instead of JSON (format mismatch)
- Detects **REPETITION**: Model stuck in repetition loop ("the the the...")
- **Stops generation immediately** when issues detected (saves 70-80% time)

**When to Enable**:
- Testing new models for compatibility
- Preventing runaway generations (37,000 token loops)
- Production environments needing reliability

**Example**:
```python
import os
os.environ['LLM_ENABLE_VALIDATION'] = 'true'
os.environ['LLM_VALIDATION_CHECK_INTERVAL'] = '20'  # Check every 20 tokens

from local_llm_sdk import LocalLLMClient
client = LocalLLMClient()

try:
    response = client.chat("Repeat 'test' 100 times")
except ValueError as e:
    # Validation caught REPETITION!
    # Early termination after ~20 chunks instead of 100+
    print(f"Caught error: {e}")
```

**Validation Results**:
```
🚨 VALIDATION ERROR: REPETITION
Model: mistralai/magistral-small-2509
Details: Detected repeating pattern: test test test test test...
Response preview: test test test test test...

💡 TIP: This is EARLY TERMINATION during streaming (stopped after 21 chunks)
```

### Validation Check Interval (LLM_VALIDATION_CHECK_INTERVAL)

**Description**: How often to check for validation issues (in tokens)

**Default**: `20` tokens

**Tuning**:
- Lower (5-10): Faster detection, higher overhead
- Default (20): Balanced
- Higher (50-100): Slower detection, lower overhead

### Max Tool Iterations (LLM_MAX_TOOL_ITERATIONS)

**Description**: Maximum number of tool execution loops before stopping

**Default**: `10` iterations

**Why it Matters**: Prevents infinite loops when model repeatedly calls tools without completing task

**Example**:
```python
import os
os.environ['LLM_MAX_TOOL_ITERATIONS'] = '5'  # Stricter limit

from local_llm_sdk import LocalLLMClient
client = LocalLLMClient()
client.register_tools_from(None)

# Will stop after 5 tool iterations even if task incomplete
response = client.chat("Complex multi-step task", use_tools=True)
```

---

## Configuration Options

### Base URL

**Description**: URL of your local LLM server

**Format**: `http://host:port/v1`

**Examples**:
- `http://localhost:1234/v1` (LM Studio, local)
- `http://192.168.1.100:1234/v1` (LM Studio, network)
- `http://localhost:11434/v1` (Ollama)
- `http://localhost:8080/v1` (LocalAI)

**Default**: `http://localhost:1234/v1`

### Model

**Description**: Name of the model to use

**Options**:
- `"auto"` - Auto-detect first available model (recommended)
- `"model-name"` - Specific model ID (e.g., "mistralai/magistral-small-2509")

**Example**:
```python
# Auto-detect
client = LocalLLMClient(model="auto")

# Specific model
client = LocalLLMClient(model="mistralai/magistral-small-2509")

# List available models
models = client.list_models()
for model in models.data:
    print(model.id)
```

**Default**: `"auto"`

### Timeout

**Description**: Maximum time to wait for LLM response (in seconds)

**Recommended Values**:
- `60` - Fast responses only
- `300` - Default, balanced (5 minutes)
- `600` - Long tasks (10 minutes)
- `1800` - Very long tasks (30 minutes)

**Example**:
```python
# Short timeout for quick responses
client = LocalLLMClient(timeout=60)

# Long timeout for complex tasks
agent_client = LocalLLMClient(timeout=600)
```

**Default**: `300` (5 minutes)

**Why 300s?** Local models are slower than cloud APIs. 300s provides balance between responsiveness and allowing complex tasks to complete.

### API Key

**Description**: Authentication key for the LLM server

**Note**: Most local servers don't require authentication. LM Studio uses a fixed key.

**Example**:
```python
# LM Studio (default)
client = LocalLLMClient(api_key="lm-studio")

# No auth needed
client = LocalLLMClient(api_key=None)

# Custom auth
client = LocalLLMClient(api_key="your-custom-key")
```

**Default**: `"lm-studio"`

### Debug Mode

**Description**: Enable verbose logging for troubleshooting

**Example**:
```python
import os
os.environ['LLM_DEBUG'] = 'true'

from local_llm_sdk import LocalLLMClient
client = LocalLLMClient()

# Now all requests/responses will be logged
response = client.chat("Hello")
```

**Default**: `false`

## Advanced Configuration

### Per-Request Configuration

Override settings for individual requests:

```python
from local_llm_sdk import create_chat_completion_request

# Custom request parameters
request = create_chat_completion_request(
    messages=[{"role": "user", "content": "Be creative!"}],
    temperature=0.9,           # Higher = more random (0.0 - 1.0)
    max_tokens=1000,           # Maximum response length
    top_p=0.95,                # Nucleus sampling
    frequency_penalty=0.5,     # Reduce repetition
    presence_penalty=0.5       # Encourage topic diversity
)

response = client.chat(request)
```

### Multiple Clients

Use different configurations for different purposes:

```python
# Fast client for simple queries
quick_client = LocalLLMClient(
    model="mistral-7b",
    timeout=60
)

# Powerful client for complex tasks
advanced_client = LocalLLMClient(
    model="mistralai/magistral-small-2509",
    timeout=600
)

# Use appropriate client for task
simple_answer = quick_client.chat("What is 2+2?")
complex_answer = advanced_client.chat("Analyze this dataset...")
```

### Tool Configuration

```python
from local_llm_sdk import create_client_with_tools
from local_llm_sdk.tools import builtin

# Client with all built-in tools
client = create_client_with_tools()

# Client with custom tools only
client = LocalLLMClient()
client.register_tool("My tool")(my_custom_tool)

# Mix of built-in and custom
client = create_client_with_tools()
client.register_tool("Custom tool")(my_tool)
```

## Configuration Files

### .env File (Recommended for Development)

Create a `.env` file in your project root:

```bash
# .env
LLM_BASE_URL=http://localhost:1234/v1
LLM_MODEL=mistralai/magistral-small-2509
LLM_TIMEOUT=300
LLM_DEBUG=false
```

Load with `python-dotenv`:

```python
from dotenv import load_dotenv
load_dotenv()  # Loads .env file

from local_llm_sdk import create_client
client = create_client()  # Uses .env values
```

### Config File (Python)

Create `config.py`:

```python
# config.py
LLM_CONFIG = {
    "base_url": "http://localhost:1234/v1",
    "model": "mistralai/magistral-small-2509",
    "timeout": 300
}
```

Use in your code:

```python
from config import LLM_CONFIG
from local_llm_sdk import LocalLLMClient

client = LocalLLMClient(**LLM_CONFIG)
```

## Platform-Specific Configuration

### Windows (LM Studio)

```bash
# PowerShell
$env:LLM_BASE_URL="http://localhost:1234/v1"
$env:LLM_MODEL="mistralai/magistral-small-2509"

# Command Prompt
set LLM_BASE_URL=http://localhost:1234/v1
set LLM_MODEL=mistralai/magistral-small-2509
```

### macOS/Linux (LM Studio or Ollama)

```bash
# .bashrc or .zshrc
export LLM_BASE_URL="http://localhost:1234/v1"
export LLM_MODEL="mistralai/magistral-small-2509"
export LLM_TIMEOUT="300"
```

### Docker

```yaml
# docker-compose.yml
version: '3.8'
services:
  app:
    build: .
    environment:
      - LLM_BASE_URL=http://ollama:11434/v1
      - LLM_MODEL=mistral
      - LLM_TIMEOUT=300
  ollama:
    image: ollama/ollama
    ports:
      - "11434:11434"
```

## Troubleshooting Configuration

### Check Current Configuration

```python
from local_llm_sdk import LocalLLMClient

client = LocalLLMClient()

print(f"Base URL: {client.base_url}")
print(f"Model: {client.model}")
print(f"Timeout: {client.timeout}s")
```

### Test Connection

```python
import requests

base_url = "http://localhost:1234/v1"

try:
    response = requests.get(f"{base_url}/models", timeout=5)
    print("✓ Server reachable")
    print(f"Available models: {[m['id'] for m in response.json()['data']]}")
except requests.exceptions.ConnectionError:
    print("✗ Cannot connect to server")
except requests.exceptions.Timeout:
    print("✗ Server timeout")
```

### Common Issues

**Issue: "Connection refused"**
- Check `LLM_BASE_URL` is correct
- Verify server is running
- Check firewall settings

**Issue: "Model not found"**
- Use `model="auto"` to auto-detect
- List available models with `client.list_models()`
- Verify model is loaded in LM Studio

**Issue: "Timeout"**
- Increase `LLM_TIMEOUT` value
- Check server isn't overloaded
- Try a faster/smaller model

## Next Steps

- **[Basic Usage Guide](basic-usage.md)** - Learn core concepts
- **[API Reference](../api-reference/client.md)** - Full API documentation
- **[Production Patterns](../guides/production-patterns.md)** - Best practices
