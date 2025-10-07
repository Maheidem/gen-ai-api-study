# Tools API Reference

The Local LLM SDK provides a powerful tool system that enables LLMs to execute functions and interact with external systems. This reference documents the complete Tools API, including the registry system, decorator usage, the unified bash tool, and custom tool creation.

## Table of Contents

- [Overview](#overview)
- [Quick Start](#quick-start)
- [ToolRegistry Class](#toolregistry-class)
- [@tool Decorator](#tool-decorator)
- [Unified Bash Tool](#unified-bash-tool)
- [Creating Custom Tools](#creating-custom-tools)
- [Error Handling](#error-handling)
- [Advanced Usage](#advanced-usage)

---

## Overview

The tool system follows the OpenAI function calling specification and provides:

- **Automatic schema generation** from Python type hints
- **Type-safe tool execution** with Pydantic validation
- **Unified bash tool** for terminal operations (Python, files, git, text processing, math)
- **Simple registration** using the `@tool` decorator
- **Error handling** with graceful degradation

### Architecture

```
┌─────────────────────────────────────────────────────────────┐
│ 1. Client adds tool schemas to chat request                 │
│    (via registry.get_schemas())                              │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│ 2. LLM analyzes task and returns tool_calls                 │
│    (OpenAI function calling format)                          │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│ 3. Client executes tools via registry.execute()             │
│    (automatic parameter validation and error handling)       │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│ 4. Results added as messages and sent back to LLM           │
│    (process repeats until task complete)                     │
└─────────────────────────────────────────────────────────────┘
```

---

## Quick Start

### Using the Unified Bash Tool

```python
from local_llm_sdk import create_client_with_tools

# Client automatically includes the bash tool
client = create_client_with_tools()

# Tools are called automatically when use_tools=True
response = client.chat("Calculate 42 * 17", use_tools=True)
print(response)  # "The result is 714"

# See which tools were used
client.print_tool_calls()
# Output:
# 🔧 Tool Execution Summary (1 call):
# ======================================================================
#   [1] bash(command="python -c 'print(42 * 17)'") → captured_result=714
# ======================================================================
```

### Registering Custom Tools

```python
from local_llm_sdk import LocalLLMClient, tool

# Define a custom tool
@tool("Calculate factorial of a number")
def factorial(n: int) -> dict:
    """Calculate n! iteratively."""
    if n <= 1:
        return {"result": 1}
    result = 1
    for i in range(2, n + 1):
        result *= i
    return {"result": result}

# Create client and register tool
client = LocalLLMClient()
client.register_tool("Calculate factorial")(factorial)

# Use the tool
response = client.chat("What is 5 factorial?", use_tools=True)
print(response)  # "5 factorial is 120"
```

---

## ToolRegistry Class

The `ToolRegistry` manages tool registration, schema generation, and execution.

### Constructor

```python
from local_llm_sdk.tools import ToolRegistry

registry = ToolRegistry()
```

**Description**: Creates a new tool registry instance with no tools registered.

### Methods

#### register(description: str = "") -> Callable

Decorator to register a function as a tool.

**Parameters:**
- `description` (str, optional): Human-readable description of what the tool does. Used by the LLM to understand when to use this tool.

**Returns:**
- Decorator function that registers the tool and returns the original function unchanged.

**Example:**

```python
from local_llm_sdk.tools import ToolRegistry

registry = ToolRegistry()

@registry.register("Add two numbers together")
def add(a: float, b: float) -> dict:
    """Add two numbers and return the sum."""
    return {"result": a + b}

# Tool is now registered and callable
```

#### get_schemas() -> List[Tool]

Get all registered tools as Pydantic `Tool` objects.

**Returns:**
- `List[Tool]`: List of Tool objects following OpenAI function calling specification.

#### execute(tool_name: str, arguments: Dict[str, Any]) -> str

Execute a registered tool with the given arguments.

**Parameters:**
- `tool_name` (str): Name of the tool to execute (must match function name)
- `arguments` (Dict[str, Any]): Arguments to pass to the tool function

**Returns:**
- `str`: JSON string containing the tool execution result or error

#### list_tools() -> List[str]

Get a list of all registered tool names.

**Returns:**
- `List[str]`: List of tool names (function names)

---

## @tool Decorator

The `@tool` decorator is a convenience wrapper around `ToolRegistry.register()` that uses a global registry.

### Syntax

```python
from local_llm_sdk.tools import tool

@tool(description: str = "")
def your_function(...) -> dict:
    """Function implementation"""
    pass
```

### Parameters

- `description` (str, optional): Human-readable description for the LLM

### Return Type

**IMPORTANT**: All tool functions must return a `dict` (JSON-serializable).

### Type Hints

The decorator automatically generates OpenAI-compatible schemas from Python type hints:

| Python Type | JSON Schema Type | Notes |
|-------------|------------------|-------|
| `int` | `"integer"` | Whole numbers |
| `float` | `"number"` | Decimals |
| `str` | `"string"` | Text |
| `bool` | `"boolean"` | True/False |
| `list`, `List` | `"array"` | Lists/arrays |
| `dict`, `Dict` | `"object"` | Objects/dicts |
| `Literal["a", "b"]` | `enum: ["a", "b"]` | Restricted choices |

---

## Unified Bash Tool

**The SDK includes ONE powerful tool that handles ALL terminal operations: the unified `bash` tool.**

### Why One Tool?

**Design Philosophy**:
- ✅ **Single interface** - No tool coordination issues
- ✅ **Full terminal capabilities** - Everything bash can do
- ✅ **Works in project directory** - Direct access to your files
- ✅ **Comprehensive LLM instructions** - Teaches best practices
- ✅ **300s timeout** - Suitable for local LLMs

**vs. Multiple Specialized Tools**:
- ❌ LLM confusion about which tool to use
- ❌ Artificial limitations (can't combine operations)
- ❌ More code to maintain
- ❌ Tool coordination overhead

### Bash Tool Capabilities

The bash tool can execute **ANY** terminal command:

#### 📊 Mathematics & Calculations

```python
client = create_client_with_tools()

response = client.chat("Calculate 42 * 17", use_tools=True)
# LLM calls: bash(command="python -c 'print(42 * 17)'")
# Returns: {"success": true, "stdout": "714\n", "captured_result": "714"}

response = client.chat("What is 5 factorial?", use_tools=True)
# LLM calls: bash(command="python -c 'import math; print(math.factorial(5))'")
# Returns: {"success": true, "stdout": "120\n", "captured_result": "120"}
```

#### 📁 File Operations

```python
# Write file
response = client.chat("Write 'Hello World' to test.txt", use_tools=True)
# LLM calls: bash(command="echo 'Hello World' > test.txt")

# Read file
response = client.chat("Read contents of test.txt", use_tools=True)
# LLM calls: bash(command="cat test.txt")

# Copy/Move/Delete
response = client.chat("Copy test.txt to backup.txt", use_tools=True)
# LLM calls: bash(command="cp test.txt backup.txt")
```

#### 📝 Text Processing

```python
# Count characters
response = client.chat("Count characters in 'hello world'", use_tools=True)
# LLM calls: bash(command="echo 'hello world' | wc -c")

# Convert to uppercase
response = client.chat("Convert 'hello' to uppercase", use_tools=True)
# LLM calls: bash(command="echo 'hello' | tr '[:lower:]' '[:upper:]'")

# Search text
response = client.chat("Search for 'error' in log.txt", use_tools=True)
# LLM calls: bash(command="grep 'error' log.txt")
```

#### 🐍 Python Execution

```python
# Inline Python
response = client.chat("Calculate sum of [1,2,3,4,5]", use_tools=True)
# LLM calls: bash(command="python -c 'print(sum([1,2,3,4,5]))'")

# Run Python script
response = client.chat("Run analysis.py", use_tools=True)
# LLM calls: bash(command="python analysis.py")
```

#### 📂 Directory Operations

```python
# List files
response = client.chat("List all Python files", use_tools=True)
# LLM calls: bash(command="ls *.py")

# Create directory
response = client.chat("Create output/results directory", use_tools=True)
# LLM calls: bash(command="mkdir -p output/results")
```

#### 🔗 Command Chaining

```python
# Multi-step operations
response = client.chat(
    "Create temp directory, write 'data' to temp/file.txt, then read it",
    use_tools=True
)
# LLM calls: bash(command="mkdir -p temp && echo 'data' > temp/file.txt && cat temp/file.txt")
```

### Bash Tool Signature

```python
def bash(command: str, timeout: int = 300) -> dict:
    """
    Execute bash commands with full terminal capabilities.

    Args:
        command: The bash command to execute
        timeout: Maximum execution time in seconds (default: 300)

    Returns:
        dict with keys:
        - success (bool): True if return_code == 0
        - stdout (str): Standard output from command
        - stderr (str): Standard error from command
        - return_code (int): Exit code (0 = success)
        - command (str): The command that was executed
        - captured_result (str, optional): Extracted result from stdout
    """
```

### Response Format

**Success Example:**
```json
{
  "success": true,
  "stdout": "714\n",
  "stderr": "",
  "return_code": 0,
  "command": "python -c 'print(42 * 17)'",
  "captured_result": "714"
}
```

**Error Example:**
```json
{
  "success": false,
  "stdout": "",
  "stderr": "python: command not found",
  "return_code": 127,
  "command": "python script.py"
}
```

### Security Considerations

- Commands run in current working directory (project context)
- Full bash capabilities (use with trusted inputs)
- 300-second timeout prevents runaway processes
- No sandboxing (for local development)

**For Production**:
- Validate/sanitize user inputs
- Consider command whitelist
- Use lower timeouts (30-60s)
- Run in containerized environment

---

## Creating Custom Tools

While the bash tool handles most operations, you can create specialized tools for:
- Complex business logic
- External API integrations
- Stateful operations
- Custom validation requirements

### Design Principles

1. **Return dictionaries**: Always return `dict` (JSON-serializable)
2. **Handle errors gracefully**: Return error dict instead of raising exceptions
3. **Use type hints**: Enable automatic schema generation
4. **Write clear descriptions**: Help the LLM understand when to use your tool
5. **Document parameters**: Use docstrings to explain arguments
6. **Keep it simple**: Each tool should do one thing well

### Example: Simple Custom Tool

```python
from local_llm_sdk.tools import tool

@tool("Reverse a string")
def reverse_string(text: str) -> dict:
    """
    Reverse the characters in a string.

    Args:
        text: The string to reverse

    Returns:
        Dictionary with original and reversed text
    """
    return {
        "original": text,
        "reversed": text[::-1],
        "length": len(text)
    }
```

### Example: Tool with API Integration

```python
from local_llm_sdk.tools import tool

@tool("Get current Bitcoin price in USD")
def get_bitcoin_price() -> dict:
    """
    Fetch the current Bitcoin price from a cryptocurrency API.

    Returns:
        Dictionary with current BTC price and timestamp
    """
    import requests
    from datetime import datetime

    try:
        response = requests.get(
            "https://api.coinbase.com/v2/prices/BTC-USD/spot",
            timeout=10
        )
        response.raise_for_status()

        data = response.json()
        price = float(data["data"]["amount"])

        return {
            "success": True,
            "price_usd": price,
            "currency": "USD",
            "timestamp": datetime.utcnow().isoformat()
        }

    except requests.RequestException as e:
        return {
            "success": False,
            "error": f"API request failed: {str(e)}"
        }
```

### Example: Tool with Validation

```python
from local_llm_sdk.tools import tool

@tool("Validate email address")
def validate_email(email: str) -> dict:
    """
    Check if an email address is valid.

    Args:
        email: Email address to validate

    Returns:
        Dictionary with validation result and details
    """
    import re

    # Simple email regex pattern
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    is_valid = bool(re.match(pattern, email))

    return {
        "email": email,
        "is_valid": is_valid,
        "has_at_sign": "@" in email,
        "has_domain": "." in email.split("@")[-1] if "@" in email else False
    }
```

---

## Error Handling

### Tool Execution Errors

Tools should **never raise exceptions**. Instead, return an error dictionary:

```python
@tool("Divide two numbers")
def safe_divide(a: float, b: float) -> dict:
    """Safely divide two numbers."""

    if b == 0:
        # DON'T raise ZeroDivisionError
        # DO return error dict
        return {
            "success": False,
            "error": "Division by zero is undefined",
            "suggestion": "Check that denominator is not zero"
        }

    return {
        "success": True,
        "result": a / b
    }
```

### Best Practices

**Error Response Format:**
```python
{
    "success": False,
    "error": "Clear error message",
    "details": "Additional context (optional)",
    "suggestion": "How to fix it (optional)"
}
```

**Success Response Format:**
```python
{
    "success": True,
    "result": "Main result value",
    # Additional fields as needed
}
```

---

## Advanced Usage

### Multiple Tool Registries

Use separate registries for different contexts or permission levels:

```python
from local_llm_sdk.tools import ToolRegistry, tool

# Admin registry with powerful tools
admin_registry = ToolRegistry()

@admin_registry.register("Execute system command")
def admin_bash(command: str) -> dict:
    """Admin-only: Execute any system command."""
    import subprocess
    result = subprocess.run(command, shell=True, capture_output=True, text=True)
    return {"success": result.returncode == 0, "output": result.stdout}

# User registry with safe tools
user_registry = ToolRegistry()

@user_registry.register("Calculate sum")
def safe_sum(numbers: list) -> dict:
    """User-safe: Calculate sum of numbers."""
    return {"result": sum(numbers)}

# Use appropriate registry based on user role
def get_client_for_user(user_role: str):
    from local_llm_sdk import LocalLLMClient

    client = LocalLLMClient()

    if user_role == "admin":
        client.tools.copy_from(admin_registry)
    else:
        client.tools.copy_from(user_registry)

    return client
```

### Tool Debugging

```python
from local_llm_sdk import create_client_with_tools

client = create_client_with_tools()

# Make request with tools
response = client.chat("Calculate 5! then convert to uppercase", use_tools=True)

# Print compact summary
client.print_tool_calls()

# Print detailed JSON
client.print_tool_calls(detailed=True)

# Access raw tool calls
for tool_call in client.last_tool_calls:
    print(f"Tool: {tool_call.function.name}")
    print(f"Args: {tool_call.function.arguments}")
```

---

## Summary

The Local LLM SDK's tool system provides:

- **Unified bash tool** for all terminal operations (Python, files, git, text, math)
- **Simple registration** with `@tool` decorator
- **Automatic schema generation** from Python type hints
- **Type-safe execution** with Pydantic validation
- **Flexible architecture** supporting custom tools
- **Robust error handling** with graceful degradation

**Key Takeaways:**

1. Use the **bash tool** for most operations (math, files, text, Python execution)
2. Create **custom tools** for external APIs, business logic, or specialized validation
3. Tools must return `dict` (JSON-serializable)
4. Handle errors gracefully (return error dict, don't raise)
5. Write clear descriptions to help the LLM understand tool usage

**Next Steps:**
- Read [Client API Reference](./client.md) for tool integration with chat
- See [Agent API Reference](./agents.md) for using tools with agents
- Check `notebooks/04-tool-calling-basics.ipynb` for interactive examples
