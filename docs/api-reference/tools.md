# Tools API Reference

The Local LLM SDK provides a powerful tool system that enables LLMs to execute functions and interact with external systems. This reference documents the complete Tools API, including the registry system, decorator usage, built-in tools, and custom tool creation.

## Table of Contents

- [Overview](#overview)
- [Quick Start](#quick-start)
- [ToolRegistry Class](#toolregistry-class)
- [@tool Decorator](#tool-decorator)
- [Built-in Tools](#built-in-tools)
- [Creating Custom Tools](#creating-custom-tools)
- [Error Handling](#error-handling)
- [Advanced Usage](#advanced-usage)

---

## Overview

The tool system follows the OpenAI function calling specification and provides:

- **Automatic schema generation** from Python type hints
- **Type-safe tool execution** with Pydantic validation
- **Built-in tools** for common operations (Python execution, file I/O, math, text processing)
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

### Using Built-in Tools

```python
from local_llm_sdk import create_client_with_tools

# Client automatically includes all built-in tools
client = create_client_with_tools()

# Tools are called automatically when use_tools=True
response = client.chat("Calculate 42 * 17", use_tools=True)
print(response)  # "The result is 714"

# See which tools were used
client.print_tool_calls()
```

### Registering Custom Tools

```python
from local_llm_sdk import LocalLLMClient, tool

# Define a custom tool
@tool("Calculate factorial of a number")
def factorial(n: int) -> dict:
    """Calculate n! recursively."""
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

**Example:**

```python
schemas = registry.get_schemas()
for schema in schemas:
    print(f"Tool: {schema.function.name}")
    print(f"Description: {schema.function.description}")
    print(f"Parameters: {schema.function.parameters}")
```

#### get_schemas_dict() -> List[Dict[str, Any]]

Get all tool schemas as dictionaries (for JSON serialization).

**Returns:**
- `List[Dict[str, Any]]`: List of tool schemas as plain dictionaries.

**Example:**

```python
import json

schemas_dict = registry.get_schemas_dict()
print(json.dumps(schemas_dict, indent=2))

# Output:
# [
#   {
#     "type": "function",
#     "function": {
#       "name": "add",
#       "description": "Add two numbers together",
#       "parameters": {
#         "type": "object",
#         "properties": {
#           "a": {"type": "number"},
#           "b": {"type": "number"}
#         },
#         "required": ["a", "b"]
#       }
#     }
#   }
# ]
```

#### execute(tool_name: str, arguments: Dict[str, Any]) -> str

Execute a registered tool with the given arguments.

**Parameters:**
- `tool_name` (str): Name of the tool to execute (must match function name)
- `arguments` (Dict[str, Any]): Arguments to pass to the tool function

**Returns:**
- `str`: JSON string containing the tool execution result or error

**Behavior:**
- If tool not found: Returns JSON with error and suggestions
- If execution succeeds: Returns JSON with success=True and result
- If execution fails: Returns JSON with success=False and error message

**Example:**

```python
# Execute a tool
result_json = registry.execute("add", {"a": 5, "b": 3})
print(result_json)
# Output: '{"success": true, "result": 8}'

# Invalid tool name
result_json = registry.execute("unknown_tool", {})
print(result_json)
# Output: '{"error": "Unknown tool: unknown_tool", "available_tools": "add, ...", "suggestion": "Did you mean one of: add, ...?"}'
```

#### list_tools() -> List[str]

Get a list of all registered tool names.

**Returns:**
- `List[str]`: List of tool names (function names)

**Example:**

```python
tool_names = registry.list_tools()
print(f"Available tools: {', '.join(tool_names)}")
# Output: "Available tools: add, multiply, subtract"
```

#### copy_from(other_registry: ToolRegistry) -> None

Copy all tools from another registry to this one.

**Parameters:**
- `other_registry` (ToolRegistry): The registry to copy tools from

**Example:**

```python
# Create two registries
registry1 = ToolRegistry()
registry2 = ToolRegistry()

# Register tools in registry1
@registry1.register("Add numbers")
def add(a: float, b: float) -> dict:
    return {"result": a + b}

# Copy to registry2
registry2.copy_from(registry1)

# registry2 now has the 'add' tool
print(registry2.list_tools())  # ['add']
```

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

### Example: Basic Tool

```python
from local_llm_sdk.tools import tool

@tool("Multiply two numbers")
def multiply(a: float, b: float) -> dict:
    """
    Multiply two numbers together.

    Args:
        a: First number
        b: Second number

    Returns:
        Dictionary with multiplication result
    """
    return {"result": a * b}
```

### Example: Tool with Literal Types

```python
from typing import Literal
from local_llm_sdk.tools import tool

@tool("Convert temperature between units")
def convert_temperature(
    value: float,
    from_unit: Literal["celsius", "fahrenheit", "kelvin"],
    to_unit: Literal["celsius", "fahrenheit", "kelvin"]
) -> dict:
    """Convert temperature between different units."""

    # Convert to Celsius first
    if from_unit == "fahrenheit":
        celsius = (value - 32) * 5/9
    elif from_unit == "kelvin":
        celsius = value - 273.15
    else:
        celsius = value

    # Convert from Celsius to target unit
    if to_unit == "fahrenheit":
        result = celsius * 9/5 + 32
    elif to_unit == "kelvin":
        result = celsius + 273.15
    else:
        result = celsius

    return {
        "original_value": value,
        "original_unit": from_unit,
        "converted_value": result,
        "converted_unit": to_unit
    }
```

### Example: Tool with Optional Parameters

```python
from local_llm_sdk.tools import tool

@tool("Search for files matching a pattern")
def find_files(pattern: str, max_results: int = 10) -> dict:
    """
    Search for files matching a pattern.

    Args:
        pattern: Glob pattern to match (e.g., "*.py")
        max_results: Maximum number of results to return (default: 10)
    """
    import glob

    matches = glob.glob(pattern, recursive=True)[:max_results]

    return {
        "pattern": pattern,
        "matches": matches,
        "count": len(matches),
        "truncated": len(glob.glob(pattern, recursive=True)) > max_results
    }
```

---

## Built-in Tools

The SDK includes six production-ready tools for common operations.

### 1. execute_python

Execute Python code safely in an isolated subprocess.

**Signature:**
```python
def execute_python(code: str, timeout: int = 30) -> dict:
    """Execute Python code in a subprocess and return the results."""
```

**Parameters:**
- `code` (str): Python code to execute
- `timeout` (int, optional): Execution timeout in seconds (default: 30)

**Returns:**
```python
{
    "success": bool,           # True if execution succeeded
    "stdout": str,             # Standard output
    "stderr": str,             # Standard error
    "status": str,             # "success", "error", or "timeout"
    "working_directory": str,  # Temp directory used for execution
    "return_code": int,        # Process exit code
    "captured_result": str     # Result variable if found (optional)
}
```

**Features:**
- Runs in isolated temporary directory
- Captures stdout, stderr, and result variables
- Automatically detects result from common variable names (`result`, `answer`, `output`)
- 30-second default timeout to prevent infinite loops
- Proper cleanup of temporary files

**Example:**

```python
from local_llm_sdk import create_client_with_tools

client = create_client_with_tools()

response = client.chat("Calculate 5 factorial using Python", use_tools=True)
print(response)
# LLM will use execute_python tool:
# code: "result = 1\nfor i in range(1, 6):\n    result *= i\nprint(f'5! = {result}')"
# Returns: {"success": true, "stdout": "5! = 120\n", "captured_result": "120", ...}
```

**Security Considerations:**
- Code runs in subprocess (isolated from main process)
- Uses temporary directory (no access to project files)
- Timeout prevents runaway processes
- Limited to Python standard library (no pip install)

---

### 2. filesystem_operation

Perform filesystem operations safely within the project directory.

**Signature:**
```python
def filesystem_operation(
    operation: Literal["create_dir", "write_file", "read_file", "list_dir", "delete_file", "check_exists"],
    path: str,
    content: str = "",
    encoding: str = "utf-8"
) -> dict:
    """Safely perform filesystem operations."""
```

**Parameters:**
- `operation` (Literal): One of: `create_dir`, `write_file`, `read_file`, `list_dir`, `delete_file`, `check_exists`
- `path` (str): File or directory path (absolute or relative to current working directory)
- `content` (str, optional): Content to write (for `write_file` operation)
- `encoding` (str, optional): File encoding (default: "utf-8")

**Returns:**

Varies by operation. Common fields:
- `success` (bool): Whether operation succeeded
- `path` (str): Resolved absolute path
- `error` (str, optional): Error message if failed

**Operations:**

#### create_dir
Create a directory (with parent directories if needed).

**Example:**
```python
result = registry.execute("filesystem_operation", {
    "operation": "create_dir",
    "path": "output/results"
})
# Returns: {"success": true, "message": "Directory created: /path/to/output/results", "path": "..."}
```

#### write_file
Write content to a file (creates parent directories if needed).

**Example:**
```python
result = registry.execute("filesystem_operation", {
    "operation": "write_file",
    "path": "output/report.txt",
    "content": "Analysis complete: 100% success rate"
})
# Returns: {"success": true, "message": "File written: ...", "path": "...", "size": 37}
```

#### read_file
Read content from a file.

**Example:**
```python
result = registry.execute("filesystem_operation", {
    "operation": "read_file",
    "path": "data/input.txt"
})
# Returns: {"success": true, "content": "file contents...", "path": "...", "size": 123}
```

#### list_dir
List contents of a directory.

**Example:**
```python
result = registry.execute("filesystem_operation", {
    "operation": "list_dir",
    "path": "data"
})
# Returns: {
#   "success": true,
#   "items": [
#     {"name": "input.txt", "path": "/path/to/data/input.txt", "type": "file", "size": 123},
#     {"name": "output", "path": "/path/to/data/output", "type": "directory", "size": null}
#   ],
#   "path": "/path/to/data",
#   "count": 2
# }
```

#### delete_file
Delete a file.

**Example:**
```python
result = registry.execute("filesystem_operation", {
    "operation": "delete_file",
    "path": "temp/old_file.txt"
})
# Returns: {"success": true, "message": "File deleted: ...", "path": "..."}
```

#### check_exists
Check if a path exists and get its type.

**Example:**
```python
result = registry.execute("filesystem_operation", {
    "operation": "check_exists",
    "path": "data/input.txt"
})
# Returns: {"success": true, "exists": true, "type": "file", "path": "..."}
```

**Security:**
- Relative paths are restricted to current working directory
- Absolute paths are allowed (user explicitly provided full path)
- Path traversal attacks prevented (e.g., `../../etc/passwd`)

---

### 3. math_calculator

Perform basic mathematical operations.

**Signature:**
```python
def math_calculator(
    arg1: float,
    arg2: float,
    operation: Literal["add", "subtract", "multiply", "divide"]
) -> dict:
    """Calculate result of mathematical operation."""
```

**Parameters:**
- `arg1` (float): First operand
- `arg2` (float): Second operand
- `operation` (Literal): One of: `add`, `subtract`, `multiply`, `divide`

**Returns:**
```python
{
    "arg1": float,        # First operand
    "arg2": float,        # Second operand
    "operation": str,     # Operation performed
    "result": float       # Calculation result
}
# Or on error:
{
    "error": str          # Error message (e.g., "Division by zero")
}
```

**Example:**

```python
from local_llm_sdk import create_client_with_tools

client = create_client_with_tools()

response = client.chat("What is 42 multiplied by 17?", use_tools=True)
print(response)
# LLM uses math_calculator:
# {"arg1": 42, "arg2": 17, "operation": "multiply", "result": 714}
```

**Supported Operations:**
- `add`: arg1 + arg2
- `subtract`: arg1 - arg2
- `multiply`: arg1 * arg2
- `divide`: arg1 / arg2 (returns error if arg2 is 0)

---

### 4. text_transformer

Transform text case (uppercase, lowercase, title case).

**Signature:**
```python
def text_transformer(
    text: str,
    transform: Literal["upper", "lower", "title"] = "upper"
) -> dict:
    """Transform text case."""
```

**Parameters:**
- `text` (str): Text to transform
- `transform` (Literal, optional): Transformation type - `upper`, `lower`, or `title` (default: "upper")

**Returns:**
```python
{
    "original": str,         # Original text
    "transformed": str,      # Transformed text
    "transform_type": str    # Type of transformation applied
}
```

**Example:**

```python
from local_llm_sdk.tools import tools

result = tools.execute("text_transformer", {
    "text": "hello world",
    "transform": "title"
})
# Returns: {"original": "hello world", "transformed": "Hello World", "transform_type": "title"}
```

**Transformations:**
- `upper`: Convert to UPPERCASE
- `lower`: Convert to lowercase
- `title`: Convert To Title Case

---

### 5. char_counter

Count characters, words, and detect numbers in text.

**Signature:**
```python
def char_counter(text: str) -> dict:
    """Count characters in the provided text."""
```

**Parameters:**
- `text` (str): Text to analyze

**Returns:**
```python
{
    "text": str,               # Original text
    "character_count": int,    # Total characters (including spaces)
    "word_count": int,         # Number of words (space-separated)
    "has_numbers": bool        # True if text contains digits
}
```

**Example:**

```python
from local_llm_sdk.tools import tools

result = tools.execute("char_counter", {
    "text": "Hello World 2024"
})
# Returns: {
#   "text": "Hello World 2024",
#   "character_count": 16,
#   "word_count": 3,
#   "has_numbers": true
# }
```

---

### 6. get_weather

Get mock weather data (demonstration tool).

**Signature:**
```python
def get_weather(
    city: str,
    units: Literal["celsius", "fahrenheit"] = "celsius"
) -> dict:
    """Get mock weather data for demonstration."""
```

**Parameters:**
- `city` (str): City name (supports: New York, London, Tokyo)
- `units` (Literal, optional): Temperature units - `celsius` or `fahrenheit` (default: "celsius")

**Returns:**
```python
{
    "city": str,            # City name
    "temperature": float,   # Temperature in requested units
    "units": str,           # Units used
    "condition": str        # Weather condition (sunny, rainy, cloudy)
}
# Or if city not found:
{
    "city": str,
    "error": str            # "City not found in mock data"
}
```

**Example:**

```python
from local_llm_sdk.tools import tools

result = tools.execute("get_weather", {
    "city": "London",
    "units": "fahrenheit"
})
# Returns: {"city": "London", "temperature": 59.0, "units": "fahrenheit", "condition": "rainy"}
```

**Note:** This is a demonstration tool with hardcoded data. In production, replace with a real weather API.

---

## Creating Custom Tools

### Design Principles

When creating tools, follow these best practices:

1. **Return dictionaries**: Always return `dict` (JSON-serializable)
2. **Handle errors gracefully**: Return error dict instead of raising exceptions
3. **Use type hints**: Enable automatic schema generation
4. **Write clear descriptions**: Help the LLM understand when to use your tool
5. **Document parameters**: Use docstrings to explain arguments
6. **Keep it simple**: Each tool should do one thing well

### Example: Simple Tool

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

### Example: Tool with Validation

```python
from local_llm_sdk.tools import tool
from typing import Literal

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

### Example: Tool with External API

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
        # Example API call (replace with real endpoint)
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
    except (KeyError, ValueError) as e:
        return {
            "success": False,
            "error": f"Invalid API response: {str(e)}"
        }
```

### Example: Tool with File Processing

```python
from local_llm_sdk.tools import tool
from typing import Literal

@tool("Analyze CSV file and return statistics")
def analyze_csv(
    filepath: str,
    column: str,
    operation: Literal["mean", "median", "sum", "count"]
) -> dict:
    """
    Read a CSV file and calculate statistics for a column.

    Args:
        filepath: Path to CSV file
        column: Column name to analyze
        operation: Statistical operation to perform

    Returns:
        Dictionary with analysis results
    """
    try:
        import csv
        from pathlib import Path
        from statistics import mean, median

        # Validate file exists
        path = Path(filepath)
        if not path.exists():
            return {"success": False, "error": f"File not found: {filepath}"}

        # Read CSV
        with open(path, 'r') as f:
            reader = csv.DictReader(f)
            rows = list(reader)

        # Validate column exists
        if column not in rows[0]:
            available = ", ".join(rows[0].keys())
            return {
                "success": False,
                "error": f"Column '{column}' not found. Available: {available}"
            }

        # Extract numeric values
        try:
            values = [float(row[column]) for row in rows if row[column]]
        except ValueError:
            return {
                "success": False,
                "error": f"Column '{column}' contains non-numeric values"
            }

        # Calculate statistic
        if operation == "mean":
            result = mean(values)
        elif operation == "median":
            result = median(values)
        elif operation == "sum":
            result = sum(values)
        elif operation == "count":
            result = len(values)
        else:
            return {"success": False, "error": f"Unknown operation: {operation}"}

        return {
            "success": True,
            "filepath": str(path),
            "column": column,
            "operation": operation,
            "result": result,
            "row_count": len(values)
        }

    except Exception as e:
        return {
            "success": False,
            "error": f"Unexpected error: {str(e)}"
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

### Registry Execution Errors

The `ToolRegistry.execute()` method automatically handles errors:

```python
# Tool not found
result = registry.execute("nonexistent_tool", {})
# Returns: '{"error": "Unknown tool: nonexistent_tool", "available_tools": "...", "suggestion": "..."}'

# Tool raises exception
@tool("Buggy tool")
def buggy_tool(x: int) -> dict:
    raise ValueError("Oops!")

result = registry.execute("buggy_tool", {"x": 5})
# Returns: '{"success": false, "error": "Oops!"}'
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

**Example: Comprehensive Error Handling**

```python
from local_llm_sdk.tools import tool

@tool("Download file from URL")
def download_file(url: str, destination: str) -> dict:
    """
    Download a file from a URL to local filesystem.

    Args:
        url: URL to download from
        destination: Local path to save file
    """
    import requests
    from pathlib import Path

    # Validate URL format
    if not url.startswith(("http://", "https://")):
        return {
            "success": False,
            "error": "Invalid URL format",
            "suggestion": "URL must start with http:// or https://"
        }

    # Validate destination path
    try:
        dest_path = Path(destination)
        dest_path.parent.mkdir(parents=True, exist_ok=True)
    except (OSError, ValueError) as e:
        return {
            "success": False,
            "error": f"Invalid destination path: {str(e)}"
        }

    # Download file
    try:
        response = requests.get(url, timeout=30, stream=True)
        response.raise_for_status()

        with open(dest_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)

        file_size = dest_path.stat().st_size

        return {
            "success": True,
            "url": url,
            "destination": str(dest_path),
            "size_bytes": file_size,
            "size_mb": round(file_size / 1024 / 1024, 2)
        }

    except requests.Timeout:
        return {
            "success": False,
            "error": "Download timed out after 30 seconds",
            "suggestion": "Check network connection or try a smaller file"
        }

    except requests.HTTPError as e:
        return {
            "success": False,
            "error": f"HTTP error: {e.response.status_code}",
            "details": str(e)
        }

    except requests.RequestException as e:
        return {
            "success": False,
            "error": "Network error",
            "details": str(e)
        }

    except OSError as e:
        return {
            "success": False,
            "error": "Failed to write file",
            "details": str(e)
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

@admin_registry.register("Delete all files in directory")
def delete_directory_contents(path: str) -> dict:
    """Admin-only: Delete all files in a directory."""
    import shutil
    shutil.rmtree(path)
    return {"success": True, "deleted": path}

# User registry with safe tools
user_registry = ToolRegistry()

@user_registry.register("List files in directory")
def list_files(path: str) -> dict:
    """User-safe: List directory contents."""
    from pathlib import Path
    files = [f.name for f in Path(path).iterdir()]
    return {"files": files}

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

### Tool Composition

Create higher-level tools by composing existing tools:

```python
from local_llm_sdk.tools import tool, tools

@tool("Download and analyze CSV file")
def download_and_analyze_csv(url: str, column: str) -> dict:
    """
    Download a CSV file and compute statistics for a column.
    Combines download_file and analyze_csv tools.
    """
    import tempfile
    import os

    # Create temp file
    temp_file = os.path.join(tempfile.gettempdir(), "temp_data.csv")

    # Download file (using filesystem_operation or custom download tool)
    # Note: In real implementation, you'd use the actual download tool
    download_result = {"success": True, "destination": temp_file}

    if not download_result.get("success"):
        return download_result  # Return error from download

    # Analyze CSV (using analyze_csv tool)
    analysis_result = tools.execute("analyze_csv", {
        "filepath": temp_file,
        "column": column,
        "operation": "mean"
    })

    # Cleanup
    try:
        os.unlink(temp_file)
    except:
        pass

    return analysis_result
```

### Dynamic Tool Registration

Register tools at runtime based on configuration:

```python
from local_llm_sdk.tools import ToolRegistry
import importlib

def load_tools_from_config(config_path: str) -> ToolRegistry:
    """
    Load tools dynamically from a configuration file.

    Config format (YAML):
    tools:
      - module: my_tools.math
        functions: [advanced_sqrt, matrix_multiply]
      - module: my_tools.text
        functions: [sentiment_analysis]
    """
    import yaml

    registry = ToolRegistry()

    with open(config_path) as f:
        config = yaml.safe_load(f)

    for tool_spec in config.get("tools", []):
        module_name = tool_spec["module"]
        function_names = tool_spec["functions"]

        # Import module
        module = importlib.import_module(module_name)

        # Register each function
        for func_name in function_names:
            func = getattr(module, func_name)
            # Assuming functions are already decorated with @tool
            # If not, you'd need to register them manually

    return registry
```

### Tool Middleware

Add logging, metrics, or validation to all tool calls:

```python
from local_llm_sdk.tools import ToolRegistry
from typing import Callable, Dict, Any
import time

class InstrumentedToolRegistry(ToolRegistry):
    """Tool registry with automatic logging and timing."""

    def __init__(self):
        super().__init__()
        self.execution_times = {}
        self.call_counts = {}

    def execute(self, tool_name: str, arguments: Dict[str, Any]) -> str:
        """Execute tool with timing and logging."""

        # Log start
        print(f"[TOOL] Executing {tool_name} with args: {arguments}")

        # Track call count
        self.call_counts[tool_name] = self.call_counts.get(tool_name, 0) + 1

        # Time execution
        start_time = time.time()
        result = super().execute(tool_name, arguments)
        elapsed = time.time() - start_time

        # Track execution time
        if tool_name not in self.execution_times:
            self.execution_times[tool_name] = []
        self.execution_times[tool_name].append(elapsed)

        # Log completion
        print(f"[TOOL] {tool_name} completed in {elapsed:.3f}s")

        return result

    def get_stats(self) -> dict:
        """Get execution statistics."""
        stats = {}
        for tool_name in self.call_counts:
            times = self.execution_times.get(tool_name, [])
            stats[tool_name] = {
                "call_count": self.call_counts[tool_name],
                "avg_time": sum(times) / len(times) if times else 0,
                "total_time": sum(times)
            }
        return stats
```

### Type-Safe Tool Calls

Use Pydantic models for tool parameters:

```python
from local_llm_sdk.tools import tool
from pydantic import BaseModel, Field, validator

class EmailConfig(BaseModel):
    """Configuration for sending emails."""
    to: str = Field(..., description="Recipient email address")
    subject: str = Field(..., description="Email subject")
    body: str = Field(..., description="Email body")
    cc: list[str] = Field(default_factory=list, description="CC recipients")

    @validator("to")
    def validate_email(cls, v):
        if "@" not in v:
            raise ValueError("Invalid email address")
        return v

@tool("Send email via SMTP")
def send_email(
    to: str,
    subject: str,
    body: str,
    cc: list = None
) -> dict:
    """
    Send an email message.

    Uses Pydantic model for validation.
    """
    # Validate parameters using Pydantic
    try:
        config = EmailConfig(to=to, subject=subject, body=body, cc=cc or [])
    except Exception as e:
        return {
            "success": False,
            "error": f"Invalid parameters: {str(e)}"
        }

    # Send email (mock implementation)
    return {
        "success": True,
        "to": config.to,
        "subject": config.subject,
        "cc": config.cc,
        "sent_at": "2024-01-01T12:00:00Z"
    }
```

---

## Complete Example: Building a Data Analysis Agent

Here's a complete example combining multiple custom tools:

```python
from local_llm_sdk import LocalLLMClient
from local_llm_sdk.tools import tool
from typing import Literal
import pandas as pd
from pathlib import Path

# Tool 1: Load dataset
@tool("Load CSV dataset into memory")
def load_dataset(filepath: str) -> dict:
    """Load a CSV file and return summary information."""
    try:
        df = pd.read_csv(filepath)
        return {
            "success": True,
            "filepath": filepath,
            "rows": len(df),
            "columns": list(df.columns),
            "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()},
            "preview": df.head(5).to_dict(orient="records")
        }
    except Exception as e:
        return {"success": False, "error": str(e)}

# Tool 2: Compute statistics
@tool("Compute summary statistics for dataset column")
def compute_statistics(filepath: str, column: str) -> dict:
    """Calculate mean, median, std for a numeric column."""
    try:
        df = pd.read_csv(filepath)

        if column not in df.columns:
            return {
                "success": False,
                "error": f"Column '{column}' not found"
            }

        if not pd.api.types.is_numeric_dtype(df[column]):
            return {
                "success": False,
                "error": f"Column '{column}' is not numeric"
            }

        return {
            "success": True,
            "column": column,
            "mean": float(df[column].mean()),
            "median": float(df[column].median()),
            "std": float(df[column].std()),
            "min": float(df[column].min()),
            "max": float(df[column].max()),
            "count": int(df[column].count())
        }
    except Exception as e:
        return {"success": False, "error": str(e)}

# Tool 3: Create visualization
@tool("Create visualization and save to file")
def create_plot(
    filepath: str,
    x_column: str,
    y_column: str,
    plot_type: Literal["scatter", "line", "bar"],
    output_path: str
) -> dict:
    """Generate a plot from dataset columns."""
    try:
        import matplotlib.pyplot as plt

        df = pd.read_csv(filepath)

        # Validate columns
        if x_column not in df.columns or y_column not in df.columns:
            return {
                "success": False,
                "error": "One or both columns not found"
            }

        # Create plot
        plt.figure(figsize=(10, 6))

        if plot_type == "scatter":
            plt.scatter(df[x_column], df[y_column])
        elif plot_type == "line":
            plt.plot(df[x_column], df[y_column])
        elif plot_type == "bar":
            plt.bar(df[x_column], df[y_column])

        plt.xlabel(x_column)
        plt.ylabel(y_column)
        plt.title(f"{plot_type.title()} Plot: {y_column} vs {x_column}")
        plt.savefig(output_path)
        plt.close()

        return {
            "success": True,
            "plot_type": plot_type,
            "output_path": output_path,
            "x_column": x_column,
            "y_column": y_column
        }
    except Exception as e:
        return {"success": False, "error": str(e)}

# Usage example
def analyze_data_with_agent():
    """Use agent to perform multi-step data analysis."""

    # Create client and register custom tools
    client = LocalLLMClient(
        base_url="http://169.254.83.107:1234/v1",
        model="mistralai/magistral-small-2509"
    )

    # Register our custom tools
    client.register_tool("Load dataset")(load_dataset)
    client.register_tool("Compute statistics")(compute_statistics)
    client.register_tool("Create plot")(create_plot)

    # Give agent a complex analysis task
    task = """
    Analyze the file 'sales_data.csv':
    1. Load the dataset and show me the columns
    2. Calculate statistics for the 'revenue' column
    3. Create a scatter plot of 'date' vs 'revenue' and save to 'revenue_plot.png'
    """

    response = client.chat(task, use_tools=True)
    print(response)

    # See which tools were used
    client.print_tool_calls(detailed=True)

if __name__ == "__main__":
    analyze_data_with_agent()
```

---

## Summary

The Local LLM SDK's tool system provides:

- **Simple registration** with `@tool` decorator
- **Automatic schema generation** from Python type hints
- **Type-safe execution** with Pydantic validation
- **Built-in tools** for Python, filesystem, math, and text operations
- **Flexible architecture** supporting custom tools and multiple registries
- **Robust error handling** with graceful degradation

**Key Takeaways:**

1. Tools must return `dict` (JSON-serializable)
2. Use type hints for automatic schema generation
3. Handle errors gracefully (return error dict, don't raise)
4. Write clear descriptions to help the LLM understand tool usage
5. Test tools independently before using with agents

For more examples, see:
- `/notebooks/07-react-agents.ipynb` - Tool usage with ReACT agents
- `/tests/test_tools.py` - Complete test suite
- `/local_llm_sdk/tools/builtin.py` - Built-in tool implementations

**Next Steps:**
- Read [Client API Reference](./client.md) for tool integration with chat
- See [Agent API Reference](./agents.md) for using tools with agents
- Check [Examples](../examples/custom-tools.md) for more custom tool patterns
