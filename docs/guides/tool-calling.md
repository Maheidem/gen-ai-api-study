# Tool Calling / Function Calling Guide

A comprehensive guide to extending your LLM's capabilities with tools in the Local LLM SDK.

## Table of Contents

- [What is Tool Calling?](#what-is-tool-calling)
- [How Tool Calling Works](#how-tool-calling-works)
- [Using Built-in Tools](#using-built-in-tools)
- [Creating Custom Tools](#creating-custom-tools)
- [Tool Execution Patterns](#tool-execution-patterns)
- [Error Handling](#error-handling)
- [Advanced Patterns](#advanced-patterns)
- [Best Practices](#best-practices)
- [Troubleshooting](#troubleshooting)

---

## What is Tool Calling?

**Tool calling** (also known as **function calling**) is a technique that allows LLMs to interact with external functions and APIs. Instead of just generating text, the LLM can:

- Execute calculations with precision
- Access real-time data from APIs
- Perform file operations
- Query databases
- Interact with external systems

### Why Use Tools?

| Without Tools | With Tools |
|---------------|------------|
| ❌ Poor at precise math (e.g., "127 × 893 ≈ 113,000") | ✅ Exact calculations (127 × 893 = 113,411) |
| ❌ No access to real-time data | ✅ Call APIs for current information |
| ❌ Can't modify files or systems | ✅ Use file operations and system calls |
| ❌ Limited to training data knowledge | ✅ Access external databases and services |
| ❌ Hallucinates function outputs | ✅ Guaranteed correct execution results |

### OpenAI Function Calling Specification

The Local LLM SDK implements the OpenAI function calling specification, which means:

- Tools are defined as JSON schemas
- LLMs decide when to call tools based on user prompts
- Tool execution happens automatically
- Results are fed back to the LLM for response generation

---

## How Tool Calling Works

### Flow Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│ 1. User Request                                                 │
│    "Calculate 127 multiplied by 893"                            │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│ 2. Client Adds Tool Schemas to Request                         │
│    {                                                            │
│      "messages": [...],                                         │
│      "tools": [                                                 │
│        {                                                        │
│          "type": "function",                                    │
│          "function": {                                          │
│            "name": "math_calculator",                           │
│            "description": "Perform math operations",            │
│            "parameters": {                                      │
│              "type": "object",                                  │
│              "properties": {                                    │
│                "arg1": {"type": "number"},                      │
│                "arg2": {"type": "number"},                      │
│                "operation": {"type": "string"}                  │
│              }                                                  │
│            }                                                    │
│          }                                                      │
│        }                                                        │
│      ]                                                          │
│    }                                                            │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│ 3. LLM Analyzes Request & Returns Tool Call                    │
│    {                                                            │
│      "role": "assistant",                                       │
│      "tool_calls": [                                            │
│        {                                                        │
│          "id": "call_123",                                      │
│          "type": "function",                                    │
│          "function": {                                          │
│            "name": "math_calculator",                           │
│            "arguments": "{\"arg1\": 127, \"arg2\": 893,         │
│                          \"operation\": \"multiply\"}"          │
│          }                                                      │
│        }                                                        │
│      ]                                                          │
│    }                                                            │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│ 4. SDK Executes Tool via ToolRegistry                          │
│    registry.execute("math_calculator", {                        │
│      "arg1": 127,                                               │
│      "arg2": 893,                                               │
│      "operation": "multiply"                                    │
│    })                                                           │
│    → Returns: {"result": 113411, "success": true}              │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│ 5. SDK Adds Tool Result as Message                             │
│    {                                                            │
│      "role": "tool",                                            │
│      "tool_call_id": "call_123",                                │
│      "content": "{\"result\": 113411, \"success\": true}"       │
│    }                                                            │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│ 6. SDK Sends Updated Conversation Back to LLM                  │
│    [                                                            │
│      {"role": "user", "content": "Calculate 127 × 893"},       │
│      {"role": "assistant", "tool_calls": [...]},                │
│      {"role": "tool", "content": "{\"result\": 113411}"}        │
│    ]                                                            │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│ 7. LLM Generates Natural Language Response                     │
│    "The result of 127 multiplied by 893 is 113,411."           │
└─────────────────────────────────────────────────────────────────┘
```

### Key Components

1. **Tool Schema**: JSON definition of the tool (name, description, parameters)
2. **Tool Call**: LLM's decision to invoke a tool with specific arguments
3. **Tool Execution**: SDK runs the actual function
4. **Tool Result**: Output returned to the LLM as a message
5. **Final Response**: LLM incorporates result into natural language answer

### Conversation State

The SDK maintains complete conversation state including tool calls:

```python
client.conversation  # Full history including tool calls and results
client.last_conversation_additions  # New messages from last call
client.last_tool_calls  # Tool calls from last request
```

---

## Using Built-in Tools

The SDK provides 6 production-ready built-in tools.

### Quick Start

```python
from local_llm_sdk import create_client_with_tools

# Create client with built-in tools automatically registered
client = create_client_with_tools(
    base_url="http://localhost:1234/v1",
    model="mistralai/magistral-small-2509"
)

# Use tools (enabled by default)
response = client.chat("Calculate 127 multiplied by 893")
print(response)  # "The result is 113,411."

# See which tools were used
client.print_tool_calls()
# Output:
# 🔧 Tool Execution Summary (1 call):
# ======================================================================
#   [1] math_calculator(arg1=127, arg2=893, operation=multiply) → result=113411
# ======================================================================
```

### Available Built-in Tools

#### 1. `math_calculator`

Performs precise mathematical operations.

```python
# Basic calculation
response = client.chat("What is 456 * 789?")
# Uses: math_calculator(arg1=456, arg2=789, operation="multiply")

# Complex expression (multiple calls)
response = client.chat("Calculate (15 + 25) * 3 / 2")
# Uses:
#   [1] math_calculator(arg1=15, arg2=25, operation=add) → 40
#   [2] math_calculator(arg1=40, arg2=3, operation=multiply) → 120
#   [3] math_calculator(arg1=120, arg2=2, operation=divide) → 60
```

**Operations**: `add`, `subtract`, `multiply`, `divide`

#### 2. `text_transformer`

Transforms text case.

```python
response = client.chat("Convert 'hello world' to uppercase")
# Uses: text_transformer(text="hello world", transform="upper")
# Returns: "HELLO WORLD"

response = client.chat("Make 'PYTHON' lowercase")
# Uses: text_transformer(text="PYTHON", transform="lower")
# Returns: "python"
```

**Transforms**: `upper`, `lower`, `title`

#### 3. `char_counter`

Counts characters and words in text.

```python
response = client.chat("How many characters in 'Hello, World!'?")
# Uses: char_counter(text="Hello, World!")
# Returns: {
#   "character_count": 13,
#   "word_count": 2,
#   "has_numbers": false
# }
```

#### 4. `execute_python`

Executes Python code in an isolated sandbox.

```python
response = client.chat("Calculate 5 factorial using Python")
# Uses: execute_python(code="...")
# Runs code in temporary directory, returns stdout/stderr

# Example with file creation
response = client.chat("Write 'Hello, World!' to a file and read it back")
# Creates file in temp dir, executes, returns result
```

**Security**: Runs in subprocess with timeout, isolated working directory.

#### 5. `filesystem_operation`

Performs file and directory operations.

```python
# Create directory
response = client.chat("Create a directory called 'output'")
# Uses: filesystem_operation(operation="create_dir", path="output")

# Write file
response = client.chat("Write 'Test content' to output/test.txt")
# Uses: filesystem_operation(
#   operation="write_file",
#   path="output/test.txt",
#   content="Test content"
# )

# Read file
response = client.chat("Read the contents of output/test.txt")
# Uses: filesystem_operation(operation="read_file", path="output/test.txt")
```

**Operations**: `create_dir`, `write_file`, `read_file`, `list_dir`, `delete_file`, `check_exists`

**Security**: Relative paths are restricted to current working directory.

#### 6. `get_weather`

Mock weather data (example tool for demonstrations).

```python
response = client.chat("What's the weather in London?")
# Uses: get_weather(city="London", units="celsius")
# Returns mock data: {"temp": 15, "condition": "rainy"}
```

**Note**: This is a demonstration tool with mock data. Replace with real API in production.

### Inspecting Tool Usage

Two methods to inspect tool execution:

#### Method 1: Quick Summary with `print_tool_calls()`

```python
response = client.chat("Calculate 5! and uppercase 'python'")

# Compact summary
client.print_tool_calls()
# Output:
# 🔧 Tool Execution Summary (2 calls):
# ======================================================================
#   [1] execute_python(code=...) → result=120
#   [2] text_transformer(text=python, transform=upper) → transformed=PYTHON
# ======================================================================

# Detailed JSON output
client.print_tool_calls(detailed=True)
# Shows full arguments and results as JSON
```

#### Method 2: Full ChatCompletion Object

```python
response = client.chat("Calculate 15 + 25", return_full_response=True)

print(f"Model: {response.model}")
print(f"Finish reason: {response.choices[0].finish_reason}")

# Access tool calls
message = response.choices[0].message
if message.tool_calls:
    for tool_call in message.tool_calls:
        print(f"Function: {tool_call.function.name}")
        print(f"Arguments: {tool_call.function.arguments}")
        print(f"ID: {tool_call.id}")
```

### Controlling Tool Usage with `tool_choice`

The `tool_choice` parameter controls when tools are used:

```python
# "auto" - Let LLM decide (default)
response = client.chat("What is 25 * 16?", tool_choice="auto")
# LLM may or may not use tools

# "required" - Force tool usage
response = client.chat("What is 25 * 16?", tool_choice="required")
# LLM MUST use at least one tool

# "none" - Prevent tool usage
response = client.chat("What is 25 * 16?", tool_choice="none")
# LLM answers without tools (may be less accurate)

# Specific tool - Force a particular tool
response = client.chat(
    "Calculate something",
    tool_choice={
        "type": "function",
        "function": {"name": "math_calculator"}
    }
)
```

**When to use:**

- **`"auto"`**: General use, balanced approach (default)
- **`"required"`**: Guaranteed tool execution, calculator apps, API wrappers
- **`"none"`**: Creative writing, brainstorming, pure reasoning
- **Specific tool**: When you know exactly which tool should be used

**Reasoning Models Note**: Models like Magistral use thinking blocks and may skip tools for simple tasks with `tool_choice="auto"`. Use `"required"` to bypass thinking and force tool usage.

---

## Creating Custom Tools

### Basic Custom Tool

Use the `@tool` decorator to create custom tools:

```python
from local_llm_sdk.tools import tool

@tool("Generates a personalized greeting for a given name")
def greet(name: str) -> dict:
    """Greet a person by name.

    Args:
        name: The person's name to greet

    Returns:
        A dictionary with the greeting message
    """
    return {
        "greeting": f"Hello, {name}! Welcome to custom tools!",
        "name": name
    }

# Register with client
client.register_tools([greet])

# Use it
response = client.chat("Please greet Alice")
# Uses: greet(name="Alice")
# Returns: "Hello, Alice! Welcome to custom tools!"
```

### Anatomy of a Tool

```python
@tool("Description for LLM to understand when to use this tool")
def tool_name(
    required_param: str,          # Required parameter
    optional_param: int = 42       # Optional with default
) -> dict:                         # Must return dict
    """
    Detailed docstring (optional but recommended).
    Helps LLM understand parameter meanings.

    Args:
        required_param: Description of required parameter
        optional_param: Description of optional parameter

    Returns:
        Dictionary with results and status
    """
    # Tool implementation
    result = do_something(required_param, optional_param)

    return {
        "result": result,
        "success": True,
        "error": None
    }
```

**Key components:**

1. **`@tool(description)`**: Registers the tool and provides LLM context
2. **Type hints**: Required! Tells LLM parameter types
3. **Docstring**: Optional but helps LLM understand usage
4. **Return dict**: Structured output for easy parsing

### Supported Parameter Types

```python
from typing import Literal, List, Dict

@tool("Demonstrates all supported parameter types")
def demo_types(
    text: str,                              # String
    count: int,                             # Integer
    amount: float,                          # Float
    enabled: bool,                          # Boolean
    items: list,                            # Array (untyped)
    config: dict,                           # Object (untyped)
    typed_list: List[str],                  # Typed array
    typed_dict: Dict[str, int],             # Typed object
    choice: Literal["option1", "option2"]   # Enum/choices
) -> dict:
    """All parameter types are automatically converted to JSON Schema."""
    return {"status": "ok"}
```

**JSON Schema generation:**

| Python Type | JSON Schema Type | Example |
|-------------|------------------|---------|
| `str` | `"string"` | `"hello"` |
| `int` | `"integer"` | `42` |
| `float` | `"number"` | `3.14` |
| `bool` | `"boolean"` | `true` |
| `list`, `List` | `"array"` | `[1, 2, 3]` |
| `dict`, `Dict` | `"object"` | `{"key": "value"}` |
| `Literal["a", "b"]` | `{"enum": ["a", "b"]}` | `"a"` or `"b"` |

### Multiple Parameters Example

```python
@tool("Calculates the area and perimeter of a rectangle")
def rectangle_area(width: float, height: float) -> dict:
    """Calculate rectangle dimensions.

    Args:
        width: The width of the rectangle
        height: The height of the rectangle

    Returns:
        Dictionary with area, perimeter, and dimensions
    """
    area = width * height
    perimeter = 2 * (width + height)

    return {
        "area": area,
        "perimeter": perimeter,
        "width": width,
        "height": height,
        "unit": "square units"
    }

client.register_tools([rectangle_area])

response = client.chat("What's the area of a 12.5 by 8.3 rectangle?")
# Uses: rectangle_area(width=12.5, height=8.3)
# Returns area and perimeter
```

### Optional Parameters with Defaults

```python
@tool("Formats a price with currency symbol")
def format_price(
    amount: float,
    currency: str = "USD",
    decimals: int = 2
) -> dict:
    """Format a price with currency symbol.

    Args:
        amount: The price amount
        currency: Currency code (USD, EUR, GBP). Default: USD
        decimals: Number of decimal places. Default: 2
    """
    symbols = {"USD": "$", "EUR": "€", "GBP": "£", "JPY": "¥"}
    symbol = symbols.get(currency, currency)
    formatted = f"{symbol}{amount:.{decimals}f}"

    return {
        "formatted_price": formatted,
        "amount": amount,
        "currency": currency
    }

# Usage examples:
client.chat("Format 1234.567")  # Uses defaults: USD, 2 decimals
client.chat("Format 99.99 in euros")  # Custom currency
client.chat("Format 1500 yen with no decimals")  # Custom decimals
```

### Enum Parameters with Literal

```python
from typing import Literal

@tool("Converts temperature between units")
def convert_temperature(
    value: float,
    from_unit: Literal["celsius", "fahrenheit", "kelvin"],
    to_unit: Literal["celsius", "fahrenheit", "kelvin"]
) -> dict:
    """Convert temperature between different units.

    Args:
        value: Temperature value to convert
        from_unit: Source unit (celsius, fahrenheit, kelvin)
        to_unit: Target unit (celsius, fahrenheit, kelvin)
    """
    # Conversion logic
    if from_unit == "celsius" and to_unit == "fahrenheit":
        result = (value * 9/5) + 32
    # ... more conversions

    return {
        "value": result,
        "from_unit": from_unit,
        "to_unit": to_unit,
        "conversion": f"{value}°{from_unit[0].upper()} = {result}°{to_unit[0].upper()}"
    }
```

**Benefits of Literal:**
- LLM sees valid choices in schema
- Type safety at runtime
- Better error messages if invalid choice used

### Registering Custom Tools

Three methods to register tools:

#### Method 1: Single Decorator on Client

```python
@client.register_tool("Description of the tool")
def my_tool(param: str) -> dict:
    return {"result": param.upper()}
```

#### Method 2: List of Functions

```python
def tool1(x: int) -> dict:
    """Tool 1 description"""
    return {"result": x * 2}

def tool2(x: int) -> dict:
    """Tool 2 description"""
    return {"result": x + 10}

# Register all at once
client.register_tools([tool1, tool2])
```

#### Method 3: Global Registry with @tool

```python
from local_llm_sdk.tools import tool

# Define tools with global decorator
@tool("Tool 1 description")
def tool1(x: int) -> dict:
    return {"result": x * 2}

@tool("Tool 2 description")
def tool2(x: int) -> dict:
    return {"result": x + 10}

# Tools are automatically in global registry
# Import them into client
client.register_tools([tool1, tool2])
```

---

## Tool Execution Patterns

### Pattern 1: Single Tool Call

Simple request, single tool execution:

```python
response = client.chat("What is 123 * 456?")
# Flow:
# 1. LLM decides to use math_calculator
# 2. SDK executes: math_calculator(123, 456, "multiply")
# 3. SDK sends result back to LLM
# 4. LLM returns: "The result is 56,088."
```

### Pattern 2: Sequential Tool Calls

LLM uses multiple tools in sequence:

```python
response = client.chat(
    "Calculate 5 factorial, convert to uppercase string, count characters"
)
# Flow:
# 1. execute_python(code="result = 1\nfor i in range(1,6): result *= i")
#    → Returns: 120
# 2. text_transformer(text="120", transform="upper")
#    → Returns: "120" (no change, already numeric)
# 3. char_counter(text="120")
#    → Returns: 3 characters

# LLM sees all results and composes final answer
```

### Pattern 3: Parallel Tool Calls

Some models can request multiple tools simultaneously:

```python
response = client.chat("What's 10*5 and also uppercase 'hello'?")
# LLM may return both tool_calls in one message:
# tool_calls: [
#   {function: {name: "math_calculator", arguments: {...}}},
#   {function: {name: "text_transformer", arguments: {...}}}
# ]

# SDK executes both, sends results, LLM combines
```

### Pattern 4: Tool Chaining in Conversation

Tools work with conversation history:

```python
history = []

# Turn 1
response1, history = client.chat_with_history(
    "Calculate 25 * 16",
    history,
    use_tools=True
)
# Uses: math_calculator → 400

# Turn 2 - References previous result
response2, history = client.chat_with_history(
    "Now add 100 to that result",
    history,
    use_tools=True
)
# LLM knows previous result was 400
# Uses: math_calculator(400, 100, "add") → 500

# Turn 3 - Build on context
response3, history = client.chat_with_history(
    "Divide it by 5",
    history,
    use_tools=True
)
# Uses: math_calculator(500, 5, "divide") → 100
```

### Pattern 5: Conditional Tool Usage

Tools called based on conditions:

```python
response = client.chat(
    "If the character count of 'hello world' is even, multiply it by 5, "
    "otherwise multiply by 3"
)
# Flow:
# 1. char_counter(text="hello world") → 11 characters
# 2. LLM checks: 11 is odd
# 3. math_calculator(11, 3, "multiply") → 33
```

### Pattern 6: Tool Result Validation

Validate tool results and retry if needed:

```python
response = client.chat(
    "Calculate 10 divided by 0, if error occurs, return 'undefined'"
)
# Flow:
# 1. math_calculator(10, 0, "divide")
#    → Returns: {"error": "Division by zero"}
# 2. LLM sees error in result
# 3. LLM returns: "The result is undefined (cannot divide by zero)"
```

### Pattern 7: Multi-Step Complex Tasks

For complex multi-step tasks, consider using agents:

```python
from local_llm_sdk.agents import ReACT

agent = ReACT(client)
result = agent.run(
    "Calculate 5!, convert to uppercase, count chars, multiply count by 2",
    max_iterations=15
)

# Agent handles:
# - Iteration 1: execute_python for factorial → 120
# - Iteration 2: text_transformer to uppercase → "120"
# - Iteration 3: char_counter → 3
# - Iteration 4: math_calculator(3, 2, "multiply") → 6
# - Final: Returns "6" with metadata

print(result.final_response)  # "6"
print(f"Iterations: {result.iterations}")  # 4
print(f"Tools used: {result.metadata['total_tool_calls']}")  # 4
```

See [ReACT Agents Guide](./react-agents.md) for details.

---

## Error Handling

### Tool-Level Error Handling

**Always return errors in the result dict, never raise exceptions:**

```python
@tool("Safely divide two numbers")
def safe_divide(numerator: float, denominator: float) -> dict:
    """Divide two numbers with error handling."""

    # Validate inputs
    if denominator == 0:
        return {
            "error": "Cannot divide by zero",
            "numerator": numerator,
            "denominator": denominator,
            "result": None
        }

    # Perform operation
    result = numerator / denominator

    return {
        "result": result,
        "numerator": numerator,
        "denominator": denominator,
        "error": None
    }
```

**Why return errors instead of raising?**

1. **LLM can understand errors** - Errors in JSON are parsed by the LLM
2. **Graceful degradation** - LLM can explain the error to the user
3. **No crashes** - Conversation continues even if tool fails
4. **Partial results** - Can return what succeeded even if something failed

### Error Patterns

#### Pattern 1: Validation Errors

```python
@tool("Get element from array by index")
def get_array_element(arr: list, index: int) -> dict:
    """Get element at index with bounds checking."""

    if index < 0 or index >= len(arr):
        return {
            "error": f"Index {index} out of bounds (array length: {len(arr)})",
            "element": None,
            "valid_indices": f"0 to {len(arr)-1}"
        }

    return {
        "element": arr[index],
        "index": index,
        "error": None
    }
```

#### Pattern 2: Try-Except with Structured Errors

```python
@tool("Parse JSON string")
def parse_json(json_string: str) -> dict:
    """Parse JSON with error handling."""
    import json

    try:
        parsed = json.loads(json_string)
        return {
            "success": True,
            "data": parsed,
            "error": None
        }
    except json.JSONDecodeError as e:
        return {
            "success": False,
            "data": None,
            "error": f"Invalid JSON: {str(e)}",
            "position": e.pos
        }
```

#### Pattern 3: Multiple Error Types

```python
@tool("Download file from URL")
def download_file(url: str, destination: str) -> dict:
    """Download file with comprehensive error handling."""
    import requests
    from pathlib import Path

    try:
        # Validate URL
        if not url.startswith(('http://', 'https://')):
            return {"error": "Invalid URL protocol", "success": False}

        # Validate destination
        dest_path = Path(destination)
        if dest_path.exists():
            return {"error": "Destination already exists", "success": False}

        # Download
        response = requests.get(url, timeout=30)
        response.raise_for_status()

        # Write file
        dest_path.write_bytes(response.content)

        return {
            "success": True,
            "destination": str(dest_path),
            "size_bytes": len(response.content),
            "error": None
        }

    except requests.RequestException as e:
        return {"error": f"Download failed: {str(e)}", "success": False}
    except IOError as e:
        return {"error": f"File write failed: {str(e)}", "success": False}
    except Exception as e:
        return {"error": f"Unexpected error: {str(e)}", "success": False}
```

### Client-Level Error Handling

Handle errors at the client level for robustness:

```python
# Retry logic for transient errors
from requests.exceptions import ConnectionError, Timeout

try:
    response = client.chat("Calculate something", use_tools=True)
except (ConnectionError, Timeout) as e:
    print(f"Connection error: {e}")
    print("Check that LM Studio is running on http://localhost:1234")
except Exception as e:
    print(f"Unexpected error: {e}")

# Check for tool execution failures
client.chat("Use a tool")
if client.last_tool_calls:
    for tool_call in client.last_tool_calls:
        # Parse tool result to check for errors
        import json
        result = json.loads(tool_call.get('result', '{}'))
        if 'error' in result and result['error']:
            print(f"Tool {tool_call['name']} failed: {result['error']}")
```

### Built-in Tool Error Examples

Built-in tools handle errors gracefully:

```python
# Division by zero
response = client.chat("Divide 10 by 0")
# math_calculator returns: {"error": "Division by zero"}
# LLM explains: "Cannot divide by zero"

# File not found
response = client.chat("Read nonexistent.txt")
# filesystem_operation returns: {"error": "File does not exist"}
# LLM explains: "The file doesn't exist"

# Python execution timeout
response = client.chat("Run infinite loop in Python")
# execute_python returns: {"error": "Code execution timed out"}
# LLM explains: "The code took too long to execute"
```

---

## Advanced Patterns

### Pattern 1: Tool Composition

Create higher-level tools by combining lower-level ones:

```python
@tool("Analyzes text comprehensively with multiple metrics")
def comprehensive_text_analysis(text: str) -> dict:
    """Combines multiple text analysis operations."""

    # Use multiple operations internally
    char_count = len(text)
    word_count = len(text.split())
    sentence_count = text.count('.') + text.count('!') + text.count('?')

    return {
        "text": text,
        "metrics": {
            "characters": char_count,
            "words": word_count,
            "sentences": sentence_count,
            "avg_word_length": round(char_count / word_count, 2) if word_count > 0 else 0
        },
        "breakdown": {
            "uppercase": sum(1 for c in text if c.isupper()),
            "lowercase": sum(1 for c in text if c.islower()),
            "digits": sum(1 for c in text if c.isdigit()),
            "spaces": text.count(' ')
        }
    }
```

### Pattern 2: Conditional Tools

Tools that behave differently based on context:

```python
@tool("Smart calculator that auto-detects operation from expression")
def smart_calculate(expression: str) -> dict:
    """Automatically parse and calculate mathematical expressions."""

    # Detect operation
    if '+' in expression:
        parts = expression.split('+')
        operation = "add"
    elif '*' in expression or '×' in expression:
        parts = expression.replace('×', '*').split('*')
        operation = "multiply"
    # ... more operations

    try:
        nums = [float(p.strip()) for p in parts]

        if operation == "add":
            result = sum(nums)
        elif operation == "multiply":
            result = 1
            for n in nums:
                result *= n

        return {
            "expression": expression,
            "operation": operation,
            "result": result,
            "error": None
        }
    except Exception as e:
        return {
            "expression": expression,
            "error": f"Could not parse: {str(e)}"
        }
```

### Pattern 3: Stateful Tools

Tools that maintain state across calls (use with caution):

```python
# Global state (consider using class-based tools for better encapsulation)
_counter_state = {"count": 0}

@tool("Increments a persistent counter")
def increment_counter(amount: int = 1) -> dict:
    """Increment global counter and return new value."""
    _counter_state["count"] += amount
    return {
        "previous_count": _counter_state["count"] - amount,
        "current_count": _counter_state["count"],
        "increment": amount
    }

@tool("Resets the counter to zero")
def reset_counter() -> dict:
    """Reset counter to zero."""
    old_count = _counter_state["count"]
    _counter_state["count"] = 0
    return {
        "previous_count": old_count,
        "current_count": 0
    }
```

**Warning**: Stateful tools can cause issues with:
- Parallel execution
- Testing and reproducibility
- Conversation replay

Consider using conversation context instead of global state when possible.

### Pattern 4: Tool Factories

Generate tools dynamically:

```python
def create_unit_converter(from_unit: str, to_unit: str, conversion_factor: float):
    """Factory to create unit conversion tools."""

    @tool(f"Convert from {from_unit} to {to_unit}")
    def converter(value: float) -> dict:
        result = value * conversion_factor
        return {
            "value": value,
            "from_unit": from_unit,
            "to_unit": to_unit,
            "result": result,
            "conversion": f"{value} {from_unit} = {result} {to_unit}"
        }

    # Rename function for unique registration
    converter.__name__ = f"{from_unit}_to_{to_unit}"
    return converter

# Create multiple converters
km_to_miles = create_unit_converter("km", "miles", 0.621371)
miles_to_km = create_unit_converter("miles", "km", 1.60934)
celsius_to_fahrenheit = create_unit_converter("°C", "°F", lambda c: c * 9/5 + 32)

client.register_tools([km_to_miles, miles_to_km, celsius_to_fahrenheit])
```

### Pattern 5: Tool Delegation

Tools that call other tools (requires client access):

```python
@tool("Performs comprehensive file analysis")
def analyze_file(filepath: str) -> dict:
    """Read file and analyze its contents."""

    # This pattern requires passing client to tools
    # Better approach: Use agents for multi-step tasks

    return {
        "message": "Use ReACT agent for multi-tool workflows",
        "suggestion": "See Pattern 6 below"
    }
```

**Better approach**: Use agents for multi-step, multi-tool tasks.

### Pattern 6: Agent-Based Multi-Tool Workflows

For complex workflows requiring multiple tools and decision-making:

```python
from local_llm_sdk.agents import ReACT

# Create agent with all necessary tools
agent = ReACT(client)

# Complex task requiring multiple tools and decisions
result = agent.run(
    task="Read data.csv, calculate average of numbers in first column, "
         "write result to output.txt, then read it back to verify",
    max_iterations=20,
    verbose=True
)

# Agent automatically:
# 1. Reads data.csv with filesystem_operation
# 2. Calculates average with execute_python or math_calculator
# 3. Writes result to output.txt
# 4. Reads output.txt to verify
# 5. Returns final confirmation

print(result.final_response)
print(f"Steps taken: {result.iterations}")
print(f"Tools used: {result.metadata['total_tool_calls']}")
```

### Pattern 7: Tool Result Caching

Cache expensive tool results:

```python
from functools import lru_cache

@lru_cache(maxsize=100)
def _expensive_computation(value: int) -> int:
    """Cached expensive computation."""
    # Simulate expensive operation
    import time
    time.sleep(2)
    return value ** 2

@tool("Performs expensive computation with caching")
def cached_compute(value: int) -> dict:
    """Compute with automatic caching."""
    result = _expensive_computation(value)
    return {
        "input": value,
        "result": result,
        "note": "Results are cached for performance"
    }
```

### Pattern 8: Batch Tool Operations

Process multiple items efficiently:

```python
@tool("Converts multiple temperatures at once")
def batch_temperature_convert(
    values: List[float],
    from_unit: Literal["celsius", "fahrenheit"],
    to_unit: Literal["celsius", "fahrenheit"]
) -> dict:
    """Convert multiple temperature values in one call."""

    def convert_one(value: float) -> float:
        if from_unit == "celsius" and to_unit == "fahrenheit":
            return (value * 9/5) + 32
        elif from_unit == "fahrenheit" and to_unit == "celsius":
            return (value - 32) * 5/9
        return value

    results = [convert_one(v) for v in values]

    return {
        "input_values": values,
        "output_values": results,
        "from_unit": from_unit,
        "to_unit": to_unit,
        "count": len(values)
    }
```

---

## Best Practices

### 1. Write Clear Descriptions

**Good descriptions help the LLM choose the right tool:**

```python
# ❌ Bad: Vague, unhelpful
@tool("Does text stuff")
def process_text(text: str) -> dict:
    return {"result": text.upper()}

# ✅ Good: Clear, specific
@tool("Converts text to uppercase letters")
def uppercase_text(text: str) -> dict:
    """Transform all characters in text to uppercase."""
    return {"result": text.upper()}
```

### 2. Use Type Hints

**Type hints are required for schema generation:**

```python
# ❌ Bad: No type hints
@tool("Add numbers")
def add(a, b):  # LLM doesn't know types!
    return {"result": a + b}

# ✅ Good: Clear type hints
@tool("Add two numbers with precision")
def add(a: float, b: float) -> dict:
    return {"result": a + b}
```

### 3. Return Structured Dictionaries

**Structured returns are easier for LLMs to parse:**

```python
# ❌ Bad: Returns raw value
@tool("Calculate square")
def square(x: float) -> float:
    return x * x

# ✅ Good: Returns structured dict
@tool("Calculate square of a number")
def square(x: float) -> dict:
    return {
        "input": x,
        "result": x * x,
        "operation": "square"
    }
```

### 4. Handle Errors Gracefully

**Return errors in dict, don't raise exceptions:**

```python
# ❌ Bad: Raises exception
@tool("Divide numbers")
def divide(a: float, b: float) -> dict:
    return {"result": a / b}  # Crashes on b=0!

# ✅ Good: Handles errors
@tool("Safely divide two numbers")
def divide(a: float, b: float) -> dict:
    if b == 0:
        return {"error": "Division by zero", "result": None}
    return {"result": a / b, "error": None}
```

### 5. Provide Helpful Docstrings

**Docstrings help both developers and LLMs:**

```python
@tool("Calculates rectangle area and perimeter")
def rectangle_metrics(width: float, height: float) -> dict:
    """
    Calculate area and perimeter of a rectangle.

    Args:
        width: Width of the rectangle in units
        height: Height of the rectangle in units

    Returns:
        Dictionary containing:
        - area: Width × height
        - perimeter: 2 × (width + height)
        - dimensions: Original width and height
    """
    return {
        "area": width * height,
        "perimeter": 2 * (width + height),
        "dimensions": {"width": width, "height": height}
    }
```

### 6. Use Literal for Enums

**Literal types provide clear options:**

```python
from typing import Literal

# ✅ Good: Clear choices
@tool("Convert text case")
def transform_text(
    text: str,
    transform: Literal["upper", "lower", "title", "capitalize"]
) -> dict:
    """Transform text case with specified transformation."""
    transformations = {
        "upper": str.upper,
        "lower": str.lower,
        "title": str.title,
        "capitalize": str.capitalize
    }
    return {"result": transformations[transform](text)}
```

### 7. Keep Tools Focused

**Single responsibility principle:**

```python
# ❌ Bad: Tool does too much
@tool("Do everything with text")
def text_operations(text: str, operation: str, case: str, count_type: str) -> dict:
    # Too many operations in one tool!
    pass

# ✅ Good: Separate focused tools
@tool("Convert text to uppercase")
def uppercase(text: str) -> dict:
    return {"result": text.upper()}

@tool("Count characters in text")
def count_chars(text: str) -> dict:
    return {"count": len(text)}

@tool("Reverse text")
def reverse_text(text: str) -> dict:
    return {"result": text[::-1]}
```

### 8. Validate Inputs

**Always validate before processing:**

```python
@tool("Get file extension from filename")
def get_extension(filename: str) -> dict:
    """Extract file extension from filename."""

    # Validate input
    if not filename or not isinstance(filename, str):
        return {"error": "Invalid filename", "extension": None}

    if '.' not in filename:
        return {"error": "No extension found", "extension": None}

    # Process
    extension = filename.split('.')[-1]

    return {
        "filename": filename,
        "extension": extension,
        "error": None
    }
```

### 9. Include Metadata in Results

**Extra context helps LLMs provide better answers:**

```python
@tool("Search for items in database")
def search_items(query: str, limit: int = 10) -> dict:
    """Search database with query string."""

    # Simulate search
    results = perform_search(query, limit)

    return {
        "query": query,
        "results": results,
        "count": len(results),
        "limit": limit,
        "has_more": len(results) == limit,  # Indicates truncation
        "timestamp": datetime.now().isoformat()
    }
```

### 10. Document Side Effects

**Make side effects explicit:**

```python
@tool("Writes data to file (WARNING: Modifies filesystem)")
def write_data(filepath: str, content: str) -> dict:
    """
    Write content to file.

    **SIDE EFFECTS**: Creates or overwrites file at filepath.

    Args:
        filepath: Path where file will be written
        content: Content to write to file

    Returns:
        Success status and file information
    """
    from pathlib import Path

    path = Path(filepath)
    path.write_text(content)

    return {
        "success": True,
        "filepath": str(path),
        "size_bytes": len(content.encode()),
        "side_effect": "File created/overwritten"
    }
```

### 11. Avoid Stateful Tools When Possible

**Prefer stateless tools for predictability:**

```python
# ⚠️ Avoid: Global state
_global_config = {}

@tool("Update global config")
def update_config(key: str, value: str) -> dict:
    _global_config[key] = value  # Problematic!
    return {"updated": key}

# ✅ Better: Stateless, use parameters
@tool("Create configuration object")
def create_config(settings: dict) -> dict:
    """Create a configuration object from settings."""
    return {
        "config": settings,
        "valid": validate_config(settings)
    }
```

### 12. Test Tools Independently

**Unit test tools before using with LLM:**

```python
def test_rectangle_area():
    """Test rectangle_area tool."""
    result = rectangle_area(10, 5)
    assert result["area"] == 50
    assert result["perimeter"] == 30
    assert result["error"] is None

def test_rectangle_area_error_handling():
    """Test error handling."""
    result = rectangle_area(-10, 5)
    assert "error" in result
    assert result["error"] is not None

# Run tests before registering with client
test_rectangle_area()
test_rectangle_area_error_handling()
```

---

## Troubleshooting

### Problem 1: Tools Not Being Called

**Symptoms:**
- LLM generates answer without using tools
- Response is approximated or incorrect
- `client.last_tool_calls` is empty

**Solutions:**

```python
# Solution 1: Use tool_choice="required"
response = client.chat(
    "Calculate 127 * 893",
    use_tools=True,
    tool_choice="required"  # Force tool usage
)

# Solution 2: Be more explicit in prompt
response = client.chat(
    "Use the calculator tool to compute exactly: 127 * 893",
    use_tools=True
)

# Solution 3: Check tools are registered
print("Registered tools:", client.tools.list_tools())
# If empty, register tools:
client.register_tools_from(None)  # Built-in tools

# Solution 4: Verify model supports function calling
# Some older/smaller models don't support tools well
# Recommended: Mistral, Qwen, Hermes, Functionary models
```

### Problem 2: Tool Not Found Error

**Symptoms:**
- Error: "Unknown tool: tool_name"
- Tool execution fails
- LLM mentions tool but SDK can't execute it

**Solutions:**

```python
# Solution 1: Verify tool is registered
print("Available tools:", client.tools.list_tools())

# Solution 2: Register the tool
@tool("Description")
def my_tool(param: str) -> dict:
    return {"result": param}

client.register_tools([my_tool])

# Solution 3: Check for name conflicts
# Tool names must be unique
# Registering same name twice overwrites

# Solution 4: Check tool decorator syntax
# ❌ Bad:
@tool  # Missing description!
def my_tool(): pass

# ✅ Good:
@tool("Tool description")
def my_tool(param: str) -> dict:
    return {"result": param}
```

### Problem 3: Type Validation Errors

**Symptoms:**
- LLM passes wrong types to tool
- Tool receives unexpected values
- Execution fails with type errors

**Solutions:**

```python
# Solution 1: Add type validation in tool
@tool("Calculate with validation")
def validated_calc(a: float, b: float) -> dict:
    # Validate types
    if not isinstance(a, (int, float)):
        return {"error": f"Expected number for 'a', got {type(a).__name__}"}
    if not isinstance(b, (int, float)):
        return {"error": f"Expected number for 'b', got {type(b).__name__}"}

    return {"result": a + b}

# Solution 2: Use Literal for strict choices
from typing import Literal

@tool("Process with strict choices")
def process(mode: Literal["fast", "slow", "balanced"]) -> dict:
    # Type is enforced at schema level
    return {"mode": mode}

# Solution 3: Coerce types in tool
@tool("Flexible number processing")
def flexible_calc(a, b) -> dict:  # No type hints
    try:
        a_num = float(a)  # Coerce to number
        b_num = float(b)
        return {"result": a_num + b_num}
    except ValueError:
        return {"error": "Could not convert inputs to numbers"}
```

### Problem 4: Tool Execution Timeout

**Symptoms:**
- Tool takes too long to execute
- Request times out
- No response returned

**Solutions:**

```python
# Solution 1: Increase client timeout
client = LocalLLMClient(
    base_url="http://localhost:1234/v1",
    model="your-model",
    timeout=600  # 10 minutes (default: 300s)
)

# Solution 2: Add timeout to tool
@tool("Execute with timeout")
def long_running_operation(data: str) -> dict:
    import signal

    def timeout_handler(signum, frame):
        raise TimeoutError("Operation timed out")

    # Set timeout
    signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(30)  # 30 second timeout

    try:
        result = expensive_operation(data)
        signal.alarm(0)  # Cancel timeout
        return {"result": result}
    except TimeoutError:
        return {"error": "Operation exceeded timeout"}

# Solution 3: Use async for long operations
# Consider running tools in background thread/process
```

### Problem 5: Tool Returns Incorrect Schema

**Symptoms:**
- LLM confused by tool parameters
- Wrong parameters passed
- Schema validation errors

**Solutions:**

```python
# Solution 1: Add type hints
# ❌ Bad:
@tool("Calculate")
def calc(a, b):  # No types!
    return {"result": a + b}

# ✅ Good:
@tool("Calculate two numbers")
def calc(a: float, b: float) -> dict:
    return {"result": a + b}

# Solution 2: Check generated schema
schemas = client.tools.get_schemas_dict()
import json
print(json.dumps(schemas, indent=2))
# Verify schema matches expectations

# Solution 3: Use complex types correctly
from typing import List, Dict

@tool("Process data")
def process(items: List[str], config: Dict[str, int]) -> dict:
    # Correctly typed list and dict
    return {"processed": len(items)}
```

### Problem 6: Tools Not Available in Chat

**Symptoms:**
- Tools registered but not used
- `use_tools=True` doesn't help
- LLM acts like tools don't exist

**Solutions:**

```python
# Solution 1: Verify tools passed to request
response = client.chat(
    "Calculate something",
    use_tools=True,  # Must be True!
    return_full_response=True
)
# Check if tools were in request (debug mode)

# Solution 2: Check model compatibility
# Not all models support function calling
# Test with known-good model: mistralai/magistral-small-2509

# Solution 3: Verify registration before chat
client.register_tools_from(None)  # Register built-ins
print("Tools:", client.tools.list_tools())  # Should not be empty
response = client.chat("Use calculator", use_tools=True)

# Solution 4: Check LM Studio configuration
# Ensure function calling is enabled in LM Studio settings
```

### Problem 7: Conversation State Lost

**Symptoms:**
- Tool results not remembered
- LLM doesn't see previous tool calls
- Context missing in multi-turn conversations

**Solutions:**

```python
# Solution 1: Use chat_with_history
history = []

response1, history = client.chat_with_history(
    "Calculate 10 * 5",
    history,
    use_tools=True
)

# Tool result is in history
response2, history = client.chat_with_history(
    "Add 20 to that",  # References previous result
    history,
    use_tools=True
)

# Solution 2: Check last_conversation_additions
client.chat("Calculate something", use_tools=True)
print("New messages:", len(client.last_conversation_additions))
# Should include: assistant + tool + assistant messages

# Solution 3: Inspect full conversation
print("Full conversation:", client.conversation)
# Verify tool results are present
```

### Problem 8: Security Concerns with execute_python

**Symptoms:**
- Worried about code injection
- Concerned about filesystem access
- Need isolation guarantees

**Solutions:**

```python
# execute_python is already sandboxed:
# - Runs in subprocess (isolated from main process)
# - Uses temporary working directory
# - Has timeout (default 30s)
# - Can't access parent process memory

# Solution 1: Increase isolation with custom tool
@tool("Execute Python in docker container")
def execute_python_docker(code: str) -> dict:
    """Execute Python in isolated Docker container."""
    import subprocess

    # Run in throwaway container
    result = subprocess.run(
        ["docker", "run", "--rm", "--network=none",
         "python:3.11-slim", "python", "-c", code],
        capture_output=True,
        text=True,
        timeout=30
    )

    return {
        "stdout": result.stdout,
        "stderr": result.stderr,
        "returncode": result.returncode
    }

# Solution 2: Restrict filesystem_operation paths
@tool("Safe file read")
def safe_read_file(filepath: str) -> dict:
    """Read file with path restrictions."""
    from pathlib import Path

    # Only allow specific directory
    safe_dir = Path("/app/safe_data")
    requested_path = (safe_dir / filepath).resolve()

    # Ensure path is within safe_dir
    if not str(requested_path).startswith(str(safe_dir)):
        return {"error": "Path outside safe directory"}

    if not requested_path.exists():
        return {"error": "File not found"}

    return {
        "content": requested_path.read_text(),
        "path": str(requested_path)
    }
```

### Problem 9: Debugging Tool Execution

**Symptoms:**
- Tool called but wrong result
- Need to see what tool actually received
- Debugging complex tool chains

**Solutions:**

```python
# Solution 1: Use verbose output
client.print_tool_calls(detailed=True)
# Shows full arguments and results as JSON

# Solution 2: Add logging to tools
import logging
logging.basicConfig(level=logging.DEBUG)

@tool("Debug tool")
def debug_tool(param: str) -> dict:
    logging.debug(f"Tool called with param: {param}")
    result = process(param)
    logging.debug(f"Tool returning: {result}")
    return result

# Solution 3: Inspect tool call history
for tool_call in client.last_tool_calls:
    print(f"Tool: {tool_call.function.name}")
    print(f"Args: {tool_call.function.arguments}")
    # Add breakpoint here for step debugging

# Solution 4: Use MLflow tracing
# See docs/guides/mlflow-observability.md for details
import mlflow
mlflow.set_tracking_uri("http://localhost:5000")

# Tool calls automatically traced with MLflow enabled
response = client.chat("Use tools", use_tools=True)
# View traces in MLflow UI at http://localhost:5000
```

### Problem 10: Performance Issues

**Symptoms:**
- Tool calls are slow
- Multiple round-trips delay response
- Latency too high for production

**Solutions:**

```python
# Solution 1: Batch operations in tools
@tool("Batch process multiple items")
def batch_process(items: List[str]) -> dict:
    """Process multiple items in one call instead of N calls."""
    results = [process_one(item) for item in items]
    return {"results": results}

# Solution 2: Use tool_choice="required" to skip decision phase
response = client.chat(
    "Calculate something",
    use_tools=True,
    tool_choice="required"  # Skip "should I use tool?" decision
)

# Solution 3: Cache expensive tool results
from functools import lru_cache

@lru_cache(maxsize=100)
def expensive_operation(key: str):
    # Results cached automatically
    return compute(key)

@tool("Cached tool")
def cached_tool(key: str) -> dict:
    return {"result": expensive_operation(key)}

# Solution 4: Optimize tool implementation
# - Use efficient algorithms
# - Minimize I/O operations
# - Avoid unnecessary computations
# - Return only needed data (not huge payloads)
```

---

## Additional Resources

### Documentation
- [OpenAI Function Calling Specification](https://platform.openai.com/docs/guides/function-calling)
- [Local LLM SDK API Reference](../api/client.md)
- [ReACT Agents Guide](./react-agents.md)
- [MLflow Observability Guide](./mlflow-observability.md)

### Example Code
- [Notebook 04: Tool Calling Basics](../../notebooks/04-tool-calling-basics.ipynb)
- [Notebook 05: Custom Tools](../../notebooks/05-custom-tools.ipynb)
- [Notebook 06: Filesystem & Code Execution](../../notebooks/06-filesystem-code-execution.ipynb)

### Tests
- [Tool Registry Tests](../../tests/test_tools.py)
- [Client Tool Integration Tests](../../tests/test_client.py)
- [Built-in Tools Tests](../../tests/test_tools.py)

---

## Summary

Tool calling extends LLM capabilities with:

- **Precision**: Exact calculations, file operations, API calls
- **Real-time data**: Access current information beyond training data
- **Automation**: Execute actions, modify systems, query databases
- **Reliability**: Guaranteed correct execution (no hallucinations)

Key takeaways:

1. Use `@tool` decorator to create custom tools
2. Type hints are required for schema generation
3. Return structured dictionaries with error handling
4. Use `tool_choice` to control when tools are used
5. Inspect execution with `print_tool_calls()` or `return_full_response=True`
6. For complex workflows, use ReACT agents
7. Handle errors gracefully within tools
8. Follow best practices for maintainable, reliable tools

Happy tool building!
