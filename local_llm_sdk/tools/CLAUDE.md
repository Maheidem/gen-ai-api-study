# local_llm_sdk/tools/

## Purpose
Tool/function calling system enabling LLMs to execute actions. Provides registry, decorators, and built-in tools following OpenAI function calling specification.

## Contents
- `__init__.py` - Exports `ToolRegistry`, `@tool` decorator
- `registry.py` - `ToolRegistry` class managing tool definitions and execution
  - Schema generation for OpenAI function calling
  - Tool execution with error handling
  - Registration from modules or individual functions
- `builtin.py` - Six production-ready tools:
  - `execute_python`: Run Python code in isolated sandbox
  - `filesystem_operation`: File/directory operations (read/write/create/delete)
  - `math_calculator`: Basic arithmetic operations
  - `text_transformer`: String transformations (upper/lower/title)
  - `char_counter`: Count characters in text
  - `get_weather`: Mock weather data (example tool)

## Relationships
- **Parent**: Used by `../client.py:LocalLLMClient` for automatic tool handling
- **Used by**: Agents when `use_tools=True` in chat calls
- **Schema format**: OpenAI Function Calling specification (JSON Schema)

## Getting Started
1. **Read `registry.py:11-50`** - Understanding `ToolRegistry.__init__()` and tool storage
2. **Check `registry.py:63-116`** - See `get_schemas()` for OpenAI function format
3. **Look at `builtin.py:11-60`** - Example tool: `execute_python` with @tool decorator
4. **Understand pattern** from `registry.py:28-61` - How `register()` works

## Creating Custom Tools
```python
from local_llm_sdk.tools import tool

@tool("Calculate fibonacci number")
def fibonacci(n: int) -> dict:
    \"\"\"
    Args:
        n: Position in fibonacci sequence
    \"\"\"
    if n <= 1:
        return {"result": n}
    a, b = 0, 1
    for _ in range(2, n+1):
        a, b = b, a+b
    return {"result": b}

# Tool is automatically registered when module is imported
```

## Tool Execution Flow
1. Client calls `chat(use_tools=True)`
2. Client adds tool schemas to request via `registry.get_schemas()`
3. LLM returns `tool_calls` in response
4. Client calls `registry.execute(tool_name, args)` for each call
5. Tool results added as messages, sent back to LLM
6. Process repeats until no more tool calls

## Important Notes
- Tools must return JSON-serializable dict
- `execute_python` runs in temp dir (isolated from project)
- `filesystem_operation` works in project directory
- Error handling: Tools return error dict, don't raise exceptions
- Typing: Use Python type hints for automatic schema generation