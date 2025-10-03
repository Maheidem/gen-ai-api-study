# local_llm_sdk/tools/

## Purpose
Tool/function calling system enabling LLMs to execute actions. Provides registry, decorators, and a powerful unified bash tool following OpenAI function calling specification.

## Contents
- `__init__.py` - Exports `ToolRegistry`, `@tool` decorator
- `registry.py` - `ToolRegistry` class managing tool definitions and execution
  - Schema generation for OpenAI function calling
  - Tool execution with error handling
  - Registration from modules or individual functions
- `builtin.py` - Production-ready unified bash tool:
  - `bash`: Execute ANY command with full terminal capabilities
    - Python execution: `python -c "code"` or `python script.py`
    - File operations: `cat`, `ls`, `mkdir`, `cp`, `mv`, `rm`, `echo > file`
    - Git operations: `git status`, `git add`, `git commit`
    - Text processing: `grep`, `sed`, `awk`, Python string ops
    - Math: Python calculations via `python -c "print(42 * 17)"`
    - Multi-step: Chain commands with `&&`

## Design Philosophy
- **Single Unified Interface**: No tool coordination issues
- **Works in Project Directory**: All commands execute in current working directory
- **300s Timeout**: Suitable for local LLMs
- **Comprehensive LLM-Friendly Description**: Teaches LLMs best practices
- **Structured Return**: `{success, stdout, stderr, return_code, command}`

## Relationships
- **Parent**: Used by `../client.py:LocalLLMClient` for automatic tool handling
- **Used by**: Agents when `use_tools=True` in chat calls
- **Schema format**: OpenAI Function Calling specification (JSON Schema)

## Getting Started
1. **Read `registry.py:11-50`** - Understanding `ToolRegistry.__init__()` and tool storage
2. **Check `registry.py:63-116`** - See `get_schemas()` for OpenAI function format
3. **Look at `builtin.py:12-89`** - The unified bash tool with comprehensive docs
4. **Understand pattern** from `registry.py:28-61` - How `register()` works

## Using the Bash Tool
```python
from local_llm_sdk import create_client_with_tools

client = create_client_with_tools()
result = client.chat("Calculate factorial of 5 and save to file", use_tools=True)

# LLM uses bash tool:
# 1. bash("python -c 'import math; print(math.factorial(5))'")
# 2. bash("echo '120' > result.txt")
```

## Creating Custom Tools
```python
from local_llm_sdk.tools import tool

@tool("Calculate fibonacci number")
def fibonacci(n: int) -> dict:
    """
    Args:
        n: Position in fibonacci sequence
    """
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
- `bash` tool executes in project directory (current working directory)
- Error handling: Tools return error dict, don't raise exceptions
- Typing: Use Python type hints for automatic schema generation
- Command chaining: Use `&&` to combine multiple operations in single bash call
