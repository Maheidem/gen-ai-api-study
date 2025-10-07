# Tool Execution Best Practices

**Last Updated**: 2025-10-06
**Status**: Production Reference

This document provides best practices for implementing tool functions in the Local LLM SDK, covering architecture, common pitfalls, error handling, and security considerations.

---

## Core Principles

1. **Tools should be simple and focused** - One clear responsibility
2. **Always return dictionaries** - Structured, JSON-serializable results
3. **Handle errors gracefully** - Return error dict, don't raise exceptions
4. **Use type hints** - Enables automatic schema generation
5. **Never execute untrusted code directly** - Use sandboxing/validation

---

## Tool Architecture

### The Unified bash Tool Pattern

The SDK provides a single, comprehensive `bash` tool that replaces multiple specialized tools.

**Why one tool instead of many?**
- ✅ Simpler mental model for LLM
- ✅ Reduces tool call overhead
- ✅ Full terminal capabilities (Python, files, git, text processing)
- ✅ Easier to maintain
- ✅ More flexible for complex tasks

**What it supports:**
```python
from local_llm_sdk.tools import builtin

# Math calculations
bash(command="python -c 'print(42 * 17)'")

# File operations
bash(command="cat myfile.txt")
bash(command="ls -la /path/to/directory")

# Text processing
bash(command="echo 'HELLO' | tr '[:upper:]' '[:lower:]'")

# Git operations
bash(command="git log --oneline -5")

# Multi-step workflows
bash(command="cd /tmp && mkdir test && cd test && touch file.txt && ls")
```

**Implementation:**
```python
@tool("Execute bash commands in a safe subprocess environment")
def bash(command: str, timeout: int = 30) -> dict:
    """
    Execute bash commands for calculations, file operations, text processing, etc.

    Args:
        command: The bash command to execute
        timeout: Maximum execution time in seconds (default: 30)

    Returns:
        dict with keys: success, stdout, stderr, return_code
    """
    import subprocess
    import sys

    try:
        result = subprocess.run(
            command,
            shell=True,
            capture_output=True,
            text=True,
            timeout=timeout,
            executable='/bin/bash'
        )

        return {
            "success": result.returncode == 0,
            "stdout": result.stdout,
            "stderr": result.stderr,
            "return_code": result.returncode
        }

    except subprocess.TimeoutExpired:
        return {
            "success": False,
            "error": f"Command timed out after {timeout} seconds"
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }
```

---

## Common Pitfalls and Solutions

### Pitfall 1: String Wrapping for Code Execution ❌

**The Problem:**

Wrapping user-provided code in triple-quoted strings breaks when code contains quotes.

**Bad Example (execute_python tool, OLD):**
```python
def execute_python(code: str) -> dict:
    """BROKEN: String wrapping approach."""
    # Wrap code in triple quotes
    exec("""
    {chr(10).join(line for line in code.split(chr(10)))}
    """, execution_namespace)
```

**Why it fails:**
```python
# User's code with docstring:
code = '''
def is_prime(n):
    """Check if number is prime"""  # ← Contains """
    return n > 1
'''

# Becomes (BROKEN):
exec("""
def is_prime(n):
    """Check if number is prime"""  # ← TERMINATES THE OUTER """
    return n > 1
""", namespace)
# Result: SyntaxError: unterminated string literal
```

**The Fix: Direct File Execution ✅**

```python
@tool("Execute Python code safely and return results")
def execute_python(code: str, timeout: int = 30) -> dict:
    """Execute Python code in a subprocess."""
    import subprocess
    import tempfile
    import os
    import sys

    try:
        # Create temp file - NO STRING WRAPPING
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write(code)  # Write raw code directly
            temp_file = f.name

        # Execute directly
        result = subprocess.run(
            [sys.executable, temp_file],
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=tempfile.gettempdir()
        )

        os.unlink(temp_file)

        return {
            "success": result.returncode == 0,
            "stdout": result.stdout,
            "stderr": result.stderr,
            "return_code": result.returncode
        }

    except subprocess.TimeoutExpired:
        return {"success": False, "error": f"Timeout after {timeout}s"}
    except Exception as e:
        return {"success": False, "error": str(e)}
```

**Why better:**
- ✅ No string wrapping = no quote escaping issues
- ✅ Simpler implementation (30 lines vs 80 lines)
- ✅ Clear error messages from actual Python interpreter
- ✅ Easier to debug

**Evidence:**
- Fixed critical bug where LLM-generated code with docstrings failed
- Reduced from repeated "unterminated string" errors to clean execution
- Test results: All Python code with quotes/docstrings now works

---

### Pitfall 2: Raising Exceptions Instead of Returning Errors ❌

**The Problem:**

Tools that raise exceptions break the agent execution flow.

**Bad Example:**
```python
@tool("Calculate division")
def divide(a: float, b: float) -> dict:
    """BROKEN: Raises exception."""
    if b == 0:
        raise ValueError("Cannot divide by zero")  # ❌ Breaks agent

    return {"result": a / b}
```

**What happens:**
```python
agent.run("Calculate 10 / 0")
# Result: Unhandled ValueError, agent crashes
# LLM never gets feedback about what went wrong
```

**The Fix: Return Error Dictionaries ✅**

```python
@tool("Calculate division")
def divide(a: float, b: float) -> dict:
    """Handle errors gracefully."""
    if b == 0:
        return {
            "success": False,
            "error": "Cannot divide by zero",
            "suggestion": "Please use a non-zero divisor"
        }

    return {
        "success": True,
        "result": a / b
    }
```

**What happens:**
```python
agent.run("Calculate 10 / 0")
# Result: LLM receives error dict
# LLM: "I cannot divide by zero. Let me try a different approach..."
# Agent continues execution instead of crashing
```

**Pattern:**
```python
@tool("Description")
def tool_function(param: str) -> dict:
    """Always return dict, never raise."""

    # Validate inputs
    if not param:
        return {"success": False, "error": "Parameter required"}

    try:
        # Execute operation
        result = do_something(param)

        return {
            "success": True,
            "result": result
        }

    except Exception as e:
        # Catch ALL exceptions
        return {
            "success": False,
            "error": str(e),
            "type": type(e).__name__
        }
```

---

### Pitfall 3: Not Using Type Hints ❌

**The Problem:**

Without type hints, schema generation fails or produces incorrect schemas.

**Bad Example:**
```python
@tool("Process data")
def process_data(data, options):  # ❌ No type hints
    """BROKEN: Schema generation can't infer types."""
    return {"result": data.upper()}
```

**Generated Schema (BROKEN):**
```json
{
  "type": "function",
  "function": {
    "name": "process_data",
    "parameters": {
      "type": "object",
      "properties": {
        "data": {"type": "string"},  // Guessed
        "options": {"type": "object"}  // Guessed
      }
    }
  }
}
```

**The Fix: Always Use Type Hints ✅**

```python
from typing import Dict, List, Optional

@tool("Process data with specific options")
def process_data(
    data: str,
    options: Optional[Dict[str, str]] = None
) -> dict:
    """
    Process data with specified options.

    Args:
        data: The input data to process
        options: Optional processing options (e.g., {"case": "upper"})

    Returns:
        Dict with processed result
    """
    if options is None:
        options = {}

    result = data.upper() if options.get("case") == "upper" else data.lower()

    return {"success": True, "result": result}
```

**Generated Schema (CORRECT):**
```json
{
  "type": "function",
  "function": {
    "name": "process_data",
    "parameters": {
      "type": "object",
      "properties": {
        "data": {"type": "string"},
        "options": {
          "type": "object",
          "additionalProperties": {"type": "string"}
        }
      },
      "required": ["data"]
    }
  }
}
```

**Supported Type Hints:**
- `str` → `{"type": "string"}`
- `int` → `{"type": "integer"}`
- `float` → `{"type": "number"}`
- `bool` → `{"type": "boolean"}`
- `List[T]` → `{"type": "array", "items": {...}}`
- `Dict[K, V]` → `{"type": "object", "additionalProperties": {...}}`
- `Optional[T]` → Makes parameter optional in schema

---

### Pitfall 4: Complex Nested String Templates ❌

**The Problem:**

Complex f-strings with double braces make debugging impossible.

**Bad Example (from old execute_python):**
```python
# Lines 169-196 of old implementation
output = f"""
{{
    "success": {{"true" if success else "false"}},
    "stdout": "{{{stdout.replace('"', '\\"')}}}"",
    "stderr": "{{{stderr.replace('"', '\\"')}}}"",
}}
"""
# Result: Impossible to debug, escaping errors
```

**The Fix: Use Simple String Concatenation or json.dumps ✅**

```python
import json

# Option 1: Build dict, serialize with json.dumps
result = {
    "success": result.returncode == 0,
    "stdout": result.stdout,
    "stderr": result.stderr
}
return result  # SDK handles JSON serialization

# Option 2: Simple concatenation if needed
if success:
    return {"success": True, "result": value}
else:
    return {"success": False, "error": error_message}
```

**Why better:**
- ✅ No escaping issues
- ✅ Python handles JSON encoding
- ✅ Easy to debug
- ✅ Type-safe

---

### Pitfall 5: Not Setting Timeouts ❌

**The Problem:**

Tools without timeouts can hang indefinitely.

**Bad Example:**
```python
@tool("Run command")
def run_command(command: str) -> dict:
    """BROKEN: No timeout."""
    result = subprocess.run(command, shell=True, capture_output=True)
    # ❌ Could hang forever on: while true; do sleep 1; done
```

**The Fix: Always Set Timeouts ✅**

```python
@tool("Run command with timeout")
def run_command(command: str, timeout: int = 30) -> dict:
    """Safe execution with timeout."""
    try:
        result = subprocess.run(
            command,
            shell=True,
            capture_output=True,
            text=True,
            timeout=timeout  # ✅ Timeout set
        )

        return {
            "success": result.returncode == 0,
            "stdout": result.stdout,
            "stderr": result.stderr
        }

    except subprocess.TimeoutExpired:
        return {
            "success": False,
            "error": f"Command exceeded {timeout} second timeout",
            "suggestion": "Try a simpler command or increase timeout"
        }
```

**Recommended Timeouts:**
- **Quick operations** (math, text processing): 10s
- **File operations** (read, write): 15s
- **Complex operations** (compilation, analysis): 30-60s
- **Never**: No timeout (infinite)

---

## Security Best Practices

### 1. Subprocess Execution

**Always use subprocess.run() with specific parameters:**

```python
# ✅ SAFE
result = subprocess.run(
    command,
    shell=True,           # Controlled shell access
    capture_output=True,  # Capture both stdout and stderr
    text=True,            # Return strings, not bytes
    timeout=30,           # Prevent hangs
    cwd=safe_directory    # Control working directory
)

# ❌ UNSAFE
os.system(command)  # No output capture, no timeout
eval(code)          # Never use for user input
exec(code)          # Only in controlled namespace
```

### 2. Input Validation

**Validate all parameters before execution:**

```python
@tool("Read file")
def read_file(filepath: str) -> dict:
    """Read file with path validation."""
    import os

    # Validate input
    if not filepath:
        return {"success": False, "error": "Filepath required"}

    # Prevent path traversal
    if ".." in filepath or filepath.startswith("/"):
        return {
            "success": False,
            "error": "Invalid path (no .. or absolute paths)"
        }

    # Check file exists
    if not os.path.exists(filepath):
        return {"success": False, "error": f"File not found: {filepath}"}

    try:
        with open(filepath, 'r') as f:
            content = f.read()

        return {"success": True, "content": content}

    except Exception as e:
        return {"success": False, "error": str(e)}
```

### 3. Resource Limits

**Prevent resource exhaustion:**

```python
@tool("Process large file")
def process_file(filepath: str, max_size_mb: int = 10) -> dict:
    """Process file with size limit."""
    import os

    # Check file size
    size_mb = os.path.getsize(filepath) / (1024 * 1024)

    if size_mb > max_size_mb:
        return {
            "success": False,
            "error": f"File too large: {size_mb:.1f}MB (max: {max_size_mb}MB)"
        }

    # Process file
    ...
```

### 4. Sandboxing

**For code execution, use controlled environments:**

```python
@tool("Execute Python code")
def execute_python(code: str, timeout: int = 30) -> dict:
    """Execute in temp directory with timeout."""
    import subprocess
    import tempfile

    # Write to temp file (sandboxed)
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
        f.write(code)
        temp_file = f.name

    try:
        # Execute in temp directory
        result = subprocess.run(
            [sys.executable, temp_file],
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=tempfile.gettempdir()  # Sandboxed working directory
        )

        return {
            "success": result.returncode == 0,
            "stdout": result.stdout,
            "stderr": result.stderr
        }

    finally:
        # Cleanup
        os.unlink(temp_file)
```

---

## Error Handling Patterns

### Pattern 1: Comprehensive Error Dict

```python
def handle_error(e: Exception, context: str = "") -> dict:
    """Standard error response format."""
    return {
        "success": False,
        "error": str(e),
        "error_type": type(e).__name__,
        "context": context,
        "suggestion": "Check input parameters and try again"
    }
```

### Pattern 2: Validation Before Execution

```python
@tool("Tool with validation")
def validated_tool(param: str, required_param: int) -> dict:
    """Tool with comprehensive validation."""

    # Validate required parameters
    if not param:
        return {"success": False, "error": "param is required"}

    if required_param < 0:
        return {"success": False, "error": "required_param must be >= 0"}

    try:
        # Execute operation
        result = perform_operation(param, required_param)

        return {"success": True, "result": result}

    except ValueError as e:
        return {"success": False, "error": f"Invalid value: {e}"}
    except FileNotFoundError as e:
        return {"success": False, "error": f"File not found: {e}"}
    except Exception as e:
        return {"success": False, "error": f"Unexpected error: {e}"}
```

### Pattern 3: Helpful Error Messages

```python
# ❌ BAD: Vague error
return {"success": False, "error": "Operation failed"}

# ✅ GOOD: Specific with suggestion
return {
    "success": False,
    "error": "File 'data.txt' not found in current directory",
    "current_directory": os.getcwd(),
    "suggestion": "Create the file or specify full path"
}
```

---

## Testing Tools

### Unit Test Pattern

```python
# tests/test_tools.py

def test_bash_command_success():
    """Test successful bash execution."""
    from local_llm_sdk.tools.builtin import bash

    result = bash(command="echo 'hello'")

    assert result["success"] is True
    assert "hello" in result["stdout"]
    assert result["return_code"] == 0

def test_bash_command_failure():
    """Test bash error handling."""
    result = bash(command="nonexistent_command_xyz")

    assert result["success"] is False
    assert result["return_code"] != 0
    assert result["stderr"]  # Should have error message

def test_bash_timeout():
    """Test timeout handling."""
    result = bash(command="sleep 100", timeout=1)

    assert result["success"] is False
    assert "timeout" in result["error"].lower()

def test_error_handling():
    """Test that errors return dicts, not exceptions."""
    result = bash(command="exit 1")

    # Should return dict, not raise
    assert isinstance(result, dict)
    assert result["success"] is False
```

---

## Documentation Standards

### Comprehensive Docstring Pattern

```python
@tool("Clear, one-line description")
def tool_name(
    required_param: str,
    optional_param: int = 10,
    timeout: int = 30
) -> dict:
    """
    Detailed description of what the tool does.

    Explain the purpose, use cases, and any important behavior.

    Args:
        required_param: Description of this parameter
        optional_param: Description with default value (default: 10)
        timeout: Maximum execution time in seconds (default: 30)

    Returns:
        Dict with the following keys:
        - success (bool): Whether operation succeeded
        - result (Any): The result if successful
        - error (str): Error message if failed

    Example:
        >>> tool_name("input", optional_param=20)
        {"success": True, "result": "processed input"}
    """
    # Implementation
    ...
```

**What to include:**
1. **One-line summary** - Clear, concise purpose
2. **Detailed description** - Use cases and behavior
3. **Args section** - Each parameter with type and description
4. **Returns section** - Structure of return dict
5. **Example** - Show usage with expected output

---

## Performance Considerations

### 1. Avoid Repeated Subprocess Calls

**Bad:**
```python
@tool("Get file info")
def get_file_info(filepath: str) -> dict:
    """SLOW: Multiple subprocess calls."""
    size = subprocess.run(["wc", "-c", filepath], ...).stdout
    lines = subprocess.run(["wc", "-l", filepath], ...).stdout
    words = subprocess.run(["wc", "-w", filepath], ...).stdout
    # ❌ 3 separate process spawns
```

**Good:**
```python
@tool("Get file info")
def get_file_info(filepath: str) -> dict:
    """FAST: Single call or native Python."""
    import os

    # Use native Python when possible
    size = os.path.getsize(filepath)

    with open(filepath) as f:
        content = f.read()
        lines = content.count('\n')
        words = len(content.split())

    return {
        "success": True,
        "size_bytes": size,
        "line_count": lines,
        "word_count": words
    }
```

### 2. Cache Expensive Operations

```python
from functools import lru_cache

@lru_cache(maxsize=128)
def expensive_operation(param: str) -> str:
    """Cache results of expensive operations."""
    # Heavy computation here
    return result

@tool("Tool with caching")
def cached_tool(param: str) -> dict:
    """Use cached expensive operation."""
    result = expensive_operation(param)
    return {"success": True, "result": result}
```

---

## Migration Guide: execute_python to bash

**Old approach (BROKEN):**
```python
from local_llm_sdk import LocalLLMClient
from local_llm_sdk.tools.builtin import execute_python

client = LocalLLMClient()
client.register_tool("Execute Python code")(execute_python)

result = client.chat("Calculate factorial of 5", use_tools=True)
# ❌ Fails with "unterminated string" if code has docstrings
```

**New approach (WORKS):**
```python
from local_llm_sdk import create_client_with_tools

# bash tool is pre-registered
client = create_client_with_tools()

# LLM will use: bash(command="python -c 'import math; print(math.factorial(5))'")
result = client.chat("Calculate factorial of 5", use_tools=True)
# ✅ Works perfectly
```

**Benefits:**
- ✅ No string wrapping issues
- ✅ Works with any code (quotes, docstrings, etc.)
- ✅ Simpler for LLM to use
- ✅ Full Python access via command line

---

## Key Takeaways

1. **Avoid string wrapping** - Use direct file execution for code
2. **Return dicts, not exceptions** - Graceful error handling
3. **Always use type hints** - Enables proper schema generation
4. **Set timeouts** - Prevent hangs
5. **Validate inputs** - Check parameters before execution
6. **Use bash tool** - One tool for many operations
7. **Clear error messages** - Help LLM and users debug
8. **Test thoroughly** - Unit tests for success and failure cases

---

**References:**
- `.scratchpad/execute_python_analysis.md` - Analysis of string wrapping bug
- `local_llm_sdk/tools/builtin.py` - bash tool implementation
- `local_llm_sdk/tools/registry.py` - Tool registration and schema generation
- `tests/test_tools.py` - Comprehensive tool tests

**Status**: Production implementation complete
**Test Coverage**: 100% for bash tool
**Known Issues**: None (execute_python bug fixed by using bash tool)
