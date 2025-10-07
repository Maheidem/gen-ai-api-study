"""
Production-ready bash tool for LM Studio.
Provides full terminal capabilities in a single unified interface.
"""

from typing import Optional
from .registry import tool
import subprocess
import sys


@tool("Execute bash commands with full terminal capabilities")
def bash(command: str, timeout: int = 300) -> dict:
    """
    Execute any bash command with full terminal capabilities.

    ═══════════════════════════════════════════════════════════════════
    WHEN TO USE THIS TOOL
    ═══════════════════════════════════════════════════════════════════

    ⚠️  USE THIS TOOL ALWAYS for ANY task unless there is another tool
        with an EXPLICIT, DIRECT relationship to the specific task.

    This is your UNIVERSAL tool for:
    • ALL calculations and mathematics
    • ALL file operations (read, write, copy, move, delete)
    • ALL text processing (count, transform, search, replace)
    • ALL code execution (Python, scripts, commands)
    • ALL system operations (list files, check status, etc.)

    ═══════════════════════════════════════════════════════════════════
    CAPABILITIES & EXAMPLES
    ═══════════════════════════════════════════════════════════════════

    📊 MATHEMATICS & CALCULATIONS
    ───────────────────────────────────────────────────────────────────
    • Simple math:
      bash(command="python -c 'print(42 + 58)'")
      bash(command="python -c 'print(12 * 8)'")

    • Complex calculations:
      bash(command="python -c 'import math; print(math.factorial(10))'")
      bash(command="python -c 'print(2 ** 100)'")
      bash(command="python -c 'print((15 + 25) * 2 / 4 - 10)'")

    📁 FILE OPERATIONS
    ───────────────────────────────────────────────────────────────────
    • Read file:
      bash(command="cat README.md")
      bash(command="head -10 file.txt")
      bash(command="tail -20 logfile.log")

    • Write file:
      bash(command="echo 'Hello World' > test.txt")
      bash(command="echo 'Line 2' >> test.txt")  # append

    • Copy/Move/Delete:
      bash(command="cp source.txt destination.txt")
      bash(command="mv old.txt new.txt")
      bash(command="rm unwanted.txt")
      bash(command="rm -rf directory/")

    • Check if file exists:
      bash(command="test -f myfile.txt && echo 'exists' || echo 'not found'")
      bash(command="ls myfile.txt")

    • File permissions:
      bash(command="chmod +x script.sh")
      bash(command="chmod 644 file.txt")

    📝 TEXT PROCESSING
    ───────────────────────────────────────────────────────────────────
    • Count characters:
      bash(command="echo 'hello world' | wc -c")
      bash(command="wc -c < file.txt")

    • Count words:
      bash(command="echo 'the quick brown fox' | wc -w")
      bash(command="wc -w file.txt")

    • Count lines:
      bash(command="wc -l file.txt")

    • Uppercase/Lowercase:
      bash(command="echo 'hello' | tr '[:lower:]' '[:upper:]'")
      bash(command="echo 'HELLO' | tr '[:upper:]' '[:lower:]'")

    • Search/Grep:
      bash(command="grep 'pattern' file.txt")
      bash(command="grep -i 'error' logfile.log")  # case-insensitive
      bash(command="grep -c 'word' file.txt")  # count occurrences

    • Replace text:
      bash(command="sed 's/old/new/g' file.txt")
      bash(command="sed -i 's/pattern/replacement/' file.txt")  # in-place

    • Extract fields:
      bash(command="echo 'a,b,c' | cut -d',' -f2")
      bash(command="awk '{print $1}' file.txt")  # first column

    🐍 PYTHON EXECUTION
    ───────────────────────────────────────────────────────────────────
    • Inline Python:
      bash(command="python -c 'print(5 * 24)'")
      bash(command="python -c 'x = [1,2,3]; print(sum(x))'")

    • Multi-line Python:
      bash(command="python -c 'import sys\\nfor i in range(5):\\n    print(i)'")

    • Run Python script:
      bash(command="python script.py")
      bash(command="python script.py arg1 arg2")

    • Python with packages:
      bash(command="python -c 'import json; print(json.dumps({\"key\": \"value\"}))'")

    📂 DIRECTORY OPERATIONS
    ───────────────────────────────────────────────────────────────────
    • List files:
      bash(command="ls")
      bash(command="ls -la")  # detailed
      bash(command="ls *.py")  # specific pattern

    • Create directory:
      bash(command="mkdir temp")
      bash(command="mkdir -p path/to/nested/dir")

    • Current directory:
      bash(command="pwd")

    • Find files:
      bash(command="find . -name '*.txt'")
      bash(command="find . -type f -mtime -1")  # modified last day

    🔗 COMMAND CHAINING & PIPING
    ───────────────────────────────────────────────────────────────────
    • Sequential (AND):
      bash(command="mkdir temp && cd temp && echo 'data' > file.txt")
      bash(command="echo 'test' > file.txt && cat file.txt")

    • Conditional (OR):
      bash(command="test -f file.txt && cat file.txt || echo 'not found'")

    • Pipes:
      bash(command="cat file.txt | grep pattern | wc -l")
      bash(command="ls -la | grep '.py' | wc -l")

    • Redirect output:
      bash(command="python script.py > output.txt")
      bash(command="python script.py 2>&1 > all_output.txt")  # stdout + stderr

    ═══════════════════════════════════════════════════════════════════
    COMMON TASK EXAMPLES
    ═══════════════════════════════════════════════════════════════════

    Task: "Calculate 5 factorial"
    → bash(command="python -c 'import math; print(math.factorial(5))'")

    Task: "Count characters in 'Testing123'"
    → bash(command="echo 'Testing123' | wc -c")

    Task: "Convert 'hello world' to uppercase"
    → bash(command="echo 'hello world' | tr '[:lower:]' '[:upper:]'")

    Task: "Write 'Hello' to test.txt then read it"
    → bash(command="echo 'Hello' > test.txt && cat test.txt")

    Task: "Calculate (15+25)*2/4-10"
    → bash(command="python -c 'print((15+25)*2/4-10)'")

    Task: "Count words in a sentence"
    → bash(command="echo 'the quick brown fox' | wc -w")

    Task: "Check if file.txt exists"
    → bash(command="test -f file.txt && echo 'yes' || echo 'no'")

    ═══════════════════════════════════════════════════════════════════
    IMPORTANT NOTES
    ═══════════════════════════════════════════════════════════════════

    • Execution Context: Commands run in the current working directory
    • Timeout: 300 seconds default (suitable for complex operations)
    • Quoting: Always quote paths with spaces: "path with spaces/file.txt"
    • Error Handling: Check return_code (0 = success, non-zero = error)
    • Output: Both stdout and stderr are captured

    ═══════════════════════════════════════════════════════════════════

    Args:
        command: The bash command to execute (required)
        timeout: Maximum execution time in seconds (default: 300)

    Returns:
        dict with keys:
        - success (bool): True if return_code == 0
        - stdout (str): Standard output from command
        - stderr (str): Standard error from command
        - return_code (int): Exit code (0 = success)
        - command (str): The command that was executed
    """
    try:
        # Execute command in current working directory
        result = subprocess.run(
            command,
            shell=True,
            capture_output=True,
            text=True,
            timeout=timeout,
            executable='/bin/bash'  # Ensure bash is used (not sh)
        )

        return {
            "success": result.returncode == 0,
            "stdout": result.stdout,
            "stderr": result.stderr,
            "return_code": result.returncode,
            "command": command
        }

    except subprocess.TimeoutExpired:
        return {
            "success": False,
            "stdout": "",
            "stderr": f"Command timed out after {timeout} seconds",
            "return_code": -1,
            "command": command,
            "error": f"Timeout after {timeout}s"
        }
    except Exception as e:
        return {
            "success": False,
            "stdout": "",
            "stderr": str(e),
            "return_code": -1,
            "command": command,
            "error": str(e)
        }
