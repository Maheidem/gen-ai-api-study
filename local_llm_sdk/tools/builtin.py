"""
All registered tools for LM Studio.
Just import this file and all tools are ready to use.
"""

from typing import Literal
from .registry import tool


@tool("Count the number of characters in a text string")
def char_counter(text: str) -> dict:
    """Count characters in the provided text."""
    return {
        "text": text,
        "character_count": len(text),
        "word_count": len(text.split()),
        "has_numbers": any(c.isdigit() for c in text)
    }


@tool("Perform mathematical calculations with two numbers")
def math_calculator(
    arg1: float,
    arg2: float,
    operation: Literal["add", "subtract", "multiply", "divide"]
) -> dict:
    """
    Calculate result of mathematical operation.
    Operations: add, subtract, multiply, divide
    """
    operations = {
        "add": lambda a, b: a + b,
        "subtract": lambda a, b: a - b,
        "multiply": lambda a, b: a * b,
        "divide": lambda a, b: a / b if b != 0 else None
    }

    if operation not in operations:
        return {"error": f"Unknown operation: {operation}"}

    if operation == "divide" and arg2 == 0:
        return {"error": "Division by zero"}

    result = operations[operation](arg1, arg2)

    return {
        "arg1": arg1,
        "arg2": arg2,
        "operation": operation,
        "result": result
    }


@tool("Get the current weather for a city (mock data)")
def get_weather(city: str, units: Literal["celsius", "fahrenheit"] = "celsius") -> dict:
    """Get mock weather data for demonstration."""
    # Mock data - in real app, would call weather API
    mock_weather = {
        "new york": {"temp": 22, "condition": "sunny"},
        "london": {"temp": 15, "condition": "rainy"},
        "tokyo": {"temp": 28, "condition": "cloudy"}
    }

    city_lower = city.lower()
    if city_lower in mock_weather:
        weather = mock_weather[city_lower]
        temp = weather["temp"]

        if units == "fahrenheit":
            temp = (temp * 9/5) + 32

        return {
            "city": city,
            "temperature": temp,
            "units": units,
            "condition": weather["condition"]
        }

    return {
        "city": city,
        "error": "City not found in mock data"
    }


@tool("Convert text to uppercase or lowercase")
def text_transformer(text: str, transform: Literal["upper", "lower", "title"] = "upper") -> dict:
    """Transform text case."""
    if transform == "upper":
        result = text.upper()
    elif transform == "lower":
        result = text.lower()
    elif transform == "title":
        result = text.title()
    else:
        return {"error": f"Unknown transform: {transform}"}

    return {
        "original": text,
        "transformed": result,
        "transform_type": transform
    }


@tool("Execute Python code safely and return results")
def execute_python(code: str, timeout: int = 30) -> dict:
    """
    Execute Python code in a subprocess and return the results.
    Supports both expressions and statements with proper output capture.
    """
    import subprocess
    import tempfile
    import os
    import sys
    from pathlib import Path

    try:
        # Create a temporary file for the code
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            # Wrap the code to capture stdout, stderr, and handle both expressions and statements
            wrapper_code = f'''
import sys
import io
import traceback
import os

# Redirect stdout and stderr
old_stdout = sys.stdout
old_stderr = sys.stderr
stdout_capture = io.StringIO()
stderr_capture = io.StringIO()
sys.stdout = stdout_capture
sys.stderr = stderr_capture

# Change to a safe working directory
import tempfile
import os
work_dir = tempfile.mkdtemp(prefix="python_exec_")
os.chdir(work_dir)

# Execute in a namespace to capture variables
execution_namespace = {{}}

try:
    # Execute the user code in the namespace
    exec("""
{chr(10).join(line for line in code.split(chr(10)))}
""", execution_namespace)

    # If we get here, execution was successful
    execution_result = "success"

    # Capture result variables (look for common names or last assignment)
    captured_result = None
    result_var_names = ['result', 'answer', 'output', '_result', 'res']

    for var_name in result_var_names:
        if var_name in execution_namespace:
            captured_result = execution_namespace[var_name]
            break

    # If no common name found, check if there's only one user-defined variable
    if captured_result is None:
        user_vars = {{k: v for k, v in execution_namespace.items()
                     if not k.startswith('_') and k not in ['__builtins__']}}
        if len(user_vars) == 1:
            captured_result = list(user_vars.values())[0]

except Exception as e:
    # Capture any exception
    execution_result = f"error: {{str(e)}}"
    traceback.print_exc()
    captured_result = None

finally:
    # Restore stdout and stderr
    sys.stdout = old_stdout
    sys.stderr = old_stderr

    # Print results as JSON for easy parsing
    import json
    result_data = {{
        "stdout": stdout_capture.getvalue(),
        "stderr": stderr_capture.getvalue(),
        "status": execution_result,
        "working_directory": work_dir
    }}

    # Add captured result if available
    if captured_result is not None:
        # Convert to string for JSON serialization
        try:
            result_data["captured_result"] = str(captured_result)
        except:
            result_data["captured_result"] = repr(captured_result)

    print(json.dumps(result_data))
'''
            f.write(wrapper_code)
            temp_file = f.name

        # Execute the code with timeout
        result = subprocess.run(
            [sys.executable, temp_file],
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=tempfile.gettempdir()
        )

        # Clean up temp file
        os.unlink(temp_file)

        # Parse the JSON output from our wrapper
        try:
            import json
            output_data = json.loads(result.stdout.strip().split('\n')[-1])

            result_dict = {
                "success": True,
                "stdout": output_data["stdout"],
                "stderr": output_data["stderr"],
                "status": output_data["status"],
                "working_directory": output_data["working_directory"],
                "return_code": result.returncode
            }

            # Include captured_result if present
            if "captured_result" in output_data:
                result_dict["captured_result"] = output_data["captured_result"]

            return result_dict
        except (json.JSONDecodeError, IndexError):
            # Fallback if JSON parsing fails
            return {
                "success": result.returncode == 0,
                "stdout": result.stdout,
                "stderr": result.stderr,
                "status": "completed" if result.returncode == 0 else "error",
                "return_code": result.returncode
            }

    except subprocess.TimeoutExpired:
        if 'temp_file' in locals():
            os.unlink(temp_file)
        return {
            "success": False,
            "error": f"Code execution timed out after {timeout} seconds",
            "stdout": "",
            "stderr": "",
            "status": "timeout"
        }
    except Exception as e:
        if 'temp_file' in locals():
            os.unlink(temp_file)
        return {
            "success": False,
            "error": str(e),
            "stdout": "",
            "stderr": "",
            "status": "error"
        }


@tool("Perform filesystem operations - create directories, read/write files, list contents")
def filesystem_operation(
    operation: Literal["create_dir", "write_file", "read_file", "list_dir", "delete_file", "check_exists"],
    path: str,
    content: str = "",
    encoding: str = "utf-8"
) -> dict:
    """
    Safely perform filesystem operations within the current working directory.
    Operations: create_dir, write_file, read_file, list_dir, delete_file, check_exists
    """
    import os
    from pathlib import Path

    try:
        # Convert to Path object for safer handling
        file_path = Path(path)

        # Security check depends on whether path is absolute or relative
        if file_path.is_absolute():
            # Absolute paths: use directly (user explicitly provided full path)
            resolved_path = file_path.resolve()
        else:
            # Relative paths: ensure they stay within current directory
            cwd = Path.cwd()
            resolved_path = (cwd / file_path).resolve()
            try:
                resolved_path.relative_to(cwd)
            except ValueError:
                return {
                    "success": False,
                    "error": f"Path '{path}' attempts to escape current working directory for security reasons"
                }

        if operation == "create_dir":
            resolved_path.mkdir(parents=True, exist_ok=True)
            return {
                "success": True,
                "message": f"Directory created: {resolved_path}",
                "path": str(resolved_path)
            }

        elif operation == "write_file":
            # Ensure parent directory exists
            resolved_path.parent.mkdir(parents=True, exist_ok=True)

            with open(resolved_path, 'w', encoding=encoding) as f:
                f.write(content)

            return {
                "success": True,
                "message": f"File written: {resolved_path}",
                "path": str(resolved_path),
                "size": len(content.encode(encoding))
            }

        elif operation == "read_file":
            if not resolved_path.exists():
                return {
                    "success": False,
                    "error": f"File does not exist: {resolved_path}"
                }

            if resolved_path.is_dir():
                return {
                    "success": False,
                    "error": f"Path is a directory, not a file: {resolved_path}"
                }

            with open(resolved_path, 'r', encoding=encoding) as f:
                file_content = f.read()

            return {
                "success": True,
                "content": file_content,
                "path": str(resolved_path),
                "size": len(file_content.encode(encoding))
            }

        elif operation == "list_dir":
            if not resolved_path.exists():
                return {
                    "success": False,
                    "error": f"Directory does not exist: {resolved_path}"
                }

            if not resolved_path.is_dir():
                return {
                    "success": False,
                    "error": f"Path is not a directory: {resolved_path}"
                }

            items = []
            for item in resolved_path.iterdir():
                items.append({
                    "name": item.name,
                    "path": str(item),
                    "type": "directory" if item.is_dir() else "file",
                    "size": item.stat().st_size if item.is_file() else None
                })

            return {
                "success": True,
                "items": items,
                "path": str(resolved_path),
                "count": len(items)
            }

        elif operation == "delete_file":
            if not resolved_path.exists():
                return {
                    "success": False,
                    "error": f"File does not exist: {resolved_path}"
                }

            if resolved_path.is_dir():
                return {
                    "success": False,
                    "error": f"Use rmdir for directories. Path is a directory: {resolved_path}"
                }

            resolved_path.unlink()
            return {
                "success": True,
                "message": f"File deleted: {resolved_path}",
                "path": str(resolved_path)
            }

        elif operation == "check_exists":
            exists = resolved_path.exists()
            file_type = None
            if exists:
                if resolved_path.is_file():
                    file_type = "file"
                elif resolved_path.is_dir():
                    file_type = "directory"
                else:
                    file_type = "other"

            return {
                "success": True,
                "exists": exists,
                "type": file_type,
                "path": str(resolved_path)
            }

        else:
            return {
                "success": False,
                "error": f"Unknown operation: {operation}. Available: create_dir, write_file, read_file, list_dir, delete_file, check_exists"
            }

    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "operation": operation,
            "path": path
        }