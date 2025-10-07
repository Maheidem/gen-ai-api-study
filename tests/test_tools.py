"""
Tests for tool registry and builtin tools.
"""

import pytest
from unittest.mock import Mock
import json
from typing import Literal

from local_llm_sdk.tools.registry import ToolRegistry, tool, tools
from local_llm_sdk.tools import builtin
from local_llm_sdk.models import Tool, Function


class TestToolRegistry:
    """Test ToolRegistry functionality."""

    def test_init_empty_registry(self):
        """Test creating an empty ToolRegistry."""
        registry = ToolRegistry()

        assert len(registry.list_tools()) == 0
        assert len(registry.get_schemas()) == 0

    def test_register_simple_function(self):
        """Test registering a simple function."""
        registry = ToolRegistry()

        @registry.register("Add two numbers")
        def add(a: float, b: float) -> dict:
            return {"result": a + b}

        tools = registry.list_tools()
        assert "add" in tools
        assert len(registry.get_schemas()) == 1

    def test_register_function_without_description(self):
        """Test registering a function without description."""
        registry = ToolRegistry()

        @registry.register()
        def test_func() -> dict:
            """This is the docstring."""
            return {"result": "test"}

        schemas = registry.get_schemas()
        assert len(schemas) == 1
        # Should use docstring or function name
        assert schemas[0].function.description in ["This is the docstring.", "Function: test_func"]

    def test_register_function_with_literal_types(self):
        """Test registering a function with Literal types."""
        registry = ToolRegistry()

        @registry.register("Process text")
        def process_text(text: str, operation: Literal["upper", "lower"]) -> dict:
            return {"result": text.upper() if operation == "upper" else text.lower()}

        schemas = registry.get_schemas_dict()
        function_schema = schemas[0]["function"]

        # Check that enum is generated for Literal type
        operation_param = function_schema["parameters"]["properties"]["operation"]
        assert "enum" in operation_param
        assert operation_param["enum"] == ["upper", "lower"]

    def test_register_function_with_optional_params(self):
        """Test registering a function with optional parameters."""
        registry = ToolRegistry()

        @registry.register("Test function")
        def test_func(required: str, optional: int = 42) -> dict:
            return {"required": required, "optional": optional}

        schemas = registry.get_schemas_dict()
        function_schema = schemas[0]["function"]

        # Check required parameters
        required_params = function_schema["parameters"]["required"]
        assert "required" in required_params
        assert "optional" not in required_params

        # Check properties
        properties = function_schema["parameters"]["properties"]
        assert "required" in properties
        assert "optional" in properties

    def test_execute_registered_function(self):
        """Test executing a registered function."""
        registry = ToolRegistry()

        @registry.register("Multiply numbers")
        def multiply(x: float, y: float) -> dict:
            return {"result": x * y}

        result = registry.execute("multiply", {"x": 3.0, "y": 4.0})
        parsed_result = json.loads(result)

        assert parsed_result["result"] == 12.0

    def test_execute_nonexistent_function(self):
        """Test executing a function that doesn't exist."""
        registry = ToolRegistry()

        result = registry.execute("nonexistent", {})
        parsed_result = json.loads(result)

        assert "error" in parsed_result
        assert "Unknown tool" in parsed_result["error"]

    def test_execute_function_with_exception(self):
        """Test executing a function that raises an exception."""
        registry = ToolRegistry()

        @registry.register("Divide numbers")
        def divide(x: float, y: float) -> dict:
            return {"result": x / y}

        result = registry.execute("divide", {"x": 10.0, "y": 0.0})
        parsed_result = json.loads(result)

        assert "error" in parsed_result
        assert "division by zero" in parsed_result["error"].lower()

    def test_execute_function_returning_non_dict(self):
        """Test executing a function that returns non-dict."""
        registry = ToolRegistry()

        @registry.register("Return string")
        def return_string() -> str:
            return "hello"

        result = registry.execute("return_string", {})
        parsed_result = json.loads(result)

        assert parsed_result["result"] == "hello"

    def test_get_schemas_returns_tool_objects(self):
        """Test that get_schemas returns Tool objects."""
        registry = ToolRegistry()

        @registry.register("Test function")
        def test_func(x: int) -> dict:
            return {"x": x}

        schemas = registry.get_schemas()
        assert len(schemas) == 1
        assert isinstance(schemas[0], Tool)
        assert schemas[0].type == "function"
        assert isinstance(schemas[0].function, Function)

    def test_get_schemas_dict_returns_serializable(self):
        """Test that get_schemas_dict returns serializable data."""
        registry = ToolRegistry()

        @registry.register("Test function")
        def test_func(x: int) -> dict:
            return {"x": x}

        schemas_dict = registry.get_schemas_dict()

        # Should be JSON-serializable
        json_str = json.dumps(schemas_dict)
        assert isinstance(json_str, str)

        # Should have expected structure
        assert len(schemas_dict) == 1
        assert "type" in schemas_dict[0]
        assert "function" in schemas_dict[0]

    def test_python_to_json_type_mapping(self):
        """Test _python_to_json_type method."""
        registry = ToolRegistry()

        # Test basic type mappings
        assert registry._python_to_json_type(int) == "integer"
        assert registry._python_to_json_type(float) == "number"
        assert registry._python_to_json_type(str) == "string"
        assert registry._python_to_json_type(bool) == "boolean"
        assert registry._python_to_json_type(list) == "array"
        assert registry._python_to_json_type(dict) == "object"

        # Test unknown type defaults to string
        class CustomType:
            pass
        assert registry._python_to_json_type(CustomType) == "string"

    def test_generate_schema_with_complex_types(self):
        """Test schema generation with complex parameter types."""
        registry = ToolRegistry()

        @registry.register("Complex function")
        def complex_func(
            text: str,
            count: int,
            enabled: bool,
            items: list,
            config: dict
        ) -> dict:
            return {"status": "ok"}

        schemas = registry.get_schemas_dict()
        properties = schemas[0]["function"]["parameters"]["properties"]

        assert properties["text"]["type"] == "string"
        assert properties["count"]["type"] == "integer"
        assert properties["enabled"]["type"] == "boolean"
        assert properties["items"]["type"] == "array"
        assert properties["config"]["type"] == "object"


class TestGlobalToolDecorator:
    """Test the global tool decorator."""

    def test_global_tool_decorator(self):
        """Test using the global @tool decorator."""
        # Note: This modifies the global registry, so we need to be careful
        initial_count = len(tools.list_tools())

        @tool("Global test function")
        def global_test_func(x: int) -> dict:
            return {"doubled": x * 2}

        # Should be added to global registry
        new_count = len(tools.list_tools())
        assert new_count == initial_count + 1
        assert "global_test_func" in tools.list_tools()

        # Test execution
        result = tools.execute("global_test_func", {"x": 5})
        parsed_result = json.loads(result)
        assert parsed_result["doubled"] == 10

    def test_tool_decorator_returns_original_function(self):
        """Test that @tool decorator returns the original function."""
        @tool("Test function")
        def original_func(x: int) -> int:
            return x * 2

        # Should still be callable as normal function
        assert original_func(5) == 10


class TestBuiltinTools:
    """Test builtin bash tool functionality."""

    def test_bash_tool_simple_command(self):
        """Test bash tool with simple command."""
        result = tools.execute("bash", {"command": "echo 'hello world'"})
        data = json.loads(result)
        assert data["success"] is True
        assert "hello world" in data["stdout"]
        assert data["return_code"] == 0

    def test_bash_tool_python_execution(self):
        """Test bash tool executing Python code."""
        result = tools.execute("bash", {"command": "python -c \"print(5 * 24)\""})
        data = json.loads(result)
        assert data["success"] is True
        assert "120" in data["stdout"]

    def test_bash_tool_python_factorial(self):
        """Test bash tool calculating factorial with Python."""
        result = tools.execute("bash", {"command": "python -c \"import math; print(math.factorial(5))\""})
        data = json.loads(result)
        assert data["success"] is True
        assert "120" in data["stdout"]

    def test_bash_tool_file_operations(self):
        """Test bash tool with file operations."""
        # Create, write, read, delete file
        result = tools.execute("bash", {
            "command": "echo 'test content' > /tmp/test_bash.txt && cat /tmp/test_bash.txt && rm /tmp/test_bash.txt"
        })
        data = json.loads(result)
        assert data["success"] is True
        assert "test content" in data["stdout"]

    def test_bash_tool_chained_commands(self):
        """Test bash tool with chained commands."""
        result = tools.execute("bash", {
            "command": "mkdir -p /tmp/bash_test_dir && echo 'content' > /tmp/bash_test_dir/file.txt && cat /tmp/bash_test_dir/file.txt && rm -rf /tmp/bash_test_dir"
        })
        data = json.loads(result)
        assert data["success"] is True
        assert "content" in data["stdout"]

    def test_bash_tool_error_handling(self):
        """Test bash tool handles errors gracefully."""
        result = tools.execute("bash", {"command": "ls /nonexistent_directory_xyz"})
        data = json.loads(result)
        assert data["success"] is False
        assert data["return_code"] != 0
        assert len(data["stderr"]) > 0

    def test_bash_tool_timeout(self):
        """Test bash tool respects timeout."""
        result = tools.execute("bash", {"command": "sleep 2", "timeout": 1})
        data = json.loads(result)
        assert data["success"] is False
        assert "timed out" in data["stderr"].lower() or "timeout" in data.get("error", "").lower()

    def test_bash_tool_math_operations(self):
        """Test bash tool for mathematical operations via Python."""
        # Addition
        result = tools.execute("bash", {"command": "python -c \"print(42 + 58)\""})
        data = json.loads(result)
        assert data["success"] is True
        assert "100" in data["stdout"]

    def test_bash_tool_text_processing(self):
        """Test bash tool for text processing."""
        # Uppercase transformation using Python
        result = tools.execute("bash", {"command": "python -c \"print('hello world'.upper())\""})
        data = json.loads(result)
        assert data["success"] is True
        assert "HELLO WORLD" in data["stdout"]

    def test_bash_tool_return_structure(self):
        """Test bash tool returns correct structure."""
        result = tools.execute("bash", {"command": "echo 'test'"})
        data = json.loads(result)

        # Check all required fields are present
        assert "success" in data
        assert "stdout" in data
        assert "stderr" in data
        assert "return_code" in data
        assert "command" in data
        assert data["command"] == "echo 'test'"

    def test_builtin_tools_have_schemas(self):
        """Test that bash tool has proper schema."""
        schemas = tools.get_schemas()

        # Should have at least the bash tool
        assert len(schemas) >= 1

        # Find the bash tool
        bash_schema = None
        for schema in schemas:
            if schema.function.name == "bash":
                bash_schema = schema
                break

        assert bash_schema is not None, "bash tool should be in schemas"

        # Schema should be a valid Tool object
        assert isinstance(bash_schema, Tool)
        assert bash_schema.type == "function"
        assert isinstance(bash_schema.function, Function)
        assert bash_schema.function.name == "bash"
        assert bash_schema.function.parameters


class TestToolSchemaGeneration:
    """Test tool schema generation edge cases."""

    def test_function_with_no_parameters(self):
        """Test schema generation for function with no parameters."""
        registry = ToolRegistry()

        @registry.register("No params function")
        def no_params() -> dict:
            return {"status": "ok"}

        schemas = registry.get_schemas_dict()
        function_schema = schemas[0]["function"]

        assert function_schema["parameters"]["type"] == "object"
        assert function_schema["parameters"]["properties"] == {}
        assert function_schema["parameters"]["required"] == []

    def test_function_with_self_parameter(self):
        """Test that self parameter is ignored in methods."""
        registry = ToolRegistry()

        class TestClass:
            @registry.register("Method test")
            def method(self, x: int) -> dict:
                return {"x": x}

        schemas = registry.get_schemas_dict()
        properties = schemas[0]["function"]["parameters"]["properties"]

        # self should be excluded
        assert "self" not in properties
        assert "x" in properties

    def test_function_with_complex_return_annotation(self):
        """Test function with complex return type annotation."""
        registry = ToolRegistry()

        from typing import Dict, Any

        @registry.register("Complex return")
        def complex_return(x: int) -> Dict[str, Any]:
            return {"result": x}

        # Should work without issues
        schemas = registry.get_schemas()
        assert len(schemas) == 1

    def test_schema_parameter_descriptions(self):
        """Test that parameter descriptions are generated."""
        registry = ToolRegistry()

        @registry.register("Documented function")
        def documented_func(param1: str, param2: int) -> dict:
            """This function does something.

            Args:
                param1: The first parameter
                param2: The second parameter
            """
            return {"result": "ok"}

        schemas = registry.get_schemas_dict()
        properties = schemas[0]["function"]["parameters"]["properties"]

        # Should have description fields
        assert "description" in properties["param1"]
        assert "description" in properties["param2"]