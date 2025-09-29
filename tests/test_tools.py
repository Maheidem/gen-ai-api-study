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
    """Test builtin tools functionality."""

    def test_math_calculator_addition(self):
        """Test math_calculator with addition."""
        result = tools.execute("math_calculator", {
            "arg1": 5.0,
            "arg2": 3.0,
            "operation": "add"
        })
        parsed_result = json.loads(result)
        assert parsed_result["result"] == 8.0

    def test_math_calculator_subtraction(self):
        """Test math_calculator with subtraction."""
        result = tools.execute("math_calculator", {
            "arg1": 10.0,
            "arg2": 3.0,
            "operation": "subtract"
        })
        parsed_result = json.loads(result)
        assert parsed_result["result"] == 7.0

    def test_math_calculator_multiplication(self):
        """Test math_calculator with multiplication."""
        result = tools.execute("math_calculator", {
            "arg1": 4.0,
            "arg2": 5.0,
            "operation": "multiply"
        })
        parsed_result = json.loads(result)
        assert parsed_result["result"] == 20.0

    def test_math_calculator_division(self):
        """Test math_calculator with division."""
        result = tools.execute("math_calculator", {
            "arg1": 15.0,
            "arg2": 3.0,
            "operation": "divide"
        })
        parsed_result = json.loads(result)
        assert parsed_result["result"] == 5.0

    def test_math_calculator_division_by_zero(self):
        """Test math_calculator division by zero."""
        result = tools.execute("math_calculator", {
            "arg1": 10.0,
            "arg2": 0.0,
            "operation": "divide"
        })
        parsed_result = json.loads(result)
        assert "error" in parsed_result

    def test_char_counter(self):
        """Test char_counter tool."""
        result = tools.execute("char_counter", {
            "text": "Hello, World!"
        })
        parsed_result = json.loads(result)

        assert parsed_result["text"] == "Hello, World!"
        assert parsed_result["character_count"] == 13
        assert parsed_result["word_count"] == 2

    def test_char_counter_empty_string(self):
        """Test char_counter with empty string."""
        result = tools.execute("char_counter", {
            "text": ""
        })
        parsed_result = json.loads(result)

        assert parsed_result["character_count"] == 0
        assert parsed_result["word_count"] == 0

    def test_text_transformer_uppercase(self):
        """Test text_transformer with uppercase."""
        result = tools.execute("text_transformer", {
            "text": "hello world",
            "transform": "upper"
        })
        parsed_result = json.loads(result)

        assert parsed_result["original"] == "hello world"
        assert parsed_result["transformed"] == "HELLO WORLD"
        assert parsed_result["transform_type"] == "upper"

    def test_text_transformer_lowercase(self):
        """Test text_transformer with lowercase."""
        result = tools.execute("text_transformer", {
            "text": "HELLO WORLD",
            "transform": "lower"
        })
        parsed_result = json.loads(result)

        assert parsed_result["transformed"] == "hello world"

    def test_text_transformer_title(self):
        """Test text_transformer with title case."""
        result = tools.execute("text_transformer", {
            "text": "hello world",
            "transform": "title"
        })
        parsed_result = json.loads(result)

        assert parsed_result["transformed"] == "Hello World"

    def test_text_transformer_invalid_transform(self):
        """Test text_transformer with invalid transform."""
        result = tools.execute("text_transformer", {
            "text": "hello",
            "transform": "invalid"
        })
        parsed_result = json.loads(result)

        assert "error" in parsed_result

    def test_weather_tool_mock(self):
        """Test weather tool (mocked since it's a placeholder)."""
        result = tools.execute("get_weather", {
            "city": "London"
        })
        parsed_result = json.loads(result)

        # Should return some weather data structure
        assert "city" in parsed_result
        assert parsed_result["city"] == "London"

    def test_builtin_tools_have_schemas(self):
        """Test that all builtin tools have proper schemas."""
        schemas = tools.get_schemas()

        # Should have multiple tools
        assert len(schemas) > 0

        # Each schema should be a valid Tool object
        for schema in schemas:
            assert isinstance(schema, Tool)
            assert schema.type == "function"
            assert isinstance(schema.function, Function)
            assert schema.function.name
            assert schema.function.parameters


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