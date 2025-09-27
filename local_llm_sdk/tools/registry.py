"""
Simple tool registry for LM Studio function calling.
KISS principle: Write function, get tool. No boilerplate.
Integrated with api_models for type safety.
"""

import json
import inspect
from typing import Callable, Dict, Any, List, get_type_hints
from ..models import Tool, Function


class ToolRegistry:
    """Dead simple tool registry. Register functions, get schemas, execute them."""

    def __init__(self):
        self._tools: Dict[str, Callable] = {}
        self._schemas: List[Tool] = []  # Now stores Pydantic Tool objects

    def register(self, description: str = ""):
        """
        Decorator to register a function as a tool.

        Usage:
            @tools.register("Adds two numbers")
            def add(a: float, b: float) -> dict:
                return {"result": a + b}
        """
        def decorator(func: Callable) -> Callable:
            # Register the function
            self._tools[func.__name__] = func

            # Generate schema from function signature
            schema = self._generate_schema(func, description)
            self._schemas.append(schema)

            return func
        return decorator

    def _generate_schema(self, func: Callable, description: str) -> Tool:
        """Generate OpenAI-compatible Tool object from function signature."""
        # Get type hints
        hints = get_type_hints(func)
        sig = inspect.signature(func)

        # Build parameters schema
        properties = {}
        required = []

        for param_name, param in sig.parameters.items():
            if param_name == "self":
                continue

            # Get type
            param_type = hints.get(param_name, Any)

            # Convert Python type to JSON Schema type
            json_type = self._python_to_json_type(param_type)

            # Build property schema
            prop_schema = {"type": json_type}

            # Add description from docstring if available
            if func.__doc__:
                # Simple extraction - could be enhanced
                prop_schema["description"] = f"Parameter: {param_name}"

            # Handle enums and Literal types
            if hasattr(param_type, "__args__"):
                # Check if it's a Literal type
                origin = getattr(param_type, "__origin__", None)
                if origin is not None:
                    # Handle typing.Literal
                    origin_name = getattr(origin, "__name__", str(origin))
                    if "Literal" in str(origin_name):
                        prop_schema["enum"] = list(param_type.__args__)

            properties[param_name] = prop_schema

            # Add to required if no default value
            if param.default == inspect.Parameter.empty:
                required.append(param_name)

        # Create Pydantic Function and Tool objects
        function = Function(
            name=func.__name__,
            description=description or func.__doc__ or f"Function: {func.__name__}",
            parameters={
                "type": "object",
                "properties": properties,
                "required": required
            }
        )

        return Tool(type="function", function=function)

    def _python_to_json_type(self, python_type) -> str:
        """Convert Python type to JSON Schema type."""
        type_map = {
            int: "integer",
            float: "number",
            str: "string",
            bool: "boolean",
            list: "array",
            dict: "object",
            List: "array",
            Dict: "object"
        }

        # Handle Union types
        if hasattr(python_type, "__origin__"):
            origin = python_type.__origin__
            if origin in type_map:
                return type_map[origin]

        # Direct type mapping
        if python_type in type_map:
            return type_map[python_type]

        # Default to string for unknown types
        return "string"

    def get_schemas(self) -> List[Tool]:
        """Get all registered Tool objects (Pydantic models)."""
        return self._schemas

    def get_schemas_dict(self) -> List[Dict[str, Any]]:
        """Get all tool schemas as dictionaries (for JSON serialization)."""
        return [tool.model_dump(exclude_none=True) for tool in self._schemas]

    def execute(self, tool_name: str, arguments: Dict[str, Any]) -> str:
        """
        Execute a registered tool and return JSON result.

        Args:
            tool_name: Name of the tool to execute
            arguments: Arguments to pass to the tool

        Returns:
            JSON string with the result
        """
        if tool_name not in self._tools:
            return json.dumps({"error": f"Unknown tool: {tool_name}"})

        try:
            result = self._tools[tool_name](**arguments)

            # Ensure result is JSON-serializable
            if isinstance(result, dict):
                return json.dumps(result)
            else:
                return json.dumps({"result": result})

        except Exception as e:
            return json.dumps({"error": str(e)})

    def list_tools(self) -> List[str]:
        """List all registered tool names."""
        return list(self._tools.keys())


# Global registry instance
tools = ToolRegistry()


# Convenience decorator for direct use
def tool(description: str = ""):
    """
    Convenience decorator for registering tools.

    Usage:
        @tool("Adds two numbers")
        def add(a: float, b: float) -> dict:
            return {"result": a + b}
    """
    return tools.register(description)