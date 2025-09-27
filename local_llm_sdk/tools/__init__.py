"""
Tool system for Local LLM SDK.

This module provides a decorator-based system for creating and registering
tools (functions) that can be called by LLMs.
"""

from .registry import ToolRegistry, tool
from . import builtin

# Create a default global registry with built-in tools
default_registry = ToolRegistry()

# Import builtin tools to register them
# This automatically registers all @tool decorated functions in builtin.py

__all__ = [
    "ToolRegistry",
    "tool",
    "builtin",
    "default_registry",
]