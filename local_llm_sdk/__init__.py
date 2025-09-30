"""
Local LLM SDK - A type-safe Python SDK for local LLM APIs (OpenAI-compatible).

This package provides a unified interface for interacting with local LLM servers
that implement the OpenAI API specification, such as LM Studio, Ollama, and others.
"""

__version__ = "0.1.0"
__author__ = "Your Name"

# Main client
from .client import LocalLLMClient, create_client, quick_chat

# Models
from .models import (
    # Chat models
    ChatMessage,
    ChatCompletion,
    ChatCompletionRequest,
    create_chat_message,
    create_chat_completion_request,

    # Model management
    ModelInfo,
    ModelList,

    # Embeddings
    EmbeddingsRequest,
    EmbeddingsResponse,

    # Tools/Functions
    Tool,
    Function,
    ToolCall,
    FunctionCall,

    # Constants
    OPENAI_MODELS,
    EMBEDDING_MODELS,
)

# Tool system
from .tools import ToolRegistry, tool

# Agent system
from . import agents
from .agents import ReACT, BaseAgent, AgentResult, AgentStatus

# Convenience re-export
Client = LocalLLMClient


# Convenience functions for quick setup
def create_client_with_tools(base_url: str = "http://localhost:1234/v1", model: str = "auto") -> LocalLLMClient:
    """
    Create a LocalLLMClient with built-in tools pre-loaded.

    This is a convenience function that creates a client and automatically
    loads all built-in tools, making it ready to use immediately.

    Args:
        base_url: Base URL for the local LLM API
        model: Model to use ("auto" to auto-detect first available)

    Returns:
        LocalLLMClient instance with tools loaded

    Example:
        >>> from local_llm_sdk import create_client_with_tools
        >>> client = create_client_with_tools()
        >>> response = client.chat("What's 2+2?")
    """
    client = LocalLLMClient(base_url, model)
    client.register_tools_from(None)  # Load built-in tools
    return client


# Pre-configured default client for immediate use
default_client = create_client_with_tools()

__all__ = [
    # Client
    "LocalLLMClient",
    "Client",
    "create_client",
    "create_client_with_tools",
    "default_client",
    "quick_chat",

    # Models
    "ChatMessage",
    "ChatCompletion",
    "ChatCompletionRequest",
    "create_chat_message",
    "create_chat_completion_request",
    "ModelInfo",
    "ModelList",
    "EmbeddingsRequest",
    "EmbeddingsResponse",
    "Tool",
    "Function",
    "ToolCall",
    "FunctionCall",

    # Tools
    "ToolRegistry",
    "tool",

    # Agents
    "agents",
    "ReACT",
    "BaseAgent",
    "AgentResult",
    "AgentStatus",

    # Constants
    "OPENAI_MODELS",
    "EMBEDDING_MODELS",
]