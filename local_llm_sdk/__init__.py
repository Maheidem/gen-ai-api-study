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

# Convenience re-export
Client = LocalLLMClient

__all__ = [
    # Client
    "LocalLLMClient",
    "Client",
    "create_client",
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

    # Constants
    "OPENAI_MODELS",
    "EMBEDDING_MODELS",
]