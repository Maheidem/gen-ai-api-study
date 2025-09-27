"""
OpenAI-compatible API models for LM Studio and OpenAI API interactions.

This module provides Pydantic models for type-safe API interactions with both
OpenAI and LM Studio endpoints. All models follow the OpenAI API specification.
"""

from typing import List, Optional, Literal, Any, Union, Dict
from pydantic import BaseModel, Field


# ============================================================================
# Models Endpoint (/v1/models)
# ============================================================================

class ModelInfo(BaseModel):
    """Information about a single model available in the API."""
    id: str
    object: str = "model"
    owned_by: str = "organization_owner"
    created: Optional[int] = None
    permission: Optional[List[Dict[str, Any]]] = None


class ModelList(BaseModel):
    """Response from the /v1/models endpoint."""
    object: str = "list"
    data: List[ModelInfo]


# ============================================================================
# Chat Completions Endpoint (/v1/chat/completions)
# ============================================================================

class FunctionCall(BaseModel):
    """Function call details within a tool call."""
    name: str
    arguments: str


class Function(BaseModel):
    """Function definition for function calling."""
    name: str
    description: Optional[str] = None
    parameters: Optional[Dict[str, Any]] = None


class Tool(BaseModel):
    """Tool definition for the chat completion request."""
    type: Literal["function"] = "function"
    function: Function


class ToolCall(BaseModel):
    """Tool call in a message."""
    id: Optional[str] = None
    type: str = "function"
    function: FunctionCall


class ChatMessage(BaseModel):
    """Message in a chat conversation."""
    role: Union[Literal["system", "user", "assistant", "tool", "function"], str]
    content: Optional[Union[str, List[Dict[str, Any]]]] = None  # Can be string or multi-modal content
    name: Optional[str] = None  # For function/tool messages
    tool_calls: Optional[List[ToolCall]] = Field(default=None)
    tool_call_id: Optional[str] = None  # For tool response messages
    function_call: Optional[FunctionCall] = None  # Deprecated, use tool_calls


class ChatCompletionChoice(BaseModel):
    """A single completion choice in the response."""
    index: int
    message: ChatMessage
    logprobs: Optional[Any] = None
    finish_reason: Optional[str] = None


class CompletionUsage(BaseModel):
    """Token usage information for the completion."""
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class ChatCompletion(BaseModel):
    """Response from the chat completions endpoint."""
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: List[ChatCompletionChoice]
    usage: CompletionUsage
    stats: Optional[Dict[str, Any]] = Field(default=None)  # LM Studio specific
    system_fingerprint: Optional[str] = None


# ============================================================================
# Chat Completions Request
# ============================================================================

class ChatCompletionRequest(BaseModel):
    """Request body for chat completions endpoint."""
    model: str
    messages: List[ChatMessage]

    # Optional parameters
    frequency_penalty: Optional[float] = Field(default=0, ge=-2.0, le=2.0)
    logit_bias: Optional[Dict[str, float]] = None
    logprobs: Optional[bool] = None
    top_logprobs: Optional[int] = Field(default=None, ge=0, le=20)
    max_tokens: Optional[int] = None
    n: Optional[int] = Field(default=1, ge=1, le=128)
    presence_penalty: Optional[float] = Field(default=0, ge=-2.0, le=2.0)
    response_format: Optional[Dict[str, str]] = None  # {"type": "json_object"}
    seed: Optional[int] = None
    stop: Optional[Union[str, List[str]]] = None
    stream: Optional[bool] = False
    temperature: Optional[float] = Field(default=1.0, ge=0, le=2)
    top_p: Optional[float] = Field(default=1.0, ge=0, le=1)
    tools: Optional[List[Tool]] = None
    tool_choice: Optional[Union[str, Dict[str, Any]]] = None  # "none", "auto", or specific tool
    user: Optional[str] = None

    # LM Studio specific (will be ignored by OpenAI)
    repeat_penalty: Optional[float] = None
    top_k: Optional[int] = None


# ============================================================================
# Streaming Response
# ============================================================================

class ChatCompletionStreamChoice(BaseModel):
    """A single choice in a streaming response chunk."""
    index: int
    delta: ChatMessage
    logprobs: Optional[Any] = None
    finish_reason: Optional[str] = None


class ChatCompletionChunk(BaseModel):
    """A single chunk in a streaming response."""
    id: str
    object: str = "chat.completion.chunk"
    created: int
    model: str
    system_fingerprint: Optional[str] = None
    choices: List[ChatCompletionStreamChoice]


# ============================================================================
# Embeddings Endpoint (/v1/embeddings)
# ============================================================================

class EmbeddingData(BaseModel):
    """Single embedding vector with metadata."""
    index: int
    embedding: List[float]
    object: str = "embedding"


class EmbeddingsRequest(BaseModel):
    """Request body for embeddings endpoint."""
    input: Union[str, List[str], List[int], List[List[int]]]
    model: str
    encoding_format: Optional[Literal["float", "base64"]] = "float"
    dimensions: Optional[int] = None  # Only some models support this
    user: Optional[str] = None


class EmbeddingsResponse(BaseModel):
    """Response from the embeddings endpoint."""
    object: str = "list"
    data: List[EmbeddingData]
    model: str
    usage: CompletionUsage


# ============================================================================
# Completions Endpoint - Legacy (/v1/completions)
# ============================================================================

class CompletionChoice(BaseModel):
    """A single completion choice in the legacy completions response."""
    text: str
    index: int
    logprobs: Optional[Any] = None
    finish_reason: Optional[str] = None


class CompletionRequest(BaseModel):
    """Request body for legacy completions endpoint."""
    model: str
    prompt: Union[str, List[str], List[int], List[List[int]]]

    # Optional parameters
    best_of: Optional[int] = Field(default=1, ge=1)
    echo: Optional[bool] = False
    frequency_penalty: Optional[float] = Field(default=0, ge=-2.0, le=2.0)
    logit_bias: Optional[Dict[str, float]] = None
    logprobs: Optional[int] = None
    max_tokens: Optional[int] = 16
    n: Optional[int] = Field(default=1, ge=1, le=128)
    presence_penalty: Optional[float] = Field(default=0, ge=-2.0, le=2.0)
    seed: Optional[int] = None
    stop: Optional[Union[str, List[str]]] = None
    stream: Optional[bool] = False
    suffix: Optional[str] = None
    temperature: Optional[float] = Field(default=1.0, ge=0, le=2)
    top_p: Optional[float] = Field(default=1.0, ge=0, le=1)
    user: Optional[str] = None


class CompletionResponse(BaseModel):
    """Response from the legacy completions endpoint."""
    id: str
    object: str = "text_completion"
    created: int
    model: str
    choices: List[CompletionChoice]
    usage: CompletionUsage
    system_fingerprint: Optional[str] = None


# ============================================================================
# Error Models
# ============================================================================

class APIError(BaseModel):
    """Error response from the API."""
    message: str
    type: str
    param: Optional[str] = None
    code: Optional[str] = None


class ErrorResponse(BaseModel):
    """Wrapper for error responses."""
    error: APIError


# ============================================================================
# Helper Functions
# ============================================================================

def create_chat_message(
    role: str,
    content: str,
    name: Optional[str] = None,
    tool_calls: Optional[List[ToolCall]] = None
) -> ChatMessage:
    """
    Helper function to create a chat message.

    Args:
        role: The role of the message sender
        content: The message content
        name: Optional name for function/tool messages
        tool_calls: Optional tool calls in the message

    Returns:
        ChatMessage instance
    """
    return ChatMessage(
        role=role,
        content=content,
        name=name,
        tool_calls=tool_calls
    )


def create_chat_completion_request(
    model: str,
    messages: List[ChatMessage],
    **kwargs
) -> ChatCompletionRequest:
    """
    Helper function to create a chat completion request.

    Args:
        model: Model ID to use
        messages: List of messages in the conversation
        **kwargs: Additional optional parameters

    Returns:
        ChatCompletionRequest instance
    """
    return ChatCompletionRequest(
        model=model,
        messages=messages,
        **kwargs
    )


# ============================================================================
# Constants
# ============================================================================

# Common OpenAI models
OPENAI_MODELS = {
    "gpt-4o": "Most capable GPT-4 Omni model",
    "gpt-4o-mini": "Small, fast, and affordable GPT-4 Omni model",
    "gpt-4-turbo": "Latest GPT-4 Turbo model",
    "gpt-4": "Standard GPT-4 model",
    "gpt-3.5-turbo": "Fast and inexpensive GPT-3.5 model",
}

# Common embedding models
EMBEDDING_MODELS = {
    "text-embedding-3-small": "Small, efficient embedding model",
    "text-embedding-3-large": "Large, powerful embedding model",
    "text-embedding-ada-002": "Legacy embedding model",
}

# Response format types
RESPONSE_FORMATS = {
    "text": {"type": "text"},
    "json_object": {"type": "json_object"},
}

# Common finish reasons
FINISH_REASONS = [
    "stop",           # Natural stop point
    "length",         # Max tokens reached
    "tool_calls",     # Model called a tool
    "content_filter", # Content was filtered
    "function_call",  # Deprecated: Model called a function
]