# Models API Reference

Type-safe Pydantic models for OpenAI-compatible API interactions.

## Overview

The `local_llm_sdk.models` module provides comprehensive Pydantic v2 models that follow the OpenAI API specification. All models enforce strict type validation and provide automatic serialization/deserialization for API requests and responses.

**Key Features:**
- Full OpenAI API specification compliance
- Type-safe request/response handling
- Support for both OpenAI and LM Studio endpoints
- Streaming response models
- Function/tool calling support
- Helper functions for common operations

**Module Location:** `/Users/maheidem/Documents/dev/gen-ai-api-study/local_llm_sdk/models.py`

## Import

```python
from local_llm_sdk.models import (
    ChatMessage,
    ChatCompletionRequest,
    ChatCompletion,
    Tool,
    ToolCall,
    ModelInfo,
    EmbeddingsRequest,
    create_chat_message,
)
```

---

## Models Endpoint

### ModelInfo

Information about a single model available in the API.

**Fields:**
- `id` (str): Model identifier (e.g., "gpt-4", "mistralai/magistral-small-2509")
- `object` (str): Object type, default: "model"
- `owned_by` (str): Organization owner, default: "organization_owner"
- `created` (Optional[int]): Unix timestamp of model creation
- `permission` (Optional[List[Dict[str, Any]]]): Model permissions

**Example:**
```python
from local_llm_sdk.models import ModelInfo

model = ModelInfo(
    id="mistralai/magistral-small-2509",
    owned_by="mistralai",
    created=1234567890
)
print(model.id)  # "mistralai/magistral-small-2509"
```

### ModelList

Response from the `/v1/models` endpoint.

**Fields:**
- `object` (str): Object type, default: "list"
- `data` (List[ModelInfo]): List of available models

**Example:**
```python
from local_llm_sdk.models import ModelList, ModelInfo

model_list = ModelList(
    data=[
        ModelInfo(id="gpt-4", owned_by="openai"),
        ModelInfo(id="gpt-3.5-turbo", owned_by="openai")
    ]
)
print(len(model_list.data))  # 2
```

---

## Chat Completions

### ChatMessage

Message in a chat conversation. Supports system, user, assistant, tool, and function roles.

**Fields:**
- `role` (Literal["system", "user", "assistant", "tool", "function"]): Message role
- `content` (Optional[Union[str, List[Dict[str, Any]]]]): Message content (text or multi-modal)
- `name` (Optional[str]): Name for function/tool messages
- `tool_calls` (Optional[List[ToolCall]]): Tool calls made by assistant
- `tool_call_id` (Optional[str]): ID linking tool response to tool call
- `function_call` (Optional[FunctionCall]): Deprecated, use `tool_calls`

**Methods:**
- `__repr__()`: Returns string representation with content preview and tool count

**Examples:**

**System Message:**
```python
from local_llm_sdk.models import ChatMessage

system_msg = ChatMessage(
    role="system",
    content="You are a helpful assistant specialized in mathematics."
)
```

**User Message:**
```python
user_msg = ChatMessage(
    role="user",
    content="What is 42 * 17?"
)
```

**Assistant with Tool Calls:**
```python
from local_llm_sdk.models import ToolCall, FunctionCall

assistant_msg = ChatMessage(
    role="assistant",
    content=None,
    tool_calls=[
        ToolCall(
            id="call_123",
            type="function",
            function=FunctionCall(
                name="calculate",
                arguments='{"expression": "42 * 17"}'
            )
        )
    ]
)
```

**Tool Response:**
```python
tool_msg = ChatMessage(
    role="tool",
    content='{"result": 714}',
    tool_call_id="call_123",
    name="calculate"
)
```

**Multi-Modal Content:**
```python
multimodal_msg = ChatMessage(
    role="user",
    content=[
        {"type": "text", "text": "What's in this image?"},
        {"type": "image_url", "image_url": {"url": "https://..."}}
    ]
)
```

### ChatCompletionRequest

Request body for `/v1/chat/completions` endpoint.

**Required Fields:**
- `model` (str): Model ID to use
- `messages` (List[ChatMessage]): Conversation messages

**Optional Parameters:**

**Sampling:**
- `temperature` (Optional[float]): Sampling temperature (0-2), default varies by model
- `top_p` (Optional[float]): Nucleus sampling (0-1)
- `top_k` (Optional[int]): Top-K sampling (LM Studio only)
- `seed` (Optional[int]): Random seed for reproducibility

**Output Control:**
- `max_tokens` (Optional[int]): Maximum tokens to generate
- `n` (Optional[int]): Number of completions (1-128)
- `stop` (Optional[Union[str, List[str]]]): Stop sequences
- `response_format` (Optional[Dict[str, str]]): Output format, e.g., `{"type": "json_object"}`

**Penalties:**
- `frequency_penalty` (Optional[float]): Penalize token frequency (-2.0 to 2.0)
- `presence_penalty` (Optional[float]): Penalize token presence (-2.0 to 2.0)
- `repeat_penalty` (Optional[float]): Penalize repetition (LM Studio only)

**Tool/Function Calling:**
- `tools` (Optional[List[Tool]]): Available tools for function calling
- `tool_choice` (Optional[Union[str, Dict[str, Any]]]): "none", "auto", "required", or specific tool

**Logging:**
- `logprobs` (Optional[bool]): Return log probabilities
- `top_logprobs` (Optional[int]): Number of top logprobs (0-20)
- `logit_bias` (Optional[Dict[str, float]]): Token bias adjustments

**Other:**
- `stream` (Optional[bool]): Enable streaming responses
- `user` (Optional[str]): User identifier for abuse monitoring

**Example:**
```python
from local_llm_sdk.models import ChatCompletionRequest, ChatMessage

request = ChatCompletionRequest(
    model="mistralai/magistral-small-2509",
    messages=[
        ChatMessage(role="system", content="You are helpful."),
        ChatMessage(role="user", content="Hello!")
    ],
    temperature=0.7,
    max_tokens=1000,
    stop=["END"]
)
```

**With Tools:**
```python
from local_llm_sdk.models import Tool, Function

request = ChatCompletionRequest(
    model="gpt-4",
    messages=[ChatMessage(role="user", content="Calculate 5!")],
    tools=[
        Tool(
            type="function",
            function=Function(
                name="factorial",
                description="Calculate factorial of a number",
                parameters={
                    "type": "object",
                    "properties": {
                        "n": {"type": "integer", "description": "Number"}
                    },
                    "required": ["n"]
                }
            )
        )
    ],
    tool_choice="auto"
)
```

### ChatCompletion

Response from `/v1/chat/completions` endpoint (non-streaming).

**Fields:**
- `id` (str): Unique completion ID
- `object` (str): Object type, default: "chat.completion"
- `created` (int): Unix timestamp
- `model` (str): Model used for completion
- `choices` (List[ChatCompletionChoice]): List of completion choices
- `usage` (Optional[CompletionUsage]): Token usage statistics
- `stats` (Optional[Dict[str, Any]]): LM Studio-specific statistics
- `system_fingerprint` (Optional[str]): System configuration fingerprint

**Methods:**
- `__repr__()`: Returns string with model, tokens, and choice count

**Example:**
```python
# Typical response structure:
{
    "id": "chatcmpl-123",
    "object": "chat.completion",
    "created": 1677652288,
    "model": "gpt-4",
    "choices": [
        {
            "index": 0,
            "message": {
                "role": "assistant",
                "content": "Hello! How can I help you?"
            },
            "finish_reason": "stop"
        }
    ],
    "usage": {
        "prompt_tokens": 10,
        "completion_tokens": 8,
        "total_tokens": 18
    }
}
```

### ChatCompletionChoice

A single completion choice in the response.

**Fields:**
- `index` (int): Choice index (for n > 1)
- `message` (ChatMessage): Generated message
- `logprobs` (Optional[Any]): Log probabilities if requested
- `finish_reason` (Optional[str]): Why generation stopped

**Finish Reasons:**
- `"stop"`: Natural stop point
- `"length"`: Max tokens reached
- `"tool_calls"`: Model called a tool
- `"content_filter"`: Content filtered

**Example:**
```python
from local_llm_sdk.models import ChatCompletionChoice, ChatMessage

choice = ChatCompletionChoice(
    index=0,
    message=ChatMessage(role="assistant", content="The answer is 42."),
    finish_reason="stop"
)
```

### CompletionUsage

Token usage information for the completion.

**Fields:**
- `prompt_tokens` (int): Tokens in prompt (>= 0)
- `completion_tokens` (Optional[int]): Tokens in completion (>= 0)
- `total_tokens` (int): Total tokens used (>= 0)

**Example:**
```python
from local_llm_sdk.models import CompletionUsage

usage = CompletionUsage(
    prompt_tokens=50,
    completion_tokens=30,
    total_tokens=80
)
print(f"Cost estimate: ${usage.total_tokens * 0.00001:.5f}")
```

---

## Tool/Function Calling

### Tool

Tool definition for function calling in chat completion requests.

**Fields:**
- `type` (Literal["function"]): Tool type, always "function"
- `function` (Function): Function definition

**Example:**
```python
from local_llm_sdk.models import Tool, Function

calculator_tool = Tool(
    type="function",
    function=Function(
        name="calculate",
        description="Evaluate mathematical expressions",
        parameters={
            "type": "object",
            "properties": {
                "expression": {
                    "type": "string",
                    "description": "Math expression to evaluate"
                }
            },
            "required": ["expression"]
        }
    )
)
```

### Function

Function definition for function calling.

**Fields:**
- `name` (str): Function name (must be unique)
- `description` (Optional[str]): Function description for LLM
- `parameters` (Optional[Dict[str, Any]]): JSON Schema of parameters

**Example:**
```python
from local_llm_sdk.models import Function

weather_function = Function(
    name="get_weather",
    description="Get current weather for a location",
    parameters={
        "type": "object",
        "properties": {
            "location": {
                "type": "string",
                "description": "City name"
            },
            "units": {
                "type": "string",
                "enum": ["celsius", "fahrenheit"],
                "description": "Temperature units"
            }
        },
        "required": ["location"]
    }
)
```

### ToolCall

Tool call made by the assistant in a message.

**Fields:**
- `id` (Optional[str]): Unique tool call ID (e.g., "call_123")
- `type` (str): Call type, default: "function"
- `function` (FunctionCall): Function call details

**Example:**
```python
from local_llm_sdk.models import ToolCall, FunctionCall

tool_call = ToolCall(
    id="call_abc123",
    type="function",
    function=FunctionCall(
        name="calculate",
        arguments='{"expression": "2 + 2"}'
    )
)
```

### FunctionCall

Function call details within a tool call.

**Fields:**
- `name` (str): Function name (min length: 1)
- `arguments` (str): JSON-encoded function arguments

**Example:**
```python
from local_llm_sdk.models import FunctionCall

function_call = FunctionCall(
    name="get_weather",
    arguments='{"location": "San Francisco", "units": "celsius"}'
)
```

**Parsing Arguments:**
```python
import json

args = json.loads(function_call.arguments)
print(args["location"])  # "San Francisco"
```

---

## Streaming Responses

### ChatCompletionChunk

A single chunk in a streaming response.

**Fields:**
- `id` (str): Completion ID (same across all chunks)
- `object` (str): Object type, default: "chat.completion.chunk"
- `created` (int): Unix timestamp
- `model` (str): Model name
- `system_fingerprint` (Optional[str]): System fingerprint
- `choices` (List[ChatCompletionStreamChoice]): Stream choices

**Example:**
```python
# Typical streaming chunk:
{
    "id": "chatcmpl-123",
    "object": "chat.completion.chunk",
    "created": 1677652288,
    "model": "gpt-4",
    "choices": [
        {
            "index": 0,
            "delta": {"role": "assistant", "content": "Hello"},
            "finish_reason": null
        }
    ]
}
```

### ChatCompletionStreamChoice

A single choice in a streaming response chunk.

**Fields:**
- `index` (int): Choice index
- `delta` (ChatMessage): Incremental message update
- `logprobs` (Optional[Any]): Log probabilities
- `finish_reason` (Optional[str]): Reason for completion (in final chunk)

**Example:**
```python
from local_llm_sdk.models import ChatCompletionStreamChoice, ChatMessage

# First chunk - role
chunk1 = ChatCompletionStreamChoice(
    index=0,
    delta=ChatMessage(role="assistant", content=""),
    finish_reason=None
)

# Content chunk
chunk2 = ChatCompletionStreamChoice(
    index=0,
    delta=ChatMessage(role=None, content="Hello"),
    finish_reason=None
)

# Final chunk
chunk3 = ChatCompletionStreamChoice(
    index=0,
    delta=ChatMessage(role=None, content=""),
    finish_reason="stop"
)
```

---

## Embeddings

### EmbeddingsRequest

Request body for `/v1/embeddings` endpoint.

**Fields:**
- `input` (Union[str, List[str], List[int], List[List[int]]]): Text(s) to embed
- `model` (str): Embedding model to use
- `encoding_format` (Optional[Literal["float", "base64"]]): Output format, default: "float"
- `dimensions` (Optional[int]): Output dimensions (if supported by model)
- `user` (Optional[str]): User identifier

**Examples:**

**Single Text:**
```python
from local_llm_sdk.models import EmbeddingsRequest

request = EmbeddingsRequest(
    input="Hello, world!",
    model="text-embedding-3-small"
)
```

**Multiple Texts:**
```python
request = EmbeddingsRequest(
    input=["First text", "Second text", "Third text"],
    model="text-embedding-3-large",
    dimensions=256
)
```

**Token IDs:**
```python
request = EmbeddingsRequest(
    input=[1234, 5678, 9012],  # Pre-tokenized
    model="text-embedding-ada-002"
)
```

### EmbeddingsResponse

Response from `/v1/embeddings` endpoint.

**Fields:**
- `object` (str): Object type, default: "list"
- `data` (List[EmbeddingData]): Embedding vectors
- `model` (str): Model used
- `usage` (CompletionUsage): Token usage

**Example:**
```python
# Typical response:
{
    "object": "list",
    "data": [
        {
            "index": 0,
            "embedding": [0.123, -0.456, 0.789, ...],
            "object": "embedding"
        }
    ],
    "model": "text-embedding-3-small",
    "usage": {
        "prompt_tokens": 5,
        "total_tokens": 5
    }
}
```

### EmbeddingData

Single embedding vector with metadata.

**Fields:**
- `index` (int): Index in the input list
- `embedding` (List[float]): Embedding vector
- `object` (str): Object type, default: "embedding"

**Example:**
```python
from local_llm_sdk.models import EmbeddingData

embedding = EmbeddingData(
    index=0,
    embedding=[0.1, 0.2, 0.3, ...],  # 1536 dimensions for ada-002
    object="embedding"
)
print(len(embedding.embedding))  # 1536
```

---

## Legacy Completions (Deprecated)

### CompletionRequest

Request body for legacy `/v1/completions` endpoint.

**Note:** This endpoint is deprecated. Use chat completions instead.

**Fields:**
- `model` (str): Model ID
- `prompt` (Union[str, List[str], List[int], List[List[int]]]): Prompt text
- `max_tokens` (Optional[int]): Maximum tokens, default: 16
- `temperature` (Optional[float]): Sampling temperature, default: 1.0
- `top_p` (Optional[float]): Nucleus sampling, default: 1.0
- `n` (Optional[int]): Number of completions, default: 1
- `stream` (Optional[bool]): Enable streaming, default: False
- `stop` (Optional[Union[str, List[str]]]): Stop sequences
- `presence_penalty` (Optional[float]): Presence penalty, default: 0
- `frequency_penalty` (Optional[float]): Frequency penalty, default: 0
- `best_of` (Optional[int]): Generate best_of and return best, default: 1
- `echo` (Optional[bool]): Echo prompt, default: False
- `suffix` (Optional[str]): Text after completion
- Other fields (logprobs, logit_bias, seed, user)

### CompletionResponse

Response from legacy `/v1/completions` endpoint.

**Fields:**
- `id` (str): Completion ID
- `object` (str): Object type, default: "text_completion"
- `created` (int): Unix timestamp
- `model` (str): Model used
- `choices` (List[CompletionChoice]): Completion choices
- `usage` (CompletionUsage): Token usage
- `system_fingerprint` (Optional[str]): System fingerprint

### CompletionChoice

A single completion choice in legacy response.

**Fields:**
- `text` (str): Generated text
- `index` (int): Choice index
- `logprobs` (Optional[Any]): Log probabilities
- `finish_reason` (Optional[str]): Completion reason

---

## Error Models

### APIError

Error information from the API.

**Fields:**
- `message` (str): Human-readable error message
- `type` (str): Error type (e.g., "invalid_request_error")
- `param` (Optional[str]): Parameter that caused the error
- `code` (Optional[str]): Error code

**Example:**
```python
from local_llm_sdk.models import APIError

error = APIError(
    message="Invalid model specified",
    type="invalid_request_error",
    param="model",
    code="model_not_found"
)
```

### ErrorResponse

Wrapper for error responses from the API.

**Fields:**
- `error` (APIError): Error details

**Example:**
```python
# Typical error response:
{
    "error": {
        "message": "The model `invalid-model` does not exist",
        "type": "invalid_request_error",
        "param": "model",
        "code": "model_not_found"
    }
}
```

---

## Helper Functions

### create_chat_message

Helper function to create a chat message.

**Signature:**
```python
def create_chat_message(
    role: str,
    content: str,
    name: Optional[str] = None,
    tool_calls: Optional[List[ToolCall]] = None
) -> ChatMessage
```

**Parameters:**
- `role` (str): Message role ("system", "user", "assistant", "tool", "function")
- `content` (str): Message content
- `name` (Optional[str]): Name for function/tool messages
- `tool_calls` (Optional[List[ToolCall]]): Tool calls in the message

**Returns:**
- `ChatMessage`: Constructed message instance

**Examples:**
```python
from local_llm_sdk.models import create_chat_message

# System message
system_msg = create_chat_message("system", "You are helpful.")

# User message
user_msg = create_chat_message("user", "Hello!")

# Tool response
tool_msg = create_chat_message(
    role="tool",
    content='{"result": 42}',
    name="calculator"
)
```

### create_chat_completion_request

Helper function to create a chat completion request.

**Signature:**
```python
def create_chat_completion_request(
    model: str,
    messages: List[ChatMessage],
    **kwargs
) -> ChatCompletionRequest
```

**Parameters:**
- `model` (str): Model ID to use
- `messages` (List[ChatMessage]): Conversation messages
- `**kwargs`: Additional optional parameters (temperature, max_tokens, tools, etc.)

**Returns:**
- `ChatCompletionRequest`: Constructed request instance

**Examples:**
```python
from local_llm_sdk.models import (
    create_chat_completion_request,
    create_chat_message
)

# Basic request
request = create_chat_completion_request(
    model="gpt-4",
    messages=[
        create_chat_message("system", "You are helpful."),
        create_chat_message("user", "What's 2+2?")
    ]
)

# With parameters
request = create_chat_completion_request(
    model="gpt-4",
    messages=[create_chat_message("user", "Hello!")],
    temperature=0.7,
    max_tokens=500,
    stop=["END"]
)
```

---

## Constants

### OPENAI_MODELS

Dictionary of common OpenAI models with descriptions.

**Value:**
```python
{
    "gpt-4o": "Most capable GPT-4 Omni model",
    "gpt-4o-mini": "Small, fast, and affordable GPT-4 Omni model",
    "gpt-4-turbo": "Latest GPT-4 Turbo model",
    "gpt-4": "Standard GPT-4 model",
    "gpt-3.5-turbo": "Fast and inexpensive GPT-3.5 model",
}
```

**Example:**
```python
from local_llm_sdk.models import OPENAI_MODELS

for model_id, description in OPENAI_MODELS.items():
    print(f"{model_id}: {description}")
```

### EMBEDDING_MODELS

Dictionary of common embedding models.

**Value:**
```python
{
    "text-embedding-3-small": "Small, efficient embedding model",
    "text-embedding-3-large": "Large, powerful embedding model",
    "text-embedding-ada-002": "Legacy embedding model",
}
```

### RESPONSE_FORMATS

Dictionary of response format configurations.

**Value:**
```python
{
    "text": {"type": "text"},
    "json_object": {"type": "json_object"},
}
```

**Example:**
```python
from local_llm_sdk.models import RESPONSE_FORMATS, ChatCompletionRequest

request = ChatCompletionRequest(
    model="gpt-4",
    messages=[...],
    response_format=RESPONSE_FORMATS["json_object"]
)
```

### FINISH_REASONS

List of possible finish reasons for completions.

**Value:**
```python
[
    "stop",           # Natural stop point
    "length",         # Max tokens reached
    "tool_calls",     # Model called a tool
    "content_filter", # Content was filtered
    "function_call",  # Deprecated: Model called a function
]
```

**Example:**
```python
from local_llm_sdk.models import FINISH_REASONS

if completion.choices[0].finish_reason == "length":
    print("Response truncated - increase max_tokens")
```

---

## Complete Usage Examples

### Basic Chat Completion

```python
from local_llm_sdk.models import (
    ChatCompletionRequest,
    ChatMessage,
    create_chat_message
)

# Create request
request = ChatCompletionRequest(
    model="gpt-4",
    messages=[
        create_chat_message("system", "You are a math tutor."),
        create_chat_message("user", "Explain the Pythagorean theorem.")
    ],
    temperature=0.7,
    max_tokens=500
)

# Send to API (using requests library)
import requests

response = requests.post(
    "https://api.openai.com/v1/chat/completions",
    headers={"Authorization": f"Bearer {api_key}"},
    json=request.model_dump(exclude_none=True)
)

# Parse response
from local_llm_sdk.models import ChatCompletion

completion = ChatCompletion(**response.json())
print(completion.choices[0].message.content)
print(f"Tokens used: {completion.usage.total_tokens}")
```

### Function Calling

```python
from local_llm_sdk.models import (
    ChatCompletionRequest,
    ChatMessage,
    Tool,
    Function,
    ToolCall,
    FunctionCall
)
import json

# Define tool
calculator_tool = Tool(
    type="function",
    function=Function(
        name="calculate",
        description="Evaluate mathematical expression",
        parameters={
            "type": "object",
            "properties": {
                "expression": {
                    "type": "string",
                    "description": "Math expression like '2 + 2'"
                }
            },
            "required": ["expression"]
        }
    )
)

# Initial request
messages = [
    ChatMessage(role="user", content="What is 42 * 17?")
]

request = ChatCompletionRequest(
    model="gpt-4",
    messages=messages,
    tools=[calculator_tool],
    tool_choice="auto"
)

# Send request, get response with tool call
# (Assume we get a response with tool_calls)

# Simulate tool execution
tool_call = ToolCall(
    id="call_123",
    type="function",
    function=FunctionCall(
        name="calculate",
        arguments='{"expression": "42 * 17"}'
    )
)

# Execute tool
args = json.loads(tool_call.function.arguments)
result = eval(args["expression"])  # 714

# Add tool result to conversation
messages.append(ChatMessage(
    role="assistant",
    content=None,
    tool_calls=[tool_call]
))

messages.append(ChatMessage(
    role="tool",
    content=json.dumps({"result": result}),
    tool_call_id=tool_call.id,
    name="calculate"
))

# Send follow-up request
final_request = ChatCompletionRequest(
    model="gpt-4",
    messages=messages
)

# Get final response: "42 * 17 equals 714."
```

### Streaming Response

```python
from local_llm_sdk.models import (
    ChatCompletionRequest,
    ChatMessage,
    ChatCompletionChunk
)
import requests

request = ChatCompletionRequest(
    model="gpt-4",
    messages=[ChatMessage(role="user", content="Count to 5")],
    stream=True
)

response = requests.post(
    "https://api.openai.com/v1/chat/completions",
    headers={"Authorization": f"Bearer {api_key}"},
    json=request.model_dump(exclude_none=True),
    stream=True
)

# Process stream
full_content = ""
for line in response.iter_lines():
    if line:
        line_text = line.decode('utf-8')
        if line_text.startswith("data: "):
            data_str = line_text[6:]
            if data_str == "[DONE]":
                break

            chunk = ChatCompletionChunk(**json.loads(data_str))
            delta_content = chunk.choices[0].delta.content
            if delta_content:
                full_content += delta_content
                print(delta_content, end="", flush=True)

print(f"\n\nFull response: {full_content}")
```

### Embeddings

```python
from local_llm_sdk.models import EmbeddingsRequest, EmbeddingsResponse
import requests

# Create request
request = EmbeddingsRequest(
    input=["Hello world", "Goodbye world"],
    model="text-embedding-3-small"
)

# Send request
response = requests.post(
    "https://api.openai.com/v1/embeddings",
    headers={"Authorization": f"Bearer {api_key}"},
    json=request.model_dump(exclude_none=True)
)

# Parse response
embeddings = EmbeddingsResponse(**response.json())

for embedding_data in embeddings.data:
    print(f"Text {embedding_data.index}: {len(embedding_data.embedding)} dimensions")
    print(f"First 5 values: {embedding_data.embedding[:5]}")

print(f"Total tokens: {embeddings.usage.total_tokens}")
```

### Error Handling

```python
from local_llm_sdk.models import (
    ChatCompletionRequest,
    ChatMessage,
    ErrorResponse
)
import requests

request = ChatCompletionRequest(
    model="invalid-model",  # This will fail
    messages=[ChatMessage(role="user", content="Hello")]
)

response = requests.post(
    "https://api.openai.com/v1/chat/completions",
    headers={"Authorization": f"Bearer {api_key}"},
    json=request.model_dump(exclude_none=True)
)

if response.status_code != 200:
    error_response = ErrorResponse(**response.json())
    print(f"Error: {error_response.error.message}")
    print(f"Type: {error_response.error.type}")
    print(f"Code: {error_response.error.code}")
    # Output:
    # Error: The model `invalid-model` does not exist
    # Type: invalid_request_error
    # Code: model_not_found
```

---

## Model Validation

All models use Pydantic v2 for strict validation:

**Type Checking:**
```python
from local_llm_sdk.models import ChatMessage

# Valid
msg = ChatMessage(role="user", content="Hello")

# Invalid - raises ValidationError
try:
    msg = ChatMessage(role="invalid_role", content="Hello")
except ValidationError as e:
    print(e)  # Role must be system/user/assistant/tool/function
```

**Field Constraints:**
```python
from local_llm_sdk.models import ChatCompletionRequest

# Invalid temperature - raises ValidationError
try:
    request = ChatCompletionRequest(
        model="gpt-4",
        messages=[...],
        temperature=3.0  # Must be 0-2
    )
except ValidationError as e:
    print(e)  # temperature must be <= 2.0
```

**Serialization:**
```python
from local_llm_sdk.models import ChatMessage

msg = ChatMessage(role="user", content="Hello")

# To dict (for API requests)
msg_dict = msg.model_dump()
# {"role": "user", "content": "Hello", "name": None, ...}

# Exclude None values
msg_dict = msg.model_dump(exclude_none=True)
# {"role": "user", "content": "Hello"}

# To JSON
msg_json = msg.model_dump_json(exclude_none=True)
# '{"role":"user","content":"Hello"}'
```

---

## Best Practices

**1. Use Helper Functions:**
```python
# Good
from local_llm_sdk.models import create_chat_message

msg = create_chat_message("user", "Hello")

# Also good but more verbose
from local_llm_sdk.models import ChatMessage

msg = ChatMessage(role="user", content="Hello")
```

**2. Exclude None Values in API Requests:**
```python
# Good - only send non-None fields
request_data = request.model_dump(exclude_none=True)

# Bad - sends all fields including None
request_data = request.model_dump()
```

**3. Type Hints for Better IDE Support:**
```python
from local_llm_sdk.models import ChatCompletion, ChatMessage
from typing import List

def process_completion(completion: ChatCompletion) -> str:
    """IDE will provide autocompletion for ChatCompletion fields."""
    return completion.choices[0].message.content

def build_conversation(messages: List[ChatMessage]) -> None:
    """IDE knows messages is a list of ChatMessage."""
    for msg in messages:
        print(f"{msg.role}: {msg.content}")
```

**4. Handle Optional Fields:**
```python
from local_llm_sdk.models import ChatCompletion

completion: ChatCompletion = get_completion()

# Good - check before accessing
if completion.usage:
    print(f"Tokens: {completion.usage.total_tokens}")

# Bad - may raise AttributeError if usage is None
print(f"Tokens: {completion.usage.total_tokens}")
```

**5. Parse Tool Arguments Safely:**
```python
import json
from local_llm_sdk.models import ToolCall

tool_call: ToolCall = get_tool_call()

# Good - handle JSON parse errors
try:
    args = json.loads(tool_call.function.arguments)
except json.JSONDecodeError:
    print("Invalid tool arguments")
    args = {}

# Bad - may raise JSONDecodeError
args = json.loads(tool_call.function.arguments)
```

---

## See Also

- [Client API Reference](client.md) - LocalLLMClient for simplified API interactions
- [Tools API Reference](tools.md) - Tool/function calling system
- [Agents API Reference](agents.md) - High-level agent framework
- [OpenAI API Documentation](https://platform.openai.com/docs/api-reference) - Official OpenAI spec

---

**Last Updated:** 2025-10-01
**SDK Version:** 0.1.0
