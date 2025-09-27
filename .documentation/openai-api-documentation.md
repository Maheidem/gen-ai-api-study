# OpenAI API Reference Documentation
*Comprehensive guide to the OpenAI API - Last Updated: January 2025*

## Table of Contents

1. [Overview](#overview)
2. [Authentication](#authentication)
3. [Rate Limits](#rate-limits)
4. [Error Codes and Handling](#error-codes-and-handling)
5. [Models and Pricing](#models-and-pricing)
6. [API Endpoints](#api-endpoints)
   - [Chat Completions](#chat-completions-api)
   - [Completions (Legacy)](#completions-api-legacy)
   - [Embeddings](#embeddings-api)
   - [Images (DALL-E)](#images-api-dall-e)
   - [Audio](#audio-api)
   - [Files](#files-api)
   - [Fine-Tuning](#fine-tuning-api)
   - [Assistants](#assistants-api)
   - [Realtime](#realtime-api)
   - [Models](#models-api)
   - [Moderation](#moderation-api)
   - [Batch](#batch-api)
   - [Vision Capabilities](#vision-capabilities)
   - [Structured Outputs](#structured-outputs)
   - [Usage](#usage-api)
7. [API Management Features](#api-management-features)
8. [Best Practices](#best-practices)
9. [Code Examples](#code-examples)
10. [Recent Updates](#recent-updates-2024-2025)
11. [References](#references)

---

## Overview

The OpenAI API provides developers with access to advanced AI models for various tasks including text generation, image creation, audio transcription, embeddings, and more. The API is RESTful and uses JSON for request/response formatting.

**Base URL**: `https://api.openai.com/v1`

**Key Features**:
- Multi-modal support (text, images, audio)
- Real-time speech-to-speech capabilities
- Fine-tuning for custom models
- Assistants API for building AI assistants
- Vector embeddings for semantic search
- Content moderation capabilities

## Authentication

### API Key Authentication

All API requests require authentication using Bearer token authentication:

```http
Authorization: Bearer YOUR_API_KEY
Content-Type: application/json
```

### Request Headers

Standard headers for API requests:

```http
POST https://api.openai.com/v1/chat/completions
Authorization: Bearer YOUR_API_KEY
Content-Type: application/json
OpenAI-Organization: YOUR_ORG_ID (optional)
```

### Example cURL Request

```bash
curl https://api.openai.com/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $OPENAI_API_KEY" \
  -d '{
    "model": "gpt-4o",
    "messages": [{"role": "user", "content": "Hello!"}]
  }'
```

## Rate Limits

### Usage Tiers

OpenAI implements a tier-based rate limiting system. New accounts start at Tier 1 and progress based on usage and payment history.

### Rate Limit Headers

API responses include rate limit information in headers:

```http
x-ratelimit-limit-requests: 500
x-ratelimit-limit-tokens: 30000
x-ratelimit-remaining-requests: 499
x-ratelimit-remaining-tokens: 29996
x-ratelimit-reset-requests: 120ms
x-ratelimit-reset-tokens: 8ms
```

### Rate Limit Types

1. **Requests per minute (RPM)**: Limits the number of API calls
2. **Tokens per minute (TPM)**: Limits the total tokens processed
3. **Images per minute**: Specific to DALL-E endpoints

### Handling Rate Limits

When rate limits are exceeded, the API returns a `429 Too Many Requests` error. Best practices:
- Implement exponential backoff
- Monitor usage via headers
- Batch requests when possible
- Use streaming for large responses

## Error Codes and Handling

### Common Error Codes

#### Client Errors (4xx)

| Code | Error | Description | Resolution |
|------|-------|-------------|------------|
| 400 | Bad Request | Invalid request format or parameters | Check request syntax and parameters |
| 401 | Unauthorized | Invalid or missing API key | Verify API key is correct and active |
| 403 | Forbidden | Access denied to resource | Check account permissions and tier |
| 404 | Not Found | Requested resource doesn't exist | Verify endpoint URL and resource ID |
| 429 | Too Many Requests | Rate limit exceeded | Implement backoff and retry logic |

#### Server Errors (5xx)

| Code | Error | Description | Resolution |
|------|-------|-------------|------------|
| 500 | Internal Server Error | OpenAI server error | Retry with exponential backoff |
| 502 | Bad Gateway | Gateway error | Retry request |
| 503 | Service Unavailable | Service temporarily unavailable | Wait and retry |

### Error Response Format

```json
{
  "error": {
    "message": "Invalid API key provided",
    "type": "invalid_request_error",
    "param": null,
    "code": "invalid_api_key"
  }
}
```

### Retry Strategy

```python
import time
import random

def retry_with_exponential_backoff(
    func,
    initial_delay: float = 1,
    exponential_base: float = 2,
    jitter: bool = True,
    max_retries: int = 10,
):
    """Retry a function with exponential backoff."""

    num_retries = 0
    delay = initial_delay

    while True:
        try:
            return func()
        except Exception as e:
            num_retries += 1
            if num_retries > max_retries:
                raise Exception(f"Maximum number of retries ({max_retries}) exceeded.")

            delay *= exponential_base * (1 + jitter * random.random())
            time.sleep(delay)
```

## Models and Pricing

### Latest Models (2024-2025)

#### GPT-4.1 Series (Newest)
- **GPT-4.1**: Most advanced model with 1M token context
- **GPT-4.1 mini**: Smaller, cost-efficient version
- **GPT-4.1 nano**: Ultra-lightweight version
- Knowledge cutoff: June 2024

#### GPT-4o Series
| Model | Context | Input Price | Output Price | Notes |
|-------|---------|-------------|--------------|-------|
| gpt-4o | 128K | $2.50/1M tokens | $10.00/1M tokens | Multimodal, fastest |
| gpt-4o-mini | 128K | $0.15/1M tokens | $0.60/1M tokens | Most cost-efficient |

#### GPT-4 and GPT-4 Turbo
| Model | Context | Input Price | Output Price |
|-------|---------|-------------|--------------|
| gpt-4-turbo | 128K | $10.00/1M tokens | $30.00/1M tokens |
| gpt-4 | 8K | $30.00/1M tokens | $60.00/1M tokens |
| gpt-4-32k | 32K | $60.00/1M tokens | $120.00/1M tokens |

#### GPT-3.5 Turbo
| Model | Context | Input Price | Output Price |
|-------|---------|-------------|--------------|
| gpt-3.5-turbo-0125 | 16K | $0.50/1M tokens | $1.50/1M tokens |
| gpt-3.5-turbo | 4K | $0.50/1M tokens | $1.50/1M tokens |

### Specialized Models

#### DALL-E (Image Generation)
| Model | Resolution | Price |
|-------|------------|-------|
| DALL-E 3 HD | 1024x1024 | $0.08/image |
| DALL-E 3 Standard | 1024x1024 | $0.04/image |
| DALL-E 2 | 1024x1024 | $0.02/image |
| DALL-E 2 | 512x512 | $0.018/image |
| DALL-E 2 | 256x256 | $0.016/image |

#### Whisper (Audio)
- Transcription: $0.006/minute
- Translation: $0.006/minute

#### Embeddings
| Model | Price |
|-------|-------|
| text-embedding-3-large | $0.13/1M tokens |
| text-embedding-3-small | $0.02/1M tokens |
| text-embedding-ada-002 | $0.10/1M tokens |

### Fine-Tuning Costs
| Model | Training | Input Usage | Output Usage |
|-------|----------|-------------|--------------|
| gpt-3.5-turbo | $8.00/1M tokens | $3.00/1M tokens | $6.00/1M tokens |
| davinci-002 | $6.00/1M tokens | $12.00/1M tokens | $12.00/1M tokens |
| babbage-002 | $0.40/1M tokens | $1.60/1M tokens | $1.60/1M tokens |

## API Endpoints

### Chat Completions API

The primary interface for interacting with GPT models.

**Endpoint**: `POST /v1/chat/completions`

#### Request Parameters

```json
{
  "model": "gpt-4o",
  "messages": [
    {
      "role": "system",
      "content": "You are a helpful assistant."
    },
    {
      "role": "user",
      "content": "Hello!"
    }
  ],
  "temperature": 0.7,
  "max_tokens": 150,
  "top_p": 1.0,
  "frequency_penalty": 0.0,
  "presence_penalty": 0.0,
  "stop": ["\n"],
  "stream": false,
  "n": 1,
  "logprobs": null,
  "response_format": { "type": "json_object" },
  "seed": 1234,
  "tools": [],
  "tool_choice": "auto",
  "user": "user-123"
}
```

#### Response Format

```json
{
  "id": "chatcmpl-abc123",
  "object": "chat.completion",
  "created": 1704067200,
  "model": "gpt-4o",
  "usage": {
    "prompt_tokens": 13,
    "completion_tokens": 17,
    "total_tokens": 30
  },
  "choices": [
    {
      "message": {
        "role": "assistant",
        "content": "Hello! How can I assist you today?"
      },
      "finish_reason": "stop",
      "index": 0
    }
  ]
}
```

#### Message Roles
- `system`: Sets the behavior of the assistant
- `user`: Messages from the end user
- `assistant`: Messages from the AI assistant
- `tool`: Results from tool/function calls

#### Advanced Features

##### Streaming
```python
stream = client.chat.completions.create(
    model="gpt-4o",
    messages=[{"role": "user", "content": "Count to 10"}],
    stream=True,
)
for chunk in stream:
    print(chunk.choices[0].delta.content, end="")
```

##### JSON Mode
```json
{
  "model": "gpt-4o",
  "response_format": { "type": "json_object" },
  "messages": [
    {
      "role": "system",
      "content": "You are a helpful assistant designed to output JSON."
    },
    {
      "role": "user",
      "content": "List three colors"
    }
  ]
}
```

##### Function Calling
```json
{
  "model": "gpt-4o",
  "messages": [{"role": "user", "content": "What's the weather in Boston?"}],
  "tools": [
    {
      "type": "function",
      "function": {
        "name": "get_weather",
        "description": "Get the current weather",
        "parameters": {
          "type": "object",
          "properties": {
            "location": {
              "type": "string",
              "description": "The city and state"
            }
          },
          "required": ["location"]
        }
      }
    }
  ]
}
```

### Completions API (Legacy)

**Note**: This endpoint is considered legacy. Use Chat Completions API instead.

**Endpoint**: `POST /v1/completions`

```json
{
  "model": "gpt-3.5-turbo-instruct",
  "prompt": "Once upon a time",
  "max_tokens": 100,
  "temperature": 0.7
}
```

### Embeddings API

Generate vector representations of text for semantic search and similarity.

**Endpoint**: `POST /v1/embeddings`

#### Request
```json
{
  "model": "text-embedding-3-small",
  "input": "The quick brown fox jumps over the lazy dog",
  "encoding_format": "float"
}
```

#### Response
```json
{
  "object": "list",
  "data": [
    {
      "object": "embedding",
      "index": 0,
      "embedding": [0.0023064255, -0.009327292, ...]
    }
  ],
  "model": "text-embedding-3-small",
  "usage": {
    "prompt_tokens": 8,
    "total_tokens": 8
  }
}
```

### Images API (DALL-E)

#### Create Image
**Endpoint**: `POST /v1/images/generations`

```json
{
  "model": "dall-e-3",
  "prompt": "A cute baby sea otter",
  "n": 1,
  "size": "1024x1024",
  "quality": "standard",
  "style": "vivid"
}
```

#### Edit Image
**Endpoint**: `POST /v1/images/edits`

```bash
curl https://api.openai.com/v1/images/edits \
  -H "Authorization: Bearer $OPENAI_API_KEY" \
  -F image="@otter.png" \
  -F mask="@mask.png" \
  -F prompt="A cute baby sea otter wearing a beret" \
  -F n=1 \
  -F size="1024x1024"
```

#### Create Image Variation
**Endpoint**: `POST /v1/images/variations`

```bash
curl https://api.openai.com/v1/images/variations \
  -H "Authorization: Bearer $OPENAI_API_KEY" \
  -F image="@otter.png" \
  -F n=2 \
  -F size="1024x1024"
```

### Audio API

#### Create Transcription (Whisper)
**Endpoint**: `POST /v1/audio/transcriptions`

```python
audio_file = open("/path/to/audio.mp3", "rb")
transcript = client.audio.transcriptions.create(
  model="whisper-1",
  file=audio_file,
  response_format="text"
)
```

#### Create Translation
**Endpoint**: `POST /v1/audio/translations`

```python
audio_file = open("/path/to/german_audio.mp3", "rb")
translation = client.audio.translations.create(
  model="whisper-1",
  file=audio_file
)
```

#### Create Speech (TTS)
**Endpoint**: `POST /v1/audio/speech`

```json
{
  "model": "tts-1",
  "input": "Hello world!",
  "voice": "alloy",
  "response_format": "mp3",
  "speed": 1.0
}
```

Available voices: `alloy`, `echo`, `fable`, `onyx`, `nova`, `shimmer`

### Files API

#### Upload File
**Endpoint**: `POST /v1/files`

```python
file = client.files.create(
  file=open("training_data.jsonl", "rb"),
  purpose="fine-tune"
)
```

#### List Files
**Endpoint**: `GET /v1/files`

```python
files = client.files.list()
```

#### Retrieve File
**Endpoint**: `GET /v1/files/{file_id}`

```python
file = client.files.retrieve("file-abc123")
```

#### Delete File
**Endpoint**: `DELETE /v1/files/{file_id}`

```python
client.files.delete("file-abc123")
```

### Fine-Tuning API

#### Create Fine-Tuning Job
**Endpoint**: `POST /v1/fine_tuning/jobs`

```json
{
  "training_file": "file-abc123",
  "validation_file": "file-xyz456",
  "model": "gpt-3.5-turbo",
  "hyperparameters": {
    "n_epochs": 3,
    "batch_size": 8,
    "learning_rate_multiplier": 0.1
  },
  "suffix": "custom-model"
}
```

#### List Fine-Tuning Jobs
**Endpoint**: `GET /v1/fine_tuning/jobs`

```python
jobs = client.fine_tuning.jobs.list()
```

#### Retrieve Fine-Tuning Job
**Endpoint**: `GET /v1/fine_tuning/jobs/{job_id}`

```python
job = client.fine_tuning.jobs.retrieve("ft-abc123")
```

#### Cancel Fine-Tuning Job
**Endpoint**: `POST /v1/fine_tuning/jobs/{job_id}/cancel`

```python
client.fine_tuning.jobs.cancel("ft-abc123")
```

### Assistants API

#### Create Assistant
**Endpoint**: `POST /v1/assistants`

```json
{
  "model": "gpt-4o",
  "name": "Math Tutor",
  "instructions": "You are a personal math tutor.",
  "tools": [
    {"type": "code_interpreter"},
    {"type": "file_search"}
  ],
  "file_ids": ["file-abc123"]
}
```

#### Create Thread
**Endpoint**: `POST /v1/threads`

```json
{
  "messages": [
    {
      "role": "user",
      "content": "I need help with calculus"
    }
  ]
}
```

#### Create Run
**Endpoint**: `POST /v1/threads/{thread_id}/runs`

```json
{
  "assistant_id": "asst_abc123",
  "instructions": "Please address the user as Jane Doe.",
  "tools": [{"type": "code_interpreter"}]
}
```

#### Assistant Pricing
- Code Interpreter: $0.03 per session
- File Search: $0.10/GB of vector storage per day
- Files stored indefinitely until manually deleted

### Realtime API

The newest addition supporting real-time speech-to-speech interactions.

**Features**:
- WebSocket-based for low latency
- Supports Session Initiation Protocol (SIP)
- Voice Activity Detection (VAD)
- Multiple voice options
- Image input support

#### Connection
```javascript
const ws = new WebSocket('wss://api.openai.com/v1/realtime', {
  headers: {
    'Authorization': 'Bearer YOUR_API_KEY',
    'OpenAI-Beta': 'realtime=v1'
  }
});
```

#### Session Configuration
```json
{
  "type": "session.update",
  "session": {
    "modalities": ["text", "audio"],
    "voice": "alloy",
    "input_audio_format": "pcm16",
    "output_audio_format": "pcm16",
    "turn_detection": {
      "type": "server_vad",
      "threshold": 0.5
    }
  }
}
```

### Models API

#### List Models
**Endpoint**: `GET /v1/models`

```python
models = client.models.list()
```

#### Retrieve Model
**Endpoint**: `GET /v1/models/{model}`

```python
model = client.models.retrieve("gpt-4o")
```

### Moderation API

OpenAI's free Moderation API helps identify potentially harmful content. The latest model (text-moderation-007) uses GPT-4o for improved accuracy.

**Endpoint**: `POST /v1/moderations`

#### Request
```json
{
  "input": "Content to moderate",
  "model": "text-moderation-007"
}
```

#### Response
```json
{
  "id": "modr-abc123",
  "model": "text-moderation-007",
  "results": [
    {
      "flagged": false,
      "categories": {
        "sexual": false,
        "hate": false,
        "harassment": false,
        "self-harm": false,
        "sexual/minors": false,
        "hate/threatening": false,
        "violence/graphic": false,
        "self-harm/intent": false,
        "self-harm/instructions": false,
        "harassment/threatening": false,
        "violence": false
      },
      "category_scores": {
        "sexual": 0.00023,
        "hate": 0.00011,
        "harassment": 0.00032,
        "self-harm": 0.00001,
        "sexual/minors": 0.00001,
        "hate/threatening": 0.00001,
        "violence/graphic": 0.00001,
        "self-harm/intent": 0.00001,
        "self-harm/instructions": 0.00001,
        "harassment/threatening": 0.00001,
        "violence": 0.00045
      }
    }
  ]
}
```

#### Moderation Categories
- **harassment**: Content that promotes, encourages, or depicts acts of harassment
- **hate**: Content that expresses, incites, or promotes hate
- **self-harm**: Content that promotes, encourages, or depicts acts of self-harm
- **sexual**: Sexual content including sexual activity or sexual services
- **violence**: Content depicting violence or physical harm

### Batch API

Process multiple requests asynchronously at 50% lower cost with higher rate limits. Batches complete within 24 hours.

**Endpoint**: `POST /v1/batches`

#### Creating a Batch

1. **Prepare JSONL file** with requests:
```jsonl
{"custom_id": "request-1", "method": "POST", "url": "/v1/chat/completions", "body": {"model": "gpt-4o", "messages": [{"role": "user", "content": "Hello!"}]}}
{"custom_id": "request-2", "method": "POST", "url": "/v1/chat/completions", "body": {"model": "gpt-4o", "messages": [{"role": "user", "content": "How are you?"}]}}
```

2. **Upload file**:
```python
batch_file = client.files.create(
  file=open("batch_requests.jsonl", "rb"),
  purpose="batch"
)
```

3. **Create batch**:
```python
batch = client.batches.create(
  input_file_id=batch_file.id,
  endpoint="/v1/chat/completions",
  completion_window="24h"
)
```

4. **Check status**:
```python
batch_status = client.batches.retrieve(batch.id)
print(batch_status.status)  # "validating", "in_progress", "completed", "failed"
```

5. **Retrieve results**:
```python
if batch_status.status == "completed":
    result_file = client.files.content(batch_status.output_file_id)
    results = result_file.read()
```

#### Batch API with Vision

The Batch API fully supports vision models for image analysis:

```jsonl
{
  "custom_id": "img-1",
  "method": "POST",
  "url": "/v1/chat/completions",
  "body": {
    "model": "gpt-4o",
    "messages": [
      {
        "role": "user",
        "content": [
          {"type": "text", "text": "What's in this image?"},
          {"type": "image_url", "image_url": {"url": "https://example.com/image.jpg"}}
        ]
      }
    ]
  }
}
```

### Vision Capabilities

GPT-4 models can understand and analyze images alongside text.

#### Image Input Format

```python
response = client.chat.completions.create(
  model="gpt-4o",
  messages=[
    {
      "role": "user",
      "content": [
        {"type": "text", "text": "What's in this image?"},
        {
          "type": "image_url",
          "image_url": {
            "url": "https://example.com/image.jpg",
            "detail": "high"  # "low", "high", or "auto"
          }
        }
      ]
    }
  ]
)
```

#### Base64 Image Input

```python
import base64

with open("image.jpg", "rb") as image_file:
    base64_image = base64.b64encode(image_file.read()).decode('utf-8')

response = client.chat.completions.create(
  model="gpt-4o",
  messages=[
    {
      "role": "user",
      "content": [
        {"type": "text", "text": "Describe this image"},
        {
          "type": "image_url",
          "image_url": {
            "url": f"data:image/jpeg;base64,{base64_image}"
          }
        }
      ]
    }
  ]
)
```

#### Vision Best Practices

- **Image size**: Resize large images to reduce token usage
- **Detail level**: Use "low" for faster, cheaper processing
- **Multiple images**: Can process multiple images in one request
- **Supported formats**: JPEG, PNG, GIF, WebP

### Structured Outputs

Ensure model outputs always match your JSON schema (100% reliability with gpt-4o-2024-08-06).

#### Using Response Format

```python
from pydantic import BaseModel

class CalendarEvent(BaseModel):
    name: str
    date: str
    participants: list[str]

response = client.chat.completions.create(
  model="gpt-4o-2024-08-06",
  messages=[
    {"role": "user", "content": "Create a calendar event for a team meeting tomorrow at 2pm with John and Sarah"}
  ],
  response_format=CalendarEvent
)
```

#### Using JSON Schema

```python
response = client.chat.completions.create(
  model="gpt-4o",
  messages=[
    {"role": "system", "content": "Extract information as JSON"},
    {"role": "user", "content": "John Doe is 30 years old and lives in NYC"}
  ],
  response_format={
    "type": "json_schema",
    "json_schema": {
      "name": "person_info",
      "strict": True,
      "schema": {
        "type": "object",
        "properties": {
          "name": {"type": "string"},
          "age": {"type": "integer"},
          "city": {"type": "string"}
        },
        "required": ["name", "age", "city"]
      }
    }
  }
)
```

#### Structured Outputs with Vision

```python
class ImageAnalysis(BaseModel):
    objects: list[str]
    scene_description: str
    dominant_colors: list[str]

response = client.chat.completions.create(
  model="gpt-4o-2024-08-06",
  messages=[
    {
      "role": "user",
      "content": [
        {"type": "text", "text": "Analyze this image"},
        {"type": "image_url", "image_url": {"url": image_url}}
      ]
    }
  ],
  response_format=ImageAnalysis
)
```

### Usage API

Monitor your OpenAI API usage programmatically.

#### Get Usage
**Endpoint**: `GET /v1/usage`

Parameters:
- `start_date`: Start of the usage period (YYYY-MM-DD)
- `end_date`: End of the usage period (YYYY-MM-DD)
- `project_id`: Filter by project
- `user_id`: Filter by user
- `api_key_id`: Filter by API key
- `model`: Filter by model

```python
# Example: Get usage for current month
usage = client.usage.retrieve(
    start_date="2025-01-01",
    end_date="2025-01-31",
    model="gpt-4o"
)
```

## API Management Features

### API Key Permissions

Configure granular permissions for API keys:

- **Read-only access**: For dashboards and monitoring
- **Endpoint restrictions**: Limit keys to specific endpoints
- **Model restrictions**: Limit keys to specific models
- **Project scoping**: Restrict keys to specific projects

### Usage Dashboard

Access the usage dashboard at: https://platform.openai.com/usage

Features:
- Daily usage breakdown
- Cost tracking by model
- Token usage visualization
- API key level metrics (with tracking enabled)
- Export functionality for custom analysis

### Organization Management

- **Roles**: Owner, Member, Reader
- **Billing**: Prepaid credits system
- **Limits**: Set spending limits per project
- **Monitoring**: Real-time usage alerts

## Best Practices

### 1. Token Management
- Monitor token usage in responses
- Estimate tokens before requests (1 token ≈ 4 characters)
- Use `max_tokens` to control costs
- Implement token counting in your application

### 2. Prompt Engineering
```python
# Good practice: Clear, specific prompts
messages = [
    {"role": "system", "content": "You are a helpful assistant that provides concise answers."},
    {"role": "user", "content": "Explain quantum computing in 2 sentences."}
]

# Use few-shot examples for consistent output
messages = [
    {"role": "system", "content": "Extract key information from text."},
    {"role": "user", "content": "Apple Inc. reported $90B in Q4 2023 revenue."},
    {"role": "assistant", "content": '{"company": "Apple Inc.", "metric": "revenue", "value": "$90B", "period": "Q4 2023"}'},
    {"role": "user", "content": "Microsoft saw 15% growth in cloud services."}
]
```

### 3. Error Handling
```python
import openai
from openai import OpenAI

client = OpenAI()

def safe_api_call(messages, max_retries=3):
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=messages
            )
            return response
        except openai.RateLimitError as e:
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)  # Exponential backoff
                continue
            raise e
        except openai.APIError as e:
            # Handle API errors
            print(f"API error: {e}")
            raise e
```

### 4. Streaming for Better UX
```python
def stream_response(messages):
    stream = client.chat.completions.create(
        model="gpt-4o",
        messages=messages,
        stream=True
    )

    full_response = ""
    for chunk in stream:
        if chunk.choices[0].delta.content is not None:
            content = chunk.choices[0].delta.content
            full_response += content
            print(content, end="", flush=True)

    return full_response
```

### 5. Caching Strategies
- Cache embeddings for repeated queries
- Store fine-tuned model IDs
- Implement conversation history management
- Use file IDs for repeated file operations

### 6. Security Best Practices
- Never expose API keys in client-side code
- Use environment variables for API keys
- Implement request validation
- Set up usage monitoring and alerts
- Use organization IDs for team management

### 7. Cost Optimization
- Use appropriate models for tasks (GPT-3.5 for simple tasks)
- Batch requests when possible
- Implement caching for repeated queries
- Monitor and set spending limits
- Use streaming to reduce perceived latency

## Code Examples

### Python SDK Example
```python
from openai import OpenAI

client = OpenAI(api_key="YOUR_API_KEY")

# Simple chat completion
response = client.chat.completions.create(
    model="gpt-4o",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What is the capital of France?"}
    ],
    temperature=0.7,
    max_tokens=100
)

print(response.choices[0].message.content)
```

### Node.js SDK Example
```javascript
import OpenAI from 'openai';

const openai = new OpenAI({
  apiKey: process.env.OPENAI_API_KEY,
});

async function main() {
  const completion = await openai.chat.completions.create({
    model: "gpt-4o",
    messages: [
      { role: "system", content: "You are a helpful assistant." },
      { role: "user", content: "What is the capital of France?" }
    ],
  });

  console.log(completion.choices[0].message.content);
}

main();
```

### Function Calling Example
```python
def get_weather(location):
    # Mock weather function
    return {"location": location, "temperature": 72, "unit": "fahrenheit"}

functions = [
    {
        "name": "get_weather",
        "description": "Get the current weather in a location",
        "parameters": {
            "type": "object",
            "properties": {
                "location": {
                    "type": "string",
                    "description": "The city and state, e.g. San Francisco, CA"
                }
            },
            "required": ["location"]
        }
    }
]

messages = [{"role": "user", "content": "What's the weather like in Boston?"}]

response = client.chat.completions.create(
    model="gpt-4o",
    messages=messages,
    functions=functions,
    function_call="auto"
)

# Check if function was called
if response.choices[0].message.function_call:
    function_name = response.choices[0].message.function_call.name
    function_args = json.loads(response.choices[0].message.function_call.arguments)

    # Execute function
    function_response = get_weather(function_args["location"])

    # Send function result back to model
    messages.append(response.choices[0].message)
    messages.append({
        "role": "function",
        "name": function_name,
        "content": json.dumps(function_response)
    })

    final_response = client.chat.completions.create(
        model="gpt-4o",
        messages=messages
    )
```

### Embedding Search Example
```python
import numpy as np
from openai import OpenAI

client = OpenAI()

def get_embedding(text, model="text-embedding-3-small"):
    response = client.embeddings.create(
        input=text,
        model=model
    )
    return response.data[0].embedding

def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

# Create embeddings for documents
documents = [
    "The cat sat on the mat.",
    "The dog played in the park.",
    "Machine learning is fascinating."
]

doc_embeddings = [get_embedding(doc) for doc in documents]

# Search query
query = "feline resting"
query_embedding = get_embedding(query)

# Find most similar document
similarities = [cosine_similarity(query_embedding, doc_emb) for doc_emb in doc_embeddings]
most_similar_idx = np.argmax(similarities)

print(f"Most similar document: {documents[most_similar_idx]}")
print(f"Similarity score: {similarities[most_similar_idx]}")
```

## Recent Updates (2024-2025)

### January 2025
- **GPT-4.1 Series Launch**: New models with 1M token context and June 2024 knowledge cutoff
- Major improvements in coding and instruction following

### December 2024
- **Realtime API Price Drop**: 60% reduction in audio token costs
- **Assistants API v2**: Only v2 version supported
- **GPT-4o Mini TTS**: New text-to-speech capabilities

### 2024 Highlights
- **Prepaid Billing**: Shift from pay-as-you-go to prepaid model
- **Enhanced Multimodal**: Improved vision and audio capabilities in GPT-4o
- **Embedding Models**: New text-embedding-3 series with better performance
- **Fine-Tuning Expansion**: GPT-4o mini fine-tuning for Tier 4+ users
- **JSON Mode**: Native JSON output formatting
- **Streaming Improvements**: Better support for real-time applications

## References

### Official Documentation
- Main API Reference: [https://platform.openai.com/docs/api-reference](https://platform.openai.com/docs/api-reference)
- Models Documentation: [https://platform.openai.com/docs/models](https://platform.openai.com/docs/models)
- Pricing: [https://openai.com/api/pricing/](https://openai.com/api/pricing/)
- Rate Limits Guide: [https://platform.openai.com/docs/guides/rate-limits](https://platform.openai.com/docs/guides/rate-limits)
- Error Codes: [https://platform.openai.com/docs/guides/error-codes](https://platform.openai.com/docs/guides/error-codes)

### SDKs and Libraries
- Python SDK: [https://pypi.org/project/openai/](https://pypi.org/project/openai/)
- Node.js SDK: npm install openai
- Community SDKs available for Go, Rust, Java, C#, and more

### Additional Resources
- OpenAI Cookbook: Examples and best practices
- API Playground: Interactive testing environment
- Community Forum: Developer discussions and support
- Status Page: API availability and incident reports

---

*Note: This documentation is based on publicly available information as of January 2025. Always refer to the official OpenAI documentation for the most current information, as APIs and pricing may change.*

## Version History
- v1.0 (January 2025): Initial comprehensive documentation
- Based on OpenAI API updates through January 2025
- Includes GPT-4.1, Realtime API, and latest pricing

---

*For questions or updates, consult the official OpenAI API documentation at platform.openai.com*