# Basic Usage

A beginner-friendly guide to using Local LLM SDK for everyday tasks.

## Overview

This guide covers the fundamental concepts and patterns you'll use in most projects. By the end, you'll understand:

- Core concepts (client, messages, conversation)
- How to send chat messages
- System prompts and behavior control
- Temperature and parameter tuning
- Model selection
- Basic error handling
- Common usage patterns

**Estimated time**: 20 minutes

## Prerequisites

- SDK installed (`pip install -e .`)
- Local LLM server running (LM Studio or Ollama)
- Completed the [Quick Start](quickstart.md) guide

## Core Concepts

### The Client

The `LocalLLMClient` is your main interface to the LLM. It manages:
- Connection to your local server
- Conversation history
- Tool execution
- Configuration and defaults

```python
from local_llm_sdk import LocalLLMClient

client = LocalLLMClient(
    base_url="http://localhost:1234/v1",  # Your LLM server
    model="auto",                          # Auto-detect first model
    timeout=300                            # 5 minutes timeout
)
```

### Messages

All LLM interactions use **messages** with two key components:
- **role**: Who's speaking (`system`, `user`, or `assistant`)
- **content**: What they're saying

```python
from local_llm_sdk import create_chat_message

# User message
user_msg = create_chat_message("user", "What is Python?")

# System message (sets behavior)
system_msg = create_chat_message("system", "You are a helpful coding tutor.")

# Assistant message (LLM's previous response)
assistant_msg = create_chat_message("assistant", "Python is a programming language...")
```

### Conversation

A **conversation** is a list of messages exchanged between you and the LLM. The client automatically maintains this history so the LLM remembers context.

```python
# The client tracks all messages
client.chat("My name is Alice")
client.chat("What's my name?")  # LLM remembers: "Your name is Alice"

# View the conversation
for msg in client.conversation:
    print(f"{msg.role}: {msg.content}")
```

## Simple Chat Examples

### Single Message

The simplest way to chat - just pass a string:

```python
from local_llm_sdk import LocalLLMClient

client = LocalLLMClient()

# Ask a question
response = client.chat("What is the capital of France?")
print(response)
# Output: "The capital of France is Paris."
```

**What happened behind the scenes:**
1. Your string was wrapped in a `user` message
2. Sent to the LLM server
3. Response extracted and returned as a clean string

### Multiple Questions

Each call automatically builds on the conversation:

```python
# First question
answer1 = client.chat("What is machine learning?")
print(answer1)

# Related follow-up (has context from first question)
answer2 = client.chat("Give me an example")
print(answer2)

# Another follow-up
answer3 = client.chat("What are the challenges?")
print(answer3)
```

The LLM sees the full conversation history and provides contextual answers.

### One-Liner for Simple Queries

For quick, one-off questions without maintaining history:

```python
from local_llm_sdk import quick_chat

# No client needed - uses defaults
answer = quick_chat("What is 2 + 2?")
print(answer)  # "4"
```

## System Prompts

**System prompts** define the LLM's personality, behavior, and constraints. Think of them as instructions you give before the conversation starts.

### Basic System Prompt

```python
response = client.chat(
    "Tell me about Python",
    system="You are a helpful programming tutor."
)
```

### Personality Examples

**Professional Advisor:**
```python
response = client.chat(
    "How should I learn programming?",
    system="You are a professional career advisor. Be concise and actionable."
)
```

**Enthusiastic Teacher:**
```python
response = client.chat(
    "What is a function?",
    system="You are an enthusiastic teacher who loves programming. Use lots of encouragement!"
)
```

**Explain Like I'm 5:**
```python
response = client.chat(
    "What is recursion?",
    system="Explain things like you're talking to a 5-year-old. Use simple words and fun analogies."
)
```

### System Prompt Best Practices

**DO:**
- Be specific about tone and style
- Set clear constraints (length, format, language)
- Define expertise level
- Specify output format if needed

```python
system = """
You are a senior Python developer with 10 years of experience.
Provide concise, production-ready code examples.
Always include error handling.
Explain your reasoning briefly.
"""
```

**DON'T:**
- Make system prompts too long (keep under 200 words)
- Contradict yourself in instructions
- Include user-specific details (put those in the user message)

```python
# Bad: Conflicting instructions
system = "Be very brief. Provide detailed, comprehensive explanations."

# Good: Clear and aligned
system = "Be concise but thorough. Explain key concepts in 2-3 sentences each."
```

## Temperature and Parameters

Temperature controls how random or creative the LLM's responses are.

### Temperature Scale

| Temperature | Behavior | Best For |
|-------------|----------|----------|
| **0.0 - 0.3** | Deterministic, focused, consistent | Math, code, facts, data extraction |
| **0.5 - 0.7** | Balanced, reliable, coherent | General chat, Q&A, tutoring |
| **0.8 - 1.2** | Creative, varied, interesting | Writing, brainstorming, ideas |
| **1.3 - 2.0** | Very creative, experimental | Poetry, artistic content, wild ideas |

**Default**: 0.7 (balanced)

### Temperature Examples

**Low Temperature (Factual Tasks):**
```python
# Math problem - want consistent, correct answer
answer = client.chat(
    "What is 127 * 893?",
    temperature=0.1
)
print(answer)  # Will give same answer every time
```

**Medium Temperature (General Chat):**
```python
# Tutoring - want helpful but natural
answer = client.chat(
    "How do I learn Python?",
    temperature=0.7
)
```

**High Temperature (Creative Tasks):**
```python
# Story writing - want variety and creativity
story = client.chat(
    "Write a one-sentence story about a robot",
    temperature=1.2
)
print(story)  # Different creative story each time
```

### Testing Temperature

Run the same prompt with different temperatures to see the effect:

```python
prompt = "Give me a creative name for a coffee shop"

for temp in [0.3, 0.7, 1.2]:
    response = client.chat(prompt, temperature=temp)
    print(f"Temperature {temp}: {response}\n")
```

### Other Important Parameters

**max_tokens** - Limit response length:
```python
response = client.chat(
    "Explain machine learning",
    max_tokens=100  # Short response
)
```

**top_p** - Alternative to temperature (nucleus sampling):
```python
response = client.chat(
    "Tell me a joke",
    top_p=0.9  # Consider top 90% of probability mass
)
```

**Combining parameters:**
```python
response = client.chat(
    "Write a product description for running shoes",
    system="You are a creative marketing copywriter.",
    temperature=0.9,
    max_tokens=150,
    top_p=0.95
)
```

## Model Selection

### Auto-Detection (Recommended)

Let the SDK automatically find and use the first available model:

```python
client = LocalLLMClient(model="auto")
# Output: "✓ Auto-detected model: mistralai/magistral-small-2509"
```

### Listing Available Models

See what models your server has loaded:

```python
models = client.list_models()

print("Available models:")
for model in models.data:
    print(f"  - {model.id}")
```

### Specifying a Model

Choose a specific model if you have multiple loaded:

```python
# Use a specific model
client = LocalLLMClient(model="mistralai/magistral-small-2509")

# Or change model on the fly
client.default_model = "qwen/qwen3-coder-30b"
```

### Model Selection Tips

**For coding tasks:**
```python
# Code-specialized models work best
client = LocalLLMClient(model="qwen/qwen3-coder-30b")
```

**For general chat:**
```python
# General-purpose models
client = LocalLLMClient(model="mistralai/magistral-small-2509")
```

**For speed:**
```python
# Smaller models = faster responses
client = LocalLLMClient(model="mistral-7b")
```

**For quality:**
```python
# Larger models = better quality (but slower)
client = LocalLLMClient(model="llama-70b")
```

## Basic Error Handling

### Connection Errors

Check if your LLM server is running:

```python
from local_llm_sdk import LocalLLMClient

try:
    client = LocalLLMClient()
    response = client.chat("Hello!")
    print(response)
except Exception as e:
    print(f"Error: {e}")
    print("\nTroubleshooting:")
    print("1. Is LM Studio or Ollama running?")
    print("2. Is a model loaded?")
    print("3. Check the base_url is correct")
```

### Verify Server Connection

```python
import requests

try:
    response = requests.get("http://localhost:1234/v1/models", timeout=5)
    if response.status_code == 200:
        print("✓ Server is running")
        models = response.json()
        print(f"✓ {len(models.get('data', []))} model(s) available")
    else:
        print(f"✗ Server returned status {response.status_code}")
except requests.exceptions.ConnectionError:
    print("✗ Cannot connect to server")
    print("  → Start LM Studio or Ollama")
except requests.exceptions.Timeout:
    print("✗ Server timeout")
    print("  → Check server is responsive")
```

### Timeout Handling

For slow models or complex tasks, increase timeout:

```python
# Default timeout is 300 seconds (5 minutes)
client = LocalLLMClient(timeout=600)  # 10 minutes

# Or set via environment variable
import os
os.environ['LLM_TIMEOUT'] = '600'
client = LocalLLMClient()
```

### Model Not Loaded

```python
try:
    client = LocalLLMClient(model="auto")
    if client.default_model is None:
        print("⚠ No model loaded in LM Studio")
        print("  1. Open LM Studio")
        print("  2. Go to 'Local Server' tab")
        print("  3. Load a model")
except Exception as e:
    print(f"Error: {e}")
```

### Graceful Degradation

```python
def chat_with_fallback(prompt, client):
    """Chat with automatic fallback on errors."""
    try:
        return client.chat(prompt)
    except requests.exceptions.Timeout:
        print("⚠ Request timed out, retrying with simpler prompt...")
        return client.chat(f"Briefly: {prompt}", max_tokens=100)
    except requests.exceptions.ConnectionError:
        return "Error: Cannot connect to LLM server. Please check it's running."
    except Exception as e:
        return f"Error: {str(e)}"

# Use it
response = chat_with_fallback("Explain quantum computing", client)
print(response)
```

## Common Patterns

### Pattern 1: Conversational Assistant

```python
from local_llm_sdk import LocalLLMClient

client = LocalLLMClient()

# Set personality once
system_prompt = "You are a helpful assistant. Be concise and friendly."

print("Chat with the AI (type 'exit' to quit)")
while True:
    user_input = input("\nYou: ")
    if user_input.lower() == 'exit':
        break

    response = client.chat(user_input, system=system_prompt)
    print(f"AI: {response}")
```

### Pattern 2: Batch Processing

```python
questions = [
    "What is Python?",
    "What is JavaScript?",
    "What is Rust?"
]

answers = []
for question in questions:
    answer = client.chat(question, system="Be concise, 1-2 sentences only.")
    answers.append({"question": question, "answer": answer})

# Display results
for item in answers:
    print(f"Q: {item['question']}")
    print(f"A: {item['answer']}\n")
```

### Pattern 3: Template-Based Generation

```python
def generate_email(recipient, topic, tone="professional"):
    """Generate an email using a template."""
    system = f"You are writing a {tone} email. Be clear and concise."
    prompt = f"Write an email to {recipient} about {topic}"

    return client.chat(prompt, system=system, temperature=0.7)

# Use it
email1 = generate_email("engineering team", "new deployment process")
email2 = generate_email("CEO", "quarterly results", tone="formal")
email3 = generate_email("colleague", "lunch plans", tone="casual")
```

### Pattern 4: Iterative Refinement

```python
def refine_until_satisfied(initial_prompt, max_attempts=3):
    """Iteratively refine a response."""
    response = client.chat(initial_prompt)
    print(f"Attempt 1:\n{response}\n")

    for i in range(2, max_attempts + 1):
        feedback = input(f"Feedback (or 'done' if satisfied): ")
        if feedback.lower() == 'done':
            return response

        response = client.chat(f"Improve this based on feedback: {feedback}")
        print(f"\nAttempt {i}:\n{response}\n")

    return response

# Use it
final = refine_until_satisfied("Write a tagline for a tech startup")
```

### Pattern 5: Structured Output Extraction

```python
def extract_info(text, info_type):
    """Extract specific information from text."""
    system = f"""
    Extract {info_type} from the text and return ONLY the extracted information.
    If not found, return 'Not found'.
    """
    return client.chat(text, system=system, temperature=0.1)

# Use it
text = "Contact John Smith at john@example.com or call (555) 123-4567"

email = extract_info(text, "email address")
phone = extract_info(text, "phone number")
name = extract_info(text, "person's name")

print(f"Email: {email}")
print(f"Phone: {phone}")
print(f"Name: {name}")
```

### Pattern 6: Configuration from Environment

```python
import os
from local_llm_sdk import LocalLLMClient

# Set via environment variables
os.environ['LLM_BASE_URL'] = 'http://localhost:1234/v1'
os.environ['LLM_MODEL'] = 'mistralai/magistral-small-2509'
os.environ['LLM_TIMEOUT'] = '600'

# Client automatically uses environment config
client = LocalLLMClient()

# Or override specific values
client = LocalLLMClient(temperature=0.8)  # Use env for base_url, model, timeout
```

### Pattern 7: Response Validation

```python
def chat_with_validation(prompt, validation_fn, max_retries=3):
    """Chat with automatic validation and retry."""
    for attempt in range(max_retries):
        response = client.chat(prompt, temperature=0.3)

        if validation_fn(response):
            return response

        print(f"Attempt {attempt + 1} failed validation, retrying...")

    raise ValueError(f"Failed to get valid response after {max_retries} attempts")

# Example: Ensure response is a number
def is_number(text):
    try:
        float(text.strip())
        return True
    except:
        return False

# Use it
result = chat_with_validation(
    "What is 127 * 893? Respond with ONLY the number.",
    is_number
)
print(f"Result: {result}")
```

## Next Steps

### Immediate Next Steps

You now understand the basics! Here's what to learn next:

1. **[Tool Calling Guide](../guides/tool-calling.md)** - Let the LLM execute functions (calculate, search files, etc.)
2. **[Conversation History](../guides/conversation-management.md)** - Advanced context management
3. **[Configuration Guide](configuration.md)** - Environment variables and advanced settings

### Interactive Learning

Work through the tutorial notebooks for hands-on practice:

```bash
cd notebooks/
jupyter notebook
```

**Recommended path:**
1. `02-basic-chat.ipynb` - Reinforce what you learned here
2. `03-conversation-history.ipynb` - Multi-turn conversations
3. `04-tool-calling-basics.ipynb` - Function calling
4. `07-react-agents.ipynb` - Build autonomous agents

### API Reference

For detailed documentation on every method and parameter:

- **[Client API Reference](../api-reference/client.md)** - All `LocalLLMClient` methods
- **[Models Reference](../api-reference/models.md)** - Message types and data structures
- **[Parameters Reference](../api-reference/parameters.md)** - Complete list of options

### Building Real Applications

Once comfortable with basics, explore production patterns:

- **[Error Handling Guide](../guides/error-handling.md)** - Robust error handling
- **[Production Patterns](../guides/production-patterns.md)** - Best practices for real apps
- **[MLflow Integration](../guides/mlflow-observability.md)** - Tracing and debugging

## Quick Reference

### Essential Imports

```python
from local_llm_sdk import (
    LocalLLMClient,           # Main client
    create_client_with_tools, # Client with built-in tools
    quick_chat,               # One-liner for simple queries
    create_chat_message,      # Create message objects
)
```

### Essential Methods

```python
# Create client
client = LocalLLMClient()

# Simple chat
response = client.chat("Hello!")

# With system prompt
response = client.chat("Hello!", system="You are helpful")

# With parameters
response = client.chat("Hello!", temperature=0.9, max_tokens=100)

# List models
models = client.list_models()

# View conversation
for msg in client.conversation:
    print(f"{msg.role}: {msg.content}")

# Reset conversation
client.conversation.clear()
```

### Common Parameters

```python
client.chat(
    message,                    # str or list of messages
    system=None,                # System prompt (str)
    temperature=0.7,            # 0.0-2.0 (creativity)
    max_tokens=None,            # Limit response length
    top_p=1.0,                  # Nucleus sampling
    use_tools=False,            # Enable tool calling
    return_full_response=False  # Return ChatCompletion object
)
```

### Environment Variables

```bash
export LLM_BASE_URL="http://localhost:1234/v1"
export LLM_MODEL="auto"
export LLM_TIMEOUT="300"
export LLM_DEBUG="true"
```

## Troubleshooting

### "Connection refused"

```bash
# Check if server is running
curl http://localhost:1234/v1/models

# If not running:
# - Start LM Studio (Local Server tab)
# - Or start Ollama (ollama serve)
```

### "No model found"

```python
# Auto-detect won't work if no models loaded
# Solution: Load a model in LM Studio first
# Then run:
client = LocalLLMClient(model="auto")
```

### "Response too slow"

```python
# Increase timeout
client = LocalLLMClient(timeout=600)  # 10 minutes

# Or use a smaller/faster model
client = LocalLLMClient(model="mistral-7b")
```

### "Response is inconsistent"

```python
# Lower temperature for more consistency
response = client.chat("Your prompt", temperature=0.1)
```

### "Response is boring"

```python
# Raise temperature for more creativity
response = client.chat("Your prompt", temperature=1.2)
```

## Summary

You've learned:

- **Core concepts**: Client, messages, and conversation
- **Simple chat**: Send messages and get responses
- **System prompts**: Control personality and behavior
- **Temperature**: Tune creativity vs consistency
- **Model selection**: Choose and auto-detect models
- **Error handling**: Graceful failures and validation
- **Common patterns**: Real-world usage examples

**Key Takeaways:**

1. Start simple with `client.chat("message")`
2. Use system prompts to set behavior
3. Match temperature to task type (low for facts, high for creativity)
4. Handle errors gracefully
5. Build on these basics with tools and agents

Ready to level up? Continue to the [Tool Calling Guide](../guides/tool-calling.md)!
