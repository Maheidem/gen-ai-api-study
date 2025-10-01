# Conversation Management Guide

Complete guide to managing multi-turn conversations with the Local LLM SDK, including context preservation, state management, and best practices.

---

## Table of Contents

1. [Overview](#overview)
2. [Multi-Turn Conversations](#multi-turn-conversations)
3. [Conversation State Management](#conversation-state-management)
4. [Using chat_with_history()](#using-chat_with_history)
5. [System Prompts](#system-prompts)
6. [Advanced Context Management](#advanced-context-management)
7. [Conversation Persistence](#conversation-persistence)
8. [Context Window Management](#context-window-management)
9. [Best Practices](#best-practices)
10. [Examples](#examples)

---

## Overview

### The Problem: Stateless Chat

Each `chat()` call is independent by default - the LLM doesn't remember previous interactions:

```python
from local_llm_sdk import LocalLLMClient

client = LocalLLMClient()

# First message
response1 = client.chat("My name is Alice.")
# "Hello Alice! How can I assist you today?"

# Second message - LLM won't remember!
response2 = client.chat("What's my name?")
# "I don't have that information. Could you tell me your name?"
```

### The Solution: Conversation History

Maintain context by passing conversation history between turns:

```python
# Start a conversation
history = []

# Turn 1
response1, history = client.chat_with_history("My name is Alice.", history)

# Turn 2 - Now it remembers!
response2, history = client.chat_with_history("What's my name?", history)
# "Your name is Alice!"
```

---

## Multi-Turn Conversations

### Basic Multi-Turn Pattern

```python
from local_llm_sdk import LocalLLMClient

client = LocalLLMClient()
history = []

# Define conversation turns
conversation = [
    "I'm planning a trip to Japan.",
    "What's the best time of year to visit?",
    "I love food. What should I try?",
    "Can you summarize your advice?"
]

# Execute conversation
for user_message in conversation:
    response, history = client.chat_with_history(user_message, history)
    print(f"You: {user_message}")
    print(f"LLM: {response}\n")

print(f"Total messages: {len(history)}")
# Total messages: 8 (4 user + 4 assistant)
```

### Understanding Message Flow

Each turn adds TWO messages to history:
1. User message (role="user")
2. Assistant response (role="assistant")

```python
# After 3 turns:
# history[0] = user: "I'm planning a trip to Japan."
# history[1] = assistant: "Great! Japan is..."
# history[2] = user: "What's the best time?"
# history[3] = assistant: "Spring and autumn are..."
# history[4] = user: "I love food..."
# history[5] = assistant: "You should try..."
```

---

## Conversation State Management

### Message Structure

Messages are `ChatMessage` objects with key attributes:

```python
from local_llm_sdk.models import ChatMessage

# Example message
message = ChatMessage(
    role="user",  # "system", "user", "assistant", "tool"
    content="Hello, world!"
)

# Access via attributes
print(message.role)     # "user"
print(message.content)  # "Hello, world!"
```

### Inspecting Conversation State

```python
# View conversation history
for i, msg in enumerate(history, 1):
    print(f"{i}. {msg.role}: {msg.content[:50]}...")

# Output:
# 1. user: My name is Alice...
# 2. assistant: Hello Alice! How can I assist you today?...
# 3. user: What's my name?...
# 4. assistant: Your name is Alice!...
```

### Tracking Conversation Additions

When using tools, the SDK tracks additional messages created:

```python
response = client.chat("Calculate 5 factorial", use_tools=True)

# Check what was added to conversation
for msg in client.last_conversation_additions:
    print(f"{msg.role}: {msg.content[:50]}...")

# Output shows:
# - assistant: [tool calls]
# - tool: {"result": 120}
# - assistant: The factorial of 5 is 120
```

---

## Using chat_with_history()

### Method Signature

```python
def chat_with_history(
    query: str,
    history: List[ChatMessage],
    **kwargs
) -> tuple[str, List[ChatMessage]]:
    """
    Chat with conversation history.

    Args:
        query: New user query
        history: Previous conversation messages
        **kwargs: Additional parameters (system, temperature, etc.)

    Returns:
        Tuple of (response_string, updated_history)
    """
```

### Basic Usage

```python
history = []

# First turn
response, history = client.chat_with_history(
    "Hello, my name is Alice.",
    history
)
print(response)  # "Hello Alice! How can I help you today?"

# Second turn
response, history = client.chat_with_history(
    "What's my name?",
    history
)
print(response)  # "Your name is Alice!"
```

### With Additional Parameters

```python
# Use custom temperature and max tokens
response, history = client.chat_with_history(
    "Write a creative story.",
    history,
    temperature=0.9,  # More creative
    max_tokens=500
)

# Enable tools
response, history = client.chat_with_history(
    "Calculate 42 * 17",
    history,
    use_tools=True
)

# Force tool usage (for reasoning models)
response, history = client.chat_with_history(
    "What is 5!",
    history,
    use_tools=True,
    tool_choice="required"
)
```

---

## System Prompts

### Adding System Prompts

System prompts set the LLM's behavior. Include them ONLY on the first turn:

```python
history = []

# First turn - include system prompt
response, history = client.chat_with_history(
    "What is a list comprehension?",
    history,
    system="You are a helpful Python tutor who explains concepts simply."
)

# Subsequent turns - no system prompt needed!
response, history = client.chat_with_history(
    "Show me an example.",
    history
)
```

### System Prompt Best Practices

**DO:**
```python
# ✅ Set once at conversation start
history = []
system = "You are a helpful coding assistant."
response, history = client.chat_with_history("Hi", history, system=system)
response, history = client.chat_with_history("Help me", history)
```

**DON'T:**
```python
# ❌ Repeat on every turn (creates duplicates)
system = "You are a helpful coding assistant."
response, history = client.chat_with_history("Hi", history, system=system)
response, history = client.chat_with_history("Help", history, system=system)  # Wrong!
```

### Default System Prompt

If no system prompt provided, SDK uses:
```python
"You are a helpful assistant with access to tools."
```

### Custom System Prompts

```python
# Coding assistant
system = """You are an expert Python developer.
- Provide clean, readable code
- Explain your reasoning
- Use type hints
- Follow PEP 8 style"""

# Quiz host
system = """You are a friendly quiz show host.
- Ask one question at a time
- Give encouraging feedback
- Keep score throughout
- Summarize results at the end"""

# Data analyst
system = """You are a data analyst assistant.
- Use tools to analyze data
- Provide clear visualizations
- Explain statistical findings
- Recommend actionable insights"""
```

---

## Advanced Context Management

### Manual History Modification

You can modify history for advanced use cases:

```python
# Build conversation
history = []
response, history = client.chat_with_history("My color is blue.", history)
response, history = client.chat_with_history("My food is pizza.", history)
response, history = client.chat_with_history("What are my favorites?", history)
# Response: "Color is blue, food is pizza"

# Remove food preference (keep only color)
modified_history = [history[0], history[1], history[-1]]

# Ask again with modified history
response, _ = client.chat_with_history(
    "What are my favorites?",
    modified_history
)
# Response: "Your favorite color is blue" (forgot about food)
```

### Sliding Window Context

Keep only the last N messages to manage token usage:

```python
MAX_HISTORY = 10  # Keep last 10 messages

history = []
for i in range(100):
    response, history = client.chat_with_history(f"Message {i}", history)

    # Trim history if too long
    if len(history) > MAX_HISTORY:
        # Keep system message (if present) + last N messages
        if history[0].role == "system":
            history = [history[0]] + history[-(MAX_HISTORY-1):]
        else:
            history = history[-MAX_HISTORY:]
```

### Summarization Pattern

For very long conversations, summarize and reset:

```python
def summarize_and_reset(client, history):
    """Summarize conversation and start fresh with summary."""

    # Request summary
    summary_prompt = "Please summarize our conversation so far."
    summary, _ = client.chat_with_history(summary_prompt, history)

    # Create new history with just the summary
    new_history = []
    new_history.append(create_chat_message(
        "system",
        f"Previous conversation summary: {summary}"
    ))

    return new_history

# Usage
if len(history) > 50:  # Too long
    history = summarize_and_reset(client, history)
```

### Removing Sensitive Information

```python
from local_llm_sdk.models import create_chat_message

def sanitize_history(history):
    """Remove messages containing sensitive information."""
    clean_history = []

    for msg in history:
        # Skip messages with sensitive keywords
        if any(word in msg.content.lower() for word in ["password", "secret", "api_key"]):
            # Replace with placeholder
            clean_msg = create_chat_message(
                msg.role,
                "[REDACTED: Sensitive information removed]"
            )
            clean_history.append(clean_msg)
        else:
            clean_history.append(msg)

    return clean_history

# Apply before using
history = sanitize_history(history)
```

---

## Conversation Persistence

### Saving Conversations

Save conversations to JSON for later use:

```python
import json
from pathlib import Path

def save_conversation(history, filename):
    """Save conversation history to JSON file."""
    # Convert ChatMessage objects to dictionaries
    data = [
        {
            "role": msg.role,
            "content": msg.content,
            "tool_calls": [
                {
                    "id": tc.id,
                    "type": tc.type,
                    "function": {
                        "name": tc.function.name,
                        "arguments": tc.function.arguments
                    }
                } for tc in (msg.tool_calls or [])
            ] if msg.tool_calls else None,
            "tool_call_id": msg.tool_call_id
        }
        for msg in history
    ]

    Path(filename).write_text(json.dumps(data, indent=2))
    print(f"Saved {len(history)} messages to {filename}")

# Usage
save_conversation(history, "conversation_2024-10-01.json")
```

### Loading Conversations

```python
from local_llm_sdk.models import ChatMessage, ToolCall, FunctionCall

def load_conversation(filename):
    """Load conversation history from JSON file."""
    data = json.loads(Path(filename).read_text())

    history = []
    for msg_data in data:
        # Reconstruct tool calls if present
        tool_calls = None
        if msg_data.get("tool_calls"):
            tool_calls = [
                ToolCall(
                    id=tc["id"],
                    type=tc["type"],
                    function=FunctionCall(
                        name=tc["function"]["name"],
                        arguments=tc["function"]["arguments"]
                    )
                )
                for tc in msg_data["tool_calls"]
            ]

        # Create message
        msg = ChatMessage(
            role=msg_data["role"],
            content=msg_data["content"],
            tool_calls=tool_calls,
            tool_call_id=msg_data.get("tool_call_id")
        )
        history.append(msg)

    print(f"Loaded {len(history)} messages from {filename}")
    return history

# Usage
history = load_conversation("conversation_2024-10-01.json")

# Continue the conversation
response, history = client.chat_with_history("Let's continue...", history)
```

### Database Persistence

For production applications, use a database:

```python
import sqlite3
from datetime import datetime

class ConversationDB:
    """Simple SQLite-based conversation storage."""

    def __init__(self, db_path="conversations.db"):
        self.conn = sqlite3.connect(db_path)
        self._create_tables()

    def _create_tables(self):
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS conversations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                role TEXT NOT NULL,
                content TEXT NOT NULL
            )
        """)
        self.conn.commit()

    def save_message(self, session_id, message):
        """Save a single message."""
        self.conn.execute(
            "INSERT INTO conversations (session_id, timestamp, role, content) VALUES (?, ?, ?, ?)",
            (session_id, datetime.now().isoformat(), message.role, message.content)
        )
        self.conn.commit()

    def load_session(self, session_id):
        """Load all messages for a session."""
        cursor = self.conn.execute(
            "SELECT role, content FROM conversations WHERE session_id = ? ORDER BY timestamp",
            (session_id,)
        )

        history = []
        for row in cursor:
            history.append(ChatMessage(role=row[0], content=row[1]))

        return history

# Usage
db = ConversationDB()
session_id = "user_alice_2024-10-01"

# Load existing conversation or start new
history = db.load_session(session_id)

# Chat
response, history = client.chat_with_history("Hello!", history)

# Save new messages
db.save_message(session_id, history[-2])  # User message
db.save_message(session_id, history[-1])  # Assistant response
```

---

## Context Window Management

### Understanding Token Limits

LLMs have context window limits (e.g., 4K, 8K, 32K tokens). Long conversations can exceed these limits.

**Symptoms:**
- Truncated responses
- Loss of early conversation context
- API errors (if server enforces strict limits)

### Estimating Token Count

Rough estimate: 1 token ≈ 4 characters for English text

```python
def estimate_tokens(history):
    """Estimate token count for conversation history."""
    total_chars = sum(len(msg.content or "") for msg in history)
    estimated_tokens = total_chars // 4
    return estimated_tokens

# Check before sending
tokens = estimate_tokens(history)
print(f"Estimated tokens: {tokens}")

if tokens > 3000:  # Approaching 4K limit
    print("Warning: Conversation is getting long!")
```

### Token Management Strategies

**1. Sliding Window (Recent Messages)**

```python
def apply_sliding_window(history, max_messages=20):
    """Keep only the last N messages."""
    if len(history) <= max_messages:
        return history

    # Preserve system message if present
    if history[0].role == "system":
        return [history[0]] + history[-(max_messages-1):]
    else:
        return history[-max_messages:]

# Apply before each turn
history = apply_sliding_window(history, max_messages=20)
response, history = client.chat_with_history(query, history)
```

**2. Importance-Based Filtering**

```python
def filter_important_messages(history, max_messages=30):
    """Keep system messages, recent messages, and important exchanges."""
    if len(history) <= max_messages:
        return history

    important = []

    # Always keep system messages
    for msg in history:
        if msg.role == "system":
            important.append(msg)

    # Keep messages with keywords
    keywords = ["error", "important", "remember", "note"]
    for msg in history:
        if any(kw in (msg.content or "").lower() for kw in keywords):
            important.append(msg)

    # Fill remaining slots with recent messages
    recent_needed = max_messages - len(important)
    recent = [msg for msg in history[-recent_needed:] if msg not in important]

    return important + recent
```

**3. Summarization + Fresh Context**

```python
def compress_history(client, history, target_size=10):
    """Summarize old messages and keep recent ones fresh."""
    if len(history) <= target_size:
        return history

    # Split into old and recent
    split_point = len(history) - target_size
    old_history = history[:split_point]
    recent_history = history[split_point:]

    # Summarize old portion
    summary_prompt = "Summarize the key points from our conversation so far."
    summary, _ = client.chat_with_history(summary_prompt, old_history)

    # Create compressed history
    compressed = []
    if history[0].role == "system":
        compressed.append(history[0])

    compressed.append(create_chat_message(
        "system",
        f"Summary of previous discussion: {summary}"
    ))
    compressed.extend(recent_history)

    return compressed
```

**4. Selective Removal**

```python
def remove_verbose_messages(history, max_length=500):
    """Remove overly verbose messages, keeping concise summaries."""
    filtered = []

    for msg in history:
        if msg.role == "system":
            filtered.append(msg)
        elif len(msg.content or "") > max_length:
            # Replace with truncated version
            truncated = create_chat_message(
                msg.role,
                msg.content[:max_length] + "... [truncated]"
            )
            filtered.append(truncated)
        else:
            filtered.append(msg)

    return filtered
```

### Dynamic Context Management

```python
class ManagedConversation:
    """Conversation with automatic context window management."""

    def __init__(self, client, max_tokens=3000, strategy="sliding_window"):
        self.client = client
        self.max_tokens = max_tokens
        self.strategy = strategy
        self.history = []

    def chat(self, query, **kwargs):
        """Chat with automatic context management."""
        # Check token count
        tokens = self._estimate_tokens()

        # Apply strategy if needed
        if tokens > self.max_tokens:
            if self.strategy == "sliding_window":
                self.history = self._apply_sliding_window()
            elif self.strategy == "summarize":
                self.history = self._summarize_and_compress()

        # Execute chat
        response, self.history = self.client.chat_with_history(
            query,
            self.history,
            **kwargs
        )

        return response

    def _estimate_tokens(self):
        total_chars = sum(len(msg.content or "") for msg in self.history)
        return total_chars // 4

    def _apply_sliding_window(self, keep=20):
        if self.history[0].role == "system":
            return [self.history[0]] + self.history[-(keep-1):]
        return self.history[-keep:]

    def _summarize_and_compress(self):
        # Implementation similar to compress_history above
        pass

# Usage
conv = ManagedConversation(client, max_tokens=2000, strategy="sliding_window")

for i in range(100):
    response = conv.chat(f"Message {i}")
    # Context automatically managed!
```

---

## Best Practices

### 1. Always Update History

```python
# ❌ WRONG: Not using returned history
history = []
response, history = client.chat_with_history("Hello", history)
response, _ = client.chat_with_history("What did I say?", history)  # Forgot to update!

# ✅ CORRECT: Always capture returned history
history = []
response, history = client.chat_with_history("Hello", history)
response, history = client.chat_with_history("What did I say?", history)
```

### 2. Use Context Appropriately

```python
# ❌ WRONG: Using history for independent tasks
history = []
response, history = client.chat_with_history("What is 2+2?", history)
response, history = client.chat_with_history("What is 5*5?", history)
# Wastes tokens on unrelated context

# ✅ CORRECT: Use simple chat for independent queries
response1 = client.chat("What is 2+2?")
response2 = client.chat("What is 5*5?")
```

### 3. Manage Token Budget

```python
# ✅ Monitor and manage conversation length
MAX_HISTORY = 30

response, history = client.chat_with_history(query, history)

if len(history) > MAX_HISTORY:
    history = history[-MAX_HISTORY:]  # Trim
```

### 4. Preserve System Messages

```python
# ✅ When trimming, always keep system messages
def safe_trim(history, max_size):
    if history[0].role == "system":
        return [history[0]] + history[-(max_size-1):]
    return history[-max_size:]
```

### 5. Handle Tool Messages

```python
# ✅ Tool messages are part of conversation flow
response = client.chat("Calculate 5!", use_tools=True)

# Get full conversation including tool interactions
for msg in client.last_conversation_additions:
    history.append(msg)
```

### 6. Use Context Managers for Sessions

```python
# ✅ Group related turns under single trace
with client.conversation("travel-planning"):
    history = []
    for query in travel_queries:
        response, history = client.chat_with_history(query, history)
```

### 7. Validate History Before Sending

```python
# ✅ Ensure history is well-formed
def validate_history(history):
    """Check history for common issues."""
    if not history:
        return True

    # Check alternating roles
    for i in range(1, len(history)):
        if history[i].role == history[i-1].role:
            if history[i].role not in ["system", "tool"]:
                print(f"Warning: Consecutive {history[i].role} messages at index {i}")

    # Check for content
    for i, msg in enumerate(history):
        if msg.role != "tool" and not msg.content:
            print(f"Warning: Empty content at index {i}")

    return True

validate_history(history)
```

---

## Examples

### Example 1: Technical Q&A Session

```python
from local_llm_sdk import LocalLLMClient

client = LocalLLMClient()
history = []

# Set technical context
system_prompt = """You are a senior software engineer specializing in Python.
Provide accurate, practical advice with code examples."""

response, history = client.chat_with_history(
    "How do I read a CSV file?",
    history,
    system=system_prompt
)
print(response)

# Follow-up questions
response, history = client.chat_with_history(
    "What about handling missing values?",
    history
)
print(response)

response, history = client.chat_with_history(
    "Show me how to export to JSON.",
    history
)
print(response)

# Final summary
response, history = client.chat_with_history(
    "Summarize the complete workflow.",
    history
)
print(response)
```

### Example 2: Interactive Story Game

```python
history = []
system_prompt = """You are a creative storyteller running an interactive adventure.
- Present choices after each scene
- Remember player decisions
- Build a coherent narrative"""

print("🎮 INTERACTIVE ADVENTURE\n")

response, history = client.chat_with_history(
    "Start an adventure in a magical forest.",
    history,
    system=system_prompt
)
print(response)

# Player makes choices
choices = [
    "I choose to investigate the mysterious light.",
    "I speak to the creature.",
    "I use the magic spell I learned."
]

for choice in choices:
    response, history = client.chat_with_history(choice, history)
    print(f"\nYou: {choice}")
    print(f"\n{response}\n")
```

### Example 3: Code Review Assistant

```python
history = []
system_prompt = """You are a code review assistant.
- Point out bugs and edge cases
- Suggest improvements
- Explain your reasoning"""

# Submit code for review
code = '''
def calculate_average(numbers):
    total = sum(numbers)
    return total / len(numbers)
'''

response, history = client.chat_with_history(
    f"Review this code:\n```python\n{code}\n```",
    history,
    system=system_prompt
)
print("Initial review:\n", response)

# Ask follow-up questions
response, history = client.chat_with_history(
    "How would you handle empty lists?",
    history
)
print("\nEdge case handling:\n", response)

response, history = client.chat_with_history(
    "Show me the improved version.",
    history
)
print("\nImproved code:\n", response)
```

### Example 4: Multi-Session Conversation

```python
import json
from pathlib import Path

class SessionManager:
    """Manage persistent conversation sessions."""

    def __init__(self, client, session_dir="sessions"):
        self.client = client
        self.session_dir = Path(session_dir)
        self.session_dir.mkdir(exist_ok=True)

    def load_or_create(self, session_id):
        """Load existing session or create new one."""
        session_file = self.session_dir / f"{session_id}.json"

        if session_file.exists():
            history = load_conversation(str(session_file))
            print(f"Loaded session: {len(history)} messages")
        else:
            history = []
            print(f"Created new session: {session_id}")

        return history

    def save(self, session_id, history):
        """Save session to disk."""
        session_file = self.session_dir / f"{session_id}.json"
        save_conversation(history, str(session_file))

    def chat(self, session_id, query, **kwargs):
        """Chat within a session with auto-save."""
        history = self.load_or_create(session_id)
        response, history = self.client.chat_with_history(query, history, **kwargs)
        self.save(session_id, history)
        return response

# Usage
manager = SessionManager(client)

# Day 1
response = manager.chat("alice_2024", "My favorite color is blue.")
print(response)

# Day 2 (later session)
response = manager.chat("alice_2024", "What's my favorite color?")
print(response)  # "Your favorite color is blue!"
```

### Example 5: Conversation Analytics

```python
def analyze_conversation(history):
    """Analyze conversation patterns and statistics."""

    stats = {
        "total_messages": len(history),
        "user_messages": sum(1 for m in history if m.role == "user"),
        "assistant_messages": sum(1 for m in history if m.role == "assistant"),
        "tool_messages": sum(1 for m in history if m.role == "tool"),
        "total_chars": sum(len(m.content or "") for m in history),
        "avg_user_length": 0,
        "avg_assistant_length": 0,
        "topics": []
    }

    # Calculate averages
    user_msgs = [m for m in history if m.role == "user"]
    asst_msgs = [m for m in history if m.role == "assistant"]

    if user_msgs:
        stats["avg_user_length"] = sum(len(m.content) for m in user_msgs) // len(user_msgs)
    if asst_msgs:
        stats["avg_assistant_length"] = sum(len(m.content) for m in asst_msgs) // len(asst_msgs)

    return stats

# Usage
stats = analyze_conversation(history)
print(json.dumps(stats, indent=2))
```

---

## Summary

**Key Takeaways:**

1. **Use `chat_with_history()`** for multi-turn conversations
2. **Always update history** with returned value
3. **System prompts** go in the first turn only
4. **Manage token budget** with sliding windows or summarization
5. **Persist conversations** with JSON or databases
6. **Monitor context size** to avoid hitting limits
7. **Use context appropriately** - not all tasks need history

**When to Use History:**
- Building on previous responses
- Refining requirements over multiple turns
- Natural conversations with context
- Working on a single topic with follow-ups

**When to Skip History:**
- Independent factual queries
- Batch operations on unrelated items
- Simple calculations
- Testing different prompts

**Next Steps:**

- Explore notebook `03-conversation-history.ipynb` for hands-on practice
- Combine with tools (see `tool-calling.md`) for powerful agents
- Use MLflow tracing (see `mlflow-observability.md`) to debug conversations
- Build production systems with proper persistence and context management
