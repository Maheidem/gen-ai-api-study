# Local LLM SDK - Architecture Overview

## Table of Contents
1. [Introduction](#introduction)
2. [System Architecture](#system-architecture)
3. [Layered Architecture](#layered-architecture)
4. [Key Architectural Patterns](#key-architectural-patterns)
5. [Component Interactions](#component-interactions)
6. [Design Principles](#design-principles)
7. [Extension Points](#extension-points)
8. [Performance Considerations](#performance-considerations)
9. [Critical Implementation Details](#critical-implementation-details)

---

## Introduction

Local LLM SDK is a **type-safe Python SDK** for interacting with local LLM APIs that implement the OpenAI specification. It provides a clean, extensible interface for working with LM Studio, Ollama, and other OpenAI-compatible servers.

**Core Technologies:**
- Python 3.12.11 with Pydantic v2 for type safety
- OpenAI API compatibility (LM Studio, Ollama, LocalAI)
- MLflow integration (optional) for tracing and observability
- Comprehensive testing (213+ unit tests + behavioral test suite)

**Primary Use Cases:**
- Local LLM interaction with type safety
- Multi-step task execution via agents
- Tool-augmented chat completions
- Observability and debugging with MLflow tracing

---

## System Architecture

### High-Level Architecture Diagram

```
┌──────────────────────────────────────────────────────────────────────────┐
│                         APPLICATION LAYER                                │
│  ┌─────────────────┐  ┌──────────────────┐  ┌────────────────────────┐ │
│  │ Jupyter         │  │ Python Scripts   │  │ FastAPI/Flask Apps     │ │
│  │ Notebooks       │  │ CLI Tools        │  │ Web Services           │ │
│  └────────┬────────┘  └────────┬─────────┘  └──────────┬─────────────┘ │
└───────────┼──────────────────────┼──────────────────────┼────────────────┘
            │                      │                      │
            └──────────────────────┴──────────────────────┘
                                   │
            ┌──────────────────────▼──────────────────────┐
            │        LOCAL LLM SDK (Core Package)         │
            │                                              │
            │  ┌────────────────────────────────────────┐ │
            │  │  Layer 5: Configuration (config.py)    │ │
            │  │  • Environment variables               │ │
            │  │  • Default values                      │ │
            │  │  • Type-safe config loading            │ │
            │  └────────────────────────────────────────┘ │
            │                    ▲                         │
            │  ┌────────────────┴─────────────────────┐  │
            │  │  Layer 4: Agent Framework (agents/)  │  │
            │  │  • BaseAgent (abstract + tracing)    │  │
            │  │  • ReACT (reasoning + acting)        │  │
            │  │  • AgentResult, AgentStatus models   │  │
            │  └────────────────┬─────────────────────┘  │
            │                    ▲                         │
            │  ┌────────────────┴─────────────────────┐  │
            │  │  Layer 3: Tool System (tools/)       │  │
            │  │  • ToolRegistry (schema + execution) │  │
            │  │  • @tool decorator                   │  │
            │  │  • Built-in tools (Python, file ops) │  │
            │  └────────────────┬─────────────────────┘  │
            │                    ▲                         │
            │  ┌────────────────┴─────────────────────┐  │
            │  │  Layer 2: Type System (models.py)    │  │
            │  │  • Pydantic v2 models                │  │
            │  │  • OpenAI API spec compliance        │  │
            │  │  • ChatMessage, Tool, ToolCall       │  │
            │  └────────────────┬─────────────────────┘  │
            │                    ▲                         │
            │  ┌────────────────┴─────────────────────┐  │
            │  │  Layer 1: Core Client (client.py)    │  │
            │  │  • LocalLLMClient (main entry point) │  │
            │  │  • Conversation state management     │  │
            │  │  • Automatic tool handling           │  │
            │  │  • MLflow tracing integration        │  │
            │  └────────────────┬─────────────────────┘  │
            └───────────────────┼──────────────────────┘
                                │
            ┌───────────────────▼──────────────────────┐
            │    EXTERNAL INTEGRATIONS (Optional)      │
            │  ┌──────────────┐    ┌────────────────┐ │
            │  │   MLflow     │    │  Custom Tools  │ │
            │  │   Tracing    │    │  & Extensions  │ │
            │  └──────────────┘    └────────────────┘ │
            └──────────────────────────────────────────┘
                                │
            ┌───────────────────▼──────────────────────┐
            │         LLM INFRASTRUCTURE                │
            │  ┌──────────┐ ┌──────────┐ ┌──────────┐ │
            │  │LM Studio │ │  Ollama  │ │ LocalAI  │ │
            │  │(Primary) │ │          │ │          │ │
            │  └──────────┘ └──────────┘ └──────────┘ │
            │   OpenAI-Compatible API (HTTP/REST)      │
            └──────────────────────────────────────────┘
```

### Data Flow

```
User Request
    │
    ▼
┌──────────────────────────────────────────────────────────────┐
│ 1. Application calls client.chat("task", use_tools=True)    │
└──────────────────────┬───────────────────────────────────────┘
                       │
                       ▼
┌──────────────────────────────────────────────────────────────┐
│ 2. LocalLLMClient prepares request                           │
│    • Builds ChatMessage objects (Pydantic validation)        │
│    • Adds conversation history                               │
│    • Includes tool schemas (if use_tools=True)               │
└──────────────────────┬───────────────────────────────────────┘
                       │
                       ▼
┌──────────────────────────────────────────────────────────────┐
│ 3. HTTP Request to LLM (LM Studio/Ollama)                    │
│    • POST /v1/chat/completions                               │
│    • Request validated via Pydantic models                   │
│    • Timeout: 300s (configurable)                            │
└──────────────────────┬───────────────────────────────────────┘
                       │
                       ▼
┌──────────────────────────────────────────────────────────────┐
│ 4. LLM Response                                              │
│    • ChatCompletion object (Pydantic parsed)                 │
│    • May contain tool_calls                                  │
└──────────────────────┬───────────────────────────────────────┘
                       │
                       ▼
                  ┌────┴────┐
                  │ Has     │
                  │ Tools?  │
                  └────┬────┘
                       │
        ┌──────────────┴──────────────┐
        │ NO                           │ YES
        │                              │
        ▼                              ▼
┌──────────────────┐    ┌──────────────────────────────────────┐
│ 6a. Return       │    │ 5. ToolRegistry executes tools       │
│     Response     │    │    • Python sandbox execution        │
│                  │    │    • File operations, math, etc.     │
│                  │    │    • Returns JSON results            │
│                  │    └──────────────────┬───────────────────┘
│                  │                       │
│                  │                       ▼
│                  │    ┌──────────────────────────────────────┐
│                  │    │ 6b. Add tool results to conversation │
│                  │    │     • Tool message role              │
│                  │    │     • Update conversation state      │
│                  │    │     • Track in last_conversation_... │
│                  │    └──────────────────┬───────────────────┘
│                  │                       │
│                  │                       ▼
│                  │    ┌──────────────────────────────────────┐
│                  │    │ 7. Send back to LLM (iterate)        │
│                  │    │    • Repeat steps 3-6 until done     │
│                  │    └──────────────────┬───────────────────┘
│                  │                       │
└──────────────────┴───────────────────────┘
                       │
                       ▼
┌──────────────────────────────────────────────────────────────┐
│ 8. Final Response to User                                    │
│    • Clean text response                                     │
│    • Full conversation preserved                             │
│    • MLflow traces recorded (if enabled)                     │
└──────────────────────────────────────────────────────────────┘
```

---

## Layered Architecture

The SDK implements a **strict 5-layer architecture** with unidirectional dependencies (higher layers depend on lower layers, never vice versa).

### Layer 1: Core Client (`client.py`)

**Responsibility:** Primary interface for all LLM interactions

**Key Components:**
- `LocalLLMClient` class - Main entry point
- Chat methods (`chat()`, `chat_streaming()`)
- Conversation state management
- Automatic tool call handling
- MLflow tracing integration

**Key Methods:**
```python
def chat(
    self,
    user_message: str,
    use_tools: bool = False,
    tool_choice: str = "auto"
) -> str:
    """Send message, handle tools automatically, return response."""

def _handle_tool_calls(
    self,
    tool_calls: List[ToolCall]
) -> tuple[str, List[ChatMessage]]:
    """Execute tools and return (content, new_messages)."""
```

**Design Highlights:**
- 300s default timeout (local models are slower)
- Preserves complete conversation history
- Tracks `last_conversation_additions` for tracing
- Graceful MLflow degradation if not installed

### Layer 2: Type System (`models.py`)

**Responsibility:** Type-safe data models for all API interactions

**Key Components:**
- Pydantic v2 models following OpenAI specification
- Strict validation for request/response types
- JSON Schema generation for API compliance

**Core Models:**
```python
class ChatMessage(BaseModel):
    role: Literal["system", "user", "assistant", "tool"]
    content: Optional[str]
    tool_calls: Optional[List[ToolCall]]
    tool_call_id: Optional[str]

class Tool(BaseModel):
    type: Literal["function"] = "function"
    function: FunctionDefinition

class ChatCompletion(BaseModel):
    id: str
    choices: List[Choice]
    usage: Usage
    model: str
```

**Benefits:**
- Compile-time type checking
- Runtime validation (catches API contract violations)
- Auto-generated JSON schemas
- Clear API documentation via types

### Layer 3: Tool System (`tools/`)

**Responsibility:** Tool registration, schema generation, and execution

**Key Components:**
- `ToolRegistry` - Central registry with schema generation
- `@tool` decorator - Simple registration pattern
- Built-in tools (Python exec, file ops, math, text processing)

**Architecture:**
```python
class ToolRegistry:
    def register_function(
        self,
        name: str,
        func: Callable,
        description: str
    ) -> None:
        """Register function with automatic schema generation."""

    def get_tool_schemas(self) -> List[Tool]:
        """Generate OpenAI-compatible tool schemas from type hints."""

    def execute_tool(
        self,
        name: str,
        arguments: dict
    ) -> dict:
        """Execute tool with JSON-serializable result."""
```

**Schema Generation Process:**
1. Extract function signature via `inspect.signature()`
2. Convert Python types to JSON Schema types
3. Identify required vs. optional parameters
4. Generate OpenAI-compatible `Tool` object
5. Cache schemas for performance

**Built-in Tools:**
- `python_executor` - Execute Python code in sandbox
- `file_reader` - Read file contents
- `file_writer` - Write files
- `math_calculator` - Mathematical operations
- `text_transformer` - Text case conversion
- `char_counter` - Character/word counting

### Layer 4: Agent Framework (`agents/`)

**Responsibility:** Multi-step task execution with reasoning

**Key Components:**
- `BaseAgent` - Abstract base with automatic tracing
- `ReACT` - Reasoning + Acting pattern implementation
- `AgentResult`, `AgentStatus` - Result types

**Agent Pattern:**
```python
class BaseAgent(ABC):
    """
    Base class with automatic MLflow tracing.

    Subclasses implement _execute() with task-specific logic.
    """
    def run(self, task: str, **kwargs) -> AgentResult:
        """Wraps _execute() with tracing and error handling."""
        with mlflow.start_span(name=f"{self.__class__.__name__}.run"):
            return self._execute(task, **kwargs)

    @abstractmethod
    def _execute(self, task: str, **kwargs) -> AgentResult:
        """Implement task execution logic."""
        pass
```

**ReACT Agent Loop:**
```python
def _execute(self, task: str, max_iterations: int = 15) -> AgentResult:
    """
    Reasoning + Acting pattern:
    1. Reason about current state
    2. Act (call tools if needed)
    3. Observe results
    4. Repeat until task complete or max iterations
    """
    for iteration in range(max_iterations):
        response = self.client.chat(
            self._build_prompt(task),
            use_tools=True
        )

        if self._should_stop(response):
            return self._build_result(response)

        # Continue iteration

    return self._build_timeout_result()
```

**Key Features:**
- Optimized system prompts for clear instructions
- Iteration tracking and metadata collection
- Stop condition detection ("TASK_COMPLETE")
- Final response extraction and cleanup
- Hierarchical MLflow tracing (agent → iterations → tools)

### Layer 5: Configuration (`config.py`)

**Responsibility:** Environment-based configuration management

**Key Components:**
- Environment variable support
- Default configuration values
- Type-safe config loading

**Configuration Options:**
```python
class Config:
    LLM_BASE_URL: str = "http://localhost:1234/v1"
    LLM_MODEL: str = "auto"  # Auto-select from server
    LLM_TIMEOUT: int = 300   # 5 minutes (local models slower)
    LLM_DEBUG: bool = False

    @classmethod
    def from_env(cls) -> "Config":
        """Load config from environment variables."""
        return cls(
            LLM_BASE_URL=os.getenv("LLM_BASE_URL", cls.LLM_BASE_URL),
            LLM_MODEL=os.getenv("LLM_MODEL", cls.LLM_MODEL),
            LLM_TIMEOUT=int(os.getenv("LLM_TIMEOUT", cls.LLM_TIMEOUT)),
            LLM_DEBUG=os.getenv("LLM_DEBUG", "false").lower() == "true"
        )
```

**Design Principles:**
- Environment variables override defaults
- Sensible defaults for local development
- Type-safe value parsing
- Clear error messages for invalid config

---

## Key Architectural Patterns

### 1. Conversation State Management

**Problem:** Tool results must persist in conversation for LLM context

**Solution:** Complete message tracking with explicit state management

```python
class LocalLLMClient:
    def __init__(self):
        self.conversation: List[ChatMessage] = []
        self.last_conversation_additions: List[ChatMessage] = []

    def _handle_tool_calls(self, tool_calls) -> tuple[str, List[ChatMessage]]:
        """
        Returns: (final_content, new_messages)

        new_messages includes:
        - Assistant message with tool_calls
        - Tool result messages (one per tool)

        This ensures tool context is preserved.
        """
        new_messages = []

        # Add assistant message
        assistant_msg = ChatMessage(role="assistant", tool_calls=tool_calls)
        new_messages.append(assistant_msg)

        # Execute tools and add results
        for tool_call in tool_calls:
            result = self.tool_registry.execute_tool(
                tool_call.function.name,
                json.loads(tool_call.function.arguments)
            )
            tool_msg = ChatMessage(
                role="tool",
                content=json.dumps(result),
                tool_call_id=tool_call.id
            )
            new_messages.append(tool_msg)

        # Update state
        self.conversation.extend(new_messages)
        self.last_conversation_additions = new_messages

        # Get final response
        final_response = self._chat_completion(self.conversation)

        return (final_response.content, new_messages)
```

**Critical Details:**
- `_handle_tool_calls()` returns tuple: `(content, new_messages)`
- New messages include both assistant and tool messages
- `last_conversation_additions` enables MLflow tracing
- Fixes "empty response bug" where tool results disappeared

### 2. Tool Execution Flow

**Pattern:** Automatic iterative tool execution until completion

```
User: "Calculate 5 factorial and convert to uppercase"
  │
  ▼
Client.chat(use_tools=True)
  │
  ├─→ Add tool schemas to request
  │
  ├─→ LLM responds with tool_calls:
  │   [{"name": "math_calculator", "arguments": {"expression": "5!"}}]
  │
  ├─→ ToolRegistry executes: math_calculator("5!")
  │   Returns: {"result": 120}
  │
  ├─→ Add tool result as message:
  │   ChatMessage(role="tool", content='{"result": 120}', ...)
  │
  ├─→ Send back to LLM with updated conversation
  │
  ├─→ LLM responds with tool_calls:
  │   [{"name": "text_transformer", "arguments": {"text": "120", "operation": "upper"}}]
  │
  ├─→ Execute: text_transformer("120", "upper")
  │   Returns: {"result": "120"}
  │
  ├─→ Add tool result and send to LLM
  │
  └─→ LLM responds with final answer (no tool_calls)
      Returns: "The factorial of 5 is 120, which in uppercase is '120'."
```

**Key Features:**
- Automatic iteration (no manual loop required)
- Tool results preserved in conversation
- Stops when LLM returns no tool_calls
- Supports multiple tools per iteration (parallel execution)

### 3. Agent Pattern

**Pattern:** Template Method + Strategy for multi-step tasks

```python
class BaseAgent:
    """
    Template Method pattern with automatic tracing.

    run() provides structure, _execute() is customizable.
    """
    def run(self, task: str, **kwargs) -> AgentResult:
        with mlflow.start_span(name=f"{self.__class__.__name__}.run") as span:
            try:
                result = self._execute(task, **kwargs)
                span.set_attribute("status", result.status)
                span.set_attribute("iterations", result.iterations)
                return result
            except Exception as e:
                span.set_attribute("error", str(e))
                return AgentResult(
                    success=False,
                    error=str(e),
                    status=AgentStatus.FAILED
                )

    @abstractmethod
    def _execute(self, task: str, **kwargs) -> AgentResult:
        """Subclass implements specific execution strategy."""
        pass

class ReACT(BaseAgent):
    """
    Strategy: Reasoning + Acting in iterative loop.
    """
    def _execute(self, task: str, max_iterations: int = 15) -> AgentResult:
        # Implementation of ReACT pattern
        pass
```

**Benefits:**
- Consistent interface across agents
- Automatic tracing for all agents
- Easy to add new agent strategies
- Clear separation of concerns

### 4. MLflow Integration (Optional)

**Pattern:** Graceful degradation with hierarchical tracing

```python
# Graceful import (falls back if MLflow not installed)
try:
    import mlflow
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False

    # No-op decorator
    class mlflow:
        @staticmethod
        def trace(func):
            return func

# Hierarchical tracing structure
Agent.run()                        # Span: "ReACT.run"
  ├─ Iteration 1                   # Span: "iteration_1"
  │   ├─ tool: math_calculator     # Span: "tool.math_calculator"
  │   └─ tool: text_transformer    # Span: "tool.text_transformer"
  ├─ Iteration 2                   # Span: "iteration_2"
  │   └─ tool: char_counter        # Span: "tool.char_counter"
  └─ Final response                # Logged as attribute
```

**Implementation:**
```python
@mlflow.trace(name="ReACT._execute")
def _execute(self, task: str, max_iterations: int = 15) -> AgentResult:
    for i in range(max_iterations):
        with mlflow.start_span(name=f"iteration_{i+1}") as span:
            response = self.client.chat(task, use_tools=True)

            # Track tool calls
            for tool_call in response.tool_calls or []:
                with mlflow.start_span(name=f"tool.{tool_call.name}"):
                    result = self.registry.execute_tool(...)

            span.set_attribute("tool_calls_count", len(tool_calls))
```

**Key Features:**
- Optional dependency (no hard requirement)
- Automatic parent-child span relationships
- Rich metadata (iterations, tool calls, errors)
- Debugging-friendly hierarchical view

---

## Component Interactions

### Client ↔ Type System

```python
# Client uses Pydantic models for type safety
def chat(self, user_message: str) -> str:
    # 1. Create message object (Pydantic validation)
    msg = ChatMessage(role="user", content=user_message)
    self.conversation.append(msg)

    # 2. Build request (Pydantic validation)
    request = ChatCompletionRequest(
        model=self.model,
        messages=self.conversation,
        temperature=0.7
    )

    # 3. Send HTTP request
    response = requests.post(
        f"{self.base_url}/chat/completions",
        json=request.model_dump(exclude_none=True),
        timeout=self.timeout
    )

    # 4. Parse response (Pydantic validation)
    completion = ChatCompletion.model_validate(response.json())

    return completion.choices[0].message.content
```

**Benefits:**
- Compile-time type checking
- Runtime validation (catches API violations early)
- Clear error messages (Pydantic ValidationError)
- Auto-generated JSON serialization

### Client ↔ Tool System

```python
# Client delegates tool execution to ToolRegistry
def _handle_tool_calls(self, tool_calls: List[ToolCall]) -> tuple[str, List[ChatMessage]]:
    new_messages = []

    for tool_call in tool_calls:
        # ToolRegistry handles execution
        result = self.tool_registry.execute_tool(
            tool_call.function.name,
            json.loads(tool_call.function.arguments)
        )

        # Client handles message creation
        tool_msg = ChatMessage(
            role="tool",
            content=json.dumps(result),
            tool_call_id=tool_call.id
        )
        new_messages.append(tool_msg)

    return (final_content, new_messages)
```

**Separation of Concerns:**
- **Client**: Message flow, conversation state, API communication
- **ToolRegistry**: Tool registration, schema generation, execution

### Agent ↔ Client

```python
# Agent uses Client for LLM interactions
class ReACT(BaseAgent):
    def __init__(self, client: LocalLLMClient):
        self.client = client  # Dependency injection

    def _execute(self, task: str, max_iterations: int = 15) -> AgentResult:
        # Agent focuses on orchestration
        for i in range(max_iterations):
            # Client handles LLM communication
            response = self.client.chat(
                self._build_prompt(task, i),
                use_tools=True
            )

            # Agent handles iteration logic
            if self._should_stop(response):
                return self._build_result(response)

        return self._build_timeout_result()
```

**Separation of Concerns:**
- **Agent**: Task orchestration, iteration logic, stopping conditions
- **Client**: LLM communication, tool handling, state management

### Tool System ↔ Type System

```python
# ToolRegistry generates Pydantic-compatible schemas
class ToolRegistry:
    def get_tool_schemas(self) -> List[Tool]:
        """Generate Tool objects (Pydantic models) from registered functions."""
        schemas = []

        for name, (func, desc) in self._tools.items():
            # Extract signature
            sig = inspect.signature(func)

            # Build JSON Schema
            parameters = {
                "type": "object",
                "properties": {},
                "required": []
            }

            for param_name, param in sig.parameters.items():
                # Convert Python type to JSON Schema
                param_type = self._python_type_to_json_schema(param.annotation)
                parameters["properties"][param_name] = {
                    "type": param_type,
                    "description": f"Parameter: {param_name}"
                }

                if param.default == inspect.Parameter.empty:
                    parameters["required"].append(param_name)

            # Create Pydantic Tool model
            tool = Tool(
                type="function",
                function=FunctionDefinition(
                    name=name,
                    description=desc,
                    parameters=parameters
                )
            )
            schemas.append(tool)

        return schemas
```

**Integration:**
- ToolRegistry generates schemas compatible with Pydantic models
- Type safety ensures LLM receives valid tool specifications
- Automatic validation of tool arguments

---

## Design Principles

### 1. Separation of Concerns

**Each layer has a single, well-defined responsibility:**

| Layer | Responsibility | Does NOT Handle |
|-------|---------------|-----------------|
| Config | Environment variables, defaults | Business logic, validation |
| Type System | Data models, validation | API communication, tool execution |
| Tool System | Tool registration, execution | LLM communication, conversation state |
| Core Client | API communication, state | Task orchestration, multi-step logic |
| Agent Framework | Multi-step orchestration | Direct API calls, tool execution |

**Benefits:**
- Easy to test (mock dependencies at layer boundaries)
- Easy to extend (add new agents without changing client)
- Clear interfaces (each layer has well-defined API)

### 2. Dependency Inversion

**Higher layers depend on abstractions, not concrete implementations:**

```python
# Good: Agent depends on client interface
class ReACT(BaseAgent):
    def __init__(self, client: LocalLLMClient):
        self.client = client  # Interface, not implementation

# Good: Client depends on tool registry interface
class LocalLLMClient:
    def __init__(self, tool_registry: ToolRegistry):
        self.tool_registry = tool_registry

# Testing: Easy to inject mocks
mock_client = Mock(spec=LocalLLMClient)
agent = ReACT(mock_client)
```

**Benefits:**
- Testability (inject mocks at boundaries)
- Flexibility (swap implementations without changing code)
- Maintainability (changes isolated to single layer)

### 3. Type Safety First

**Pydantic models enforce correctness at compile-time and runtime:**

```python
# Compile-time: IDE catches type errors
client = LocalLLMClient(base_url="http://localhost:1234")
response: str = client.chat("Hello")  # Type-safe

# Runtime: Pydantic validates API responses
completion = ChatCompletion.model_validate(response_json)
# Raises ValidationError if response doesn't match OpenAI spec
```

**Benefits:**
- Catch errors early (before deployment)
- Self-documenting code (types are documentation)
- Safe refactoring (type checker catches breakages)

### 4. Conversation State as First-Class Citizen

**All messages preserved for context and debugging:**

```python
# Bad: Losing context
def chat(self, message: str) -> str:
    response = api_call([ChatMessage(role="user", content=message)])
    return response.content  # No history!

# Good: Preserving context
def chat(self, message: str) -> str:
    self.conversation.append(ChatMessage(role="user", content=message))
    response = api_call(self.conversation)
    self.conversation.append(response.message)
    self.last_conversation_additions = [response.message]
    return response.content
```

**Benefits:**
- LLM has full context (better responses)
- Debugging (inspect full conversation)
- Tracing (MLflow can see message flow)
- Reproducibility (can replay conversations)

### 5. Fail Fast, Fail Clearly

**Errors should be caught early with clear messages:**

```python
# Tool execution errors return error dict (don't raise)
def execute_tool(self, name: str, arguments: dict) -> dict:
    if name not in self._tools:
        return {
            "error": f"Tool '{name}' not found",
            "available_tools": list(self._tools.keys())
        }

    try:
        result = self._tools[name](**arguments)
        return {"result": result}
    except Exception as e:
        return {
            "error": str(e),
            "tool": name,
            "arguments": arguments
        }

# Pydantic validation provides detailed errors
try:
    msg = ChatMessage(role="invalid_role", content="Hello")
except ValidationError as e:
    print(e.json())
    # {
    #   "loc": ["role"],
    #   "msg": "Input should be 'system', 'user', 'assistant', or 'tool'",
    #   "type": "literal_error"
    # }
```

**Benefits:**
- Faster debugging (errors point to exact issue)
- Better UX (users get actionable error messages)
- Safer code (errors don't cascade)

### 6. Graceful Degradation

**Optional features fail gracefully:**

```python
# MLflow is optional
try:
    import mlflow
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False

    # Provide no-op alternatives
    class mlflow:
        @staticmethod
        def trace(func):
            return func

# Usage works regardless
@mlflow.trace(name="agent.run")  # No-op if MLflow unavailable
def run(self, task: str) -> AgentResult:
    # Works with or without MLflow
    pass
```

**Benefits:**
- Simpler setup (MLflow not required for basic usage)
- Production-ready (works in constrained environments)
- Progressive enhancement (add features as needed)

---

## Extension Points

### 1. Custom Tools

**Adding domain-specific tools is simple:**

```python
from local_llm_sdk import tool

@tool("Fetch stock price for a given ticker symbol")
def get_stock_price(ticker: str) -> dict:
    """
    Fetch current stock price from API.

    Args:
        ticker: Stock ticker symbol (e.g., 'AAPL', 'GOOGL')

    Returns:
        Dict with 'price', 'currency', 'timestamp'
    """
    # Implementation
    response = requests.get(f"https://api.example.com/stocks/{ticker}")
    data = response.json()

    return {
        "price": data["current_price"],
        "currency": "USD",
        "timestamp": data["timestamp"]
    }

# Register with client
client.register_tool("Stock price lookup")(get_stock_price)

# Use in conversation
response = client.chat("What's the price of AAPL?", use_tools=True)
```

**Requirements for Custom Tools:**
- Function must have type hints (for schema generation)
- Must return JSON-serializable dict
- Should handle errors gracefully (return error dict)
- Docstring becomes tool description for LLM

### 2. Custom Agents

**Create new agent strategies by extending BaseAgent:**

```python
from local_llm_sdk.agents import BaseAgent, AgentResult, AgentStatus

class PlanExecuteAgent(BaseAgent):
    """
    Plan-Execute pattern:
    1. Generate complete plan upfront
    2. Execute each step sequentially
    3. Adapt plan if steps fail
    """

    def _execute(self, task: str, **kwargs) -> AgentResult:
        # Step 1: Generate plan
        plan_prompt = f"Generate step-by-step plan for: {task}"
        plan_response = self.client.chat(plan_prompt, use_tools=False)
        plan_steps = self._parse_plan(plan_response)

        # Step 2: Execute each step
        results = []
        for i, step in enumerate(plan_steps):
            with mlflow.start_span(name=f"step_{i+1}"):
                result = self.client.chat(step, use_tools=True)
                results.append(result)

                # Adapt if step fails
                if self._is_failure(result):
                    return self._handle_failure(step, result)

        # Step 3: Synthesize final answer
        final_response = self._synthesize_results(results)

        return AgentResult(
            success=True,
            final_response=final_response,
            status=AgentStatus.COMPLETED,
            iterations=len(plan_steps),
            conversation=self.client.conversation,
            metadata={"plan": plan_steps, "results": results}
        )
```

**BaseAgent provides:**
- Automatic MLflow tracing
- Consistent error handling
- Standard result format
- Client integration

### 3. Custom Configuration Sources

**Extend Config class for different sources:**

```python
from local_llm_sdk.config import Config
import yaml

class YAMLConfig(Config):
    """Load configuration from YAML file."""

    @classmethod
    def from_yaml(cls, path: str) -> "YAMLConfig":
        with open(path) as f:
            data = yaml.safe_load(f)

        return cls(
            LLM_BASE_URL=data.get("llm_base_url", cls.LLM_BASE_URL),
            LLM_MODEL=data.get("llm_model", cls.LLM_MODEL),
            LLM_TIMEOUT=data.get("llm_timeout", cls.LLM_TIMEOUT),
            LLM_DEBUG=data.get("llm_debug", cls.LLM_DEBUG)
        )

# Usage
config = YAMLConfig.from_yaml("config.yaml")
client = LocalLLMClient(
    base_url=config.LLM_BASE_URL,
    model=config.LLM_MODEL,
    timeout=config.LLM_TIMEOUT
)
```

### 4. Custom Pydantic Models

**Extend type system for custom API features:**

```python
from local_llm_sdk.models import ChatMessage
from pydantic import BaseModel, Field

class ExtendedChatMessage(ChatMessage):
    """Add custom metadata to messages."""

    metadata: dict = Field(default_factory=dict)
    tags: List[str] = Field(default_factory=list)
    importance: Literal["low", "medium", "high"] = "medium"

# Use in custom client
class CustomClient(LocalLLMClient):
    def chat_with_metadata(
        self,
        message: str,
        tags: List[str] = None,
        importance: str = "medium"
    ) -> str:
        msg = ExtendedChatMessage(
            role="user",
            content=message,
            tags=tags or [],
            importance=importance
        )
        self.conversation.append(msg)
        # Continue with normal flow...
```

### 5. Middleware / Interceptors

**Add cross-cutting concerns via middleware pattern:**

```python
class LoggingMiddleware:
    """Log all API requests/responses."""

    def __init__(self, client: LocalLLMClient):
        self.client = client
        self._original_chat = client.chat
        client.chat = self._wrapped_chat

    def _wrapped_chat(self, message: str, **kwargs) -> str:
        logger.info(f"Request: {message}")

        start_time = time.time()
        response = self._original_chat(message, **kwargs)
        duration = time.time() - start_time

        logger.info(f"Response: {response} (took {duration:.2f}s)")
        return response

# Usage
client = LocalLLMClient(...)
LoggingMiddleware(client)

# All chat calls now logged
response = client.chat("Hello")
```

**Other Middleware Ideas:**
- Rate limiting
- Caching responses
- Request/response transformation
- Performance monitoring
- Cost tracking

---

## Performance Considerations

### 1. Timeout Configuration

**Local models are slower than cloud APIs:**

```python
# Default: 300s (5 minutes)
client = LocalLLMClient(timeout=300)

# Reasoning models may need more
client = LocalLLMClient(timeout=600)  # 10 minutes

# Fast models can use less
client = LocalLLMClient(timeout=60)   # 1 minute
```

**Factors Affecting Timeout:**
- Model size (larger models slower)
- Hardware (GPU vs CPU)
- Context length (longer context = slower)
- Tool complexity (complex tools take time)

**Recommendations:**
- Development: 300s (safe default)
- Production with fast models: 60-120s
- Reasoning/large models: 600-900s
- Set timeout based on P99 latency in testing

### 2. Conversation History Management

**Long conversations impact performance:**

```python
# Problem: Unbounded history
for i in range(1000):
    client.chat(f"Message {i}")  # History grows unbounded

# Solution 1: Periodic reset
if len(client.conversation) > 20:
    client.conversation = client.conversation[-10:]  # Keep last 10

# Solution 2: Summary compression
if len(client.conversation) > 30:
    summary = client.chat("Summarize our conversation so far")
    client.conversation = [
        ChatMessage(role="system", content=f"Previous context: {summary}")
    ]
```

**Best Practices:**
- Monitor conversation length
- Reset or compress after 20-50 messages
- Use system message summaries for long contexts
- Consider conversation windows (sliding window of N messages)

### 3. Tool Execution Optimization

**Tools can be performance bottlenecks:**

```python
# Problem: Synchronous tool execution
for tool_call in tool_calls:
    result = execute_tool(tool_call)  # Blocks

# Solution: Parallel execution
import concurrent.futures

with concurrent.futures.ThreadPoolExecutor() as executor:
    futures = [
        executor.submit(execute_tool, tool_call)
        for tool_call in tool_calls
    ]
    results = [f.result() for f in futures]
```

**Optimization Strategies:**
- Cache tool results when appropriate
- Parallelize independent tool calls
- Set timeouts on tool execution
- Use async tools for I/O-bound operations

### 4. Pydantic Validation Performance

**Validation has overhead, but necessary:**

```python
# Validation overhead: ~0.1-1ms per message
# Trade-off: Safety vs. Speed

# For high-throughput scenarios, consider:
# 1. Reuse model instances (avoid re-parsing)
msg = ChatMessage(role="user", content="Hello")
# Reuse: msg.model_copy(update={"content": "New message"})

# 2. Use model_validate() instead of __init__() when parsing JSON
# Faster: ChatCompletion.model_validate(json_data)
# Slower: ChatCompletion(**json.loads(json_data))

# 3. Disable validation in production (unsafe, not recommended)
# ChatMessage.model_validate(data, strict=False)
```

**Recommendations:**
- Keep validation enabled (safety > speed)
- Profile before optimizing
- Pydantic overhead usually negligible vs. LLM latency

### 5. MLflow Tracing Overhead

**Tracing adds minimal overhead:**

```python
# Overhead: ~1-5ms per span creation
# Negligible compared to LLM latency (100ms - 60s)

# For extreme performance needs, disable tracing:
import os
os.environ["MLFLOW_ENABLE_TRACING"] = "false"

# Or conditionally trace:
if DEBUG_MODE:
    with mlflow.start_span(name="debug_span"):
        result = expensive_operation()
else:
    result = expensive_operation()  # No tracing overhead
```

**Best Practices:**
- Enable tracing in development/staging
- Selectively enable in production (sample traces)
- Use MLflow's built-in sampling features
- Monitor tracing overhead via MLflow UI

### 6. Memory Management

**Large conversations can consume significant memory:**

```python
# Problem: Large conversation history
# Each message: ~0.5-2KB
# 1000 messages: ~0.5-2MB
# Multiple clients: Memory adds up

# Solution 1: Explicit cleanup
del client.conversation[:-10]  # Keep only last 10 messages

# Solution 2: Use context managers for short-lived clients
def process_request(user_input: str) -> str:
    with LocalLLMClient(...) as client:
        return client.chat(user_input)
    # Client garbage collected, memory freed

# Solution 3: Periodic garbage collection
import gc
gc.collect()  # Force collection after large operations
```

---

## Critical Implementation Details

### 1. Conversation State Tracking (`local_llm_sdk/client.py:458-510`)

**The "Empty Response Bug" Fix:**

```python
def _handle_tool_calls(self, tool_calls: List[ToolCall]) -> tuple[str, List[ChatMessage]]:
    """
    CRITICAL: Must return (content, new_messages) tuple.

    Bug we fixed:
    - Previously returned only content string
    - Tool result messages weren't tracked
    - MLflow traces missed tool context
    - Conversation state incomplete

    Solution:
    - Return both content AND new messages
    - Caller updates last_conversation_additions
    - Full state preserved for tracing
    """
    new_messages = []

    # Add assistant message with tool_calls
    assistant_msg = ChatMessage(
        role="assistant",
        content=None,
        tool_calls=tool_calls
    )
    new_messages.append(assistant_msg)
    self.conversation.append(assistant_msg)

    # Execute tools and add results
    for tool_call in tool_calls:
        result = self.tool_registry.execute_tool(
            tool_call.function.name,
            json.loads(tool_call.function.arguments)
        )

        tool_msg = ChatMessage(
            role="tool",
            content=json.dumps(result),
            tool_call_id=tool_call.id
        )
        new_messages.append(tool_msg)
        self.conversation.append(tool_msg)

    # Get final response from LLM
    final_response = self._send_request(self.conversation)
    final_msg = ChatMessage(
        role="assistant",
        content=final_response
    )
    new_messages.append(final_msg)
    self.conversation.append(final_msg)

    # CRITICAL: Return tuple
    return (final_response, new_messages)


def chat(self, message: str, use_tools: bool = False) -> str:
    """
    Uses _handle_tool_calls() return value correctly.
    """
    # ... initial setup ...

    if response.tool_calls:
        # Unpack tuple
        content, new_messages = self._handle_tool_calls(response.tool_calls)

        # Track for tracing
        self.last_conversation_additions = new_messages

        return content
    else:
        return response.content
```

**Why This Matters:**
- MLflow tracing needs full conversation context
- Debugging requires complete message history
- Reproducibility depends on state preservation
- Tool results must persist for LLM context

### 2. Tool Schema Generation (`local_llm_sdk/tools/registry.py:40-95`)

**Automatic JSON Schema from Python Types:**

```python
def _python_type_to_json_schema(self, python_type) -> str:
    """
    Convert Python type hints to JSON Schema types.

    Supported conversions:
    - str → "string"
    - int → "integer"
    - float → "number"
    - bool → "boolean"
    - list, List → "array"
    - dict, Dict → "object"
    - Any, None → "string" (fallback)
    """
    type_map = {
        str: "string",
        int: "integer",
        float: "number",
        bool: "boolean",
        list: "array",
        dict: "object"
    }

    # Handle typing module types
    origin = getattr(python_type, "__origin__", None)
    if origin is not None:
        python_type = origin

    return type_map.get(python_type, "string")


def get_tool_schemas(self) -> List[Tool]:
    """
    Generate OpenAI-compatible tool schemas.

    Process:
    1. Inspect function signature
    2. Extract parameter types and defaults
    3. Build JSON Schema
    4. Create Pydantic Tool object
    """
    schemas = []

    for name, (func, description) in self._tools.items():
        sig = inspect.signature(func)

        parameters = {
            "type": "object",
            "properties": {},
            "required": []
        }

        for param_name, param in sig.parameters.items():
            # Skip self/cls parameters
            if param_name in ("self", "cls"):
                continue

            # Convert type
            param_type = self._python_type_to_json_schema(param.annotation)

            # Add to schema
            parameters["properties"][param_name] = {
                "type": param_type,
                "description": self._extract_param_doc(func, param_name)
            }

            # Track required parameters
            if param.default == inspect.Parameter.empty:
                parameters["required"].append(param_name)

        # Create Tool model
        tool = Tool(
            type="function",
            function=FunctionDefinition(
                name=name,
                description=description,
                parameters=parameters
            )
        )
        schemas.append(tool)

    return schemas
```

**Why This Matters:**
- Automatic schema generation reduces errors
- Type hints ensure LLM gets correct parameter types
- Required vs. optional handled automatically
- Pydantic validates schemas before sending to LLM

### 3. ReACT Agent Loop (`local_llm_sdk/agents/react.py:99-148`)

**Optimized System Prompt and Iteration Logic:**

```python
REACT_SYSTEM_PROMPT = """You are a helpful AI assistant using the ReACT pattern.

For each task:
1. REASON about what needs to be done
2. ACT by calling appropriate tools
3. OBSERVE the results
4. REPEAT steps 1-3 until task is complete
5. Respond with final answer followed by "TASK_COMPLETE"

IMPORTANT:
- Use ONE tool per iteration (do NOT chain multiple tools at once)
- After each tool call, wait for the result before proceeding
- Only say "TASK_COMPLETE" after you have the final answer

Example good iteration:
Thought: I need to calculate 5 factorial first.
Action: Call math_calculator with "5!"
[Wait for result: 120]
Thought: Now I need to convert "120" to uppercase.
Action: Call text_transformer with "120", "upper"
[Wait for result: "120"]
Thought: Task complete. TASK_COMPLETE
"""


def _execute(self, task: str, max_iterations: int = 15, verbose: bool = False) -> AgentResult:
    """
    Execute task using ReACT pattern.

    Key optimizations:
    - Clear system prompt (prevents tool cramming)
    - Iteration counting and metadata tracking
    - Stop condition detection
    - Final response cleanup
    """
    iteration_count = 0
    tool_call_count = 0

    # Build initial prompt
    full_prompt = f"{REACT_SYSTEM_PROMPT}\n\nTask: {task}"

    for i in range(max_iterations):
        iteration_count = i + 1

        with mlflow.start_span(name=f"iteration_{iteration_count}") as span:
            # Get response from LLM
            response = self.client.chat(full_prompt, use_tools=True)

            # Track tool calls
            if self.client.last_conversation_additions:
                for msg in self.client.last_conversation_additions:
                    if hasattr(msg, "tool_calls") and msg.tool_calls:
                        tool_call_count += len(msg.tool_calls)

            span.set_attribute("tool_calls", tool_call_count)

            # Check stop condition
            if self._should_stop(response):
                final_response = self._extract_final_answer(response)

                return AgentResult(
                    success=True,
                    final_response=final_response,
                    iterations=iteration_count,
                    status=AgentStatus.COMPLETED,
                    conversation=self.client.conversation,
                    metadata={
                        "total_tool_calls": tool_call_count,
                        "stop_reason": "task_complete"
                    }
                )

            if verbose:
                print(f"[Iteration {iteration_count}] {response}")

    # Max iterations reached
    return AgentResult(
        success=False,
        final_response="Task did not complete within max iterations",
        iterations=iteration_count,
        status=AgentStatus.MAX_ITERATIONS_REACHED,
        conversation=self.client.conversation,
        metadata={
            "total_tool_calls": tool_call_count,
            "stop_reason": "max_iterations"
        }
    )


def _should_stop(self, response: str) -> bool:
    """Check if agent should stop (task complete)."""
    return "TASK_COMPLETE" in response.upper()


def _extract_final_answer(self, response: str) -> str:
    """
    Extract final answer and remove TASK_COMPLETE marker.

    Before: "The answer is 120. TASK_COMPLETE"
    After:  "The answer is 120."
    """
    # Remove TASK_COMPLETE (case-insensitive)
    cleaned = re.sub(r"\s*TASK_COMPLETE\s*", "", response, flags=re.IGNORECASE)
    return cleaned.strip()
```

**Why This Matters:**
- Clear prompts prevent "tool cramming" bug (all tools in one iteration)
- Iteration tracking enables performance analysis
- Stop condition prevents infinite loops
- Final response cleanup improves UX

---

## Summary

The Local LLM SDK architecture is built on these core principles:

1. **Layered Architecture** - Clear separation of concerns across 5 layers
2. **Type Safety** - Pydantic models ensure correctness at compile and runtime
3. **Extensibility** - Easy to add tools, agents, and custom behavior
4. **Observability** - MLflow tracing provides debugging and performance insights
5. **State Management** - Complete conversation history for context and reproducibility
6. **Performance** - Optimized for local LLM constraints (timeouts, memory, iteration)

**Key Design Patterns:**
- Template Method (BaseAgent)
- Strategy (multiple agent types)
- Dependency Injection (client → tools, agent → client)
- Repository (ToolRegistry)
- Builder (Pydantic models)

**Extension Points:**
- Custom tools via `@tool` decorator
- Custom agents via `BaseAgent` subclass
- Custom configuration sources
- Middleware for cross-cutting concerns
- Extended Pydantic models for domain-specific features

**Production Considerations:**
- 300s timeout default (adjust based on model)
- Conversation history management (compress/reset after 20-50 messages)
- Tool execution optimization (parallel when possible)
- MLflow tracing overhead (minimal, ~1-5ms per span)
- Memory management (cleanup large conversations)

This architecture supports both simple use cases (basic chat) and complex scenarios (multi-step agent tasks) while maintaining clean code, type safety, and excellent debugging capabilities.
