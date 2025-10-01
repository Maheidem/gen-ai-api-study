# Agents API Reference

## Overview

The Local LLM SDK agents framework provides production-ready implementations of agentic patterns for complex multi-step tasks. Agents orchestrate sequences of LLM calls and tool executions to accomplish goals that require reasoning, planning, and action.

**Key Features:**
- Automatic MLflow tracing integration
- Full conversation state management
- Extensible base class for custom agents
- Built-in ReACT (Reasoning + Acting) implementation
- Type-safe result objects with metadata

**When to Use Agents:**
- Multi-step tasks requiring sequential tool use
- Tasks that need planning and reasoning
- Complex workflows with decision points
- Tasks where you need execution observability

## Quick Start

```python
from local_llm_sdk import create_client_with_tools

# Create client with tools
client = create_client_with_tools()

# Use ReACT agent via convenience method
result = client.react(
    task="Calculate 5 factorial, convert to uppercase, count characters",
    max_iterations=15,
    verbose=True
)

print(f"Status: {result.status}")
print(f"Iterations: {result.iterations}")
print(f"Answer: {result.final_response}")
print(f"Tools used: {result.metadata['total_tool_calls']}")
```

**Output:**
```
Status: AgentStatus.SUCCESS
Iterations: 4
Answer: The text "120" has 3 characters.
Tools used: 3
```

---

## Agent Framework Architecture

```
┌─────────────────────────────────────────────────────────────┐
│ BaseAgent (Abstract)                                        │
│ - Automatic conversation context management                │
│ - MLflow tracing wrapper                                    │
│ - Error handling and result formatting                     │
└─────────────────────────────────────────────────────────────┘
                            ↓ extends
┌─────────────────────────────────────────────────────────────┐
│ ReACT Agent                                                 │
│ - Reasoning + Acting loop                                   │
│ - Sequential tool execution                                 │
│ - Stop condition detection                                  │
└─────────────────────────────────────────────────────────────┘
                            ↓ returns
┌─────────────────────────────────────────────────────────────┐
│ AgentResult                                                 │
│ - status: SUCCESS | FAILED | MAX_ITERATIONS | ERROR        │
│ - iterations: int                                           │
│ - final_response: str                                       │
│ - conversation: List[ChatMessage]                           │
│ - metadata: Dict[str, Any]                                  │
└─────────────────────────────────────────────────────────────┘
```

---

## BaseAgent

Abstract base class that all agents inherit from. Provides automatic MLflow tracing, conversation context management, and error handling.

### Class Definition

```python
class BaseAgent(ABC):
    """
    Abstract base class for all agents.

    Agents encapsulate specific interaction patterns (ReACT, Chain-of-Thought, etc.)
    and handle MLflow tracing automatically.
    """
```

### Constructor

```python
def __init__(self, client: LocalLLMClient, name: Optional[str] = None)
```

**Parameters:**
- `client` (LocalLLMClient): Client instance with tools configured
- `name` (str, optional): Name for this agent instance (defaults to class name)

**Attributes:**
- `client` (LocalLLMClient): The LLM client used for inference
- `name` (str): Agent instance name
- `metadata` (Dict[str, Any]): Agent-level metadata storage

**Example:**
```python
from local_llm_sdk import LocalLLMClient
from local_llm_sdk.agents import BaseAgent

client = LocalLLMClient(base_url="http://localhost:1234/v1")

class MyAgent(BaseAgent):
    def _execute(self, task: str, **kwargs) -> AgentResult:
        # Implementation here
        pass

agent = MyAgent(client, name="my-custom-agent")
```

### Methods

#### run()

Execute the agent on a given task.

```python
def run(self, task: str, **kwargs) -> AgentResult
```

**Parameters:**
- `task` (str): The task description/prompt for the agent
- `**kwargs`: Additional arguments passed to `_execute()`

**Returns:**
- `AgentResult`: Execution result with status, iterations, response, and metadata

**Behavior:**
1. Creates MLflow conversation context automatically
2. Calls subclass `_execute()` implementation
3. Handles exceptions and wraps in `AgentResult`
4. Adds agent name and class to result metadata

**Example:**
```python
result = agent.run(
    task="Implement a bubble sort function",
    max_iterations=20,
    temperature=0.7
)

if result.success:
    print(f"Completed in {result.iterations} iterations")
    print(result.final_response)
else:
    print(f"Failed: {result.error}")
```

**MLflow Tracing:**
```
ReACT_Calculate_5_factorial...
├─ iteration_1
│  ├─ tool_math_calculator
│  └─ observation
├─ iteration_2
│  ├─ tool_text_transformer
│  └─ observation
└─ iteration_3
   ├─ tool_char_counter
   └─ observation
```

#### _execute() (Abstract)

Implement agent-specific execution logic. Must be overridden by subclasses.

```python
@abstractmethod
def _execute(self, task: str, **kwargs) -> AgentResult
```

**Parameters:**
- `task` (str): The task description/prompt
- `**kwargs`: Agent-specific parameters (e.g., `max_iterations`, `temperature`)

**Returns:**
- `AgentResult`: Result object with execution details

**Example Implementation:**
```python
class SimpleAgent(BaseAgent):
    def _execute(self, task: str, **kwargs) -> AgentResult:
        max_iterations = kwargs.get('max_iterations', 10)
        messages = [{"role": "user", "content": task}]

        for i in range(max_iterations):
            response = self.client.chat(
                messages=messages,
                use_tools=True
            )

            # Check if task complete
            if "done" in response.lower():
                return AgentResult(
                    status=AgentStatus.SUCCESS,
                    iterations=i + 1,
                    final_response=response,
                    metadata={"method": "simple"}
                )

            messages.append({"role": "assistant", "content": response})

        return AgentResult(
            status=AgentStatus.MAX_ITERATIONS,
            iterations=max_iterations,
            final_response=response
        )
```

#### _sanitize_task_name()

Create a safe name from task description for MLflow tracing.

```python
def _sanitize_task_name(self, task: str, max_length: int = 30) -> str
```

**Parameters:**
- `task` (str): Task description
- `max_length` (int): Maximum length of sanitized name (default: 30)

**Returns:**
- `str`: Sanitized task name safe for use in trace/conversation names

**Behavior:**
- Extracts first line of task
- Removes special characters
- Replaces spaces with underscores
- Truncates to `max_length`

**Example:**
```python
task = "Calculate the factorial of 5 and convert to uppercase"
safe_name = agent._sanitize_task_name(task)
# Result: "Calculate_the_factorial_of..."
```

---

## ReACT Agent

Implementation of the ReACT (Reasoning + Acting) pattern for multi-step task completion with tool use.

### Pattern Overview

ReACT interleaves reasoning traces with task-specific actions:

1. **Reasoning**: Think about what needs to be done
2. **Acting**: Use tools to take actions
3. **Observation**: Observe tool results
4. **Repeat**: Continue until task complete

### Class Definition

```python
class ReACT(BaseAgent):
    """
    ReACT agent that combines reasoning and acting with tool use.

    The agent follows an iterative loop where it reasons about the task,
    executes tools one at a time, observes results, and continues until
    completion.
    """
```

### Constructor

```python
def __init__(
    self,
    client: LocalLLMClient,
    name: str = "ReACT",
    system_prompt: Optional[str] = None
)
```

**Parameters:**
- `client` (LocalLLMClient): Client instance with tools registered
- `name` (str): Agent name (default: "ReACT")
- `system_prompt` (str, optional): Custom system prompt (uses optimized default if None)

**Example:**
```python
from local_llm_sdk import create_client_with_tools
from local_llm_sdk.agents import ReACT

client = create_client_with_tools()

# Use default system prompt
agent = ReACT(client)

# Or customize the prompt
custom_prompt = """You are an AI that solves tasks step-by-step.
Use tools carefully and explain your reasoning."""

agent = ReACT(client, system_prompt=custom_prompt)
```

### Default System Prompt

The ReACT agent uses an optimized system prompt that emphasizes:

**Critical Rules:**
1. Use EXACTLY ONE tool per response - NEVER call multiple tools at once
2. After EVERY tool call, wait for the result before deciding next step
3. Break complex tasks into small, focused steps across multiple iterations
4. Think step-by-step: explain → call tool → observe → explain → next tool

**Instructions:**
- `execute_python` for testing/computing → returns output
- `filesystem_operation` for file writes → saves to project
- After each tool result, explain what you learned before next action
- When task is complete, state the FINAL ANSWER clearly, then say TASK_COMPLETE

**Example Pattern:**
```
User: Calculate 5 factorial, convert to uppercase, count characters
You: I'll calculate 5 factorial first. [calls math_calculator ONLY]
You: Got 120. Now converting to uppercase... [calls text_transformer ONLY]
You: Got "120". Now counting characters... [calls char_counter ONLY]
You: The answer is: The text "120" has 3 characters. TASK_COMPLETE
```

> Note: Each "You:" line represents a SEPARATE response after receiving a tool result.

### Methods

#### run()

Execute the ReACT agent (inherits from `BaseAgent.run()` with custom `_execute()`).

```python
def run(
    self,
    task: str,
    max_iterations: int = 15,
    stop_condition: Optional[Callable[[str], bool]] = None,
    temperature: float = 0.7,
    verbose: bool = True
) -> AgentResult
```

**Parameters:**
- `task` (str): The task description/prompt
- `max_iterations` (int): Maximum iterations to run (default: 15)
- `stop_condition` (Callable, optional): Custom stop condition function
- `temperature` (float): Sampling temperature (default: 0.7)
- `verbose` (bool): Print progress information (default: True)

**Returns:**
- `AgentResult`: Result with status, iterations, response, and full conversation

**Example - Basic Usage:**
```python
agent = ReACT(client)
result = agent.run(
    task="Calculate 42 * 17 and format as JSON",
    max_iterations=10
)

print(result.final_response)
# Output: {"result": 714}
```

**Example - Custom Stop Condition:**
```python
def stop_when_json(content: str) -> bool:
    """Stop when response contains valid JSON."""
    import json
    try:
        json.loads(content)
        return True
    except:
        return False

result = agent.run(
    task="Format the number 42 as JSON",
    stop_condition=stop_when_json,
    max_iterations=5
)
```

**Example - Low Temperature for Deterministic:**
```python
result = agent.run(
    task="Calculate the sum of 1 to 100",
    temperature=0.1,  # More deterministic
    verbose=False
)
```

**Example - Verbose Output:**
```python
result = agent.run(
    task="Create file 'test.txt' with content 'Hello World'",
    verbose=True
)

# Console output:
# ================================================================================
# ReACT Agent: Starting task
# Max iterations: 15
# Task: Create file 'test.txt' with content 'Hello World'
# ================================================================================
#
# Iteration 1/15
# ----------------------------------------
# Response: I'll create the file using filesystem_operation...
# Tools used: 1
#   - filesystem_operation
#
# Iteration 2/15
# ----------------------------------------
# Response: File created successfully. TASK_COMPLETE
#
# ================================================================================
# ✓ Task completed successfully in 2 iterations
# ================================================================================
```

#### _execute()

Internal execution logic for ReACT loop. Called by `BaseAgent.run()`.

```python
def _execute(
    self,
    task: str,
    max_iterations: int = 15,
    stop_condition: Optional[Callable[[ChatMessage], bool]] = None,
    temperature: float = 0.7,
    verbose: bool = True
) -> AgentResult
```

**Implementation Details:**
1. Initializes conversation with system prompt + user task
2. Enters iteration loop (up to `max_iterations`)
3. Each iteration:
   - Calls `client.chat(use_tools=True)` for LLM response
   - Automatically handles tool execution
   - Extends conversation with `last_conversation_additions` (includes tool results)
   - Checks stop conditions
4. Returns `AgentResult` with appropriate status

**Conversation State Management:**
```python
# Messages are preserved across iterations:
messages = [
    system_message,          # System prompt
    user_message,            # Initial task
    assistant_message_1,     # First response with tool_calls
    tool_result_message_1,   # Tool execution result
    assistant_message_2,     # Second response
    ...
]

# Full history available in result.conversation
```

#### _should_stop()

Check if the agent should stop iteration loop.

```python
def _should_stop(
    self,
    content: str,
    stop_condition: Optional[Callable[[str], bool]] = None
) -> bool
```

**Parameters:**
- `content` (str): The response content to check
- `stop_condition` (Callable, optional): Custom stop function

**Returns:**
- `bool`: True if agent should stop

**Default Behavior:**
- Checks custom `stop_condition` if provided
- Falls back to checking for "TASK_COMPLETE" in content (case-insensitive)

**Example:**
```python
# Custom stop conditions
def stop_on_error(content: str) -> bool:
    return "error" in content.lower()

def stop_on_code(content: str) -> bool:
    return "```python" in content

# Use with agent
result = agent.run(task="...", stop_condition=stop_on_code)
```

#### _count_tool_calls()

Count total tool calls in conversation history.

```python
def _count_tool_calls(self, messages: List[ChatMessage]) -> int
```

**Parameters:**
- `messages` (List[ChatMessage]): Conversation messages

**Returns:**
- `int`: Total number of tool calls

**Usage:**
```python
# Accessed via result metadata
result = agent.run(task="...")
print(f"Tools used: {result.metadata['total_tool_calls']}")
```

#### _extract_final_answer()

Extract final answer from content, removing TASK_COMPLETE marker.

```python
def _extract_final_answer(self, content: str) -> str
```

**Parameters:**
- `content` (str): The response content

**Returns:**
- `str`: Cleaned final answer without TASK_COMPLETE

**Example:**
```python
content = "The answer is 42. TASK_COMPLETE"
answer = agent._extract_final_answer(content)
# Result: "The answer is 42."
```

---

## AgentResult

Result object returned by all agent executions.

### Class Definition

```python
@dataclass
class AgentResult:
    """
    Result of an agent execution.
    """
    status: AgentStatus
    iterations: int
    final_response: str
    conversation: List[Any] = field(default_factory=list)
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
```

### Attributes

| Attribute | Type | Description |
|-----------|------|-------------|
| `status` | `AgentStatus` | Execution status (SUCCESS, FAILED, MAX_ITERATIONS, ERROR) |
| `iterations` | `int` | Number of iterations completed |
| `final_response` | `str` | Final response content from the agent |
| `conversation` | `List[ChatMessage]` | Full conversation history including tool messages |
| `error` | `str`, optional | Error message if status is ERROR |
| `metadata` | `Dict[str, Any]` | Additional execution metadata |

### Properties

#### success

```python
@property
def success(self) -> bool:
    """Check if agent completed successfully."""
    return self.status == AgentStatus.SUCCESS
```

**Example:**
```python
result = agent.run(task="...")

if result.success:
    print(f"Task completed: {result.final_response}")
else:
    print(f"Task failed with status: {result.status}")
```

### Metadata Keys

The `metadata` dictionary typically contains:

| Key | Type | Description |
|-----|------|-------------|
| `agent_name` | `str` | Name of the agent instance |
| `agent_class` | `str` | Agent class name (e.g., "ReACT") |
| `total_tool_calls` | `int` | Total number of tool calls made |
| `final_iteration` | `int` | Final iteration number (on success) |
| `max_iterations_reached` | `bool` | True if stopped due to max iterations |
| `failed_at_iteration` | `int` | Iteration number where error occurred |
| `error_type` | `str` | Python exception class name |

**Example:**
```python
result = agent.run(task="Multi-step task")

print(f"Agent: {result.metadata['agent_name']}")
print(f"Iterations: {result.iterations}")
print(f"Tools: {result.metadata['total_tool_calls']}")
print(f"Avg tools/iteration: {result.metadata['total_tool_calls'] / result.iterations:.1f}")
```

### Example Results

**Successful Execution:**
```python
AgentResult(
    status=AgentStatus.SUCCESS,
    iterations=3,
    final_response="The factorial of 5 is 120.",
    conversation=[...],  # 7 messages
    error=None,
    metadata={
        'agent_name': 'ReACT',
        'agent_class': 'ReACT',
        'total_tool_calls': 1,
        'final_iteration': 3
    }
)
```

**Max Iterations Reached:**
```python
AgentResult(
    status=AgentStatus.MAX_ITERATIONS,
    iterations=15,
    final_response="Still calculating...",
    conversation=[...],
    error=None,
    metadata={
        'max_iterations_reached': True,
        'total_tool_calls': 12
    }
)
```

**Error:**
```python
AgentResult(
    status=AgentStatus.ERROR,
    iterations=2,
    final_response="",
    conversation=[...],
    error="Connection timeout",
    metadata={
        'failed_at_iteration': 2,
        'error_type': 'TimeoutError'
    }
)
```

---

## AgentStatus

Enumeration of possible agent execution statuses.

### Enum Definition

```python
class AgentStatus(str, Enum):
    """Status of agent execution."""
    SUCCESS = "success"
    FAILED = "failed"
    MAX_ITERATIONS = "max_iterations"
    STOPPED = "stopped"
    ERROR = "error"
```

### Values

| Status | Value | Description |
|--------|-------|-------------|
| `SUCCESS` | `"success"` | Task completed successfully |
| `FAILED` | `"failed"` | Task failed to complete |
| `MAX_ITERATIONS` | `"max_iterations"` | Reached iteration limit without completion |
| `STOPPED` | `"stopped"` | Execution stopped early (e.g., user interrupt) |
| `ERROR` | `"error"` | Exception occurred during execution |

### Usage

```python
from local_llm_sdk.agents import AgentStatus

result = agent.run(task="...")

# Check status
if result.status == AgentStatus.SUCCESS:
    print("Task completed!")
elif result.status == AgentStatus.MAX_ITERATIONS:
    print("Need more iterations")
elif result.status == AgentStatus.ERROR:
    print(f"Error: {result.error}")

# Pattern matching (Python 3.10+)
match result.status:
    case AgentStatus.SUCCESS:
        print("Success!")
    case AgentStatus.MAX_ITERATIONS:
        print("Need more iterations")
    case AgentStatus.ERROR:
        print(f"Error: {result.error}")
```

---

## Creating Custom Agents

Extend `BaseAgent` to create agents with custom behavior patterns.

### Basic Template

```python
from local_llm_sdk.agents import BaseAgent, AgentResult, AgentStatus
from typing import List

class MyCustomAgent(BaseAgent):
    """
    Custom agent that implements [describe pattern].
    """

    def __init__(self, client, name: str = "MyCustomAgent", **config):
        super().__init__(client, name)
        # Store custom configuration
        self.config = config

    def _execute(self, task: str, **kwargs) -> AgentResult:
        """
        Execute custom agent logic.

        Args:
            task: The task to accomplish
            **kwargs: Agent-specific parameters

        Returns:
            AgentResult with execution details
        """
        # Your implementation here
        # 1. Initialize conversation
        # 2. Run your custom loop/pattern
        # 3. Return AgentResult

        return AgentResult(
            status=AgentStatus.SUCCESS,
            iterations=1,
            final_response="Completed",
            metadata={"custom_metric": 42}
        )
```

### Example: Chain-of-Thought Agent

```python
from local_llm_sdk.agents import BaseAgent, AgentResult, AgentStatus
from local_llm_sdk.models import create_chat_message

class ChainOfThought(BaseAgent):
    """
    Agent that uses chain-of-thought prompting without tools.
    Focuses on step-by-step reasoning.
    """

    SYSTEM_PROMPT = """You are a logical thinker.
    Break down problems into steps and reason through them carefully.
    Show your work and explain your reasoning."""

    def _execute(
        self,
        task: str,
        max_steps: int = 5,
        temperature: float = 0.7
    ) -> AgentResult:
        messages = [
            create_chat_message("system", self.SYSTEM_PROMPT),
            create_chat_message("user", f"Think step-by-step:\n{task}")
        ]

        for step in range(max_steps):
            response = self.client.chat(
                messages=messages,
                temperature=temperature,
                return_full_response=True
            )

            content = response.choices[0].message.content
            messages.append(response.choices[0].message)

            # Check if reasoning is complete
            if "therefore" in content.lower() or "conclusion" in content.lower():
                return AgentResult(
                    status=AgentStatus.SUCCESS,
                    iterations=step + 1,
                    final_response=content,
                    conversation=messages,
                    metadata={"reasoning_steps": step + 1}
                )

        return AgentResult(
            status=AgentStatus.MAX_ITERATIONS,
            iterations=max_steps,
            final_response=messages[-1].content,
            conversation=messages,
            metadata={"reasoning_steps": max_steps}
        )


# Usage
agent = ChainOfThought(client)
result = agent.run(
    task="What is the sum of all prime numbers less than 20?",
    max_steps=10
)
```

### Example: Agentic Workflow

```python
class WorkflowAgent(BaseAgent):
    """
    Agent that executes a predefined workflow with validation steps.
    """

    def __init__(self, client, workflow: List[dict], name: str = "Workflow"):
        super().__init__(client, name)
        self.workflow = workflow  # List of {step, tool, validation}

    def _execute(self, task: str, **kwargs) -> AgentResult:
        messages = [create_chat_message("user", task)]
        completed_steps = []

        for i, step_config in enumerate(self.workflow):
            step_name = step_config["step"]
            tool_name = step_config.get("tool")
            validate = step_config.get("validation")

            # Execute step with specific tool
            prompt = f"Step {i+1}: {step_name}"
            response = self.client.chat(
                messages=[create_chat_message("user", prompt)],
                use_tools=True
            )

            # Validate if needed
            if validate and not validate(response):
                return AgentResult(
                    status=AgentStatus.FAILED,
                    iterations=i + 1,
                    final_response=f"Validation failed at step: {step_name}",
                    metadata={
                        "failed_step": step_name,
                        "completed_steps": completed_steps
                    }
                )

            completed_steps.append(step_name)

        return AgentResult(
            status=AgentStatus.SUCCESS,
            iterations=len(self.workflow),
            final_response="All workflow steps completed",
            metadata={"completed_steps": completed_steps}
        )


# Usage
workflow = [
    {"step": "Validate input", "validation": lambda r: "valid" in r.lower()},
    {"step": "Process data", "tool": "execute_python"},
    {"step": "Generate report", "tool": "filesystem_operation"}
]

agent = WorkflowAgent(client, workflow=workflow)
result = agent.run(task="Process user data")
```

---

## Usage Examples

### Example 1: Simple Math Task

```python
from local_llm_sdk import create_client_with_tools
from local_llm_sdk.agents import ReACT

client = create_client_with_tools()
agent = ReACT(client)

result = agent.run(
    task="Calculate the factorial of 7",
    max_iterations=5,
    verbose=True
)

print(f"\nResult: {result.final_response}")
print(f"Completed in {result.iterations} iterations")
```

### Example 2: Multi-Step Text Processing

```python
task = """
Process this text: 'hello world'
1. Convert to uppercase
2. Count the characters
3. Calculate the sum of character codes
"""

result = agent.run(task=task, max_iterations=10)

if result.success:
    print(f"Final answer: {result.final_response}")
    print(f"Tool calls made: {result.metadata['total_tool_calls']}")
else:
    print(f"Failed: {result.status}")
```

### Example 3: File Operations

```python
task = """
Create a Python file that implements bubble sort:
1. Create file 'bubble_sort.py'
2. Write the function implementation
3. Add docstring and type hints
"""

result = agent.run(task=task, max_iterations=15, temperature=0.5)

# Check the conversation to see what was created
for msg in result.conversation:
    if msg.role == "tool":
        print(f"Tool result: {msg.content[:100]}...")
```

### Example 4: Error Handling

```python
try:
    result = agent.run(
        task="Divide 100 by zero",
        max_iterations=3
    )

    if result.status == AgentStatus.ERROR:
        print(f"Error occurred: {result.error}")
        print(f"Failed at iteration: {result.metadata.get('failed_at_iteration')}")
    elif result.success:
        print(f"Result: {result.final_response}")

except Exception as e:
    print(f"Unexpected error: {e}")
```

### Example 5: Batch Processing

```python
tasks = [
    "Calculate 5!",
    "Convert 'test' to uppercase",
    "Count characters in 'hello'"
]

results = []
for i, task in enumerate(tasks, 1):
    print(f"\n--- Task {i}/{len(tasks)} ---")
    result = agent.run(task=task, max_iterations=5, verbose=False)
    results.append({
        "task": task,
        "success": result.success,
        "answer": result.final_response,
        "iterations": result.iterations
    })

# Summary
successful = sum(1 for r in results if r["success"])
print(f"\nCompleted {successful}/{len(tasks)} tasks")
```

### Example 6: Custom Stop Condition

```python
def stop_on_json(content: str) -> bool:
    """Stop when valid JSON is found."""
    import json
    try:
        json.loads(content)
        return True
    except:
        return False

result = agent.run(
    task="Format the result as JSON: {'value': 42}",
    stop_condition=stop_on_json,
    max_iterations=10
)

import json
data = json.loads(result.final_response)
print(f"Parsed value: {data['value']}")
```

### Example 7: Accessing Conversation History

```python
result = agent.run(
    task="Calculate 3! + 4!",
    max_iterations=10
)

# Analyze conversation
print(f"Total messages: {len(result.conversation)}")

for i, msg in enumerate(result.conversation):
    print(f"\nMessage {i+1} ({msg.role}):")

    if msg.role == "system":
        print("  [System prompt]")
    elif msg.role == "user":
        print(f"  {msg.content}")
    elif msg.role == "assistant":
        if hasattr(msg, 'tool_calls') and msg.tool_calls:
            print(f"  [Called {len(msg.tool_calls)} tool(s)]")
        else:
            print(f"  {msg.content[:100]}...")
    elif msg.role == "tool":
        print(f"  [Tool result: {msg.content[:50]}...]")
```

### Example 8: Integration with MLflow

```python
import mlflow

mlflow.set_experiment("agent-experiments")

with mlflow.start_run(run_name="factorial-task"):
    result = agent.run(
        task="Calculate 10 factorial",
        max_iterations=5
    )

    # Log metrics
    mlflow.log_metric("iterations", result.iterations)
    mlflow.log_metric("tool_calls", result.metadata['total_tool_calls'])
    mlflow.log_metric("success", 1 if result.success else 0)

    # Log parameters
    mlflow.log_param("agent_type", "ReACT")
    mlflow.log_param("max_iterations", 5)

    # Log result
    mlflow.log_text(result.final_response, "answer.txt")

# View in MLflow UI with hierarchical traces
```

---

## Best Practices

### 1. Choose Appropriate Max Iterations

```python
# Simple tasks: 5-10 iterations
result = agent.run(task="Calculate 5!", max_iterations=5)

# Multi-step tasks: 10-20 iterations
result = agent.run(task="Process and analyze data", max_iterations=15)

# Complex workflows: 20-30 iterations
result = agent.run(task="Implement full algorithm", max_iterations=25)
```

### 2. Use Temperature Wisely

```python
# Deterministic tasks (math, code): Low temperature
result = agent.run(
    task="Calculate sum of 1 to 100",
    temperature=0.1
)

# Creative tasks: Higher temperature
result = agent.run(
    task="Generate story outline",
    temperature=0.9
)
```

### 3. Always Check Result Status

```python
result = agent.run(task="...")

# DON'T assume success
# print(result.final_response)  # Might be empty!

# DO check status first
if result.success:
    print(result.final_response)
elif result.status == AgentStatus.MAX_ITERATIONS:
    print("Need more iterations, got:", result.final_response)
else:
    print(f"Error: {result.error}")
```

### 4. Use Verbose Mode During Development

```python
# Development: See what's happening
result = agent.run(task="...", verbose=True)

# Production: Disable verbose, use MLflow for observability
result = agent.run(task="...", verbose=False)
```

### 5. Leverage Metadata for Analysis

```python
results = []
for task in tasks:
    result = agent.run(task=task, max_iterations=10)
    results.append(result)

# Analyze performance
avg_iterations = sum(r.iterations for r in results) / len(results)
avg_tools = sum(r.metadata['total_tool_calls'] for r in results) / len(results)
success_rate = sum(r.success for r in results) / len(results)

print(f"Avg iterations: {avg_iterations:.1f}")
print(f"Avg tools/task: {avg_tools:.1f}")
print(f"Success rate: {success_rate:.1%}")
```

### 6. Handle Timeouts

```python
from local_llm_sdk import LocalLLMClient

# Increase timeout for long-running tasks
client = LocalLLMClient(
    base_url="http://localhost:1234/v1",
    timeout=600  # 10 minutes
)

agent = ReACT(client)
result = agent.run(
    task="Complex multi-step task",
    max_iterations=30
)
```

### 7. Use Stop Conditions for Specific Formats

```python
def stop_on_code_block(content: str) -> bool:
    """Stop when code block is generated."""
    return "```python" in content and "```" in content[content.index("```python")+10:]

result = agent.run(
    task="Write a function to reverse a string",
    stop_condition=stop_on_code_block,
    max_iterations=10
)
```

### 8. Preserve Conversation Context

```python
# The conversation includes ALL messages (tools, results, etc.)
result = agent.run(task="Multi-step task")

# You can inspect or save the full conversation
with open("conversation.json", "w") as f:
    import json
    json.dump([{
        "role": msg.role,
        "content": msg.content
    } for msg in result.conversation], f, indent=2)
```

### 9. Custom System Prompts for Domain Tasks

```python
# Financial analysis agent
financial_prompt = """You are a financial analyst assistant.
When analyzing data:
1. Always validate input ranges
2. Use precise decimal calculations
3. Format currency properly
4. Include confidence intervals
Use tools carefully and show all calculations."""

agent = ReACT(client, system_prompt=financial_prompt)
result = agent.run(task="Calculate portfolio returns")
```

### 10. Test with Behavioral Assertions

```python
# Property-based testing (not exact matching)
result = agent.run(task="Calculate 5! + 4!")

# Test invariants, not exact outputs
assert result.success, "Task should succeed"
assert result.iterations > 1, "Multi-step task should use multiple iterations"
assert result.metadata['total_tool_calls'] >= 2, "Should call calculator at least twice"
assert "144" in result.final_response, "Final answer should be present"
assert "TASK_COMPLETE" not in result.final_response, "Marker should be stripped"
```

---

## MLflow Tracing

All agents automatically create MLflow traces when the `mlflow` package is installed.

### Trace Hierarchy

```
Conversation: ReACT_Calculate_factorial
├─ iteration_1
│  ├─ chat_completion
│  ├─ tool_math_calculator
│  └─ observation
├─ iteration_2
│  ├─ chat_completion
│  └─ final_response
└─ metadata
   ├─ iterations: 2
   ├─ tool_calls: 1
   └─ status: success
```

### Viewing Traces

```python
# Traces are automatically logged
result = agent.run(task="...")

# View in MLflow UI
# mlflow ui --port 5000
# http://localhost:5000
```

### Custom Trace Attributes

```python
from local_llm_sdk.agents import BaseAgent, AgentResult, AgentStatus
import mlflow

class TracedAgent(BaseAgent):
    def _execute(self, task: str, **kwargs) -> AgentResult:
        # Add custom attributes to trace
        with mlflow.start_span(name="custom_step") as span:
            span.set_attributes({
                "custom_metric": 42,
                "task_type": "analysis"
            })

            # Your logic here
            result = self.client.chat(task)

        return AgentResult(...)
```

---

## Troubleshooting

### Issue: Agent Stops After 1 Iteration

**Problem**: Agent completes in 1 iteration with multiple tool calls.

**Cause**: LLM is cramming all tools into first response instead of iterating.

**Solution**:
1. Use optimized system prompt (default ReACT prompt)
2. Lower temperature for more structured behavior
3. Use `tool_choice="required"` for reasoning models

```python
# Good: Default ReACT prompt emphasizes ONE tool per iteration
agent = ReACT(client)  # Uses optimized prompt

# Also try lower temperature
result = agent.run(task="...", temperature=0.3)
```

### Issue: "TASK_COMPLETE" Appears in Final Response

**Problem**: Result contains "TASK_COMPLETE" marker.

**Cause**: `_extract_final_answer()` not being called properly.

**Solution**: This should be automatic in ReACT agent. If using custom agent, ensure you strip the marker:

```python
import re

def clean_response(content: str) -> str:
    return re.sub(r'\s*TASK_COMPLETE\s*', '', content, flags=re.IGNORECASE).strip()
```

### Issue: Max Iterations Reached

**Problem**: Agent reaches `max_iterations` without completing.

**Cause**: Task is too complex or agent is stuck in loop.

**Solutions**:
```python
# 1. Increase max_iterations
result = agent.run(task="...", max_iterations=30)

# 2. Simplify the task
result = agent.run(task="Step 1 only: ...", max_iterations=10)

# 3. Check conversation to see where it's stuck
for msg in result.conversation:
    print(f"{msg.role}: {msg.content[:100]}")
```

### Issue: Empty Final Response

**Problem**: `result.final_response` is empty string.

**Cause**: Agent stopped before generating final answer.

**Solution**: Check status and conversation:

```python
if not result.final_response:
    print(f"Status: {result.status}")
    print(f"Last message: {result.conversation[-1].content if result.conversation else 'None'}")

    if result.status == AgentStatus.ERROR:
        print(f"Error: {result.error}")
```

### Issue: Tool Results Not Preserved

**Problem**: Conversation missing tool result messages.

**Cause**: Not using `last_conversation_additions` from client.

**Solution**: ReACT agent handles this automatically. If creating custom agent:

```python
def _execute(self, task: str, **kwargs) -> AgentResult:
    messages = [...]

    response = self.client.chat(messages=messages, use_tools=True)

    # IMPORTANT: Use last_conversation_additions
    if hasattr(self.client, 'last_conversation_additions'):
        messages.extend(self.client.last_conversation_additions)
    else:
        messages.append(response.choices[0].message)

    return AgentResult(conversation=messages, ...)
```

### Issue: MLflow Traces Not Appearing

**Problem**: No traces in MLflow UI.

**Solutions**:
```python
# 1. Check MLflow is installed
import mlflow
print(mlflow.__version__)

# 2. Set experiment explicitly
mlflow.set_experiment("my-experiment")

# 3. Run agent
result = agent.run(task="...")

# 4. Check MLflow UI
# mlflow ui --port 5000
```

---

## API Summary

### Classes

| Class | Purpose |
|-------|---------|
| `BaseAgent` | Abstract base class for all agents |
| `ReACT` | Reasoning + Acting agent implementation |
| `AgentResult` | Result dataclass with status and metadata |
| `AgentStatus` | Enum of execution statuses |

### Key Methods

| Method | Class | Description |
|--------|-------|-------------|
| `run(task, **kwargs)` | `BaseAgent` | Execute agent on task |
| `_execute(task, **kwargs)` | `BaseAgent` | Abstract method for agent logic |
| `run(task, max_iterations, ...)` | `ReACT` | Execute ReACT loop |
| `success` | `AgentResult` | Check if execution succeeded |

### Import Paths

```python
# Main imports
from local_llm_sdk.agents import BaseAgent, ReACT, AgentResult, AgentStatus

# Via client convenience
from local_llm_sdk import create_client_with_tools
client = create_client_with_tools()
result = client.react(task="...")  # Shortcut to ReACT agent
```

---

## Related Documentation

- [Client API Reference](./client.md) - LocalLLMClient for LLM interactions
- [Tools API Reference](./tools.md) - Tool/function calling system
- [Configuration Guide](../guides/configuration.md) - Environment variables and setup
- [Behavioral Testing Guide](../guides/testing.md) - Testing agent behavior with live LLMs
- [MLflow Integration Guide](../guides/mlflow.md) - Observability and tracing

---

## Version Information

**SDK Version**: 0.1.0
**Last Updated**: 2025-10-01
**Python Compatibility**: 3.12+
**Dependencies**: `pydantic>=2.0`, `requests`, `mlflow` (optional)
