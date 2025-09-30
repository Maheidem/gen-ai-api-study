# local_llm_sdk/agents/

## Purpose
Production-ready agent framework with automatic MLflow tracing. Implements agentic patterns like ReACT (Reasoning + Acting) for complex multi-step tasks.

## Contents
- `__init__.py` - Public exports (`BaseAgent`, `ReACT`, `AgentResult`, `AgentStatus`)
- `base.py` - `BaseAgent` abstract class with automatic conversation context and tracing
- `models.py` - `AgentResult` and `AgentStatus` dataclasses for result handling
- `react.py` - `ReACT` agent implementation with optimized prompt and tool loop

## Relationships
- **Parent**: Imported from `local_llm_sdk.agents`
- **Uses**: `LocalLLMClient` from `../client.py` for LLM calls and tool execution
- **Uses**: MLflow for hierarchical trace logging (optional dependency)
- **Extends**: New agents inherit from `BaseAgent` and implement `_execute()`

## Getting Started
1. **Read `base.py` first** - Understanding `BaseAgent` is key
   - `run()` method wraps execution with conversation context
   - `_execute()` is abstract - subclasses implement task logic
2. **Then `react.py`** - See ReACT implementation:
   - Optimized system prompt (lines 38-53)
   - Iteration loop with tool calls (lines 99-148)
   - Stop condition handling (lines 135-148)
3. **Check `models.py`** - Result types returned by agents

## Usage Pattern
```python
from local_llm_sdk import create_client_with_tools

client = create_client_with_tools()

# One-liner via client
result = client.react(task="Your task here", max_iterations=15)

# Or direct agent instantiation
from local_llm_sdk.agents import ReACT
agent = ReACT(client, system_prompt="Custom prompt...")
result = agent.run(task="Your task", max_iterations=10)
```

## Key Features
- Full conversation state management (includes tool results)
- Automatic MLflow tracing under single parent span
- Customizable system prompts
- Stop conditions (lambda or default "TASK_COMPLETE")
- Metadata tracking (iterations, tool counts, status)

## Creating Custom Agents
Extend `BaseAgent` and implement `_execute()`:
```python
class MyAgent(BaseAgent):
    def _execute(self, task: str, **kwargs) -> AgentResult:
        # Your agent logic here
        return AgentResult(status=AgentStatus.SUCCESS, ...)
```