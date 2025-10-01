# ReACT Agents Guide

## Overview

**ReACT (Reasoning + Acting)** is a powerful pattern for building autonomous agents that can solve complex, multi-step tasks by iteratively thinking, acting with tools, and observing results.

This guide shows you how to use the ReACT agent implementation in Local LLM SDK.

## What is ReACT?

ReACT agents combine two key capabilities:

1. **Reasoning**: Breaking down complex tasks into steps
2. **Acting**: Using tools to execute those steps

**Traditional Approach (Single-Shot):**
```
User: "Calculate 5 factorial, convert to uppercase, count characters"
LLM: "The answer is..." ← Tries to do everything at once, often fails
```

**ReACT Approach (Iterative):**
```
Iteration 1: [Think] "I need to calculate 5 factorial first"
            [Act] Call math_calculator(expression="5!")
            [Observe] Result: 120

Iteration 2: [Think] "Now I need to convert '120' to uppercase"
            [Act] Call text_transformer(text="120", operation="uppercase")
            [Observe] Result: "120" (unchanged, but validated)

Iteration 3: [Think] "Now I need to count the characters"
            [Act] Call char_counter(text="120")
            [Observe] Result: 3 characters

Iteration 4: [Think] "I have all the information I need"
            [Act] Generate final answer: "The factorial of 5 is 120..."
```

## Quick Start

### Basic Usage

```python
from local_llm_sdk import create_client_with_tools
from local_llm_sdk.agents import ReACT

# Create client with tools
client = create_client_with_tools()

# Create ReACT agent
agent = ReACT(client)

# Run a complex task
result = agent.run(
    task="Calculate 5 factorial, convert to uppercase, count characters",
    max_iterations=15
)

print(result.final_response)
print(f"Completed in {result.iterations} iterations")
print(f"Used {result.metadata['total_tool_calls']} tools")
```

### Output

```
Final Response: The factorial of 5 is 120. When converted to uppercase it remains "120" (since it's already numbers). The string "120" contains 3 characters.

Completed in 4 iterations
Used 3 tools
```

## ReACT Agent API

### Constructor

```python
ReACT(
    client: LocalLLMClient,
    system_prompt: Optional[str] = None,
    name: str = "react-agent"
)
```

**Parameters:**
- `client`: LocalLLMClient instance with tools registered
- `system_prompt`: Custom system prompt (uses optimized default if None)
- `name`: Agent name for tracing/logging

**Example:**
```python
from local_llm_sdk import LocalLLMClient
from local_llm_sdk.agents import ReACT

client = LocalLLMClient()
agent = ReACT(client, name="data-analysis-agent")
```

### run() Method

```python
agent.run(
    task: str,
    max_iterations: int = 10,
    verbose: bool = False,
    stop_condition: Optional[Callable] = None
) -> AgentResult
```

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `task` | `str` | Required | Task description for the agent |
| `max_iterations` | `int` | `10` | Maximum iterations before stopping |
| `verbose` | `bool` | `False` | Print iteration details |
| `stop_condition` | `Optional[Callable]` | `None` | Custom stop condition function |

**Returns:**
- `AgentResult`: Object containing final response, status, iterations, and metadata

**Example:**
```python
result = agent.run(
    task="Analyze this data file and create a summary",
    max_iterations=20,
    verbose=True
)

if result.success:
    print(result.final_response)
else:
    print(f"Task failed: {result.error}")
```

## AgentResult Object

### Properties

```python
@dataclass
class AgentResult:
    success: bool                    # Whether task completed successfully
    final_response: str              # Agent's final answer
    iterations: int                  # Number of iterations used
    conversation: List[Message]      # Full conversation history
    status: AgentStatus              # SUCCESS, FAILED, or MAX_ITERATIONS
    error: Optional[str]             # Error message if failed
    metadata: Dict[str, Any]         # Additional info (tool calls, etc.)
```

### Usage

```python
result = agent.run("Calculate 10 factorial")

# Check success
if result.success:
    print("✓ Task completed successfully")

# Get final answer
print(result.final_response)

# Check efficiency
print(f"Iterations: {result.iterations}")
print(f"Tool calls: {result.metadata['total_tool_calls']}")

# Inspect conversation
for msg in result.conversation:
    print(f"{msg.role}: {msg.content}")

# Check status
if result.status == AgentStatus.MAX_ITERATIONS:
    print("Warning: Reached max iterations")
```

## Complete Examples

### Example 1: Math and Text Processing

```python
from local_llm_sdk import create_client_with_tools
from local_llm_sdk.agents import ReACT

# Setup
client = create_client_with_tools()
agent = ReACT(client)

# Task: Multi-step calculation and formatting
task = """
1. Calculate 7 factorial
2. Convert the result to uppercase
3. Count how many characters are in the result
"""

result = agent.run(task, max_iterations=15, verbose=True)

print("=" * 60)
print("FINAL RESULT")
print("=" * 60)
print(result.final_response)
print(f"\nIterations: {result.iterations}")
print(f"Tool calls: {result.metadata['total_tool_calls']}")
```

**Output:**
```
Iteration 1: Calculating 7 factorial...
  → Tool: math_calculator(expression="7!")
  → Result: 5040

Iteration 2: Converting to uppercase...
  → Tool: text_transformer(text="5040", operation="uppercase")
  → Result: "5040"

Iteration 3: Counting characters...
  → Tool: char_counter(text="5040")
  → Result: 4 characters

Iteration 4: Generating final answer...
  → Response: "The factorial of 7 is 5040..."

============================================================
FINAL RESULT
============================================================
The factorial of 7 is 5040. When converted to uppercase, it remains "5040" (since it contains only numbers). The result contains 4 characters.

Iterations: 4
Tool calls: 3
```

### Example 2: File Processing

```python
from local_llm_sdk import create_client_with_tools
from local_llm_sdk.agents import ReACT

client = create_client_with_tools()
agent = ReACT(client)

task = """
1. Read the file /path/to/data.txt
2. Count how many lines it has
3. Convert the content to uppercase
4. Write the result to /path/to/output.txt
"""

result = agent.run(task, max_iterations=20)

if result.success:
    print("✓ File processed successfully")
    print(result.final_response)
else:
    print(f"✗ Processing failed: {result.error}")
```

### Example 3: Data Analysis

```python
from local_llm_sdk import create_client_with_tools
from local_llm_sdk.agents import ReACT

client = create_client_with_tools()
agent = ReACT(client)

task = """
Analyze the sales data in sales.csv:
1. Read the file
2. Calculate the sum of all sales
3. Find the average sale amount
4. Identify the highest sale
5. Create a summary report
"""

result = agent.run(task, max_iterations=25, verbose=True)

print(result.final_response)
# Prints: "Based on the analysis of sales.csv:
#          - Total sales: $1,250,000
#          - Average sale: $5,000
#          - Highest sale: $50,000
#          - Number of transactions: 250"
```

### Example 4: Code Generation

```python
from local_llm_sdk import create_client_with_tools
from local_llm_sdk.agents import ReACT, tool

# Add custom code execution tool
@tool("Executes Python code and returns the output")
def execute_python(code: str) -> dict:
    import subprocess
    try:
        result = subprocess.run(
            ["python", "-c", code],
            capture_output=True,
            text=True,
            timeout=10
        )
        return {
            "stdout": result.stdout,
            "stderr": result.stderr,
            "returncode": result.returncode
        }
    except subprocess.TimeoutExpired:
        return {"error": "Execution timed out"}

client = create_client_with_tools()
agent = ReACT(client)

task = """
1. Write a Python function to calculate fibonacci numbers
2. Test it with n=10
3. Verify the result is correct
"""

result = agent.run(task, max_iterations=15)
print(result.final_response)
```

## Advanced Usage

### Custom System Prompts

```python
from local_llm_sdk.agents import ReACT

custom_prompt = """
You are a data analysis expert. When given tasks:
1. Break them into clear, logical steps
2. Use appropriate tools for each step
3. Validate results before proceeding
4. Provide detailed explanations

Available tools: {tool_descriptions}

When finished, respond with: TASK_COMPLETE
"""

agent = ReACT(client, system_prompt=custom_prompt)
result = agent.run("Analyze the quarterly sales data...")
```

### Custom Stop Conditions

```python
from local_llm_sdk.agents import ReACT

def custom_stop_condition(response: str) -> bool:
    """Stop when agent says 'ANALYSIS_COMPLETE' or 'ERROR:'."""
    return "ANALYSIS_COMPLETE" in response or "ERROR:" in response

agent = ReACT(client)
result = agent.run(
    task="Analyze data and report findings",
    max_iterations=50,
    stop_condition=custom_stop_condition
)
```

### Verbose Mode for Debugging

```python
agent = ReACT(client)

result = agent.run(
    task="Complex multi-step task",
    max_iterations=20,
    verbose=True  # Prints each iteration
)

# Output shows:
# Iteration 1/20: [Thinking about the task...]
# Iteration 2/20: [Calling tool: math_calculator]
# Iteration 3/20: [Processing result: 42]
# ...
```

### Error Handling

```python
from local_llm_sdk.agents import ReACT, AgentStatus

agent = ReACT(client)

result = agent.run("Risky task that might fail", max_iterations=10)

if result.status == AgentStatus.SUCCESS:
    print("✓ Task completed successfully")
    print(result.final_response)

elif result.status == AgentStatus.MAX_ITERATIONS:
    print("⚠ Reached maximum iterations")
    print(f"Progress so far: {result.final_response}")

elif result.status == AgentStatus.FAILED:
    print(f"✗ Task failed: {result.error}")

# Check metadata for diagnostics
print(f"Iterations used: {result.iterations}")
print(f"Tools called: {result.metadata['total_tool_calls']}")
```

## Best Practices

### 1. Provide Clear Task Descriptions

```python
# ✅ Good: Specific, actionable task
task = """
1. Read the CSV file at /data/sales.csv
2. Calculate the sum of the 'amount' column
3. Calculate the average
4. Write results to /data/summary.txt
"""

# ❌ Bad: Vague task
task = "Analyze sales data"
```

### 2. Set Appropriate Max Iterations

```python
# Simple tasks: 5-10 iterations
result = agent.run("Calculate 5 factorial", max_iterations=5)

# Medium tasks: 10-20 iterations
result = agent.run("Process CSV and create summary", max_iterations=15)

# Complex tasks: 20-50 iterations
result = agent.run("Analyze data, generate insights, create report", max_iterations=30)
```

### 3. Use Verbose Mode During Development

```python
# Development: See what agent is doing
result = agent.run(task, verbose=True)

# Production: Quiet execution
result = agent.run(task, verbose=False)
```

### 4. Check Result Status

```python
result = agent.run(task)

# Always check success before using result
if not result.success:
    log_error(f"Agent failed: {result.error}")
    return

# Only use final_response if successful
process_result(result.final_response)
```

### 5. Monitor Tool Usage

```python
result = agent.run(task)

# Check if agent is efficient
if result.metadata['total_tool_calls'] > result.iterations:
    print("⚠ Agent is using multiple tools per iteration")

# Check if agent got stuck
if result.iterations == result.metadata['max_iterations']:
    print("⚠ Agent reached maximum iterations")
```

## Troubleshooting

### Agent Gets Stuck in Loops

**Problem**: Agent repeats the same actions

**Solution**: Add stop condition or reduce max_iterations

```python
def stop_on_repeat(response: str, history: List[str]) -> bool:
    """Stop if response repeats too many times."""
    if history.count(response) >= 3:
        return True
    return False

result = agent.run(task, stop_condition=stop_on_repeat)
```

### Agent Doesn't Use Tools

**Problem**: Agent tries to answer without calling tools

**Solution**: Improve task description and system prompt

```python
task = """
IMPORTANT: Use the available tools for each step.

1. Use math_calculator to compute 5 factorial
2. Use text_transformer to convert result to uppercase
3. Use char_counter to count characters

Do not calculate these manually.
"""
```

### Agent Stops Prematurely

**Problem**: Agent stops before completing task

**Solution**: Check stop condition or increase max_iterations

```python
# Remove stop condition
result = agent.run(task, max_iterations=20, stop_condition=None)

# Or use custom stop condition
def only_stop_on_complete(response: str) -> bool:
    return "TASK_COMPLETE" in response and "all steps" in response.lower()

result = agent.run(task, stop_condition=only_stop_on_complete)
```

### Response Quality Issues

**Problem**: Agent gives incomplete or incorrect answers

**Solution**: Use better model or improve prompts

```python
# Use more capable model
client = LocalLLMClient(model="mistralai/magistral-small-2509")

# Or improve system prompt
custom_prompt = """
You are a meticulous assistant. For each task:
1. Break it into clear steps
2. Execute each step completely
3. Verify results before proceeding
4. Provide detailed final answer

Think carefully before each action.
"""

agent = ReACT(client, system_prompt=custom_prompt)
```

## Performance Tips

### 1. Tool Selection

```python
# ✅ Efficient: Register only needed tools
from local_llm_sdk import LocalLLMClient, tool

client = LocalLLMClient()
client.register_tool(math_calculator)
client.register_tool(text_transformer)

# ❌ Inefficient: Register all tools (slower LLM decisions)
client = create_client_with_tools()  # Loads all 6 built-in tools
```

### 2. Iteration Limits

```python
# ✅ Efficient: Match complexity to limit
simple_task = agent.run("Calculate 5!", max_iterations=5)
complex_task = agent.run("Full analysis", max_iterations=30)

# ❌ Inefficient: Too high limit (wastes tokens)
simple_task = agent.run("Calculate 5!", max_iterations=100)
```

### 3. Prompt Optimization

```python
# ✅ Efficient: Concise, clear instructions
system_prompt = """
Break tasks into steps. Use tools. Be concise.
Stop when done (say 'TASK_COMPLETE').
"""

# ❌ Inefficient: Verbose prompt (slow, costly)
system_prompt = """
You are an incredibly helpful and thoughtful assistant...
[5 more paragraphs of instructions]
"""
```

## MLflow Integration

ReACT agents automatically create MLflow traces:

```python
import mlflow
from local_llm_sdk.agents import ReACT

# Enable MLflow tracking
mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("react-agents")

agent = ReACT(client)

# Run creates hierarchical trace
with mlflow.start_run():
    result = agent.run("Complex task", max_iterations=20)

# View trace in MLflow UI:
# - Parent span: agent.run()
# - Child spans: Each iteration
# - Grandchild spans: Tool calls
```

## Next Steps

- **[Agents API Reference](../api-reference/agents.md)** - Complete API documentation
- **[Tool Calling Guide](tool-calling.md)** - Master function calling
- **[MLflow Tracing Guide](mlflow-tracing.md)** - Observability and debugging
- **[Production Patterns](production-patterns.md)** - Best practices for deployment

## Related Examples

- **Notebook**: `notebooks/07-react-agents.ipynb` - Interactive tutorial
- **Mini-Project**: `notebooks/10-mini-project-code-helper.ipynb` - ReACT code assistant
- **Mini-Project**: `notebooks/11-mini-project-data-analyzer.ipynb` - ReACT data analyzer
