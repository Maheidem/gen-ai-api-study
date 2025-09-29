# ReACT Agent Guide - Local LLM SDK

## Overview

This guide shows how to use the new **ReACT (Reasoning, Action, Observation)** agent implementation in the `local_llm_sdk`. The ReACT pattern allows AI agents to solve complex, multi-step tasks by iteratively thinking, acting with tools, and observing results.

## Quick Start

### 1. Install and Setup

```bash
cd gen-ai-api-study
pip install -e .
```

### 2. Start LM Studio

Make sure LM Studio is running on `http://169.254.83.107:1234` with the `mistralai/magistral-small-2509` model loaded.

### 3. Run the Notebook

```bash
cd notebooks
jupyter notebook react-agent-flow.ipynb
```

## New Tools Added

### `execute_python(code: str, timeout: int = 30)`

Safely execute Python code in a subprocess and return results.

**Example:**
```python
result = client.tools.execute('execute_python', {
    'code': 'print("Hello World")\nresult = 2 + 2'
})
# Returns: {"success": true, "stdout": "Hello World\n", "stderr": "", ...}
```

**Features:**
- Sandboxed execution in temporary directory
- Timeout protection (default 30 seconds)
- Captures stdout, stderr, and execution status
- Safe error handling

### `filesystem_operation(operation, path, content="", encoding="utf-8")`

Perform safe filesystem operations within the current working directory.

**Operations:**
- `create_dir` - Create directories
- `write_file` - Write content to files
- `read_file` - Read file contents
- `list_dir` - List directory contents
- `delete_file` - Delete files
- `check_exists` - Check if path exists

**Example:**
```python
# Create a directory
client.tools.execute('filesystem_operation', {
    'operation': 'create_dir',
    'path': 'my_project'
})

# Write a file
client.tools.execute('filesystem_operation', {
    'operation': 'write_file',
    'path': 'my_project/hello.py',
    'content': 'print("Hello from ReACT!")'
})
```

**Security:**
- Restricts operations to current working directory
- Prevents path traversal attacks
- Validates all paths before operations

## ReACT Agent Class

The `ReACTAgent` class implements the full ReACT loop:

```python
from local_llm_sdk import LocalLLMClient

# Create client and register tools
client = LocalLLMClient(
    base_url="http://169.254.83.107:1234/v1",
    model="mistralai/magistral-small-2509"
)
client.register_tools_from(None)  # Load built-in tools

# Create ReACT agent
agent = ReACTAgent(client, max_iterations=10, verbose=True)

# Execute a complex task
task = """
Create a Python script that calculates fibonacci numbers,
test it with different inputs, and save the results.
"""

conversation = agent.think_and_act(task)
```

### Agent Parameters

- `client`: LocalLLMClient instance with tools
- `max_iterations`: Maximum number of ReACT cycles (default: 10)
- `verbose`: Print detailed progress (default: True)

### How It Works

1. **Think**: Agent reasons about the task and plans next steps
2. **Act**: Agent calls appropriate tools to gather information or perform actions
3. **Observe**: Agent examines tool results and updates understanding
4. **Iterate**: Process repeats until task is complete or max iterations reached

## Example Tasks

### Simple Task
```python
task = """
Create a 'hello_world' directory and write a Python script
that prints 'Hello from ReACT Agent!' then execute it.
"""
```

### Complex Task
```python
task = """
Implement a sorting algorithm project:
1. Create project structure
2. Implement bubble sort algorithm
3. Create test cases
4. Benchmark against built-in sort
5. Save results and analysis
"""
```

## Best Practices

### 1. Task Design
- Break complex tasks into clear, actionable steps
- Specify expected outputs and file locations
- Include error handling requirements

### 2. Tool Usage
- Use `execute_python` for computational tasks
- Use `filesystem_operation` for project organization
- Combine tools for complex workflows

### 3. Monitoring
- Enable verbose mode for debugging
- Set appropriate iteration limits
- Save conversation logs for analysis

### 4. Safety
- All code execution is sandboxed
- Filesystem operations are restricted to current directory
- Timeout protection prevents infinite loops

## Troubleshooting

### Common Issues

**"Tool not found" errors:**
```python
# Make sure tools are registered
client.register_tools_from(None)
print(client.tools.list_tools())  # Should show execute_python, filesystem_operation
```

**Agent stops too early:**
```python
# Increase iteration limit
agent = ReACTAgent(client, max_iterations=15)
```

**LM Studio connection errors:**
```bash
# Check LM Studio is running
curl http://169.254.83.107:1234/v1/models
```

### Performance Tips

- Use lower temperature (0.1-0.3) for more focused reasoning
- Provide clear, specific task descriptions
- Monitor token usage for long conversations
- Use shorter iteration limits for simple tasks

## Example Output

```
🎯 TASK: Create a simple calculator script

🔄 Iteration 1
🤖 Thinking: I need to create a calculator script. Let me start by creating a directory...
🔧 Action: filesystem_operation({"operation": "create_dir", "path": "calculator_project"})
📊 Observation: {"success": true, "message": "Directory created..."}

🔄 Iteration 2
🤖 Thinking: Now I'll write the calculator script...
🔧 Action: filesystem_operation({"operation": "write_file", "path": "calculator_project/calc.py", "content": "def add(a, b):\n    return a + b\n..."})
📊 Observation: {"success": true, "message": "File written..."}

✅ Task completed in 3 iterations!
```

## Next Steps

- Explore the full notebook: `notebooks/react-agent-flow.ipynb`
- Try your own complex tasks
- Add custom tools for specific domains
- Build applications using the ReACT pattern

## Resources

- [ReACT Paper](https://arxiv.org/abs/2210.03629) - Original research
- [LM Studio Docs](https://lmstudio.ai/docs) - Local LLM setup
- [Local LLM SDK](../README.md) - Main SDK documentation

---

🚀 **Happy coding with ReACT!** The combination of reasoning and action-taking gives AI agents powerful problem-solving capabilities.