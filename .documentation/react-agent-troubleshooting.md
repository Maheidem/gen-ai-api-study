# ReACT Agent Troubleshooting Guide

**Last Updated**: 2025-10-06
**Status**: Production Reference

This guide provides systematic troubleshooting for ReACT agent issues, covering common problems, root causes, and proven fixes.

---

## Common Issues Overview

| Issue | Symptom | Root Cause | Fix |
|-------|---------|-----------|-----|
| **No Tool Calls** | Agent reaches MAX_ITERATIONS without calling tools | Model doesn't understand prompting | Model-specific prompts |
| **Empty Responses** | After iteration 3-4, responses are empty | Tool keeps failing, model exhausted strategies | Fix underlying tool issue |
| **Syntax Errors** | "unterminated string" errors repeated | Tool implementation bug (quote escaping) | Simplified tool implementation |
| **Format Drift** | XML in responses instead of JSON | Model reverts to native format | Validation + prompt anchoring |
| **Infinite Loops** | Same tool called repeatedly | Agent stuck in reasoning loop | Loop detection + history pruning |

---

## Issue 1: No Tool Calls (Mistral)

### Symptoms
```
Status: AgentStatus.MAX_ITERATIONS
Iterations: 10-20
Tool Calls: 0  ← Problem
Final Response: ... (empty or incomplete)
```

### Root Cause

**ReACT Agent Prompting Incompatibility:**

The ReACT agent uses a specific system prompt designed for qwen3. This prompt doesn't effectively guide mistral to call tools. Instead, mistral:
1. Generates [THINK] blocks (reasoning)
2. Never invokes the bash tool
3. Hits max_iterations limit
4. Fails without completing the task

**This is NOT a technical bug** - it's a behavioral/prompting issue.

### Evidence

**What Works:**
```python
from local_llm_sdk import LocalLLMClient

client = LocalLLMClient(model="mistralai/magistral-small-2509")
client.register_tools_from(builtin)
result = client.chat("Use bash to calculate 2 + 2", use_tools=True)
# ✅ SUCCESS! Result: "The result of calculating 2 + 2 is:\n\n4"
```

**What Doesn't Work:**
```python
from local_llm_sdk.agents import ReACT

agent = ReACT(client)  # Same client as above
result = agent.run("Calculate 2 + 2")
# ❌ FAILURE: MAX_ITERATIONS, 0 tool calls
```

### Fixes

**Fix 1: Use qwen3 for ReACT Agents (Recommended)**
```python
# qwen3 works excellently with ReACT prompting
client = LocalLLMClient(model="qwen/qwen-2.5-coder-7b-instruct")
agent = ReACT(client)
result = agent.run("Calculate factorial of 5")
# ✅ Works perfectly
```

**Fix 2: Model-Specific Prompting (Future Enhancement)**
```python
class ReACT(BaseAgent):
    def __init__(self, client):
        super().__init__(client)
        self.system_prompt = self._get_model_specific_prompt()

    def _get_model_specific_prompt(self):
        """Return optimized prompt based on model."""
        model_name = self.client.model.lower()

        if "mistral" in model_name:
            return MISTRAL_OPTIMIZED_PROMPT
        elif "qwen" in model_name:
            return QWEN_OPTIMIZED_PROMPT
        else:
            return DEFAULT_PROMPT
```

**Fix 3: Direct SDK Usage (Workaround)**
```python
# For mistral, skip ReACT agent entirely
client = LocalLLMClient(model="mistralai/magistral-small-2509")
client.register_tools_from(builtin)

# Use direct tool calling
result = client.chat("Calculate 5 factorial", use_tools=True)
# ✅ Works with mistral
```

---

## Issue 2: Empty Responses After Iteration 3-4

### Symptoms
```
Iteration 1: "syntax error due to an unterminated string"
Iteration 2: "string formatting is still causing issues"
Iteration 3: "The error persists"
Iteration 4-15: [EMPTY]  ← Problem
```

### Root Cause

**Tool Implementation Bug Causing Repeated Failures:**

The pattern is clear:
1. **Iteration 1-3**: LLM tries different approaches, all hit syntax errors from tool
2. **Iteration 4**: LLM tries yet another approach, gets error again
3. **Iteration 5-15**: LLM has exhausted strategies, returns empty content

**Why empty?** The model likely:
- Ran out of known strategies to fix the syntax error
- Hit some internal failure mode (timeout, confusion)
- Started generating content but SDK/agent didn't capture it properly

### Evidence

This pattern occurs when tools have implementation bugs that cause consistent failures across multiple attempts.

**Example: execute_python tool bug** (see Issue 3 for details)

### Fixes

**Fix 1: Identify and Fix the Underlying Tool Issue**
```python
# Step 1: Enable verbose mode to see tool errors
result = agent.run("task", verbose=True)

# Step 2: Check tool call results
for msg in agent.conversation:
    if msg.get("role") == "tool":
        print(f"Tool result: {msg.get('content')}")

# Step 3: Fix the tool that's returning errors
```

**Fix 2: Add Better Error Context to Agent**
```python
def _handle_tool_calls(self, tool_calls):
    """Execute tools with better error context."""
    for tool_call in tool_calls:
        try:
            result = self.execute_tool(tool_call)
        except Exception as e:
            # Add clear error context
            result = {
                "success": False,
                "error": str(e),
                "suggestion": "The tool failed. Try a different approach."
            }
    return result
```

**Fix 3: Add Circuit Breaker for Repeated Errors**
```python
class ReACT(BaseAgent):
    def __init__(self, client, max_consecutive_errors=3):
        super().__init__(client)
        self.consecutive_errors = 0
        self.max_consecutive_errors = max_consecutive_errors

    def _execute(self, task, max_iterations=15, verbose=False):
        for i in range(max_iterations):
            # ... generate response ...

            if tool_failed:
                self.consecutive_errors += 1

                if self.consecutive_errors >= self.max_consecutive_errors:
                    raise RuntimeError(
                        f"Tool failed {self.consecutive_errors} times in a row. "
                        f"Last error: {last_error}"
                    )
            else:
                self.consecutive_errors = 0  # Reset on success
```

---

## Issue 3: Repeated Syntax Errors (execute_python)

### Symptoms
```
Tool result: {"success": false, "error": "SyntaxError: unterminated string"}
Agent tries again...
Tool result: {"success": false, "error": "SyntaxError: unterminated string"}
Agent tries again...
[Repeats until MAX_ITERATIONS]
```

### Root Cause

**Critical Bug in execute_python Tool** (now fixed):

The tool wrapped user code in triple-quoted strings. If user's code contained triple quotes (e.g., in docstrings), the wrapper broke:

```python
# Tool implementation (OLD, BROKEN):
exec("""
{chr(10).join(line for line in code.split(chr(10)))}
""", execution_namespace)

# User's code with docstring:
def is_prime(n):
    """Check if number is prime"""  # ← Contains """
    if n < 2:
        return False
    ...

# Becomes (BROKEN):
exec("""
def is_prime(n):
    """Check if number is prime"""  # ← TERMINATES THE OUTER """
    if n < 2:
        return False
""", namespace)
# Result: SyntaxError: unterminated string literal
```

### Fixes

**Fix 1: Direct File Execution (IMPLEMENTED)**
```python
@tool("Execute Python code safely and return results")
def execute_python(code: str, timeout: int = 30) -> dict:
    """Execute Python code in a subprocess."""
    import subprocess
    import tempfile
    import os
    import sys

    try:
        # Create temp file - NO STRING WRAPPING
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write(code)  # Write raw code directly
            temp_file = f.name

        # Execute directly
        result = subprocess.run(
            [sys.executable, temp_file],
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=tempfile.gettempdir()
        )

        os.unlink(temp_file)

        return {
            "success": result.returncode == 0,
            "stdout": result.stdout,
            "stderr": result.stderr,
            "return_code": result.returncode
        }

    except subprocess.TimeoutExpired:
        return {"success": False, "error": f"Timeout after {timeout}s"}
    except Exception as e:
        return {"success": False, "error": str(e)}
```

**Why better:**
- No string wrapping = no quote escaping issues
- Simpler implementation (reduced from ~80 lines to ~30 lines)
- Clear error messages from actual Python interpreter
- Easier to debug

**Fix 2: Escape Triple Quotes (Partial, Not Recommended)**
```python
# Only if you must keep current approach
safe_code = code.replace('"""', r'\"\"\"').replace("'''", r"\'\'\'")
exec(f"""
{safe_code}
""", namespace)
```

**Problems:**
- Still fragile (doesn't handle all edge cases)
- Complex escaping logic needed for nested strings
- Better to avoid string wrapping entirely

---

## Issue 4: Format Drift (XML in Responses)

### Symptoms
```json
{
  "role": "assistant",
  "content": "<tool_call>\n<function=bash>\n<parameter>command</parameter>\n"
}
```

Instead of:
```json
{
  "role": "assistant",
  "tool_calls": [{
    "type": "function",
    "function": {
      "name": "bash",
      "arguments": "{\"command\": \"...\"}"
    }
  }]
}
```

### Root Cause

**Model Reverts to Native Format:**

Some models (like qwen3) are trained on XML schema for tool calls. During long conversations or under certain conditions, they may revert to their native format instead of maintaining the API's expected JSON format.

**For qwen3 specifically:** LM Studio usually converts XML → JSON automatically, but validation can catch edge cases where conversion fails.

### Fixes

**Fix 1: Validation with Early Termination (IMPLEMENTED)**
```python
import os

# Enable streaming and validation
os.environ['LLM_STREAM'] = 'true'
os.environ['LLM_ENABLE_VALIDATION'] = 'true'

from local_llm_sdk import LocalLLMClient

client = LocalLLMClient()

try:
    response = client.chat("Your task here")
except ValueError as e:
    # Validation caught XML_DRIFT during streaming!
    # Early termination after ~20 chunks (instead of full generation)
    print(f"Caught error: {e}")
    # Output: "🚨 VALIDATION ERROR: XML_DRIFT"
    #         "💡 TIP: This is EARLY TERMINATION during streaming"
```

**Benefits:**
- Stops bad responses immediately (saves 70-80% generation time)
- Works during generation (not just after)
- Clear error messages with actionable suggestions

**Fix 2: Format Anchoring in System Prompt**
```python
system_prompt = """You are a helpful assistant with access to tools.

CRITICAL FORMAT REQUIREMENTS:
1. When calling tools, ALWAYS use valid JSON function calling format
2. NEVER use XML-like syntax (<tool_call>, <function>, etc.)
3. NEVER put tool call information in the content field
4. Tool calls MUST use the tool_calls array structure

Example of CORRECT format:
{
  "role": "assistant",
  "tool_calls": [{
    "type": "function",
    "function": {
      "name": "tool_name",
      "arguments": "{\\"param\\": \\"value\\"}"
    }
  }]
}

Example of INCORRECT format (DO NOT USE):
{
  "role": "assistant",
  "content": "<tool_call>\\n<function=tool_name>\\n..."
}
"""
```

**Fix 3: Response Validation Before History Append**
```python
def validate_and_append_response(response, conversation_history):
    """Validate response before adding to history."""

    # Check for XML-like patterns in content
    if response.get("content"):
        content = response["content"]
        if "<tool_call>" in content or "<function=" in content:
            logger.error("XML format detected in response")
            # DON'T append malformed response
            return handle_malformed_response(response, conversation_history)

    # Check for valid tool calls
    if hasattr(response, 'tool_calls'):
        for tool_call in response.tool_calls:
            try:
                args = json.loads(tool_call.function.arguments)
            except json.JSONDecodeError as e:
                logger.error(f"Malformed tool call arguments: {e}")
                return handle_malformed_response(response, conversation_history)

    # Validation passed - safe to append
    conversation_history.append(response)
    return conversation_history
```

---

## Issue 5: Infinite Loops

### Symptoms
```
Iteration 1: Call tool X with args Y
Iteration 2: Call tool X with args Y (identical)
Iteration 3: Call tool X with args Y (identical)
...
Iteration 15: MAX_ITERATIONS reached
```

### Root Cause

**Agent Stuck in Reasoning Loop:**

The agent believes it needs to call the same tool repeatedly, either because:
1. Tool result doesn't satisfy stop condition
2. Agent misinterprets tool result
3. System prompt doesn't clearly define success criteria
4. History corruption confusing the agent

### Fixes

**Fix 1: Loop Detection (RECOMMENDED)**
```python
def _detect_loop(self):
    """Detect if agent is stuck in loop."""
    if len(self.conversation) < 6:
        return False

    # Check if last 3 tool calls are identical
    recent_tool_calls = [
        msg.get("tool_calls")
        for msg in self.conversation[-6:]
        if msg.get("tool_calls")
    ]

    if len(recent_tool_calls) >= 3:
        if recent_tool_calls[-1] == recent_tool_calls[-2] == recent_tool_calls[-3]:
            logger.warning("Loop detected: identical tool calls")
            return True

    return False

# In agent run loop:
if self._detect_loop():
    logger.warning("Loop detected - recovering")
    self._recover_from_loop()
    continue
```

**Fix 2: History Pruning on Loop Detection**
```python
def _recover_from_loop(self):
    """Recover from detected loop."""
    # Keep system + last 3 valid messages
    system = [m for m in self.conversation if m["role"] == "system"]
    recent = self.conversation[-3:]
    self.conversation = system + recent

    # Add reset prompt
    self.conversation.append({
        "role": "user",
        "content": "The previous approach isn't working. Try a different approach."
    })
```

**Fix 3: Clear Success Criteria in System Prompt**
```python
system_prompt = """You are a helpful assistant with access to tools.

TASK COMPLETION:
- When you have successfully completed the task, respond with your final answer
- Include "TASK_COMPLETE" at the end of your final response
- Do NOT keep calling tools after the task is done
- If a tool fails 3 times, try a different approach or ask for help

EXAMPLE:
Task: Calculate 5 factorial
Thought: I'll use bash to calculate this
Action: Use bash tool
Result: 120
Final Answer: The factorial of 5 is 120. TASK_COMPLETE
"""
```

---

## Debugging Workflow

### Step 1: Enable Verbose Mode

```python
result = agent.run("your task", verbose=True)
```

**What you'll see:**
- Each iteration's thought process
- Tool calls and their arguments
- Tool results
- Agent's reasoning

### Step 2: Inspect Conversation History

```python
for i, msg in enumerate(agent.conversation):
    print(f"\n=== Message {i} ===")
    print(f"Role: {msg.get('role')}")

    if msg.get("content"):
        print(f"Content: {msg['content'][:200]}...")

    if msg.get("tool_calls"):
        print(f"Tool calls: {len(msg['tool_calls'])}")
        for tc in msg["tool_calls"]:
            print(f"  - {tc['function']['name']}: {tc['function']['arguments']}")
```

### Step 3: Check Tool Results

```python
tool_messages = [
    msg for msg in agent.conversation
    if msg.get("role") == "tool"
]

for msg in tool_messages:
    print(f"\nTool: {msg.get('name')}")
    print(f"Result: {msg.get('content')}")
```

### Step 4: Check LM Studio Logs (if applicable)

```bash
# Stream logs in real-time
lms log stream | grep -E "error|tool|POST"

# Or check specific request/response
lms log stream --source model --filter output
```

### Step 5: Validate Model Behavior

```python
# Test if model works with direct SDK usage
from local_llm_sdk import create_client_with_tools

client = create_client_with_tools()
result = client.chat("Simple task: calculate 2+2", use_tools=True)
print(f"Direct SDK result: {result}")

# If this works but agent doesn't, it's a prompting issue
# If this also fails, it's a tool or model compatibility issue
```

---

## Prevention Best Practices

### 1. Use Validation

```python
import os
os.environ['LLM_STREAM'] = 'true'
os.environ['LLM_ENABLE_VALIDATION'] = 'true'
os.environ['LLM_VALIDATION_CHECK_INTERVAL'] = '20'
```

### 2. Set Reasonable Limits

```python
agent = ReACT(client)
result = agent.run(
    task="your task",
    max_iterations=15,  # Prevent runaway execution
    verbose=False        # Set True for debugging
)
```

### 3. Choose Compatible Models

- ✅ **qwen3**: Excellent for ReACT agents
- ⚠️ **mistral**: Use direct SDK, not ReACT agent
- ⚠️ **granite**: Unknown ReACT compatibility, works with direct SDK

### 4. Monitor for Patterns

```python
# Track consecutive tool failures
consecutive_failures = 0

for iteration in range(max_iterations):
    result = execute_tool(tool_call)

    if not result.get("success"):
        consecutive_failures += 1

        if consecutive_failures >= 3:
            logger.error("Tool failing repeatedly - aborting")
            break
    else:
        consecutive_failures = 0  # Reset on success
```

### 5. Clear System Prompts

- Define success criteria explicitly
- Provide examples of correct format
- Include stop conditions
- Explain what to do when tools fail

---

## Quick Reference: 5 Essential Fixes

1. **Model Selection**: Use qwen3 for ReACT agents, mistral for direct SDK
2. **Tool Validation**: Fix underlying tool bugs that cause repeated errors
3. **Loop Detection**: Implement identical tool call detection
4. **Format Validation**: Enable streaming validation to catch XML drift early
5. **Clear Prompts**: Define success criteria and stop conditions explicitly

---

**Sources:**
- `.scratchpad/tool-calling-investigation-2025-10-06.md` - Comprehensive mistral investigation
- `.scratchpad/execute_python_analysis.md` - Tool implementation bug analysis
- `.scratchpad/handoffs/general-programmer-agent-2025-10-06-14-30-00-SUCCESS.md` - Environment variable fix
- `local_llm_sdk/agents/react.py` - ReACT agent implementation
- Production testing across 642+ requests with qwen3
