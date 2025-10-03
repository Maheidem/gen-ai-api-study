---
title: Tool Return Patterns for LLM Agents - Code Execution & Filesystem Operations
date: 2025-10-01
tags: [llm, agents, tools, best-practices]
status: completed
confidence: high
---

> **Note (October 2025):** This research document discusses `execute_python` and `filesystem_operation` as separate tools. The Local LLM SDK has since evolved to use a unified `bash` tool that provides full terminal capabilities (Python execution, file operations, git, text processing, etc.). The principles and best practices in this document still apply to the bash tool's return structure. See `local_llm_sdk/tools/builtin.py` for the current implementation.

# Executive Summary

Research into official implementations from Anthropic, OpenAI, LangChain, LlamaIndex, and Microsoft Semantic Kernel reveals consistent patterns for tool return values in LLM agent systems. The key principle: **return high-signal information that agents can act on, exclude low-level technical details unless needed for downstream operations.**

**Critical findings:**
- Token efficiency is paramount: Anthropic limits tool responses to 25,000 tokens by default
- Natural language identifiers perform better than UUIDs/technical IDs
- Simple string returns work for most cases; structured JSON when complexity requires it
- Full paths should be excluded from returns unless agent needs them for subsequent operations
- Tools can implement response_format parameters for concise vs. detailed modes

# Tool 1: Code Execution (execute_python)

## What Official Implementations Return

### Anthropic Claude Code Execution Tool
**Source:** [Code execution tool - Claude API](https://docs.claude.com/en/docs/agents-and-tools/tool-use/code-execution-tool)

**Bash command response structure:**
```json
{
  "type": "bash_code_execution_tool_result",
  "tool_use_id": "srvtoolu_xyz789",
  "content": {
    "type": "bash_code_execution_result",
    "stdout": "output text here",
    "stderr": "",
    "return_code": 0
  }
}
```

**What they return:**
- `stdout`: String output from successful execution
- `stderr`: String error messages if execution fails
- `return_code`: 0 for success, non-zero for failure
- Files created are returned separately via Files API with file_id

### LangChain Python REPL Tool
**Source:** [Python REPL | LangChain](https://python.langchain.com/docs/integrations/tools/python/)

**Return structure:**
```python
python_repl.run("print(1+1)")
# Returns: '2\n'
```

**What they return:**
- Simple string containing whatever was printed
- That's it - minimal and focused

**Key insight from docs:**
> "The interface will only return things that are printed - therefore, if you want to use it to calculate an answer, make sure to have it print out the answer."

### Best Practices from Anthropic
**Source:** [Writing effective tools for AI agents](https://www.anthropic.com/engineering/writing-tools-for-agents)

**Token efficiency:**
- Restrict tool responses to 25,000 tokens by default
- Implement truncation with helpful instructions
- Encourage agents to pursue token-efficient strategies

**Example of truncated response:**
```
Output (showing first 100 lines, 500 more lines omitted):
[actual output here]

Note: Output truncated. Use pagination or filtering to retrieve specific sections.
```

## Recommendations for execute_python

### ✅ KEEP:
- **stdout**: Essential - this is the primary output
  - **Why**: Agent needs to see the result of code execution
  - **Format**: String, with reasonable truncation (max 5,000-10,000 tokens)

- **stderr**: Essential - shows errors and warnings
  - **Why**: Agent needs to understand what went wrong
  - **Format**: String, with reasonable truncation

- **return_code**: Essential - indicates success/failure
  - **Why**: Clear signal for success (0) vs failure (non-zero)
  - **Format**: Integer

- **success**: Essential - boolean convenience field
  - **Why**: Easier for agent to check than return_code != 0
  - **Format**: Boolean

- **status**: Useful - high-level status message
  - **Why**: Human-readable summary: "success", "error", "timeout"
  - **Format**: String enum

### ❌ REMOVE:
- **working_directory**: Remove - not actionable
  - **Why**: Temporary directory path like `/var/folders/br/8w257jc979z20gbrd0l_s6yh0000gn/T/tmp5vrwtfec/` provides no value to agent
  - **Reason**: Agent can't use this path, files are ephemeral, wastes tokens
  - **Exception**: If implementing container reuse (like Anthropic), return a `container_id` instead

- **captured_result**: Redundant with stdout
  - **Why**: This duplicates information already in stdout
  - **Reason**: Wastes tokens, confuses agent about which to use
  - **Keep**: Only if you need to distinguish printed output from return value

### 🔄 MODIFY:
- **Truncate long outputs**:
  - Limit stdout/stderr to ~5,000 tokens each
  - Add message: "Output truncated to 5,000 tokens. Use filtering for specific sections."

- **Add helpful error messages**:
  ```python
  # ❌ Bad
  {"success": False, "stderr": "NameError: name 'x' is not defined"}

  # ✅ Good
  {
    "success": False,
    "stderr": "NameError: name 'x' is not defined",
    "message": "Variable 'x' is not defined. Define it before use: x = value"
  }
  ```

## Simplified Return Structure

```python
{
  "success": True,
  "status": "success",  # "success" | "error" | "timeout"
  "stdout": "120",
  "stderr": "",
  "return_code": 0
}
```

**Removed fields:**
- `working_directory`: Not actionable by agent
- `captured_result`: Redundant with stdout

**Token savings:** ~40-60 tokens per tool call (depending on path length)

---

# Tool 2: Filesystem Operations (filesystem_operation)

## What Official Implementations Return

### Anthropic Claude Text Editor Tool
**Source:** [Code execution tool - Claude API](https://docs.claude.com/en/docs/agents-and-tools/tool-use/code-execution-tool)

**Create file response:**
```json
{
  "type": "text_editor_code_execution_tool_result",
  "tool_use_id": "srvtoolu_xyz",
  "content": {
    "type": "text_editor_code_execution_result",
    "is_file_update": false
  }
}
```

**View file response:**
```json
{
  "type": "text_editor_code_execution_tool_result",
  "tool_use_id": "srvtoolu_xyz",
  "content": {
    "type": "text_editor_code_execution_result",
    "file_type": "text",
    "content": "{\"setting\": \"value\"}",
    "numLines": 4,
    "startLine": 1,
    "totalLines": 4
  }
}
```

**Key insight:** No absolute paths in responses!

### OpenAI Function Calling Pattern
**Source:** [How to call functions with chat models](https://github.com/openai/openai-cookbook/blob/main/examples/How_to_call_functions_with_chat_models.ipynb)

**Tool result format:**
```python
messages.append({
    "role": "tool",
    "tool_call_id": tool_call_id,
    "name": tool_function_name,
    "content": results  # Simple string or JSON string
})
```

**Key insight:** The `content` field is the actual result - can be simple string or JSON

### Model Context Protocol Specification
**Source:** [Tools - Model Context Protocol](https://modelcontextprotocol.io/specification/2025-06-18/server/tools)

**Result structure:**
```typescript
interface ToolResult {
  content: Array<TextContent | ImageContent | EmbeddedResource>;
  isError?: boolean;
}
```

**What they return:**
- Content can be text, images, or embedded resources
- Simple isError flag for error states
- No complex metadata by default

### Anthropic Best Practices - Meaningful Context
**Source:** [Writing effective tools for AI agents](https://www.anthropic.com/engineering/writing-tools-for-agents)

**Principles:**
> "Tool implementations should take care to return only high signal information back to agents. They should prioritize contextual relevance over flexibility, and eschew low-level technical identifiers (for example: `uuid`, `256px_image_url`, `mime_type`)."

> "Fields like `name`, `image_url`, and `file_type` are much more likely to directly inform agents' downstream actions and responses."

**Natural language > Technical IDs:**
> "Agents also tend to grapple with natural language names, terms, or identifiers significantly more successfully than they do with cryptic identifiers. We've found that merely resolving arbitrary alphanumeric UUIDs to more semantically meaningful and interpretable language (or even a 0-indexed ID scheme) significantly improves Claude's precision in retrieval tasks by reducing hallucinations."

## Recommendations for filesystem_operation

### ✅ KEEP:
- **success**: Essential - boolean status
  - **Why**: Clear signal of operation success
  - **Format**: Boolean

- **message**: Essential - human-readable summary
  - **Why**: Agent needs to understand what happened
  - **Format**: String like "File written successfully" or "File created: data.txt"
  - **Improvement**: Make messages more natural: "Created data.txt (1.5 KB)" vs "File written: /long/path/data.txt"

- **size**: Useful - file size for context
  - **Why**: Agent can understand scale of operation
  - **Format**: Integer (bytes) or human-readable "1.5 KB"

- **operation**: Add this - what was done
  - **Why**: Context for multi-operation tools
  - **Format**: String enum: "create" | "read" | "write" | "delete"

### ❌ REMOVE:
- **path**: Remove absolute paths
  - **Why**: `/var/folders/br/8w257jc979z20gbrd0l_s6yh0000gn/T/tmp5vrwtfec/data_analysis.txt` is not actionable
  - **Reason**: Agent can't use temp paths, wastes ~60-100 tokens
  - **Replace with**: Filename only: "data_analysis.txt"
  - **Exception**: If agent needs path for subsequent operations, use relative path or simple identifier

### 🔄 MODIFY:
- **Use relative paths when needed**:
  ```python
  # ❌ Bad - absolute temp path
  {"path": "/var/folders/br/.../tmp5vrwtfec/data_analysis.txt"}

  # ✅ Good - filename only
  {"filename": "data_analysis.txt"}

  # ✅ Also good - relative path for structure
  {"path": "output/data_analysis.txt"}
  ```

- **Add response_format parameter** (advanced):
  ```python
  # Concise mode (72 tokens)
  {
    "success": True,
    "message": "Created data_analysis.txt (1 KB)"
  }

  # Detailed mode (120 tokens) - when agent needs technical details
  {
    "success": True,
    "operation": "create",
    "filename": "data_analysis.txt",
    "size": 1024,
    "size_human": "1.0 KB",
    "file_type": "text",
    "encoding": "utf-8",
    "message": "File created successfully"
  }
  ```

## Simplified Return Structure

### For Write/Create Operations:
```python
{
  "success": True,
  "message": "Created data_analysis.txt (1.5 KB)",
  "filename": "data_analysis.txt",
  "size": 1536
}
```

**Removed fields:**
- `path`: Absolute temporary path not useful to agent

**Token savings:** ~60-100 tokens per tool call

### For Read Operations:
```python
{
  "success": True,
  "content": "file contents here...",
  "filename": "config.json",
  "lines": 24
}
```

### For Delete Operations:
```python
{
  "success": True,
  "message": "Deleted data_analysis.txt"
}
```

---

# Cross-Cutting Best Practices

## 1. Token Efficiency is Critical
**Source:** [Writing effective tools for AI agents](https://www.anthropic.com/engineering/writing-tools-for-agents)

- Claude Code limits tool responses to 25,000 tokens
- One optimization achieved 62% cost reduction via tool improvements
- JSON mode can save 15% on whitespace tokens

**Action:** Set maximum response size limits and truncate intelligently

## 2. Simple Strings > Complex JSON (When Possible)
**Sources:**
- [LangChain Python REPL](https://python.langchain.com/docs/integrations/tools/python/)
- [Model Context Protocol](https://modelcontextprotocol.io/specification/2025-06-18/server/tools)

**Pattern:**
```python
# Simple case - just return the answer
return "42"

# Complex case - return structured data when needed
return json.dumps({"result": 42, "units": "seconds"})
```

**When to use each:**
- Simple string: Single value, status message, error description
- Structured JSON: Multiple related values, need for downstream operations

## 3. Natural Language > Technical Identifiers
**Source:** [Writing effective tools for AI agents](https://www.anthropic.com/engineering/writing-tools-for-agents)

**Examples:**
```python
# ❌ Bad - cryptic UUIDs
{"user_id": "550e8400-e29b-41d4-a716-446655440000"}

# ✅ Good - natural language
{"user": "Jane Smith"}

# ✅ Also good - simple numeric ID
{"user": "Jane Smith", "id": 42}
```

## 4. Helpful Error Messages
**Source:** [Writing effective tools for AI agents](https://www.anthropic.com/engineering/writing-tools-for-agents)

**Pattern:**
```python
# ❌ Unhelpful
{
  "success": False,
  "error": "ValidationError: Invalid input"
}

# ✅ Helpful - actionable guidance
{
  "success": False,
  "error": "Invalid date format",
  "message": "Date must be in YYYY-MM-DD format. Example: 2025-10-01",
  "received": "10/01/2025"
}
```

## 5. Implement Concise/Detailed Response Modes (Advanced)
**Source:** [Writing effective tools for AI agents](https://www.anthropic.com/engineering/writing-tools-for-agents)

**Pattern:**
```python
def filesystem_operation(operation, path, response_format="concise"):
    result = perform_operation(operation, path)

    if response_format == "concise":
        return {
            "success": True,
            "message": f"Created {filename}"
        }
    else:  # detailed
        return {
            "success": True,
            "operation": operation,
            "filename": filename,
            "size": size,
            "file_type": file_type,
            "message": "File created successfully"
        }
```

**Token savings:** 30-60% depending on use case (example in research showed 72 vs 206 tokens)

## 6. Paths: Relative > Absolute, Filename > Path
**Sources:**
- [Absolute vs Relative Path](https://linuxhandbook.com/absolute-vs-relative-path/)
- [Writing effective tools for AI agents](https://www.anthropic.com/engineering/writing-tools-for-agents)

**Hierarchy (best to worst):**
1. **Filename only**: "data.txt" (when no structure needed)
2. **Relative path**: "output/data.txt" (preserves structure)
3. **Simple identifier**: "file_123" (when path not meaningful)
4. **Absolute path**: Only if agent needs it for downstream operations

**Rationale:**
- Absolute paths are not portable
- Temporary paths like `/var/folders/.../tmp5vrwtfec/` provide zero value to agent
- Relative paths maintain structure without system-specific details

---

# Implementation Checklist

## For execute_python tool:

- [ ] Return only: success, status, stdout, stderr, return_code
- [ ] Remove: working_directory (unless container reuse needed)
- [ ] Remove: captured_result (redundant with stdout)
- [ ] Truncate stdout/stderr to ~5,000 tokens max
- [ ] Add truncation message if output exceeds limit
- [ ] Make error messages actionable with examples

## For filesystem_operation tool:

- [ ] Return only: success, message, filename, size
- [ ] Remove: Absolute paths (replace with filename or relative path)
- [ ] Use natural language messages: "Created data.txt (1.5 KB)"
- [ ] Consider response_format parameter for concise/detailed modes
- [ ] Make error messages actionable with examples

## General improvements:

- [ ] Set token limit for all tool responses (5,000-25,000)
- [ ] Use simple strings for single values
- [ ] Use JSON for multiple related values
- [ ] Prefer natural language over technical IDs
- [ ] Test with real agent to measure token usage
- [ ] Run evaluations to measure performance impact

---

# Key Metrics to Track

After implementing changes, measure:

1. **Token usage per tool call** (should decrease 30-50%)
2. **Agent task completion rate** (should maintain or improve)
3. **Number of tool calls per task** (should maintain or decrease)
4. **Error recovery rate** (should improve with better messages)

---

# References

## Primary Sources (Official Documentation)

1. **[Code execution tool - Claude API](https://docs.claude.com/en/docs/agents-and-tools/tool-use/code-execution-tool)**
   - Accessed: 2025-10-01 17:30 UTC
   - Type: Official Documentation
   - Reliability: ⭐⭐⭐⭐⭐
   - Used for: Anthropic's code execution and file operation response structures

2. **[Writing effective tools for AI agents - Anthropic](https://www.anthropic.com/engineering/writing-tools-for-agents)**
   - Accessed: 2025-10-01 17:35 UTC
   - Type: Official Engineering Blog
   - Reliability: ⭐⭐⭐⭐⭐
   - Used for: Best practices, token efficiency, natural language vs IDs, response_format pattern

3. **[Python REPL | LangChain](https://python.langchain.com/docs/integrations/tools/python/)**
   - Accessed: 2025-10-01 17:20 UTC
   - Type: Official Documentation
   - Reliability: ⭐⭐⭐⭐⭐
   - Used for: Simple string return pattern for code execution

4. **[How to call functions with chat models - OpenAI Cookbook](https://github.com/openai/openai-cookbook/blob/main/examples/How_to_call_functions_with_chat_models.ipynb)**
   - Accessed: 2025-10-01 17:25 UTC
   - Type: Official Code Examples
   - Reliability: ⭐⭐⭐⭐⭐
   - Used for: Tool result format in content field

5. **[Tools - Model Context Protocol](https://modelcontextprotocol.io/specification/2025-06-18/server/tools)**
   - Accessed: 2025-10-01 17:30 UTC
   - Type: Protocol Specification
   - Reliability: ⭐⭐⭐⭐⭐
   - Used for: Standard tool result structure with content types

## Secondary Sources (Best Practices & Analysis)

6. **[Building Efficient LangChain Agents - Medium](https://medium.com/@heyamit10/building-efficient-langchain-agents-a-step-by-step-guide-76073898f7ff)**
   - Accessed: 2025-10-01 17:22 UTC
   - Type: Technical Article
   - Reliability: ⭐⭐⭐⭐
   - Used for: Token efficiency strategies, 62% cost reduction case study

7. **[Absolute vs Relative Path in Linux](https://linuxhandbook.com/absolute-vs-relative-path/)**
   - Accessed: 2025-10-01 17:33 UTC
   - Type: Technical Reference
   - Reliability: ⭐⭐⭐⭐
   - Used for: Path portability considerations

8. **[OpenAI Function Calling Tutorial - DataCamp](https://www.datacamp.com/tutorial/open-ai-function-calling-tutorial)**
   - Accessed: 2025-10-01 17:38 UTC
   - Type: Technical Tutorial
   - Reliability: ⭐⭐⭐⭐
   - Used for: Structured vs unstructured return patterns

## Additional Context

9. **[LangChain Token Usage Tracking](https://python.langchain.com/docs/how_to/chat_token_usage_tracking/)**
   - Accessed: 2025-10-01 17:24 UTC
   - Type: Official Documentation
   - Reliability: ⭐⭐⭐⭐⭐
   - Used for: Token tracking and optimization strategies

10. **[Claude Code Best Practices - Anthropic](https://www.anthropic.com/engineering/claude-code-best-practices)**
    - Accessed: 2025-10-01 17:18 UTC
    - Type: Official Documentation
    - Reliability: ⭐⭐⭐⭐⭐
    - Used for: General agent tool design principles

---

# Version History

- v1.0 (2025-10-01): Initial research completed
  - Analyzed official implementations from 5 major frameworks
  - Established clear recommendations for both tool types
  - Provided actionable checklists and implementation guidance
