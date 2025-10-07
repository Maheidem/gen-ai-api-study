# Model Compatibility Guide

**Last Updated**: 2025-10-06
**Status**: Production Reference

This guide provides comprehensive compatibility information for various LLM models when used with the Local LLM SDK, including quirks, workarounds, and production recommendations.

---

## Quick Reference Matrix

| Model | Tool Calling | ReACT Agent | Format | Notes |
|-------|-------------|-------------|---------|-------|
| **qwen3-coder-30b** | ✅ Perfect | ✅ Excellent | XML → JSON (auto) | Recommended for agents |
| **mistralai/magistral-small-2509** | ✅ Perfect | ⚠️ Prompting issues | JSON | Direct `chat()` works perfectly |
| **granite-4.0** | ✅ Perfect | ⚠️ Unknown | XML (SDK parses) | Tool calls in content field |

---

## qwen3: Full Compatibility (100% Success Rate)

### Status
✅ **FULLY COMPATIBLE** - 642+ production requests validated, 0 errors

### Technical Details

**Model Behavior:**
- qwen3 is trained on XML schema for tool calls
- Outputs tool calls in XML format: `<tool_call><function=...></function></tool_call>`
- This is by design, not a bug

**LM Studio API Layer:**
- Automatically detects XML format in model output
- Converts XML → OpenAI-compatible JSON format
- Returns proper `tool_calls` array to SDK

**What SDK Receives:**
```json
{
  "choices": [{
    "message": {
      "tool_calls": [{
        "id": "call_abc123",
        "type": "function",
        "function": {
          "name": "bash",
          "arguments": "{\"command\": \"expr 2 + 2\"}"
        }
      }]
    }
  }]
}
```

**SDK never sees the XML** - receives clean JSON from API.

### Architecture Flow

```
┌─────────────────────────────────────────────┐
│ qwen3 Model                                 │
│ Generates: <tool_call>XML format</tool_call>│
└──────────────────┬──────────────────────────┘
                   │
                   ↓
┌─────────────────────────────────────────────┐
│ LM Studio API Layer                         │
│ • Detects XML in model output               │
│ • Converts XML → OpenAI JSON                │
│ • Returns JSON to client                    │
└──────────────────┬──────────────────────────┘
                   │
                   ↓
┌─────────────────────────────────────────────┐
│ Local LLM SDK                               │
│ • Receives OpenAI JSON format               │
│ • Parses tool_calls normally                │
│ • Executes tools                            │
│ • Zero compatibility issues                 │
└─────────────────────────────────────────────┘
```

### Validation System

**XML_DRIFT Validator:**
- Checks for XML patterns in **API response** (what SDK receives)
- Validates SDK input, not raw model output
- With qwen3: LM Studio converts XML before SDK sees it
- SDK receives JSON (no XML patterns present)

**When XML_DRIFT Would Trigger:**
- If LM Studio's conversion fails (extremely rare)
- If model outputs malformed XML that can't convert
- If API returns raw XML without conversion (API bug)

### Test Results

**642 Production Requests:**
- ✅ 0 XML_DRIFT errors
- ✅ 0 tool call failures
- ✅ 0 parsing errors
- ✅ 100% success rate

**Validated Features:**
- ✅ Basic tool calling
- ✅ Multi-step tool workflows
- ✅ ReACT agent patterns
- ✅ Streaming with validation
- ✅ Complex notebook scenarios

### Production Recommendation

**qwen3 is production-ready:**
- ✅ Use qwen3-coder-30b without concerns
- ✅ All SDK features work perfectly
- ✅ No special configuration needed
- ✅ Excellent for ReACT agents
- ✅ Validation enabled works flawlessly

---

## mistral: Perfect SDK Compatibility, Prompting Challenges

### Status
✅ **SDK COMPATIBLE** - Direct tool calling works perfectly
⚠️ **AGENT PROMPTING** - ReACT agent requires optimization

### Investigation Summary

**Initial Report (False Premise):**
User believed tool calls were "returning as text" for mistral, suggesting a parsing/format issue.

**Actual Finding:**
Tool calling works perfectly at SDK level. Issue is **ReACT agent prompting compatibility**, not a technical bug.

### What Works ✅

**1. Raw API Response**
```json
{
  "choices": [{
    "message": {
      "role": "assistant",
      "content": "[THINK]...[/THINK]",
      "tool_calls": [{
        "type": "function",
        "id": "764311257",
        "function": {
          "name": "bash",
          "arguments": "{\"command\":\"expr 2 + 2\"}"
        }
      }]
    }
  }]
}
```
**Status**: ✅ Perfect OpenAI format

**2. Pydantic Validation**
```python
completion = ChatCompletion.model_validate(mistral_json)
# ✅ SUCCESS - No validation errors
```
**Status**: ✅ Models parse correctly

**3. Direct SDK Usage**
```python
client.chat("Use bash to calculate 2 + 2", use_tools=True)
# ✅ Returns: "4"
```
**Status**: ✅ Tool execution works

### What Doesn't Work ❌

**4. ReACT Agent Behavior**
```
Status: AgentStatus.MAX_ITERATIONS
Iterations: 10-20
Tool Calls: 0  ← ❌ PROBLEM
Final Response: ... (empty or incomplete)
```
**Status**: ❌ Agent doesn't trigger tools

### Root Cause

**ReACT Agent Prompting Incompatibility:**

The ReACT agent uses a specific system prompt to instruct the model to use tools. This prompt works for qwen3 but doesn't effectively guide mistral to call tools. Instead, mistral:
1. Generates [THINK] blocks (reasoning)
2. Never invokes tools
3. Hits max_iterations limit
4. Fails without completing the task

### Comparison: qwen3 vs Mistral

| Aspect | qwen3 | Mistral | Verdict |
|--------|-------|---------|---------|
| **Raw API Format** | OpenAI JSON | OpenAI JSON | ✅ Identical |
| **LM Studio Conversion** | XML → JSON | Direct JSON | ✅ Both work |
| **SDK Parsing** | ✅ Works | ✅ Works | ✅ No difference |
| **Tool Execution** | ✅ Works | ✅ Works | ✅ No difference |
| **ReACT Agent** | ✅ Calls tools | ❌ 0 tool calls | ❌ **This is the issue** |

### Production Recommendations

**When to Use Mistral:**
- ✅ Direct SDK usage: `client.chat(use_tools=True)`
- ✅ Custom agent implementations with tailored prompts
- ✅ Single-turn tool calling workflows

**When NOT to Use Mistral:**
- ❌ Current ReACT agent implementation
- ❌ Multi-step agentic workflows (without prompt optimization)

**Future Solutions:**

**Option 1: Model-Specific Prompting**
- Detect model name in ReACT agent
- Use different system prompts for different models
- Optimize prompts per model family

**Option 2: Prompt Tuning for Mistral**
- Research mistral-specific tool calling patterns
- Adjust ReACT prompt to be more explicit
- Test with different phrasing/instructions

**Option 3: Document Limitation**
- Mark mistral as "partial support" for ReACT agents
- Recommend qwen3 for agent workflows
- Document direct `client.chat(use_tools=True)` works fine

---

## granite: XML Format Parsing

### Status
✅ **COMPATIBLE** - SDK parses granite's XML format

### Problem Discovered

**Granite's Format:**
```
<tool_call>
{"name": "bash", "arguments": "{\"command\":\"echo 4\"}"}
</tool_call>
```

**Issue:**
- LM Studio returns this in `content` field
- `tool_calls` array is empty `[]`
- SDK needed to parse this generic XML format

### Solution Implemented

**Enhanced `_parse_text_tool_calls()` in `local_llm_sdk/client.py`:**

**Fixed regex pattern (line 369):**
- Old: `r'<tool_call>\s*(\{[^}]+\})\s*</tool_call>'` ❌ Stopped at first `}`
- New: `r'<tool_call>\s*(.*?)\s*</tool_call>'` ✅ Captures full JSON

**Why it failed initially:**
- Granite's arguments are JSON string: `"arguments": "{\"command\":\"echo 4\"}"`
- Old regex: `[^}]+` stopped at first `}` in arguments
- Only captured: `{"name": "bash", "arguments": "`
- JSON parsing failed silently

**New approach:**
- Capture everything between tags
- Parse as JSON
- Handle dict or string arguments
- Works for nested JSON

### Test Results

✅ **Direct Testing:**
```bash
python /tmp/test_granite_final.py
# Output:
# Test 1: Simple calculation
# Result: The result of 2 + 2 is **4**.
# ✅ Tool executed: bash

# Test 2: More complex task
# Result: Here are the files and directories...
# ✅ Tool executed: bash
```

✅ **Unit Tests**: Added 6 comprehensive tests in `tests/test_client.py`
- `test_parse_xml_tool_calls_basic` ✅
- `test_parse_xml_tool_calls_with_nested_json` ✅
- `test_parse_xml_tool_calls_unregistered_tool` ✅
- `test_parse_xml_tool_calls_multiple` ✅
- `test_parse_xml_tool_calls_malformed` ✅
- `test_parse_xml_and_bracket_formats` ✅

### Production Recommendation

**Granite tool calling now works!** The SDK can parse granite's XML format with nested JSON arguments.

---

## Supported Format Summary

The SDK now supports multiple tool call formats:

1. ✅ **OpenAI JSON** (standard)
   - Standard `tool_calls` array in API response
   - Used by most models

2. ✅ **Bracket format** (legacy)
   - Format: `[TOOL_CALLS]tool[ARGS]{json}`
   - Parsed by SDK for backward compatibility

3. ✅ **XML format** (NEW!)
   - Format: `<tool_call>{"name":"...","arguments":"..."}</tool_call>`
   - Supports granite and similar models
   - Handles nested JSON in arguments

---

## Debugging with LM Studio Logs

### Recommended: LM Studio CLI (Cross-Platform)

```bash
# Stream live logs (works on Mac, Windows, Linux)
lms log stream

# Filter for specific events
lms log stream | grep -E "POST|error|stream"

# Use during tests to see real-time activity
lms log stream &  # Run in background
pytest tests/test_validation_live.py -v
```

### Alternative: Direct File Access (OS-Specific)

- **Mac**: `~/.lmstudio/logs/`
- **Windows/WSL**: `/mnt/c/Users/mahei/.cache/lm-studio/server-logs/YYYY-MM/`

```bash
# Mac
tail -f ~/.lmstudio/logs/main.log

# Windows/WSL
tail -f /mnt/c/Users/mahei/.cache/lm-studio/server-logs/$(date +%Y-%m)/*.log
```

### Capture Both Input and Output

**Output only (see model responses):**
```bash
lms log stream --source model --filter output
```

**Both (full picture):**
```bash
# Terminal 1: Input logs
lms log stream --source model --filter input > input.log

# Terminal 2: Output logs
lms log stream --source model --filter output > output.log
```

### What You'll See

- **Input logs**: Tool schemas sent TO model (varies by model)
- **Output logs**: Model responses (XML for qwen3/granite, JSON for others)
- **API responses**: What SDK receives (always JSON if conversion works)

---

## Key Takeaways

1. **Always test at multiple levels**: API → Parsing → SDK → Application
2. **Verify assumptions**: Original beliefs about format issues were incorrect
3. **Model behavior varies**: Same API spec ≠ same prompting needs
4. **LM Studio handles quirks**: API layer converts model-specific formats to OpenAI JSON
5. **SDK is robust**: Handles multiple format variations gracefully

---

## Testing Strategy

**What to Test:**
- ❌ NOT tool calling parsing (proven to work across models)
- ✅ **ReACT prompting variants with different models**
- ✅ **Model-specific prompt optimization**
- ✅ **Alternative agent patterns for models with prompting challenges**

---

**Investigation Sources:**
- `.scratchpad/qwen3-xml-json-analysis.md` - qwen3 deep dive with 642 request analysis
- `.scratchpad/tool-calling-investigation-2025-10-06.md` - Comprehensive mistral/granite investigation
- `local_llm_sdk/client.py` - XML parsing implementation
- `tests/test_client.py` - Comprehensive XML parsing tests
