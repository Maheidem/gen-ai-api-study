# local_llm_sdk/

## Purpose
Core SDK package providing type-safe Python interface for local LLM APIs (OpenAI-compatible servers like LM Studio, Ollama).

## Contents
- `__init__.py` - Package exports and convenience functions
- `client.py` - `LocalLLMClient` main class with chat, tools, and conversation management
- `models.py` - Pydantic models following OpenAI API specification
- `config.py` - Configuration management (env vars, defaults)
- `agents/` - Production-ready agent framework (ReACT, etc.)
- `tools/` - Tool/function calling system with registry and built-in tools
- `utils/` - Shared utility functions

## Relationships
- **Parent**: Root package, imported by users as `from local_llm_sdk import LocalLLMClient`
- **Children**: Agents use client.py, tools register with client, utils provide helpers
- **External**: Interacts with local LLM servers via HTTP (OpenAI-compatible endpoints)

## Getting Started
1. Read `__init__.py` first - shows public API and convenience functions
2. Then `client.py:58-110` - LocalLLMClient.__init__() and configuration
3. Check `models.py` for understanding request/response types
4. Explore `agents/` for high-level agent patterns (ReACT)
5. Look at `tools/` to understand function calling system

## Key Features
- Automatic tool handling with `chat(use_tools=True)`
- MLflow tracing integration for observability
- Full conversation state management (tool results preserved)
- 300s default timeout for local models
- Thinking block extraction for reasoning models
