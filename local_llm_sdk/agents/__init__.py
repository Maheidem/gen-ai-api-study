"""
Agent module for Local LLM SDK.

Provides agent patterns for common workflows like ReACT, Chain-of-Thought, etc.

Usage:
    from local_llm_sdk.agents import ReACT, AgentResult

    agent = ReACT(client)
    result = agent.run("Your task here", max_iterations=15)

Or use the convenience method on the client:
    result = client.react("Your task here")
"""

from .models import AgentResult, AgentStatus
from .base import BaseAgent
from .react import ReACT

__all__ = [
    # Data models
    "AgentResult",
    "AgentStatus",

    # Base class
    "BaseAgent",

    # Agent implementations
    "ReACT",
]