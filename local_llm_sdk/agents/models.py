"""
Data models for agent results and configuration.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
from enum import Enum


class AgentStatus(str, Enum):
    """Status of agent execution."""
    SUCCESS = "success"
    FAILED = "failed"
    MAX_ITERATIONS = "max_iterations"
    STOPPED = "stopped"
    ERROR = "error"


@dataclass
class AgentResult:
    """
    Result of an agent execution.

    Attributes:
        status: Status of the execution
        iterations: Number of iterations completed
        final_response: Final response content from the agent
        conversation: Full conversation history (messages)
        error: Error message if status is ERROR
        metadata: Additional metadata about the execution
    """
    status: AgentStatus
    iterations: int
    final_response: str
    conversation: List[Any] = field(default_factory=list)
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def success(self) -> bool:
        """Check if agent completed successfully."""
        return self.status == AgentStatus.SUCCESS

    def __repr__(self) -> str:
        status_emoji = "✓" if self.success else "✗"
        return (f"AgentResult({status_emoji} status={self.status.value}, "
                f"iterations={self.iterations}, "
                f"response_length={len(self.final_response)})")