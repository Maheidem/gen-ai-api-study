"""
Base agent class for all agent implementations.
"""

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Optional, Dict, Any

# MLflow tracing support (optional, graceful degradation)
try:
    import mlflow
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False
    class mlflow:
        @staticmethod
        def trace(name=None, span_type=None, attributes=None):
            def decorator(func):
                return func
            return decorator

from .models import AgentResult, AgentStatus

if TYPE_CHECKING:
    from ..client import LocalLLMClient


class BaseAgent(ABC):
    """
    Abstract base class for all agents.

    Agents encapsulate specific interaction patterns (ReACT, Chain-of-Thought, etc.)
    and handle MLflow tracing automatically. Subclasses implement the `_execute()`
    method with their specific logic.

    Usage:
        class MyAgent(BaseAgent):
            def _execute(self, task: str, **kwargs) -> AgentResult:
                # Your agent logic here
                return AgentResult(...)

        agent = MyAgent(client, name="my-agent")
        result = agent.run("Do something")
    """

    def __init__(self, client: 'LocalLLMClient', name: Optional[str] = None):
        """
        Initialize the agent.

        Args:
            client: LocalLLMClient instance with tools configured
            name: Name for this agent (defaults to class name)
        """
        self.client = client
        self.name = name or self.__class__.__name__
        self.metadata: Dict[str, Any] = {}

    def run(self, task: str, **kwargs) -> AgentResult:
        """
        Execute the agent on the given task.

        This method handles the MLflow tracing wrapper and conversation context.
        Subclasses should implement `_execute()` with their specific logic.

        Args:
            task: The task description/prompt for the agent
            **kwargs: Additional arguments passed to _execute()

        Returns:
            AgentResult with execution details

        Example:
            agent = ReACT(client)
            result = agent.run("Implement sorting algorithm", max_iterations=15)
            print(f"Completed in {result.iterations} iterations")
        """
        # Create a safe task name for the conversation/trace
        safe_task_name = self._sanitize_task_name(task)
        conversation_name = f"{self.name}_{safe_task_name}"

        # Use conversation context for unified tracing
        with self.client.conversation(conversation_name):
            try:
                result = self._execute(task, **kwargs)
                # Add agent name to metadata
                result.metadata["agent_name"] = self.name
                result.metadata["agent_class"] = self.__class__.__name__
                return result
            except Exception as e:
                # Handle any errors gracefully
                return AgentResult(
                    status=AgentStatus.ERROR,
                    iterations=0,
                    final_response="",
                    error=str(e),
                    metadata={
                        "agent_name": self.name,
                        "agent_class": self.__class__.__name__,
                        "error_type": type(e).__name__
                    }
                )

    @abstractmethod
    def _execute(self, task: str, **kwargs) -> AgentResult:
        """
        Execute the agent's specific logic.

        Subclasses must implement this method with their agent behavior.

        Args:
            task: The task description/prompt
            **kwargs: Additional agent-specific parameters

        Returns:
            AgentResult with execution details
        """
        pass

    def _sanitize_task_name(self, task: str, max_length: int = 30) -> str:
        """
        Create a safe name from task description for tracing.

        Args:
            task: Task description
            max_length: Maximum length of the sanitized name

        Returns:
            Sanitized task name
        """
        # Take first line, strip whitespace, limit length
        first_line = task.split('\n')[0].strip()
        if len(first_line) > max_length:
            first_line = first_line[:max_length] + "..."

        # Replace spaces with underscores, remove special chars
        safe_name = ''.join(c if c.isalnum() or c in '_- ' else '' for c in first_line)
        safe_name = safe_name.replace(' ', '_')

        return safe_name or "unnamed_task"

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name='{self.name}')"