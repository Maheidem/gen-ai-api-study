"""
ReACT (Reasoning + Acting) agent implementation.

ReACT combines reasoning traces and task-specific actions in an interleaved manner,
allowing for greater synergy between the two. The agent can use tools to gather information
and take actions while reasoning about the task.
"""

from typing import Optional, Callable, List
from ..models import create_chat_message, ChatMessage
from .base import BaseAgent
from .models import AgentResult, AgentStatus


class ReACT(BaseAgent):
    """
    ReACT agent that combines reasoning and acting with tool use.

    The agent follows the ReACT pattern:
    1. Reasoning: Think about what needs to be done
    2. Acting: Use tools to take actions
    3. Observation: Observe the results
    4. Repeat until task is complete

    Usage:
        client = create_client_with_tools()
        agent = ReACT(client)

        result = agent.run(
            task="Implement a sorting algorithm",
            max_iterations=15,
            stop_condition=lambda resp: "TASK_COMPLETE" in resp.content.upper()
        )

        print(f"Completed in {result.iterations} iterations")
    """

    DEFAULT_SYSTEM_PROMPT = """AI assistant with tool access. Be CONCISE.

Tools:
- execute_python: Test/compute in isolated environment (no file I/O to project)
- filesystem_operation: Create/read/write files in project directory
- math_calculator, text_transformer, char_counter, get_weather

CRITICAL RULES:
1. Use EXACTLY ONE tool per response - NEVER call multiple tools at once
2. After EVERY tool call, wait for the result before deciding next step
3. Break complex tasks into small, focused steps across multiple iterations
4. Think step-by-step: explain → call tool → observe → explain → next tool

Instructions:
- execute_python for testing/computing → returns output
- filesystem_operation for file writes → saves to project
- After each tool result, explain what you learned before next action
- When task is complete, state the FINAL ANSWER clearly, then say TASK_COMPLETE

Example (note: each line is a SEPARATE response after tool result):
User: Calculate 5 factorial, convert to uppercase, count characters
You: I'll calculate 5 factorial first. [calls math_calculator ONLY]
You: Got 120. Now converting to uppercase... [calls text_transformer ONLY]
You: Got "120". Now counting characters... [calls char_counter ONLY]
You: The answer is: The text "120" has 3 characters. TASK_COMPLETE

Quality over speed. ONE tool at a time. Multiple iterations expected."""

    def __init__(self, client, name: str = "ReACT", system_prompt: Optional[str] = None):
        """
        Initialize ReACT agent.

        Args:
            client: LocalLLMClient instance
            name: Name for this agent instance
            system_prompt: Custom system prompt (uses default if None)
        """
        super().__init__(client, name)
        self.system_prompt = system_prompt or self.DEFAULT_SYSTEM_PROMPT

    def _execute(
        self,
        task: str,
        max_iterations: int = 15,
        stop_condition: Optional[Callable[[ChatMessage], bool]] = None,
        temperature: float = 0.7,
        verbose: bool = True
    ) -> AgentResult:
        """
        Execute the ReACT agent loop.

        Args:
            task: The task description/prompt
            max_iterations: Maximum number of iterations to run
            stop_condition: Optional function that returns True when task is complete
            temperature: Sampling temperature for the model
            verbose: Whether to print progress information

        Returns:
            AgentResult with execution details
        """
        # Initialize conversation
        messages: List[ChatMessage] = [
            create_chat_message("system", self.system_prompt),
            create_chat_message("user", task)
        ]

        if verbose:
            print(f"{'='*80}")
            print(f"{self.name} Agent: Starting task")
            print(f"Max iterations: {max_iterations}")
            print(f"Task: {task[:100]}{'...' if len(task) > 100 else ''}")
            print(f"{'='*80}\n")

        # Run ReACT loop
        for iteration in range(max_iterations):
            if verbose:
                print(f"\nIteration {iteration + 1}/{max_iterations}")
                print("-" * 40)

            try:
                # Get response from model (automatic tool handling)
                response = self.client.chat(
                    messages=messages,
                    use_tools=True,
                    return_full_response=True,
                    temperature=temperature
                )

                # Extract assistant message
                assistant_message = response.choices[0].message
                content = assistant_message.content or ""

                # Show progress
                if verbose:
                    content_preview = content[:150] + "..." if len(content) > 150 else content
                    print(f"Response: {content_preview}")

                    # Show tool calls if any
                    if hasattr(assistant_message, 'tool_calls') and assistant_message.tool_calls:
                        print(f"Tools used: {len(assistant_message.tool_calls)}")
                        for tc in assistant_message.tool_calls:
                            print(f"  - {tc.function.name}")

                # Add to conversation history - use full conversation state if available
                # This includes tool result messages that were created during automatic tool execution
                if hasattr(self.client, 'last_conversation_additions') and self.client.last_conversation_additions:
                    # Extend with ALL messages: assistant_with_tools + tool_results + final_assistant
                    messages.extend(self.client.last_conversation_additions)
                else:
                    # Fallback: just append assistant message (no tools were used)
                    messages.append(assistant_message)

                # Check stop conditions
                if self._should_stop(content, stop_condition):
                    if verbose:
                        print(f"\n{'='*80}")
                        print(f"✓ Task completed successfully in {iteration + 1} iterations")
                        print(f"{'='*80}")

                    # Extract final answer (remove TASK_COMPLETE marker)
                    final_answer = self._extract_final_answer(content)

                    return AgentResult(
                        status=AgentStatus.SUCCESS,
                        iterations=iteration + 1,
                        final_response=final_answer,
                        conversation=messages,
                        metadata={
                            "total_tool_calls": self._count_tool_calls(messages),
                            "final_iteration": iteration + 1
                        }
                    )

            except Exception as e:
                if verbose:
                    print(f"\n✗ Error in iteration {iteration + 1}: {e}")

                return AgentResult(
                    status=AgentStatus.ERROR,
                    iterations=iteration + 1,
                    final_response="",
                    conversation=messages,
                    error=str(e),
                    metadata={
                        "failed_at_iteration": iteration + 1,
                        "error_type": type(e).__name__
                    }
                )

        # Max iterations reached without completion
        if verbose:
            print(f"\n{'='*80}")
            print(f"⚠ Reached maximum iterations ({max_iterations}) without completion")
            print(f"{'='*80}")

        return AgentResult(
            status=AgentStatus.MAX_ITERATIONS,
            iterations=max_iterations,
            final_response=messages[-1].content if messages else "",
            conversation=messages,
            metadata={
                "max_iterations_reached": True,
                "total_tool_calls": self._count_tool_calls(messages)
            }
        )

    def _should_stop(
        self,
        content: str,
        stop_condition: Optional[Callable[[str], bool]] = None
    ) -> bool:
        """
        Check if the agent should stop.

        Args:
            content: The response content
            stop_condition: Optional custom stop condition function

        Returns:
            True if agent should stop
        """
        # Check custom stop condition
        if stop_condition and stop_condition(content):
            return True

        # Default: check for TASK_COMPLETE
        if "TASK_COMPLETE" in content.upper():
            return True

        return False

    def _count_tool_calls(self, messages: List[ChatMessage]) -> int:
        """
        Count total tool calls in conversation.

        Args:
            messages: List of chat messages

        Returns:
            Total number of tool calls
        """
        count = 0
        for msg in messages:
            if hasattr(msg, 'tool_calls') and msg.tool_calls:
                count += len(msg.tool_calls)
        return count

    def _extract_final_answer(self, content: str) -> str:
        """
        Extract the final answer from content, removing TASK_COMPLETE marker.

        Args:
            content: The response content

        Returns:
            Cleaned final answer
        """
        import re

        # Remove TASK_COMPLETE (case insensitive) and surrounding whitespace
        cleaned = re.sub(r'\s*TASK_COMPLETE\s*$', '', content, flags=re.IGNORECASE)
        cleaned = re.sub(r'\s*TASK_COMPLETE\s*', '', cleaned, flags=re.IGNORECASE)

        # If nothing left after removing TASK_COMPLETE, return original
        if not cleaned.strip():
            return content

        return cleaned.strip()