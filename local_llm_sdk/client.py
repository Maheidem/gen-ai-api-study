"""
Unified LM Studio client with full type safety and tool support.
Combines api_models, tools, and provides a clean interface.
"""

import requests
import json
from typing import List, Optional, Dict, Any, Union

# MLflow tracing support (optional, graceful degradation)
try:
    import mlflow
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False
    # Create no-op decorator if MLflow not available
    class mlflow:
        @staticmethod
        def trace(name=None, span_type=None, attributes=None):
            def decorator(func):
                return func
            return decorator

        @staticmethod
        def start_span(name, span_type=None, attributes=None):
            from contextlib import contextmanager
            @contextmanager
            def _span():
                yield
            return _span()
from .models import (
    ChatCompletion,
    ChatCompletionRequest,
    ChatMessage,
    Tool,
    ToolCall,
    create_chat_message,
    create_chat_completion_request,
    ModelList,
    EmbeddingsRequest,
    EmbeddingsResponse
)
from .tools.registry import ToolRegistry


class LocalLLMClient:
    """
    Type-safe client for LM Studio API with integrated tool support.

    Usage:
        client = LocalLLMClient("http://localhost:1234/v1", "model-name")
        client.register_tools_from(registered_tools)
        response = client.chat("What is 2+2?")
    """

    def __init__(self, base_url: str = None, model: str = None, timeout: int = None):
        """
        Initialize Local LLM client.

        Args:
            base_url: Base URL for local LLM API (e.g., LM Studio, Ollama).
                     Falls back to LLM_BASE_URL env var or "http://localhost:1234/v1"
            model: Default model to use. Use "auto" to automatically detect the first available model,
                   or specify a model name explicitly. Falls back to LLM_MODEL env var or "auto"
            timeout: Request timeout in seconds. Falls back to LLM_TIMEOUT env var or 30
        """
        from .config import get_default_config

        # Load config from environment variables
        config = get_default_config()

        # Use provided values or fall back to config
        self.base_url = (base_url or config["base_url"]).rstrip('/')
        self.timeout = timeout if timeout is not None else config["timeout"]
        model = model or config["model"]
        self.tools = ToolRegistry()
        self.last_tool_calls = []  # Track tool calls from last request
        self.last_thinking = ""  # Track thinking blocks from last request

        # API endpoints
        self.endpoints = {
            "models": f"{self.base_url}/models",
            "chat": f"{self.base_url}/chat/completions",
            "completions": f"{self.base_url}/completions",
            "embeddings": f"{self.base_url}/embeddings"
        }

        # Auto-detect model if requested
        if model == "auto":
            try:
                models = self.list_models()
                if models.data and len(models.data) > 0:
                    self.default_model = models.data[0].id
                    print(f"✓ Auto-detected model: {self.default_model}")
                else:
                    self.default_model = None
                    print("⚠ Warning: No models found. Please specify a model or load one in your LLM server.")
            except Exception as e:
                self.default_model = None
                print(f"⚠ Warning: Could not auto-detect model ({str(e)}). You'll need to specify model per request.")
        else:
            self.default_model = model

    def register_tool(self, description: str = ""):
        """
        Decorator to register a tool with the client.

        Usage:
            @client.register_tool("Add two numbers")
            def add(a: float, b: float) -> dict:
                return {"result": a + b}
        """
        return self.tools.register(description)

    def register_tools_from(self, module):
        """
        Import tools from a module or registry.

        Args:
            module: Module containing tool-decorated functions or a ToolRegistry instance
        """
        if module is None:
            # Load built-in tools
            from .tools import registry as global_registry
            # Import builtin to trigger decorator registration
            from .tools import builtin
            # Copy registered tools to our client's registry using public API
            self.tools.copy_from(global_registry.tools)
        elif hasattr(module, 'tools') and hasattr(module.tools, 'copy_from'):
            # It's a module with a tools registry attribute
            self.tools.copy_from(module.tools)
        else:
            # Assume it's a registry directly
            from .tools import registry as global_registry
            from .tools import builtin
            self.tools.copy_from(global_registry.tools)

        return self

    def _extract_thinking(self, content: str) -> tuple[str, str]:
        """
        Extract thinking blocks from model content.

        Args:
            content: Raw content from model that may contain [THINK]...[/THINK] blocks

        Returns:
            tuple of (clean_content, thinking_content)
        """
        import re

        if not content:
            return content, ""

        # Extract thinking blocks
        pattern = r'\[THINK\](.*?)\[/THINK\]'
        thinking_matches = re.findall(pattern, content, re.DOTALL)
        thinking = '\n'.join(match.strip() for match in thinking_matches) if thinking_matches else ""

        # Remove thinking blocks from content
        clean_content = re.sub(pattern, '', content, flags=re.DOTALL).strip()

        return clean_content, thinking

    def _parse_text_tool_calls(self, content: str) -> List[ToolCall]:
        """
        Parse text-based tool calls from content when model uses [TOOL_CALLS]format.

        Extracts tool calls in format: [TOOL_CALLS]tool_name[ARGS]{json_args}

        Args:
            content: Raw content that may contain text-based tool calls

        Returns:
            List of ToolCall objects parsed from text
        """
        import re
        import uuid

        if not content or '[TOOL_CALLS]' not in content:
            return []

        tool_calls = []

        # Pattern to match: [TOOL_CALLS]tool_name[ARGS]{...}
        pattern = r'\[TOOL_CALLS\](\w+)\[ARGS\](\{[^}]+\})'
        matches = re.findall(pattern, content)

        for tool_name, args_json in matches:
            try:
                # Verify this is a registered tool
                if tool_name not in self.tools.list_tools():
                    continue

                # Create a ToolCall object
                from .models import ToolCall, FunctionCall
                tool_call = ToolCall(
                    id=str(uuid.uuid4().hex[:9]),  # Generate ID like LM Studio
                    type="function",
                    function=FunctionCall(
                        name=tool_name,
                        arguments=args_json
                    )
                )
                tool_calls.append(tool_call)

            except Exception:
                # Skip malformed tool calls
                continue

        return tool_calls

    def list_models(self) -> ModelList:
        """Get list of available models from LM Studio."""
        response = requests.get(self.endpoints["models"], timeout=5)
        response.raise_for_status()
        return ModelList.model_validate(response.json())

    @mlflow.trace(name="chat", span_type="CHAT_MODEL")
    def chat(
        self,
        messages: Union[str, List[ChatMessage]],
        model: str = None,
        temperature: float = 0.7,
        max_tokens: int = None,
        use_tools: bool = True,
        stream: bool = False,
        return_full: bool = False,
        include_thinking: bool = False,
        **kwargs
    ) -> Union[str, ChatCompletion]:
        """
        Send chat completion request with automatic tool handling.

        Args:
            messages: Either a string query or list of ChatMessage objects
            model: Model to use (overrides default)
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            use_tools: Whether to include registered tools
            stream: Whether to stream the response
            return_full: Force returning full ChatCompletion object
            include_thinking: Whether to include thinking blocks in response
            **kwargs: Additional parameters for ChatCompletionRequest

        Returns:
            String response (simple mode) or ChatCompletion object (detailed mode)
        """
        # Clear tool calls and thinking from previous request
        self.last_tool_calls = []
        self.last_thinking = ""

        # Convert string to messages if needed
        if isinstance(messages, str):
            messages = [
                create_chat_message("system", "You are a helpful assistant with access to tools."),
                create_chat_message("user", messages)
            ]

        # Build request
        request = create_chat_completion_request(
            model=model or self.default_model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            stream=stream,
            **kwargs
        )

        # Add tools if available and requested
        if use_tools and self.tools.list_tools():
            request.tools = self.tools.get_schemas()
            request.tool_choice = "auto"

        # Send request
        response = self._send_request(request)

        # Check for tool calls - structured or text-based
        message = response.choices[0].message
        has_structured_tools = message.tool_calls and len(message.tool_calls) > 0

        # If no structured tool calls but content has [TOOL_CALLS], parse them
        if not has_structured_tools and use_tools and message.content:
            text_tool_calls = self._parse_text_tool_calls(message.content)
            if text_tool_calls:
                # Add parsed tool calls to the message
                message.tool_calls = text_tool_calls
                has_structured_tools = True

        # Handle tool calls if present
        if has_structured_tools and use_tools:
            # Store tool calls before they're lost in the second request
            self.last_tool_calls = response.choices[0].message.tool_calls

            # Extract thinking from the first response (which may have tool calls + thinking)
            first_content = response.choices[0].message.content or ""
            _, first_thinking = self._extract_thinking(first_content)
            if first_thinking:
                self.last_thinking = first_thinking

            response = self._handle_tool_calls(response, messages)

        # Extract thinking blocks from final response content
        content = response.choices[0].message.content or ""
        clean_content, final_thinking = self._extract_thinking(content)

        # Combine thinking from first response and final response
        if final_thinking:
            if self.last_thinking:
                self.last_thinking += "\n\n" + final_thinking
            else:
                self.last_thinking = final_thinking

        # Update response content (remove thinking blocks unless requested)
        if not include_thinking and self.last_thinking:
            # Create a copy of the response with cleaned content
            response.choices[0].message.content = clean_content
        elif include_thinking and self.last_thinking:
            # Include thinking with clear separation
            if clean_content:
                response.choices[0].message.content = f"**Thinking:**\n{self.last_thinking}\n\n**Response:**\n{clean_content}"
            else:
                response.choices[0].message.content = f"**Thinking:**\n{self.last_thinking}"

        # Return simple string or full response based on context
        if not return_full and (isinstance(messages[0], str) or len(messages) <= 2):
            # Simple mode: return just the content
            return response.choices[0].message.content
        else:
            # Advanced mode: return full ChatCompletion
            return response

    @mlflow.trace(name="send_request", span_type="LLM")
    def _send_request(self, request: ChatCompletionRequest, retry_count: int = 3) -> ChatCompletion:
        """
        Send request to local LLM API with automatic retry on connection errors.

        Args:
            request: The chat completion request to send
            retry_count: Number of retry attempts on connection failures (default: 3)

        Returns:
            ChatCompletion response from the API

        Raises:
            ConnectionError or Timeout after all retries are exhausted
        """
        import time
        from requests.exceptions import ConnectionError, Timeout

        last_error = None

        for attempt in range(retry_count):
            try:
                response = requests.post(
                    self.endpoints["chat"],
                    json=request.model_dump(exclude_none=True),
                    timeout=self.timeout
                )
                response.raise_for_status()
                return ChatCompletion.model_validate(response.json())

            except (ConnectionError, Timeout) as e:
                last_error = e
                if attempt < retry_count - 1:
                    wait_time = 2 ** attempt  # Exponential backoff: 1, 2, 4 seconds
                    print(f"⚠ Connection failed, retrying in {wait_time}s... (attempt {attempt + 1}/{retry_count})")
                    time.sleep(wait_time)
                else:
                    print(f"✗ Connection failed after {retry_count} attempts")
                    raise

        # Should never reach here, but for type safety
        if last_error:
            raise last_error

    @mlflow.trace(name="handle_tool_calls", span_type="AGENT")
    def _handle_tool_calls(
        self,
        response: ChatCompletion,
        original_messages: List[ChatMessage]
    ) -> ChatCompletion:
        """Handle tool calls and get final response."""
        # Store original tool calls to preserve them
        original_tool_calls = response.choices[0].message.tool_calls

        # Build conversation with tool calls
        messages = original_messages.copy()
        messages.append(response.choices[0].message)

        # Execute each tool call
        for tool_call in response.choices[0].message.tool_calls:
            # Create a span for each tool execution
            with mlflow.start_span(
                name=f"tool_{tool_call.function.name}",
                span_type="TOOL"
            ) as span:
                # Execute tool
                result = self.tools.execute(
                    tool_call.function.name,
                    json.loads(tool_call.function.arguments)
                )

                # Add tool response to conversation
                tool_message = create_chat_message("tool", result)
                tool_message.tool_call_id = tool_call.id
                messages.append(tool_message)

        # Get final response
        final_request = create_chat_completion_request(
            model=self.default_model,
            messages=messages,
            temperature=0.7
        )

        final_response = self._send_request(final_request)

        # Preserve the tool calls from the first response on the final response
        # This allows tracking and observability even after automatic tool execution
        final_response.choices[0].message.tool_calls = original_tool_calls

        return final_response

    def embeddings(
        self,
        input: Union[str, List[str]],
        model: str = None
    ) -> EmbeddingsResponse:
        """
        Generate embeddings for text.

        Args:
            input: Text or list of texts to embed
            model: Embedding model to use

        Returns:
            EmbeddingsResponse with embedding vectors
        """
        request = EmbeddingsRequest(
            input=input,
            model=model or self.default_model
        )

        response = requests.post(
            self.endpoints["embeddings"],
            json=request.model_dump(exclude_none=True),
            timeout=30
        )
        response.raise_for_status()

        return EmbeddingsResponse.model_validate(response.json())

    def chat_simple(self, query: str) -> str:
        """
        Simple chat without tool support or conversation history.

        Args:
            query: User query

        Returns:
            Model's response as string
        """
        return self.chat(query, use_tools=False)

    def chat_with_history(
        self,
        query: str,
        history: List[ChatMessage],
        **kwargs
    ) -> tuple[str, List[ChatMessage]]:
        """
        Chat with conversation history.

        Args:
            query: New user query
            history: Previous conversation messages
            **kwargs: Additional parameters

        Returns:
            Tuple of (response, updated_history)
        """
        # Add new user message to history
        new_history = history.copy()
        new_history.append(create_chat_message("user", query))

        # Get response
        response = self.chat(new_history, **kwargs)

        # Add assistant response to history
        if isinstance(response, ChatCompletion):
            new_history.append(response.choices[0].message)
            return response.choices[0].message.content, new_history
        else:
            new_history.append(create_chat_message("assistant", response))
            return response, new_history

    def __repr__(self) -> str:
        tools_count = len(self.tools.list_tools())
        return f"LocalLLMClient(base_url='{self.base_url}', model='{self.default_model}', tools={tools_count})"


# Convenience functions
def create_client(base_url: str = "http://localhost:1234/v1", model: str = None) -> LocalLLMClient:
    """Create a new local LLM client."""
    return LocalLLMClient(base_url, model)


def quick_chat(query: str, base_url: str = "http://localhost:1234/v1", model: str = "auto") -> str:
    """
    Quick one-off chat with auto-configuration and built-in tools.

    This convenience function creates a client, auto-detects the model if needed,
    loads built-in tools, and returns a simple string response.

    Args:
        query: The message to send
        base_url: Base URL for the local LLM API
        model: Model to use ("auto" to auto-detect)

    Returns:
        String response from the model

    Example:
        >>> from local_llm_sdk import quick_chat
        >>> response = quick_chat("What's 2+2?")
    """
    client = LocalLLMClient(base_url, model)
    # Auto-load built-in tools for better responses
    client.register_tools_from(None)
    return client.chat(query, use_tools=True)