"""
Unified LM Studio client with full type safety and tool support.
Combines api_models, tools, and provides a clean interface.
"""

import requests
import json
from typing import List, Optional, Dict, Any, Union
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

    def __init__(self, base_url: str = "http://localhost:1234/v1", model: str = None):
        """
        Initialize LM Studio client.

        Args:
            base_url: Base URL for LM Studio API
            model: Default model to use (can be overridden per request)
        """
        self.base_url = base_url.rstrip('/')
        self.default_model = model
        self.tools = ToolRegistry()

        # API endpoints
        self.endpoints = {
            "models": f"{self.base_url}/models",
            "chat": f"{self.base_url}/chat/completions",
            "completions": f"{self.base_url}/completions",
            "embeddings": f"{self.base_url}/embeddings"
        }

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
        Import tools from a module (like registered_tools).

        Args:
            module: Module containing tool-decorated functions
        """
        # Module import automatically registers tools via decorators
        return self

    def list_models(self) -> ModelList:
        """Get list of available models from LM Studio."""
        response = requests.get(self.endpoints["models"], timeout=5)
        response.raise_for_status()
        return ModelList.model_validate(response.json())

    def chat(
        self,
        messages: Union[str, List[ChatMessage]],
        model: str = None,
        temperature: float = 0.7,
        max_tokens: int = None,
        use_tools: bool = True,
        stream: bool = False,
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
            **kwargs: Additional parameters for ChatCompletionRequest

        Returns:
            String response (simple mode) or ChatCompletion object (detailed mode)
        """
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

        # Handle tool calls if present
        if response.choices[0].message.tool_calls and use_tools:
            response = self._handle_tool_calls(response, messages)

        # Return simple string or full response based on context
        if isinstance(messages[0], str) or len(messages) <= 2:
            # Simple mode: return just the content
            return response.choices[0].message.content
        else:
            # Advanced mode: return full ChatCompletion
            return response

    def _send_request(self, request: ChatCompletionRequest) -> ChatCompletion:
        """Send request to LM Studio and return typed response."""
        response = requests.post(
            self.endpoints["chat"],
            json=request.model_dump(exclude_none=True),
            timeout=30
        )
        response.raise_for_status()
        return ChatCompletion.model_validate(response.json())

    def _handle_tool_calls(
        self,
        response: ChatCompletion,
        original_messages: List[ChatMessage]
    ) -> ChatCompletion:
        """Handle tool calls and get final response."""
        # Build conversation with tool calls
        messages = original_messages.copy()
        messages.append(response.choices[0].message)

        # Execute each tool call
        for tool_call in response.choices[0].message.tool_calls:
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

        return self._send_request(final_request)

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


def quick_chat(query: str, base_url: str = "http://localhost:1234/v1", model: str = "default") -> str:
    """Quick one-off chat without creating a client."""
    client = LocalLLMClient(base_url, model)
    return client.chat_simple(query)