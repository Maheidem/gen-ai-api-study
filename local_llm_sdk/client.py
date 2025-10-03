"""
Unified LM Studio client with full type safety and tool support.
Combines api_models, tools, and provides a clean interface.
"""

import requests
import json
from typing import List, Optional, Dict, Any, Union
from contextlib import contextmanager

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
from .utils.streaming_validator import StreamingValidator


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
            timeout: Request timeout in seconds. Falls back to LLM_TIMEOUT env var or 300.
                     Local models may need longer timeouts (120-600s) for complex tasks
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
        self.last_conversation_additions = []  # Track additional messages from tool execution

        # Initialize validation and recovery systems
        # Initialize streaming validation (simplified from 950 lines to 150 lines)
        self.config = config
        self.enable_validation = config.get("enable_validation", False)  # Disabled by default

        if self.enable_validation:
            check_interval = config.get("validation_check_interval", 20)
            self.streaming_validator = StreamingValidator(check_interval=check_interval)
        else:
            self.streaming_validator = None

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

    @contextmanager
    def conversation(self, name: str = "conversation"):
        """
        Context manager for multi-turn conversations with unified tracing.

        Groups multiple chat() calls under a single parent trace for better observability
        in MLflow. This is useful for agent loops, chat sessions, or any multi-turn interaction.

        Args:
            name: Name for the conversation trace (default: "conversation")

        Usage:
            with client.conversation("react-agent-task"):
                for i in range(10):
                    response = client.chat(messages, use_tools=True)
                    messages.append(response.choices[0].message)

        This creates a hierarchy in MLflow:
            conversation
            ├─ chat (iteration 1)
            │  ├─ send_request
            │  └─ handle_tool_calls
            ├─ chat (iteration 2)
            ...
        """
        # Create a parent span for grouping chat calls
        with mlflow.start_span(name=name, span_type="CHAIN"):
            # Update the current trace to set a proper request preview/name
            # This ensures MLflow UI shows the conversation name instead of "null"
            try:
                mlflow.update_current_trace(
                    request_preview=f"Conversation: {name}",
                    tags={"conversation_name": name, "trace_type": "grouped_conversation"}
                )
            except Exception:
                # If update fails (no active trace, etc.), silently continue
                # The span name will still provide some organization
                pass

            yield self

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

    def register_tools(self, tools: List[callable]) -> 'LocalLLMClient':
        """
        Register multiple tools at once from a list of functions.

        Args:
            tools: List of callable functions to register as tools

        Usage:
            def add(a: float, b: float) -> dict:
                \"\"\"Add two numbers\"\"\"
                return {"result": a + b}

            def multiply(a: float, b: float) -> dict:
                \"\"\"Multiply two numbers\"\"\"
                return {"result": a * b}

            client.register_tools([add, multiply])
        """
        for func in tools:
            # Use the function's docstring as description
            description = func.__doc__ or f"Function: {func.__name__}"
            description = description.strip()

            # Manually register the function
            self.tools._tools[func.__name__] = func
            schema = self.tools._generate_schema(func, description)
            self.tools._schemas.append(schema)

        return self

    def print_tool_calls(self, detailed: bool = False):
        """
        Print a summary of tool calls from the last chat request.

        Args:
            detailed: If True, show full arguments and results. If False, show summary only.

        Usage:
            response = client.chat("What is 5 * 10?")
            client.print_tool_calls()  # Shows: "🔧 Called bash(command="python -c 'print(5 * 10)'") → result=50"
        """
        if not self.last_tool_calls:
            print("ℹ️  No tools were called in the last request")
            return

        print(f"\n🔧 Tool Execution Summary ({len(self.last_tool_calls)} call{'s' if len(self.last_tool_calls) > 1 else ''}):")
        print("=" * 70)

        for i, tool_call in enumerate(self.last_tool_calls, 1):
            func_name = tool_call.function.name

            # Parse arguments
            import json
            try:
                args = json.loads(tool_call.function.arguments)
            except:
                args = tool_call.function.arguments

            # Find the result from conversation additions
            result = None
            for msg in self.last_conversation_additions:
                if hasattr(msg, 'tool_call_id') and msg.tool_call_id == tool_call.id:
                    try:
                        result = json.loads(msg.content)
                    except:
                        result = msg.content
                    break

            # Format output
            if detailed:
                print(f"\n[{i}] {func_name}")
                print(f"    Arguments: {json.dumps(args, indent=6)}")
                if result:
                    print(f"    Result: {json.dumps(result, indent=6)}")
            else:
                # Compact format
                args_str = ", ".join(f"{k}={v}" for k, v in args.items()) if isinstance(args, dict) else str(args)
                result_str = ""
                if result and isinstance(result, dict):
                    # Show the most relevant result field (priority order)
                    if 'captured_result' in result:
                        result_str = f" → captured_result={result['captured_result']}"
                    elif 'result' in result:
                        result_str = f" → result={result['result']}"
                    elif 'error' in result:
                        result_str = f" → ❌ {result['error']}"
                    elif 'success' in result and not result['success']:
                        result_str = f" → success=False"
                    else:
                        # Show first key-value pair
                        key, val = next(iter(result.items()))
                        result_str = f" → {key}={val}"

                print(f"  [{i}] {func_name}({args_str}){result_str}")

        print("=" * 70 + "\n")

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

        # Extract thinking blocks - use negative lookahead to avoid matching across multiple [THINK] tags
        pattern = r'\[THINK\]((?:(?!\[THINK\]).)*?)\[/THINK\]'
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
        tool_choice: str = "auto",
        stream: bool = None,  # None = use config default
        return_full_response: bool = False,
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
            tool_choice: Control tool usage: "auto" (model decides), "required" (force tool use),
                        "none" (no tools), or dict with {"type": "function", "function": {"name": "tool_name"}}
            stream: Whether to stream the response
            return_full_response: Force returning full ChatCompletion object instead of just content string
            include_thinking: Whether to include thinking blocks in response
            **kwargs: Additional parameters for ChatCompletionRequest.
                     Special kwargs when messages is a string:
                     - system: Custom system prompt (default: "You are a helpful assistant with access to tools.")

        Returns:
            String response (simple mode) or ChatCompletion object (detailed mode)

        Note:
            For reasoning models (like Magistral), use tool_choice="required" to force immediate tool usage
            and bypass internal reasoning. This prevents the model from solving simple problems mentally.
        """
        # Clear tool calls, thinking, and conversation additions from previous request
        self.last_tool_calls = []
        self.last_thinking = ""
        self.last_conversation_additions = []

        # Track if original input was a simple string (for return type decision)
        was_simple_string = isinstance(messages, str)

        # Convert string to messages if needed
        if was_simple_string:
            # Extract system prompt from kwargs if provided, otherwise use default
            system_prompt = kwargs.pop('system', "You are a helpful assistant with access to tools.")
            messages = [
                create_chat_message("system", system_prompt),
                create_chat_message("user", messages)
            ]

        # Build request - require model to be specified
        selected_model = model or self.default_model
        if not selected_model:
            raise ValueError(
                "No model specified. Please provide a model parameter or set a default model during client initialization.\n"
                "Example: client.chat('Hello', model='your-model-name')\n"
                "Or: client = LocalLLMClient(base_url='...', model='your-model-name')"
            )

        # Safety: If validation enabled and no max_tokens set, use reasonable limit to prevent runaway generation
        if self.enable_validation and max_tokens is None:
            max_tokens = 2048  # Prevent 37,000 token repetition loops
            import logging
            logger = logging.getLogger(__name__)
            logger.debug(f"Validation enabled: Setting max_tokens={max_tokens} to prevent runaway generation")

        # Use config default for stream if not explicitly provided
        if stream is None:
            stream = self.config.get("stream", False)

        request = create_chat_completion_request(
            model=selected_model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            stream=stream,
            **kwargs
        )

        # Add tools if available and requested
        if use_tools and self.tools.list_tools():
            request.tools = self.tools.get_schemas()
            request.tool_choice = tool_choice

        # Send request
        response = self._send_request(request)

        # Simplified validation (replaces 950-line system with 150 lines)
        if self.enable_validation and self.streaming_validator:
            content = response.choices[0].message.content or ""

            # Run validation checks on complete response
            result = self.streaming_validator.add_chunk(content)
            final_result = self.streaming_validator.finalize()

            if result.should_stop or final_result.should_stop:
                error = result if result.should_stop else final_result

                # Fail fast with helpful error message
                print(f"\n🚨 VALIDATION ERROR: {error.error_type}")
                print(f"Model: {selected_model}")
                print(f"Details: {error.details}")
                print(f"Response preview: {content[:200]}...")
                print(f"\n💡 Suggestion: Try a different model (e.g., mistralai/magistral-small-2509)")

                raise ValueError(
                    f"Model response format incompatible: {error.error_type}\n"
                    f"This model may not support the expected format.\n"
                    f"Suggestion: Use a different model or disable validation."
                )

            # Reset validator for next request
            self.streaming_validator.reset()

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

            # Get both response and full conversation state
            response, full_conversation_messages = self._handle_tool_calls(
                response,
                messages,
                tool_choice=tool_choice
            )

            # Store the additional messages that were created during tool execution
            # These are: assistant_with_tool_calls + tool_result_messages + final_assistant
            self.last_conversation_additions = full_conversation_messages[len(messages):]

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
        if return_full_response:
            # User explicitly requested full response
            return response
        elif was_simple_string or len(messages) <= 2:
            # Simple string input OR short conversation (system+user) -> return simple string output
            # This maintains backward compatibility for simple use cases
            return response.choices[0].message.content
        else:
            # Complex message list input -> return full ChatCompletion for programmatic use
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
        # If streaming is requested, use streaming handler
        if request.stream:
            return self._send_streaming_request(request, retry_count)

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

    @mlflow.trace(name="send_streaming_request", span_type="LLM")
    def _send_streaming_request(self, request: ChatCompletionRequest, retry_count: int = 3) -> ChatCompletion:
        """
        Send streaming request to local LLM API with SSE parsing and validation.

        Args:
            request: The chat completion request to send (with stream=True)
            retry_count: Number of retry attempts on connection failures (default: 3)

        Returns:
            ChatCompletion response assembled from streamed chunks

        Raises:
            ConnectionError or Timeout after all retries are exhausted
            ValueError: If validation detects an error during streaming
        """
        import time
        from requests.exceptions import ConnectionError, Timeout

        last_error = None

        for attempt in range(retry_count):
            try:
                # Make streaming request
                response = requests.post(
                    self.endpoints["chat"],
                    json=request.model_dump(exclude_none=True),
                    timeout=self.timeout,
                    stream=True  # Enable streaming
                )
                response.raise_for_status()

                # Parse SSE stream
                accumulated_content = ""
                finish_reason = None
                model_name = None
                chunk_count = 0

                # Reset validator if enabled
                if self.enable_validation and self.streaming_validator:
                    self.streaming_validator.reset()

                for line in response.iter_lines():
                    if not line:
                        continue

                    # Decode bytes to string
                    line_str = line.decode('utf-8') if isinstance(line, bytes) else line

                    # SSE format: "data: {json}"
                    if line_str.startswith('data: '):
                        data_str = line_str[6:]  # Remove "data: " prefix

                        # Check for stream end
                        if data_str.strip() == '[DONE]':
                            break

                        # Parse chunk JSON
                        try:
                            chunk_data = json.loads(data_str)
                        except json.JSONDecodeError:
                            # Skip malformed chunks
                            continue

                        # Extract model name from first chunk
                        if chunk_count == 0 and 'model' in chunk_data:
                            model_name = chunk_data['model']

                        # Extract content delta
                        if 'choices' in chunk_data and len(chunk_data['choices']) > 0:
                            choice = chunk_data['choices'][0]

                            # Get content delta
                            delta = choice.get('delta', {})
                            content_delta = delta.get('content', '')

                            if content_delta:
                                accumulated_content += content_delta
                                chunk_count += 1

                                # Run validation if enabled
                                if self.enable_validation and self.streaming_validator:
                                    validation_result = self.streaming_validator.add_chunk(content_delta)

                                    if validation_result.should_stop:
                                        # Early termination!
                                        error_msg = f"🚨 VALIDATION ERROR: {validation_result.error_type}"
                                        if validation_result.details:
                                            error_msg += f"\n📋 Details: {validation_result.details}"
                                        error_msg += f"\n\n💡 TIP: This is EARLY TERMINATION during streaming (stopped after {chunk_count} chunks)"

                                        # Close the response stream
                                        response.close()

                                        raise ValueError(error_msg)

                            # Get finish reason
                            if 'finish_reason' in choice and choice['finish_reason']:
                                finish_reason = choice['finish_reason']

                # Assemble final ChatCompletion
                # Use the model from chunks or request
                final_model = model_name or request.model or self.default_model

                # Create message
                from .models import ChatMessage, ChatCompletionChoice, CompletionUsage

                message = ChatMessage(
                    role="assistant",
                    content=accumulated_content
                )

                choice = ChatCompletionChoice(
                    index=0,
                    message=message,
                    finish_reason=finish_reason or "stop"
                )

                # Create final response
                completion = ChatCompletion(
                    id=f"chatcmpl-streaming-{int(time.time())}",
                    object="chat.completion",
                    created=int(time.time()),
                    model=final_model,
                    choices=[choice],
                    usage=CompletionUsage(
                        prompt_tokens=0,  # Not available in streaming
                        completion_tokens=chunk_count,
                        total_tokens=chunk_count
                    )
                )

                return completion

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
        original_messages: List[ChatMessage],
        tool_choice: str = "auto"
    ) -> tuple[ChatCompletion, List[ChatMessage]]:
        """
        Handle tool calls and get final response with full conversation state.

        Returns:
            tuple: (final_response, complete_messages_list)
                - final_response: The ChatCompletion after tool execution
                - complete_messages_list: Full conversation including tool results
        """
        # Store original tool calls to preserve them
        original_tool_calls = response.choices[0].message.tool_calls

        # Build conversation with tool calls
        messages = original_messages.copy()
        messages.append(response.choices[0].message)

        # Execute each tool call
        for tool_call in response.choices[0].message.tool_calls:
            # Create a span for each tool execution
            tool_args = json.loads(tool_call.function.arguments)

            with mlflow.start_span(
                name=f"tool_{tool_call.function.name}",
                span_type="TOOL",
                attributes={
                    "tool_name": tool_call.function.name,
                    "tool_call_id": tool_call.id,
                }
            ) as span:
                # Log inputs to the span
                if MLFLOW_AVAILABLE:
                    span.set_inputs(tool_args)

                # Execute tool
                result = self.tools.execute(
                    tool_call.function.name,
                    tool_args
                )

                # Log outputs to the span
                if MLFLOW_AVAILABLE:
                    try:
                        result_dict = json.loads(result) if isinstance(result, str) else result
                        span.set_outputs(result_dict)
                    except:
                        span.set_outputs({"result": result})

                # Add tool response to conversation
                tool_message = create_chat_message("tool", result)
                tool_message.tool_call_id = tool_call.id
                messages.append(tool_message)

        # Get final response
        if not self.default_model:
            raise ValueError("No model available for tool response. This should not happen.")

        final_request = create_chat_completion_request(
            model=self.default_model,
            messages=messages,
            temperature=0.7
        )

        # CRITICAL: Include tools and tool_choice in follow-up request
        # Without this, the LLM doesn't know tools exist and may output malformed XML
        if self.tools.list_tools():
            final_request.tools = self.tools.get_schemas()
            final_request.tool_choice = tool_choice

        final_response = self._send_request(final_request)

        # Preserve the tool calls from the first response on the final response
        # This allows tracking and observability even after automatic tool execution
        final_response.choices[0].message.tool_calls = original_tool_calls

        # Return both the response and the complete conversation state
        # The messages list includes: original + assistant_with_tools + tool_results + final_assistant
        return final_response, messages

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

    def react(
        self,
        task: str,
        max_iterations: int = 15,
        stop_condition: callable = None,
        temperature: float = 0.7,
        verbose: bool = True,
        **kwargs
    ):
        """
        Run a ReACT (Reasoning + Acting) agent for the given task.

        This is a convenience method that creates and runs a ReACT agent.
        The agent will use tools to accomplish the task through iterative
        reasoning and acting cycles.

        Args:
            task: The task description/prompt
            max_iterations: Maximum number of iterations (default: 15)
            stop_condition: Optional function that returns True when done
            temperature: Sampling temperature (default: 0.7)
            verbose: Whether to print progress (default: True)
            **kwargs: Additional arguments passed to the agent

        Returns:
            AgentResult with execution details

        Example:
            result = client.react(
                task="Implement a sorting algorithm",
                max_iterations=15
            )

            if result.success:
                print(f"Completed in {result.iterations} iterations")
                print(result.final_response)
        """
        from .agents import ReACT

        agent = ReACT(self)
        return agent.run(
            task=task,
            max_iterations=max_iterations,
            stop_condition=stop_condition,
            temperature=temperature,
            verbose=verbose,
            **kwargs
        )

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