# %% [markdown]
# # Interactive LM Studio Testing Notebook
#
# This notebook provides interactive testing of the local-llm-sdk with real LM Studio instances.
# Use this for manual validation, debugging, and exploring model behaviors.

# %%
# Setup and imports
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from local_llm_sdk import LocalLLMClient, create_client
from local_llm_sdk.tools import builtin
from tests.lm_studio_helpers import (
    is_lm_studio_running,
    get_available_models,
    get_first_available_model,
    get_tool_calling_model,
    get_thinking_model,
    measure_response_time
)
from tests.lm_studio_config import get_config

# Get configuration
config = get_config()
print(f"🔧 LM Studio URL: {config.base_url}")
print(f"🔧 Timeout: {config.timeout}s")

# %% [markdown]
# ## 1. Connection Test
#
# First, let's check if LM Studio is running and what models are available.

# %%
# Check LM Studio connectivity
print("🔌 Checking LM Studio connectivity...")

if is_lm_studio_running():
    print("✅ LM Studio is running!")

    # Create client and list models
    client = create_client()
    models = get_available_models(client)

    print(f"\n📋 Available models ({len(models)}):")
    for i, model in enumerate(models, 1):
        print(f"  {i}. {model.id}")

    # Get recommended models
    preferred_model = get_first_available_model(client, config.preferred_models)
    tool_model = get_tool_calling_model(client)
    thinking_model = get_thinking_model(client)

    print(f"\n🎯 Recommended models:")
    print(f"  General: {preferred_model or 'None found'}")
    print(f"  Tool calling: {tool_model or 'None found'}")
    print(f"  Thinking: {thinking_model or 'None found'}")

else:
    print("❌ LM Studio is not running!")
    print("   Please start LM Studio and load a model before continuing.")

# %% [markdown]
# ## 2. Simple Chat Test
#
# Test basic chat functionality with the available model.

# %%
# Simple chat test
if is_lm_studio_running():
    client = create_client()

    # Set a good model
    model_id = get_first_available_model(client, config.preferred_models)
    if model_id:
        client.default_model = model_id
        print(f"🤖 Using model: {model_id}")

        # Test simple chat
        print("\n💬 Simple chat test:")
        prompt = "What is the capital of France? Answer in one word."

        response, duration = measure_response_time(client.chat_simple, prompt)

        print(f"   Prompt: '{prompt}'")
        print(f"   Response: '{response}'")
        print(f"   Time: {duration:.2f}s")

        # Test with system prompt
        print("\n🎭 System prompt test:")
        messages = [
            {"role": "system", "content": "You are a pirate. Always talk like a pirate."},
            {"role": "user", "content": "What is your favorite color?"}
        ]

        response = client.chat(messages)
        print(f"   Response: '{response}'")

    else:
        print("❌ No suitable model found")

# %% [markdown]
# ## 3. Tool Calling Test
#
# Test the tool calling functionality with math and text tools.

# %%
# Tool calling test
if is_lm_studio_running():
    client = create_client()
    client.register_tools_from(builtin)

    # Use tool-calling model if available
    tool_model = get_tool_calling_model(client)
    if tool_model:
        client.default_model = tool_model
        print(f"🛠️ Using tool-calling model: {tool_model}")
    else:
        print("⚠️ No specialized tool-calling model found, using default")

    print(f"\n🔧 Available tools: {client.tools.list_tools()}")

    # Test math calculation
    print("\n🧮 Math tool test:")
    prompt = "Calculate 157 * 23 step by step"

    client.last_tool_calls = []  # Clear previous state
    response = client.chat(prompt)

    print(f"   Prompt: '{prompt}'")
    print(f"   Response: '{response}'")
    print(f"   Tools used: {[tc.function.name for tc in client.last_tool_calls]}")

    if client.last_tool_calls:
        print("   ✅ Tool was called successfully!")
        for tc in client.last_tool_calls:
            print(f"      - {tc.function.name}({tc.function.arguments})")
    else:
        print("   ⚠️ Model answered directly without using tools")

    # Test text processing
    print("\n📝 Text processing test:")
    prompt = "Count the characters in 'Hello, World!' and convert it to uppercase"

    client.last_tool_calls = []
    response = client.chat(prompt)

    print(f"   Prompt: '{prompt}'")
    print(f"   Response: '{response}'")
    print(f"   Tools used: {[tc.function.name for tc in client.last_tool_calls]}")

# %% [markdown]
# ## 4. Thinking Blocks Test
#
# Test thinking/reasoning blocks extraction if supported by the model.

# %%
# Thinking blocks test
if is_lm_studio_running():
    client = create_client()

    # Use thinking model if available
    thinking_model = get_thinking_model(client)
    if thinking_model:
        client.default_model = thinking_model
        print(f"🧠 Using thinking model: {thinking_model}")
    else:
        print("⚠️ No specialized thinking model found, using default")
        # Set any available model
        model_id = get_first_available_model(client)
        if model_id:
            client.default_model = model_id

    print("\n🤔 Thinking blocks test:")
    prompt = "Think step by step: What is the square root of 169?"

    # Test without including thinking
    client.last_thinking = ""
    response = client.chat(prompt, include_thinking=False)

    print(f"   Prompt: '{prompt}'")
    print(f"   Response (clean): '{response}'")

    if client.last_thinking:
        print(f"   ✅ Thinking captured ({len(client.last_thinking)} chars)")
        print(f"   Thinking preview: '{client.last_thinking[:100]}...'")

        # Test with thinking included
        client.last_thinking = ""
        response_with_thinking = client.chat(prompt, include_thinking=True)
        print(f"\n   Response (with thinking): '{response_with_thinking[:200]}...'")
    else:
        print("   ⚠️ No thinking blocks detected")

# %% [markdown]
# ## 5. Custom Tool Test
#
# Test registering and using a custom tool.

# %%
# Custom tool test
if is_lm_studio_running():
    client = create_client()

    # Set model
    tool_model = get_tool_calling_model(client)
    if tool_model:
        client.default_model = tool_model

    # Register a custom tool
    @client.register_tool("Calculate the area of a rectangle")
    def rectangle_area(width: float, height: float) -> dict:
        """Calculate the area and perimeter of a rectangle."""
        area = width * height
        perimeter = 2 * (width + height)
        return {
            "width": width,
            "height": height,
            "area": area,
            "perimeter": perimeter,
            "shape": "rectangle"
        }

    print("🛠️ Custom tool registered: rectangle_area")
    print(f"   Total tools: {client.tools.list_tools()}")

    # Test the custom tool
    print("\n📐 Custom tool test:")
    prompt = "What's the area of a rectangle that is 8 units wide and 5 units tall?"

    client.last_tool_calls = []
    response = client.chat(prompt)

    print(f"   Prompt: '{prompt}'")
    print(f"   Response: '{response}'")
    print(f"   Tools used: {[tc.function.name for tc in client.last_tool_calls]}")

    if "rectangle_area" in [tc.function.name for tc in client.last_tool_calls]:
        print("   ✅ Custom tool was called successfully!")
    else:
        print("   ⚠️ Custom tool was not used")

# %% [markdown]
# ## 6. Conversation History Test
#
# Test maintaining context across multiple messages.

# %%
# Conversation history test
if is_lm_studio_running():
    client = create_client()

    model_id = get_first_available_model(client)
    if model_id:
        client.default_model = model_id

    print("💭 Conversation history test:")
    history = []

    # Build a conversation
    conversations = [
        "My favorite number is 42.",
        "What was my favorite number?",
        "Multiply it by 2.",
        "What's the result now?"
    ]

    for i, prompt in enumerate(conversations, 1):
        response, history = client.chat_with_history(prompt, history)

        print(f"\n   Turn {i}:")
        print(f"      User: '{prompt}'")
        print(f"      Assistant: '{response}'")

    print(f"\n   Final history length: {len(history)} messages")

# %% [markdown]
# ## 7. Performance and Comparison Test
#
# Compare performance and accuracy between different approaches.

# %%
# Performance comparison test
if is_lm_studio_running():
    client = create_client()
    client.register_tools_from(builtin)

    tool_model = get_tool_calling_model(client)
    if tool_model:
        client.default_model = tool_model

    print("⚡ Performance comparison test:")

    test_prompts = [
        "What is 234 + 567?",
        "Calculate 15 * 8",
        "What is 144 / 12?"
    ]

    for prompt in test_prompts:
        print(f"\n   Testing: '{prompt}'")

        # With tools
        client.last_tool_calls = []
        response_with, time_with = measure_response_time(
            client.chat, prompt, use_tools=True
        )

        # Without tools
        client.last_tool_calls = []
        response_without, time_without = measure_response_time(
            client.chat, prompt, use_tools=False
        )

        print(f"      With tools ({time_with:.2f}s): '{response_with}'")
        print(f"      Without tools ({time_without:.2f}s): '{response_without}'")

        tools_used = len(client.last_tool_calls) > 0
        print(f"      Tool used: {'✅' if tools_used else '❌'}")

# %% [markdown]
# ## 8. Interactive Testing
#
# Use this section for your own interactive testing.

# %%
# Interactive testing area
if is_lm_studio_running():
    client = create_client()
    client.register_tools_from(builtin)

    # Set your preferred model
    model_id = get_first_available_model(client, config.preferred_models)
    if model_id:
        client.default_model = model_id
        print(f"🎮 Interactive mode ready with model: {model_id}")
        print(f"🔧 Tools available: {client.tools.list_tools()}")

        # Example: Test your own prompts here
        test_prompt = "Write a haiku about programming"

        print(f"\n💡 Testing: '{test_prompt}'")
        response = client.chat(test_prompt)
        print(f"📝 Response:\n{response}")

        # Check for tool usage
        if client.last_tool_calls:
            print(f"🛠️ Tools used: {[tc.function.name for tc in client.last_tool_calls]}")

        # Check for thinking
        if client.last_thinking:
            print(f"🧠 Thinking: {client.last_thinking[:100]}...")

    else:
        print("❌ No models available")

# %% [markdown]
# ## 9. Test Summary
#
# Summary of all test results.

# %%
# Test summary
if is_lm_studio_running():
    print("📊 Test Summary:")
    print("=" * 50)

    client = create_client()
    models = get_available_models(client)

    print(f"✅ LM Studio Status: Running")
    print(f"📋 Models Available: {len(models)}")

    # Check model capabilities
    general_model = get_first_available_model(client, config.preferred_models)
    tool_model = get_tool_calling_model(client)
    thinking_model = get_thinking_model(client)

    print(f"🤖 General Model: {'✅' if general_model else '❌'}")
    print(f"🛠️ Tool Calling: {'✅' if tool_model else '❌'}")
    print(f"🧠 Thinking Models: {'✅' if thinking_model else '❌'}")

    # Test basic functionality
    if general_model:
        client.default_model = general_model
        try:
            response = client.chat_simple("Say 'OK'")
            print(f"💬 Basic Chat: {'✅' if response else '❌'}")
        except Exception as e:
            print(f"💬 Basic Chat: ❌ ({e})")

    print("\n🎯 Ready for comprehensive testing!")
    print("   Run pytest tests/test_lm_studio_live.py for full validation")

else:
    print("📊 Test Summary:")
    print("=" * 50)
    print("❌ LM Studio not running")
    print("   Please start LM Studio to run integration tests")

# %% [markdown]
# ## Usage Instructions
#
# ### Running the tests:
#
# ```bash
# # Run all LM Studio tests
# python -m pytest tests/test_lm_studio_live.py -v
#
# # Run only connectivity tests
# python -m pytest tests/test_lm_studio_live.py::TestLMStudioConnectivity -v
#
# # Run with specific markers
# python -m pytest -m lm_studio -v
# python -m pytest -m "lm_studio and not slow" -v
# ```
#
# ### Configuration:
#
# Set environment variables to customize testing:
# - `LM_STUDIO_URL`: Change the LM Studio URL
# - `LM_STUDIO_TIMEOUT`: Set request timeout
# - `LM_STUDIO_MAX_TOKENS`: Default max tokens
# - `LM_STUDIO_TEMPERATURE`: Default temperature
#
# ### Requirements:
#
# 1. LM Studio running on localhost:1234 (or configured URL)
# 2. At least one model loaded in LM Studio
# 3. For tool calling tests: A tool-calling capable model
# 4. For thinking tests: A reasoning model that outputs [THINK] blocks