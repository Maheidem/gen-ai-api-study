# %% [markdown]
# # Tool Calling with Local LLMs - Complete Guide
# 
# This notebook demonstrates how to use function calling (tools) with local LLMs through the `local_llm_sdk` package.
# 
# ## What You'll Learn
# 1. How to set up and register tools
# 2. Using built-in tools (math, text, weather, etc.)
# 3. Creating custom tools with the `@tool` decorator
# 4. How the LLM decides when to use tools
# 5. Debugging and testing tools directly
# 6. Real-world conversation examples with multiple tools
# 
# ## Prerequisites
# - LM Studio running with a model that supports function calling
# - The `local_llm_sdk` package installed (`pip install -e ..` from notebooks directory)

# %%
!pip install -e .. --force-reinstall -q

# %% [markdown]
# ## 1. Setup and Initialization

# %%
# Import the SDK and tools
from local_llm_sdk import LocalLLMClient, create_chat_message
from local_llm_sdk.tools import builtin

# Create client with your LM Studio server
client = LocalLLMClient(
    base_url="http://169.254.83.107:1234/v1",
    model="mistralai/magistral-small-2509"  # Replace with your model
)

print(f"✅ Client created: {client}")
print(f"📍 Server: {client.base_url}")
print(f"🤖 Model: {client.default_model}")

# %% [markdown]
# ## 2. Register Built-in Tools
# 
# The SDK comes with several pre-built tools. Let's register them and see what's available.

# %%
# Register all built-in tools at once
client.register_tools_from(builtin)

# List all registered tools
print("🧰 Registered Tools:")
print("=" * 50)
for tool_name in client.tools.list_tools():
    print(f"  • {tool_name}")

print(f"\n📊 Total tools available: {len(client.tools.list_tools())}")

# %% [markdown]
# ## 3. Inspect Tool Schemas
# 
# Let's see what each tool does and what parameters it expects.

# %%
# Get detailed schema for each tool
print("📋 Tool Details:")
print("=" * 50)

for tool in client.tools.get_schemas():
    func = tool.function
    print(f"\n🔧 {func.name}")
    print(f"   Description: {func.description}")
    
    # Show parameters
    if func.parameters and 'properties' in func.parameters:
        props = func.parameters['properties']
        print(f"   Parameters:")
        for param_name, param_info in props.items():
            param_type = param_info.get('type', 'unknown')
            param_desc = param_info.get('description', '')
            required = "*" if param_name in func.parameters.get('required', []) else ""
            print(f"     - {param_name}{required} ({param_type}): {param_desc}")

# %% [markdown]
# ## 4. Test Each Built-in Tool
# 
# Let's test each tool with the LLM making the decision to use them.

# %% [markdown]
# ### 4.1 Math Calculator Tool

# %%
# Test math operations and show when tools are actually used
math_queries = [
    "What is 15 plus 27?",
    "Calculate 100 divided by 7",
    "Multiply 13 by 9",
    "What's 50 minus 18?"
]

print("🧮 Math Calculator Tests (with tool detection):")
print("=" * 50)

for query in math_queries:
    # Make the request
    response = client.chat(query)

    print(f"Q: {query}")

    # Check if tools were ACTUALLY called using client.last_tool_calls
    if client.last_tool_calls:
        print(f"   ✅ TOOLS ACTUALLY USED:")
        for tc in client.last_tool_calls:
            print(f"      🔧 {tc.function.name}({tc.function.arguments})")
    else:
        print(f"   ❌ NO TOOL USED - Model answered directly")

    print(f"A: {response}")
    print("-" * 30)

# %% [markdown]
# ### 4.2 Character Counter Tool

# %%
# Test character counting
text_queries = [
    "How many characters are in 'Hello, World!'?",
    "Count the characters in 'The quick brown fox jumps over the lazy dog'",
    "Tell me the character count of 'Python'"
]

print("📝 Character Counter Tests:")
print("=" * 50)

for query in text_queries:
    response = client.chat(query)

    print(f"Q: {query}")

    # Check if tools were ACTUALLY called using client.last_tool_calls
    if client.last_tool_calls:
        print(f"   ✅ TOOLS ACTUALLY USED:")
        for tc in client.last_tool_calls:
            print(f"      🔧 {tc.function.name}({tc.function.arguments})")
    else:
        print(f"   ❌ NO TOOL USED - Model answered directly")

    print(f"A: {response}")
    print("-" * 30)

# %% [markdown]
# ### 4.3 Text Transformer Tool

# %%
# Test text transformation
transform_queries = [
    "Convert 'hello world' to uppercase",
    "Make 'PYTHON ROCKS' lowercase",
    "Transform 'the quick brown fox' to title case"
]

print("🔤 Text Transformer Tests:")
print("=" * 50)

for query in transform_queries:
    response = client.chat(query)

    print(f"Q: {query}")

    # Check if tools were ACTUALLY called using client.last_tool_calls
    if client.last_tool_calls:
        print(f"   ✅ TOOLS ACTUALLY USED:")
        for tc in client.last_tool_calls:
            print(f"      🔧 {tc.function.name}({tc.function.arguments})")
    else:
        print(f"   ❌ NO TOOL USED - Model answered directly")

    print(f"A: {response}")
    print("-" * 30)

# %% [markdown]
# ### 4.4 Weather Tool (Mock Data)

# %%
# Test weather queries
weather_queries = [
    "What's the weather in New York?",
    "Tell me the temperature in London in Fahrenheit",
    "How's the weather in Tokyo?"
]

print("🌤️ Weather Tool Tests (Mock Data):")
print("=" * 50)

for query in weather_queries:
    response = client.chat(query)

    print(f"Q: {query}")

    # Check if tools were ACTUALLY called using client.last_tool_calls
    if client.last_tool_calls:
        print(f"   ✅ TOOLS ACTUALLY USED:")
        for tc in client.last_tool_calls:
            print(f"      🔧 {tc.function.name}({tc.function.arguments})")
    else:
        print(f"   ❌ NO TOOL USED - Model answered directly")

    print(f"A: {response}")
    print("-" * 30)

# %%
# Demonstrate the CORRECT way to detect tool usage
# The model DOES use tools, but the final response doesn't show tool_calls
# We need to check client.last_tool_calls instead

test_queries = [
    "What is 2 plus 2?",  # Simple math - might not use tool
    "Calculate 47 * 89",   # Complex math - should use tool
    "How many letters in 'cat'?",  # Simple count - might not use tool
    "Count the characters in 'supercalifragilisticexpialidocious'"  # Complex - should use tool
]

print("🔍 Tool Usage Detection (Corrected Method):")
print("=" * 50)

for query in test_queries:
    # Make the request
    response = client.chat(query)
    
    print(f"\n❓ Query: {query}")
    
    # Check if tools were ACTUALLY called using client.last_tool_calls
    if client.last_tool_calls:
        # Tools were used!
        print(f"✅ TOOLS ACTUALLY USED:")
        for tc in client.last_tool_calls:
            print(f"   🔧 {tc.function.name}({tc.function.arguments})")
    else:
        # No tools used - model answered directly
        print("❌ NO TOOLS USED - Model generated answer directly")
    
    print(f"💬 Final Answer: {response}")
    print("-" * 50)

# %% [markdown]
# ## 4.5 Detecting When Tools Are Actually Used
# 
# It's important to know if the model is using tools or just generating answers. Let's see how to detect tool usage.

# %% [markdown]
# ## 5. Create Custom Tools
# 
# Now let's create our own custom tools using the simple `@tool` decorator.

# %%
# Create a custom tool for reversing strings
@client.register_tool("Reverse a text string")
def reverse_string(text: str) -> dict:
    """Reverse the order of characters in a string."""
    return {
        "original": text,
        "reversed": text[::-1],
        "is_palindrome": text == text[::-1]
    }

print("✅ Custom tool 'reverse_string' registered!")

# Test the custom tool
test_responses = [
    client.chat("Reverse the text 'hello world'"),
    client.chat("Is 'racecar' a palindrome? Reverse it to check"),
    client.chat("Reverse 'Python SDK'")
]

print("\n🔄 Reverse String Tool Tests:")
print("=" * 50)
for i, response in enumerate(test_responses, 1):
    print(f"Test {i}: {response}")
    print("-" * 30)

# %%
# Create a more complex custom tool
@client.register_tool("Analyze text statistics")
def text_analyzer(text: str, include_vowels: bool = True) -> dict:
    """Analyze various statistics about a text string."""
    vowels = 'aeiouAEIOU'
    
    stats = {
        "text": text,
        "length": len(text),
        "words": len(text.split()),
        "sentences": text.count('.') + text.count('!') + text.count('?'),
        "uppercase_letters": sum(1 for c in text if c.isupper()),
        "lowercase_letters": sum(1 for c in text if c.islower()),
        "digits": sum(1 for c in text if c.isdigit()),
        "spaces": text.count(' ')
    }
    
    if include_vowels:
        stats["vowels"] = sum(1 for c in text if c in vowels)
        stats["consonants"] = sum(1 for c in text if c.isalpha() and c not in vowels)
    
    return stats

print("✅ Custom tool 'text_analyzer' registered!")

# Test the analyzer
analysis_query = "Analyze the text 'The Quick Brown Fox Jumps Over The Lazy Dog 123!'"
response = client.chat(analysis_query)
print(f"\n📊 Text Analysis:")
print("=" * 50)
print(f"Q: {analysis_query}")
print(f"A: {response}")

# %% [markdown]
# ## 6. Direct Tool Execution (Without LLM)
# 
# Sometimes you want to test tools directly without going through the LLM.

# %%
# Execute tools directly for debugging
print("🔧 Direct Tool Execution (No LLM):")
print("=" * 50)

# Test char_counter directly
result = client.tools.execute('char_counter', {'text': 'Hello, World!'})
print(f"char_counter('Hello, World!'): {result}")

# Test math_calculator directly
result = client.tools.execute('math_calculator', {
    'arg1': 10, 
    'arg2': 5, 
    'operation': 'multiply'
})
print(f"\nmath_calculator(10, 5, 'multiply'): {result}")

# Test custom reverse_string directly
result = client.tools.execute('reverse_string', {'text': 'level'})
print(f"\nreverse_string('level'): {result}")

# Test text_transformer directly
result = client.tools.execute('text_transformer', {
    'text': 'python rocks',
    'transform': 'title'
})
print(f"\ntext_transformer('python rocks', 'title'): {result}")

# %% [markdown]
# ## 7. Complex Conversations with Multiple Tools
# 
# Let's demonstrate a conversation where the LLM uses multiple tools to answer complex queries.

# %%
# Complex multi-tool query
complex_queries = [
    "Calculate 15 * 3, then tell me how many characters are in the answer when written as 'forty-five'",
    "What's the weather in London? Also convert the city name to uppercase",
    "Reverse 'hello', count its characters, and tell me if it's a palindrome"
]

print("🎯 Complex Multi-Tool Queries:")
print("=" * 50)

for query in complex_queries:
    print(f"\n❓ Query: {query}")
    response = client.chat(query)
    print(f"💡 Response: {response}")
    print("=" * 50)

# %% [markdown]
# ## 8. Conversation with History and Tools
# 
# Maintain context across multiple tool-using interactions.

# %%
# Start a conversation with history
history = []

print("💬 Conversation with Context and Tools:")
print("=" * 50)

# First query
response1, history = client.chat_with_history(
    "Calculate 25 times 4", 
    history
)
print(f"User: Calculate 25 times 4")
print(f"Assistant: {response1}\n")

# Follow-up using previous result
response2, history = client.chat_with_history(
    "Now add 50 to that result", 
    history
)
print(f"User: Now add 50 to that result")
print(f"Assistant: {response2}\n")

# Another follow-up
response3, history = client.chat_with_history(
    "Convert the final number to text and count its characters", 
    history
)
print(f"User: Convert the final number to text and count its characters")
print(f"Assistant: {response3}\n")

print(f"📚 Conversation length: {len(history)} messages")

# %% [markdown]
# ## 9. Error Handling and Edge Cases

# %%
# Test error handling
print("⚠️ Error Handling Tests:")
print("=" * 50)

# Division by zero
response = client.chat("What is 10 divided by 0?")
print(f"Division by zero: {response}\n")

# Invalid operation
try:
    result = client.tools.execute('math_calculator', {
        'arg1': 10,
        'arg2': 5,
        'operation': 'invalid_op'
    })
    print(f"Invalid operation result: {result}\n")
except Exception as e:
    print(f"Invalid operation error: {e}\n")

# Non-existent city in weather
response = client.chat("What's the weather in Atlantis?")
print(f"Non-existent city: {response}")

# %% [markdown]
# ## 10. Tool Schema Export
# 
# Export tool schemas for documentation or debugging.

# %% [markdown]
# ## Summary
# 
# ### What We Learned:
# 
# 1. **Tool Registration** - Simple decorator pattern with `@client.register_tool()`
# 2. **Built-in Tools** - Math, text, weather tools ready to use
# 3. **Custom Tools** - Create any function and register it as a tool
# 4. **Automatic Schema Generation** - Type hints → OpenAI schemas
# 5. **Direct Execution** - Test tools without LLM for debugging
# 6. **Multi-Tool Queries** - LLM can use multiple tools in one response
# 7. **Conversation Context** - Maintain history across tool-using interactions
# 8. **Error Handling** - Graceful handling of edge cases
# 
# ### Best Practices:
# 
# - **Use Type Hints** - They automatically generate the schema
# - **Return Dicts** - Tools should return dictionaries with clear keys
# - **Descriptive Names** - Use clear function and parameter names
# - **Handle Errors** - Return error messages in the result dict
# - **Test Directly** - Use `client.tools.execute()` for debugging
# 
# ### Next Steps:
# 
# - Create domain-specific tools for your use case
# - Integrate with external APIs in your tools
# - Build complex multi-tool workflows
# - Experiment with different models and their tool-calling capabilities

# %%
# Demonstrate the CORRECT way to detect tool usage
# The model DOES use tools, but the final response doesn't show tool_calls
# We need to check client.last_tool_calls instead

test_queries = [
    "What is 2 plus 2?",  # Simple math - might not use tool
    "Calculate 47 * 89",   # Complex math - should use tool
    "How many letters in 'cat'?",  # Simple count - might not use tool
    "Count the characters in 'supercalifragilisticexpialidocious'"  # Complex - should use tool
]

print("🔍 Tool Usage Detection (Corrected Method):")
print("=" * 50)

for query in test_queries:
    # Make the request
    response = client.chat(query)
    
    print(f"\n❓ Query: {query}")
    
    # Check if tools were ACTUALLY called using client.last_tool_calls
    if client.last_tool_calls:
        # Tools were used!
        print(f"✅ TOOLS ACTUALLY USED:")
        for tc in client.last_tool_calls:
            print(f"   🔧 {tc.function.name}({tc.function.arguments})")
    else:
        # No tools used - model answered directly
        print("❌ NO TOOLS USED - Model generated answer directly")
    
    print(f"💬 Final Answer: {response}")
    print("-" * 50)


