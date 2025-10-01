# Local LLM SDK Documentation

Welcome to the **Local LLM SDK** documentation. This SDK provides a type-safe Python interface for interacting with local LLM servers that implement the OpenAI API specification (LM Studio, Ollama, LocalAI, etc.).

## 📚 Documentation Structure

### Getting Started
- **[Installation](getting-started/installation.md)** - Setup and dependencies
- **[Quick Start](getting-started/quickstart.md)** - Your first chat in 5 minutes
- **[Configuration](getting-started/configuration.md)** - Environment variables and settings
- **[Basic Usage](getting-started/basic-usage.md)** - Core concepts and patterns

### API Reference
- **[Client API](api-reference/client.md)** - LocalLLMClient methods and properties
- **[Models](api-reference/models.md)** - Pydantic models reference
- **[Tools](api-reference/tools.md)** - Tool system and built-in tools
- **[Agents](api-reference/agents.md)** - Agent framework (ReACT, BaseAgent)
- **[Configuration](api-reference/configuration.md)** - Config module reference

### Guides
- **[Tool Calling](guides/tool-calling.md)** - Complete guide to tool/function calling
- **[ReACT Agents](guides/react-agents.md)** - Building autonomous agents
- **[Conversation Management](guides/conversation-management.md)** - Multi-turn conversations
- **[MLflow Tracing](guides/mlflow-tracing.md)** - Observability and debugging
- **[Production Patterns](guides/production-patterns.md)** - Error handling, retries, best practices
- **[Custom Tools](guides/custom-tools.md)** - Creating your own tools
- **[Migration from OpenAI](guides/migration-openai.md)** - Switching from OpenAI SDK

### Architecture
- **[Overview](architecture/overview.md)** - System architecture and design
- **[Client Architecture](architecture/client.md)** - How the client works internally
- **[Tool System](architecture/tool-system.md)** - Tool registry and execution
- **[Agent Framework](architecture/agent-framework.md)** - Agent patterns and implementation
- **[Conversation State](architecture/conversation-state.md)** - Message handling deep dive

### Contributing
- **[Development Guide](contributing/development.md)** - Setting up development environment
- **[Testing Guide](contributing/testing.md)** - Writing and running tests
- **[Behavioral Testing](contributing/behavioral-testing.md)** - LLM behavior validation
- **[Code Style](contributing/code-style.md)** - Formatting and conventions
- **[Pull Request Guide](contributing/pull-requests.md)** - Contributing workflow

## 🚀 Quick Links

### For Users
- [Installation & Setup](getting-started/installation.md)
- [Quick Start Guide](getting-started/quickstart.md)
- [Tool Calling Tutorial](guides/tool-calling.md)
- [ReACT Agents Tutorial](guides/react-agents.md)

### For Developers
- [Development Setup](contributing/development.md)
- [Architecture Overview](architecture/overview.md)
- [Testing Guide](contributing/testing.md)
- [Code Style Guide](contributing/code-style.md)

### For Researchers
- [API Compatibility Research](./../.documentation/research-index.md)
- [OpenAI vs LM Studio Comparison](./../.documentation/lm_studio_openai_api_comparison.md)

## 📖 Interactive Tutorials

The SDK includes 11 progressive Jupyter notebooks in `notebooks/`:

**Level 1: Foundations** (30 min)
- `01-installation-setup.ipynb` - Setup and verification
- `02-basic-chat.ipynb` - Simple chat interactions
- `03-conversation-history.ipynb` - Multi-turn conversations

**Level 2: Core Features** (45 min)
- `04-tool-calling-basics.ipynb` - Using built-in tools
- `05-custom-tools.ipynb` - Creating your own tools
- `06-filesystem-code-execution.ipynb` - File I/O and code execution

**Level 3: Advanced** (60 min)
- `07-react-agents.ipynb` - ReACT pattern for complex tasks
- `08-mlflow-observability.ipynb` - Tracing and debugging
- `09-production-patterns.ipynb` - Error handling and best practices

**Level 4: Projects** (60 min)
- `10-mini-project-code-helper.ipynb` - Code review assistant
- `11-mini-project-data-analyzer.ipynb` - Data analysis pipeline

## 🔧 Quick Reference

### Installation
```bash
pip install -e .
```

### Basic Usage
```python
from local_llm_sdk import LocalLLMClient

client = LocalLLMClient(
    base_url="http://localhost:1234/v1",
    model="your-model"
)

response = client.chat("Hello!")
print(response)
```

### With Tools
```python
from local_llm_sdk import create_client_with_tools

client = create_client_with_tools()
response = client.chat("Calculate 42 * 17", use_tools=True)
client.print_tool_calls()
```

### ReACT Agent
```python
from local_llm_sdk.agents import ReACT

agent = ReACT(client)
result = agent.run("Complex multi-step task", max_iterations=15)
print(result.final_response)
```

## 🆘 Getting Help

- **Issues**: [GitHub Issues](https://github.com/Maheidem/gen-ai-api-study/issues)
- **Discussions**: [GitHub Discussions](https://github.com/Maheidem/gen-ai-api-study/discussions)
- **API Reference**: See [api-reference/](api-reference/) directory

## 📝 License

MIT License - see LICENSE file for details.

## 🙏 Acknowledgments

Built for compatibility with OpenAI API specification. Special thanks to the open-source LLM community.
