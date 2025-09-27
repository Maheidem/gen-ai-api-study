# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a Generative AI API study repository that has evolved into **Local LLM SDK** - a type-safe Python SDK for interacting with local LLM APIs that implement the OpenAI specification. The project provides a clean, extensible interface for working with LM Studio, Ollama, and other OpenAI-compatible servers.

## Study Approach

The repository follows a structured research methodology:
1. **API Documentation**: Comprehensive reference documents for both OpenAI and LM Studio APIs
2. **Practical Testing**: Jupyter notebooks with hands-on API interaction examples
3. **Comparative Analysis**: Side-by-side feature comparison and compatibility assessment
4. **Code Examples**: Pydantic models and type-safe API client implementations

## Development Environment

### Python Setup
- **Python Version**: 3.12.11
- **Key Libraries**:
  - `pydantic`: Type-safe data models for API responses
  - `requests`: HTTP client for API interactions
  - `openai`: Official OpenAI SDK
  - `ipykernel`: Jupyter notebook support

### LM Studio Configuration
- **Base URL**: `http://169.254.83.107:1234/v1` (local network)
- **Authentication**: Fixed key "lm-studio"
- **Available Models**: Check via `/v1/models` endpoint

## Project Structure

```
gen-ai-api-study/
├── local_llm_sdk/              # Main Python package
│   ├── __init__.py            # Package exports
│   ├── client.py              # LocalLLMClient implementation
│   ├── models.py              # Pydantic models (OpenAI spec)
│   ├── tools/                 # Tool system
│   │   ├── __init__.py
│   │   ├── registry.py        # Tool registry and decorator
│   │   └── builtin.py         # Built-in tools
│   └── utils/                 # Utility functions
├── notebooks/                  # Jupyter notebooks
│   ├── api-hello-world-local.ipynb
│   ├── tool-use-math-calculator.ipynb
│   └── tool-use-simplified.ipynb
├── .documentation/            # Research documentation
├── setup.py                   # Package configuration
├── requirements.txt           # Dependencies
└── README.md                  # Project documentation
```

## Common Development Tasks

### Installing the Package
```bash
# Install in development mode
pip install -e .

# Install dependencies
pip install -r requirements.txt
```

### Using the SDK
```python
from local_llm_sdk import LocalLLMClient, create_client

# Create client
client = LocalLLMClient(
    base_url="http://169.254.83.107:1234/v1",
    model="your-model"
)

# Simple chat
response = client.chat("Hello!")
```

### Running Notebooks
```bash
# Navigate to notebooks directory
cd notebooks/

# Start Jupyter
jupyter notebook
```

### Adding New Tools
```python
from local_llm_sdk import tool

@tool("Description of your tool")
def your_tool(param: str) -> dict:
    return {"result": param.upper()}
```

## Code Architecture

### Package Architecture
- **`local_llm_sdk/`**: Main package with clean separation of concerns
  - `client.py`: Main LocalLLMClient class
  - `models.py`: Pydantic models following OpenAI spec
  - `tools/`: Tool registration and execution system
- **`notebooks/`**: Interactive examples and tutorials
- **`.documentation/`**: Research documents with citations

### Key Components
1. **LocalLLMClient**: Type-safe client with automatic tool handling
2. **Tool System**: Decorator-based tool registration
3. **Pydantic Models**: Full OpenAI API specification coverage

### Documentation Standards
- All research includes source citations with timestamps
- Reliability ratings (1-5 stars) for each source
- Code examples in Python, TypeScript, and cURL
- Performance benchmarks and cost analysis

## API Compatibility Notes

### Fully Supported in LM Studio
- Chat completions (`/v1/chat/completions`)
- Streaming responses (SSE format)
- Embeddings (`/v1/embeddings`)
- Model listing (`/v1/models`)

### Partially Supported
- Function calling (~60% compatibility, model-dependent)
- JSON mode (~50% compatibility)

### Not Supported in LM Studio
- Assistants API
- Fine-tuning endpoints
- Image generation (DALL-E)
- Audio transcription/generation
- Content moderation

## Research Methodology

When adding new API comparisons:
1. Test both platforms with identical requests
2. Document response format differences
3. Note performance characteristics
4. Calculate cost implications
5. Update compatibility matrix
6. Include migration code examples

## Tips for Future Development

### When Testing New Endpoints
- Always use Pydantic models for response validation
- Test with multiple models when available
- Document timeout requirements (LM Studio can be slower)
- Include error handling examples

### When Writing Documentation
- Follow the existing citation format in `.documentation/`
- Include practical use cases
- Provide cost-benefit analysis
- Note hardware requirements for local deployment

### Common Pitfalls to Avoid
- LM Studio models may have different context windows than OpenAI
- Streaming implementation varies by model
- Function calling syntax differs between models
- Local models require significant RAM (8-64GB depending on model size)