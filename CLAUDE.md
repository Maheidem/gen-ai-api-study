# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a Generative AI API study repository focused on comparing and documenting OpenAI and LM Studio API capabilities. The project aims to create comprehensive documentation and practical examples for developers transitioning between cloud-based (OpenAI) and local (LM Studio) AI model deployments.

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

## Common Development Tasks

### Running Jupyter Notebooks
```bash
# Start Jupyter kernel
jupyter notebook api-hello-world-local.ipynb

# Or use VS Code's Jupyter extension
code api-hello-world-local.ipynb
```

### Testing API Connections
```python
# Test LM Studio connection
import requests
base_url = "http://169.254.83.107:1234/v1"
response = requests.get(f"{base_url}/models")
print(response.json())

# Test OpenAI connection (requires API key)
from openai import OpenAI
client = OpenAI()
response = client.models.list()
```

### Adding New API Examples
1. Create Pydantic models for request/response validation
2. Test endpoint with raw requests first
3. Document differences from OpenAI implementation
4. Add to comparison matrix in `lm_studio_openai_api_comparison.md`

## Code Architecture

### Repository Structure
- **Root Level**: Main documentation and Jupyter notebooks
- **`.documentation/`**: Research documents with citations and methodology
- **Notebooks**: Interactive API testing and validation

### Key Documents
1. **`openai-api-documentation.md`**: Complete OpenAI API reference (31KB)
2. **`lm_studio_openai_api_comparison.md`**: Detailed comparison with ~85% compatibility analysis (28KB)
3. **`api-hello-world-local.ipynb`**: Working examples with Pydantic models for type safety

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