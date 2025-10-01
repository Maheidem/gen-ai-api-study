# Installation

This guide walks you through installing the Local LLM SDK and setting up your local LLM server.

## Prerequisites

- **Python 3.8+** (Python 3.12.11 recommended)
- **pip** package manager
- **Local LLM Server** (LM Studio, Ollama, or other OpenAI-compatible server)

## Step 1: Install the SDK

### From Source (Recommended for Development)

```bash
# Clone the repository
git clone https://github.com/Maheidem/gen-ai-api-study.git
cd gen-ai-api-study

# Install in development mode
pip install -e .

# Install dependencies
pip install -r requirements.txt
```

### From PyPI (Coming Soon)

```bash
pip install local-llm-sdk
```

## Step 2: Verify Installation

```python
import local_llm_sdk
print(local_llm_sdk.__version__)  # Should print: 0.1.0
```

## Step 3: Set Up Local LLM Server

Choose one of the following local LLM servers:

### Option A: LM Studio (Recommended)

**1. Download LM Studio**
- Visit [https://lmstudio.ai](https://lmstudio.ai)
- Download for your OS (Windows, macOS, Linux)
- Install and launch

**2. Download a Model**
- Click "Search" in LM Studio
- Search for: `mistralai/Magistral-Small-2509` (recommended)
- Click download and wait for completion

**3. Start the Server**
- Click "Local Server" tab
- Click "Start Server"
- Default URL: `http://localhost:1234`
- Load your model in the server

**4. Test Connection**
```bash
curl http://localhost:1234/v1/models
```

### Option B: Ollama

**1. Install Ollama**
```bash
# macOS
brew install ollama

# Linux
curl -fsSL https://ollama.com/install.sh | sh

# Windows
# Download from https://ollama.com/download
```

**2. Start Ollama**
```bash
ollama serve
```

**3. Download a Model**
```bash
ollama pull mistral
```

**4. Enable OpenAI Compatibility**
Ollama runs on port 11434 by default. Use:
```python
client = LocalLLMClient(
    base_url="http://localhost:11434/v1",
    model="mistral"
)
```

### Option C: LocalAI

Follow the [LocalAI installation guide](https://localai.io/docs/getting-started/).

## Step 4: Configuration

### Environment Variables

Create a `.env` file or set environment variables:

```bash
# LM Studio (default)
export LLM_BASE_URL="http://localhost:1234/v1"
export LLM_MODEL="mistralai/magistral-small-2509"
export LLM_TIMEOUT="300"
export LLM_DEBUG="false"
```

Or for Ollama:
```bash
export LLM_BASE_URL="http://localhost:11434/v1"
export LLM_MODEL="mistral"
```

### Python Configuration

Alternatively, configure in code:

```python
from local_llm_sdk import LocalLLMClient

client = LocalLLMClient(
    base_url="http://localhost:1234/v1",
    model="your-model-name",
    timeout=300
)
```

## Step 5: Verify Setup

Run this test script to verify everything works:

```python
from local_llm_sdk import LocalLLMClient

# Create client
client = LocalLLMClient(
    base_url="http://localhost:1234/v1",
    model="auto"  # Auto-detect first available model
)

# Test connection
response = client.chat("Hello! Please respond with 'Hello, World!'")
print(response)

# Should print: "Hello, World!" or similar greeting
```

If you see a response, congratulations! 🎉 Your setup is complete.

## Optional Dependencies

### For Development
```bash
pip install -e ".[dev]"
```

Includes:
- `pytest` - Testing framework
- `pytest-cov` - Coverage reporting
- `black` - Code formatter
- `isort` - Import sorter
- `mypy` - Type checker

### For Notebooks
```bash
pip install -e ".[notebooks]"
```

Includes:
- `jupyter` - Notebook server
- `ipykernel` - Jupyter kernel
- `matplotlib` - Plotting

### For MLflow Tracing
```bash
pip install mlflow>=3.0.0
```

MLflow is optional. The SDK gracefully degrades if not installed.

## Troubleshooting

### "Connection refused" Error

**Problem**: Cannot connect to LLM server.

**Solutions**:
1. Verify server is running:
   ```bash
   curl http://localhost:1234/v1/models
   ```
2. Check the correct port (1234 for LM Studio, 11434 for Ollama)
3. Ensure firewall isn't blocking connections

### "No model loaded" Error

**Problem**: Server has no model loaded.

**Solutions**:
1. In LM Studio: Click "Load Model" in server tab
2. In Ollama: Run `ollama pull mistral`
3. Use `model="auto"` to auto-detect first available model

### Import Errors

**Problem**: `ModuleNotFoundError: No module named 'local_llm_sdk'`

**Solutions**:
1. Ensure you ran `pip install -e .` in the project directory
2. Check your Python environment (use `which python`)
3. Try reinstalling: `pip uninstall local-llm-sdk && pip install -e .`

### Timeout Errors

**Problem**: Requests timing out.

**Solutions**:
1. Increase timeout: `LocalLLMClient(timeout=600)`
2. Check model is loaded and not still downloading
3. Try a smaller, faster model

## Next Steps

- **[Quick Start Guide](quickstart.md)** - Your first chat in 5 minutes
- **[Configuration Guide](configuration.md)** - Advanced configuration options
- **[Basic Usage Guide](basic-usage.md)** - Core concepts and patterns

## Getting Help

- **GitHub Issues**: [Report bugs or request features](https://github.com/Maheidem/gen-ai-api-study/issues)
- **Documentation**: See other guides in `/docs`
- **Interactive Tutorials**: Check out `/notebooks` directory
