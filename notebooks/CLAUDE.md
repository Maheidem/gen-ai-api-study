# notebooks/

## Purpose
Educational tutorial series for learning the Local LLM SDK. Progressive notebooks from beginner to advanced, with hands-on exercises and mini-projects.

## 📚 Learning Path

### Level 1: Foundations (30 minutes)
**Start here if you're new to the SDK**

- **01-installation-setup.ipynb** (10 min) - Install SDK, connect to LM Studio, verify setup
- **02-basic-chat.ipynb** (10 min) - Simple chat, system prompts, temperature control
- **03-conversation-history.ipynb** (10 min) - Multi-turn conversations with context

### Level 2: Core Features (60 minutes)
**Learn about tools and function calling**

- **04-tool-calling-basics.ipynb** (15 min) - Using built-in tools (math, text, etc.)
- **05-custom-tools.ipynb** (15 min) - Creating your own tools with @tool decorator
- **05b-validation-safeguards.ipynb** (15 min) - Model validation and error prevention
- **06-filesystem-code-execution.ipynb** (15 min) - File I/O and code execution tools

### Level 3: Advanced (60 minutes)
**Production-ready patterns and observability**

- **07-react-agents.ipynb** (20 min) - ReACT pattern for multi-step tasks
- **08-mlflow-observability.ipynb** (20 min) - Tracing and debugging with MLflow
- **09-production-patterns.ipynb** (20 min) - Error handling, retries, configuration

### Level 4: Projects (60 minutes)
**Build complete applications**

- **10-mini-project-code-helper.ipynb** (30 min) - Code review assistant agent
- **11-mini-project-data-analyzer.ipynb** (30 min) - Data analysis pipeline

## 🎯 Quick Start

**Complete Beginner?**
→ Start with `01-installation-setup.ipynb`

**Already have SDK working?**
→ Jump to `02-basic-chat.ipynb`

**Want to build agents?**
→ Complete notebooks 1-6 first, then `07-react-agents.ipynb`

**Need production patterns?**
→ Go to `09-production-patterns.ipynb`

## Relationships
- **Parent**: Example code using `../local_llm_sdk/` package
- **Demonstrates**: All SDK features in practical scenarios
- **Requirements**: LM Studio or compatible OpenAI-spec server running locally

## Running Notebooks

### Setup
```bash
# Install dependencies
pip install -r requirements.txt

# Start Jupyter
jupyter notebook

# Navigate to notebooks/ directory
```

### Requirements
- Python 3.9+
- LM Studio running on http://localhost:1234 (or configure BASE_URL)
- At least one model loaded in LM Studio

### First Time?
1. Start with `01-installation-setup.ipynb`
2. Follow the notebooks in order (01 → 02 → 03 → ...)
3. Complete exercises in each notebook
4. Build the mini-projects (10-11) to solidify learning

### MLflow Tracing Setup (for notebooks 07-08)

**Important:** Notebooks run from the `notebooks/` directory but MLflow UI should serve from the project root.

**Issue:** By default, notebooks write traces to `notebooks/mlruns/` but MLflow UI serves from `<project-root>/mlruns/`, causing traces to appear missing.

**Solution:** Add this cell **after imports** in notebooks 07-08:

```python
import mlflow
import os

# Point to project root mlruns directory
project_root = os.path.dirname(os.path.abspath(os.getcwd()))
mlflow.set_tracking_uri(f"file://{project_root}/mlruns")

print(f"✅ MLflow tracking URI: {mlflow.get_tracking_uri()}")
```

**Start MLflow UI from project root:**
```bash
cd /path/to/gen-ai-api-study
mlflow ui --port 5000
```

Then open http://127.0.0.1:5000 to view traces.

**Verification:**
1. Run a traced operation in notebook
2. Check MLflow UI - should see experiment and traces
3. If not visible, check tracking URI matches MLflow UI directory

## Educational Design

Each notebook includes:
- ✅ **Learning Objectives** - Clear goals with checkboxes
- ✅ **Prerequisites** - What to complete first
- ✅ **Estimated Time** - Plan your learning session
- ✅ **Explanations** - Concepts before code
- ✅ **Examples** - Runnable code you can try
- ✅ **Exercises** - Hands-on practice with solutions
- ✅ **Common Pitfalls** - Mistakes to avoid
- ✅ **Summary** - What you learned
- ✅ **Next Steps** - Where to go next

## Archive

Old notebooks (pre-refactor) are in `_archive/` directory for reference.

## Getting Help

- **SDK Issues**: Check `../local_llm_sdk/CLAUDE.md` for package documentation
- **Connection Problems**: See troubleshooting in `01-installation-setup.ipynb`
- **LM Studio Help**: Visit https://lmstudio.ai/docs

## Contributing

Found an issue or have a suggestion?
1. Test your fix in a notebook
2. Document the change clearly
3. Submit feedback or PR

## Notes
- All notebooks assume LM Studio on localhost:1234 (configurable)
- Notebooks are designed to run independently but build on each other
- Exercises have solutions in collapsed cells (click to expand)
- Projects (10-11) create files in the notebooks directory
- Total learning time: ~3.25 hours for complete series