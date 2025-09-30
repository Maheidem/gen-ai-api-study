# notebooks/

## Purpose
Educational tutorial series for learning the Local LLM SDK. Progressive notebooks from beginner to advanced, with hands-on exercises and mini-projects.

## 📚 Learning Path

### Level 1: Foundations (30 minutes)
**Start here if you're new to the SDK**

- **01-installation-setup.ipynb** (10 min) - Install SDK, connect to LM Studio, verify setup
- **02-basic-chat.ipynb** (10 min) - Simple chat, system prompts, temperature control
- **03-conversation-history.ipynb** (10 min) - Multi-turn conversations with context

### Level 2: Core Features (45 minutes)
**Learn about tools and function calling**

- **04-tool-calling-basics.ipynb** (15 min) - Using built-in tools (math, text, etc.)
- **05-custom-tools.ipynb** (15 min) - Creating your own tools with @tool decorator
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
- Total learning time: ~3 hours for complete series