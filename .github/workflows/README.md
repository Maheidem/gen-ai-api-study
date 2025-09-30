# GitHub Actions Workflows

## Overview

This directory contains CI/CD workflows for the Local LLM SDK project.

## Workflows

### 1. `test-unit.yml` - Fast Unit Tests ⚡

**Trigger**: Every push and pull request
**Duration**: ~2 minutes
**Tests**: All tests except `live_llm` (mocked tests only)

```bash
# What it runs (equivalent local command):
pytest tests/ -v  # Skips live_llm by default
```

**Matrix Testing**: Python 3.9, 3.10, 3.11, 3.12

**Coverage**: Generates coverage reports uploaded to Codecov

### 2. `test-behavioral.yml` - Behavioral Tests 🤖

**Trigger**:
- Nightly at 2:00 AM UTC (scheduled)
- Manual via workflow_dispatch

**Duration**: ~15-30 minutes
**Tests**: Behavioral and golden dataset tests with real LLM

```bash
# What it runs (equivalent local command):
pytest tests/ -m "live_llm and behavioral" -v
pytest tests/ -m "live_llm and golden" -v
```

**Status**: 🚧 Requires LLM server setup (see below)

## Running Tests Locally

### Unit Tests (Fast)
```bash
# Run all unit tests (skips live_llm)
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=local_llm_sdk --cov-report=html
```

### Behavioral Tests (Requires LM Studio)
```bash
# Start LM Studio on http://169.254.83.107:1234
# Then run:

# All behavioral tests
pytest tests/ -m "live_llm and behavioral" -v

# Golden dataset only
pytest tests/ -m "live_llm and golden" -v

# Specific test
pytest tests/test_agents_behavioral.py::TestReACTBehavior::test_multi_step_iteration_pattern -v
```

### Manual Test Selection
```bash
# Run behavioral tests WITHOUT golden dataset
pytest tests/ -m "live_llm and behavioral and not golden" -v

# Run only slow tests
pytest tests/ -m "slow" -v

# Run everything including live_llm (override default skip)
pytest tests/ -m "" -v
```

## Setting Up LLM Server in CI/CD

### Option 1: Docker with Ollama (Recommended for CI)

```yaml
# Add to test-behavioral.yml:
- name: Start Ollama
  run: |
    docker run -d -p 11434:11434 --name ollama ollama/ollama
    docker exec ollama ollama pull mistral:7b

- name: Wait for Ollama
  run: |
    timeout 120 bash -c 'until curl -s http://localhost:11434/api/tags; do sleep 2; done'

- name: Run behavioral tests
  env:
    LLM_BASE_URL: http://localhost:11434/v1
  run: |
    pytest tests/ -m "live_llm and behavioral" -v
```

### Option 2: Mock LLM Server (Faster, Limited Value)

```yaml
# Add to test-behavioral.yml:
- name: Start Mock LLM Server
  run: |
    python tests/mock_llm_server.py &
    sleep 5

- name: Run behavioral tests
  env:
    LLM_BASE_URL: http://localhost:8000/v1
  run: |
    pytest tests/ -m "live_llm and behavioral" -v
```

**Trade-off**: Mock server is fast but defeats the purpose of behavioral testing (validating real LLM behavior).

### Option 3: Self-Hosted Runner (Best, Requires Infrastructure)

Set up a self-hosted GitHub Actions runner with LM Studio pre-installed:

```yaml
jobs:
  behavioral-tests:
    runs-on: [self-hosted, lm-studio]  # Custom label
```

**Requirements**:
- Machine with sufficient RAM (16GB+ for Mistral-7B)
- LM Studio or Ollama installed
- Model pre-loaded
- Network accessible from runner

## Test Output Artifacts

Both workflows upload artifacts on failure:

- **test-results-{python-version}**: Test cache and results
- **coverage-report**: HTML coverage report
- **behavioral-test-results**: Behavioral test outputs

Access via: Actions → Workflow Run → Artifacts

## Notifications

Currently, the workflows log failures but don't send notifications.

### TODO: Add Notification Integration

```yaml
# Example: Slack notification
- name: Send Slack notification
  if: failure()
  uses: 8398a7/action-slack@v3
  with:
    status: ${{ job.status }}
    webhook_url: ${{ secrets.SLACK_WEBHOOK }}
    text: "Behavioral tests failed - potential LLM regression detected"
```

## Monitoring Success Rates

Golden dataset tests track success rates over time:

- **single_tool_tasks**: ≥90% expected
- **multi_tool_tasks**: ≥80% expected
- **complex_tasks**: ≥70% expected
- **edge_cases**: ≥80% expected

**TODO**: Integrate with MLflow to track these metrics over time.

## Troubleshooting

### "Tests are being skipped"

By default, `live_llm` tests are skipped. This is intentional for fast CI.

To run them: `pytest tests/ -m "live_llm" -v`

### "LM Studio connection refused"

Check:
1. LM Studio is running on http://169.254.83.107:1234
2. A model is loaded
3. Server is not rate-limiting

Test connection: `curl http://169.254.83.107:1234/v1/models`

### "Behavioral tests failing with timeout"

Increase timeout in test:
```python
@pytest.mark.timeout(600)  # 10 minutes
def test_complex_task():
    ...
```

Or globally in pytest.ini:
```ini
[pytest]
timeout = 300
```

## Future Enhancements

- [ ] Automated LLM server provisioning in CI
- [ ] MLflow tracking integration for metrics
- [ ] Success rate trend visualization
- [ ] Automated PR comments with test results
- [ ] Performance regression detection
- [ ] Weekly golden dataset review
