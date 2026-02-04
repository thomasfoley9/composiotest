# Composio Tool Router Simulator

[![PyPI version](https://badge.fury.io/py/composio-tool-router-sim.svg)](https://badge.fury.io/py/composio-tool-router-sim)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A benchmarking tool that tests how different LLM models perform when using **Composio's Tool Router** (the system powering Rube). Compare models across accuracy, speed, cost, and tool selection quality.

## What is Tool Router?

Tool Router is Composio's agentic system that:
- Searches 10,000+ tools automatically based on natural language
- Plans multi-step workflows
- Executes tools with proper authentication
- Powers the Rube MCP server

## Installation

```bash
pip install composio-tool-router-sim
```

## Quick Start

```bash
# Set your Composio API key
export COMPOSIO_API_KEY="your-api-key"

# Run a single task across all models
tool-router-sim run --task "List my 5 most recent unread emails"

# Dry run (no API calls - for testing)
tool-router-sim run --task "List my unread emails" --dry-run

# Run a benchmark suite
tool-router-sim benchmark --suite simple_tasks

# Interactive mode
tool-router-sim interactive
```

## Supported Models

### Via Vercel AI Gateway
| Model | Description |
|-------|-------------|
| `claude-sonnet-4` | Anthropic's balanced model with excellent tool use |
| `claude-haiku-4.5` | Fast and cost-effective Claude model |
| `gpt-4o` | OpenAI's flagship multimodal model |
| `gpt-4o-mini` | Cost-effective OpenAI model |
| `gemini-2.0-flash` | Google's fast multimodal model |

### Via Groq (Fast Inference)
| Model | Description |
|-------|-------------|
| `llama-3.3-70b` | Meta's large versatile model |
| `llama-3.1-8b` | Fast, small Llama model |
| `mixtral-8x7b` | Mistral's mixture of experts |
| `gemma2-9b` | Google's instruction-tuned Gemma |

## CLI Commands

### Run a Single Task

```bash
# Test all models
tool-router-sim run --task "Send a Slack message to #general"

# Test specific models
tool-router-sim run --task "..." --models claude-sonnet-4,gpt-4o,llama-3.3-70b

# Export results
tool-router-sim run --task "..." --output results.json
```

### Run Benchmark Suites

```bash
# Built-in suites
tool-router-sim benchmark --suite simple_tasks
tool-router-sim benchmark --suite multi_step_tasks
tool-router-sim benchmark --suite edge_cases

# Custom suite from JSON file
tool-router-sim benchmark --suite my_tasks.json

# Save results to directory
tool-router-sim benchmark --suite simple_tasks --output-dir ./results
```

### Other Commands

```bash
# List available models with pricing
tool-router-sim list-models

# List built-in benchmark tasks
tool-router-sim list-tasks

# Interactive mode
tool-router-sim interactive
```

## Sample Output

```
╔══════════════════════════════════════════════════════════════════════════════╗
║              COMPOSIO TOOL ROUTER - MODEL BENCHMARK RESULTS                  ║
║              Task: "List my 5 most recent unread emails"                     ║
╠══════════════════════════════════════════════════════════════════════════════╣
║ Model                    │ Tool Selection │ Execution │ Speed  │ Cost   │ Grade ║
╠══════════════════════════╪════════════════╪═══════════╪════════╪════════╪═══════╣
║ claude-sonnet-4          │ 100%           │ 100%      │ 1.8s   │ $0.004 │ A+    ║
║ gpt-4o                   │ 100%           │ 100%      │ 2.1s   │ $0.005 │ A+    ║
║ llama-3.3-70b            │ 95%            │ 90%       │ 0.4s   │ $0.001 │ A     ║
║ gpt-4o-mini              │ 90%            │ 85%       │ 1.2s   │ $0.001 │ B+    ║
║ llama-3.1-8b             │ 70%            │ 60%       │ 0.2s   │ $0.0002│ C     ║
╚══════════════════════════════════════════════════════════════════════════════╝

🏆 BEST TOOL SELECTION: claude-sonnet-4, gpt-4o (100%)
⚡ FASTEST: llama-3.1-8b (0.2s)
💰 CHEAPEST: llama-3.1-8b ($0.0002)
⭐ BEST OVERALL: claude-sonnet-4
```

## Evaluation Metrics

### Tool Selection Score
- Found relevant tools via search
- Selected correct tool for the task
- Avoided irrelevant tools
- Proper search query formulation

### Execution Score
- Correct parameters passed
- Successful execution
- Complete results returned
- Efficient (minimal API calls)

### Planning Score
- Logical step order (search → connect → execute)
- Prerequisites handled
- Minimal steps taken

## Creating Custom Benchmark Tasks

Create a JSON file with your tasks:

```json
{
  "name": "My Custom Tasks",
  "description": "Custom benchmark suite",
  "tasks": [
    {
      "id": "custom_gmail",
      "task": "List my 5 most recent unread emails",
      "expected_tools": ["GMAIL_FETCH_EMAILS"],
      "expected_params": {
        "max_results": 5,
        "query": "is:unread"
      },
      "required_keywords": ["gmail", "email", "unread"],
      "difficulty": "easy"
    }
  ]
}
```

Then run:

```bash
tool-router-sim benchmark --suite my_tasks.json
```

## Python API

```python
from tool_router_sim.composio_client import create_client
from tool_router_sim.simulator.vercel_runner import create_vercel_runner
from tool_router_sim.evaluator.scorer import CompositeScorer

# Create client
client = create_client(dry_run=False)

# Create runner for a specific model
runner = create_vercel_runner(client, "claude-sonnet-4")

# Run simulation
result = runner.simulate("List my unread emails")

# Score the result
scorer = CompositeScorer(expected_tools=["GMAIL_FETCH_EMAILS"])
score = scorer.score(result)

print(f"Grade: {score.grade}")
print(f"Tool Selection: {score.tool_selection.score * 100}%")
print(f"Execution: {score.execution.score * 100}%")
```

## Environment Variables

| Variable | Description |
|----------|-------------|
| `COMPOSIO_API_KEY` | Your Composio API key (required) |

## Development

```bash
# Clone the repo
git clone https://github.com/composio/tool-router-simulator
cd tool-router-simulator

# Install in dev mode
pip install -e ".[dev]"

# Run tests
pytest

# Format code
black tool_router_sim/
ruff check tool_router_sim/
```

## License

MIT License - see [LICENSE](LICENSE) for details.

## Links

- [Composio Documentation](https://docs.composio.dev)
- [Composio GitHub](https://github.com/composio)
- [Report Issues](https://github.com/composio/tool-router-simulator/issues)
