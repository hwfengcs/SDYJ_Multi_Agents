# SDYJ Multi Agents

English | [中文](README.md)

[![CI](https://github.com/hwfengcs/SDYJ_Multi_Agents/actions/workflows/ci.yml/badge.svg)](https://github.com/hwfengcs/SDYJ_Multi_Agents/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)

SDYJ Multi Agents is a LangGraph-based deep research assistant. It turns an open-ended research request into a traceable workflow: intent classification, research planning, human review, multi-source retrieval, and synthesized Markdown or HTML reports.

The project is designed to demonstrate practical AI agent engineering: workflow control, tool use, human-in-the-loop approval, provider-agnostic LLM integration, source-grounded reporting, tests, and CI.

## Highlights

- **Multi-agent workflow**: Coordinator, Planner, Researcher, and Rapporteur cooperate through a shared state.
- **Human-in-the-loop planning**: generated plans can be approved or revised before execution.
- **Provider-agnostic LLM layer**: DeepSeek, OpenAI, Claude, and Gemini share one interface.
- **Multi-source retrieval**: Tavily, arXiv, and an MCP adapter for external tools.
- **Evidence-grounded reports**: retrieved results are normalized into `E1/E2/...` evidence items with URL deduplication, domain, query, date, and score metadata.
- **Agent observability**: every run can persist a JSON trace with nodes, LLM calls, tool calls, latency, errors, and report metrics.
- **Reproducible evaluation**: hard built-in agent scenarios support offline canned-evidence evaluation and real DeepSeek live evaluation.
- **Engineering-first repo**: Python packaging, CLI entry point, tests, CI, docs, and examples.
- **Internship-ready story**: emphasizes traceability, evaluation, tool reliability, and source grounding.

## Architecture

```text
User Query
    |
    v
Coordinator -- classify intent / initialize state
    |
    v
Planner -- build structured research plan
    |
    v
Human Review -- approve or request changes
    |
    v
Researcher -- Tavily / arXiv / MCP retrieval
    |
    v
Rapporteur -- synthesize Markdown or HTML report
```

See [docs/architecture.md](docs/architecture.md) for the full design notes.

## Quick Start

### 1. Install

```bash
git clone https://github.com/hwfengcs/SDYJ_Multi_Agents.git
cd SDYJ_Multi_Agents
python -m pip install -e ".[dev]"
```

You can also install runtime dependencies only:

```bash
python -m pip install -r requirements.txt
```

### 2. Configure Environment Variables

```bash
copy .env.example .env
```

Fill at least one LLM API key. DeepSeek is the recommended first provider:

```bash
LLM_PROVIDER=deepseek
LLM_MODEL=deepseek-v4-flash
DEEPSEEK_API_KEY=sk-...
TAVILY_API_KEY=tvly-...
```

Claude and Gemini use the official variable names:

```bash
ANTHROPIC_API_KEY=sk-ant-...
GOOGLE_API_KEY=AIza...
```

Legacy aliases `CLAUDE_API_KEY` and `GEMINI_API_KEY` are still supported.

### 3. Run

```bash
python main.py config-info
python main.py list-models deepseek
python main.py research "Summarize recent trends in AI agent evaluation"
```

After editable installation:

```bash
sdyj research "Compare the design trade-offs of LangGraph, AutoGen, and CrewAI"
```

Common options:

```bash
python main.py research \
  --provider deepseek \
  --model deepseek-v4-flash \
  --max-iterations 3 \
  --output-format markdown \
  --auto-approve \
  "How should RAG agents be evaluated for reliability?"
```

Run without a query to open the interactive menu:

```bash
python main.py
```

### 4. Inspect Traces

Runs write JSON traces to `outputs/traces/`:

```bash
python main.py inspect-run
python main.py inspect-run <run-id>
```

### 5. Evaluate

Offline evaluation uses deterministic hard scenarios and canned evidence, so no real API key is required:

```bash
python main.py list-scenarios
python main.py eval --max-scenarios 1 --max-iterations 2
```

Live DeepSeek evaluation uses `DEEPSEEK_API_KEY` from `.env` while still defaulting to canned evidence for reproducibility:

```bash
python main.py eval \
  --live \
  --provider deepseek \
  --model deepseek-v4-flash \
  --scenario agent_reliability_hard \
  --max-iterations 2
```

Add `--live-search` if you also want to evaluate real retrieval.

## Project Layout

```text
SDYJ_Agents/
  agents/       # Coordinator / Planner / Researcher / Rapporteur
  cli/          # argparse CLI and interactive menu
  llm/          # provider-agnostic LLM wrappers
  prompts/      # Jinja prompt templates
  tools/        # Tavily, arXiv, MCP adapters
  workflow/     # LangGraph graph and state
  utils/        # config, logging, evidence, tracing
  evaluation/   # scenarios, metrics, eval runner
docs/           # architecture, evaluation, roadmap context
examples/       # small reproducible examples
tests/          # unit tests with fake LLM/search
```

## Output

- Markdown: best for version control, editing, and report drafts.
- HTML: best for browser-based sharing and demos.

Generated artifacts are written to `outputs/`, which is ignored by git:

- `outputs/research_report_*.md|html`: research reports.
- `outputs/traces/*.json`: run traces.
- `outputs/eval_reports/`: evaluation reports and summaries.

Repository examples are available in [examples/sample_report.md](examples/sample_report.md), [examples/sample_trace.json](examples/sample_trace.json), and [examples/eval_summary.json](examples/eval_summary.json).

## Test

```bash
pytest
ruff check SDYJ_Agents tests
```

Unit tests use fake LLM and fake search implementations, so real API keys are not required.

## Roadmap

Near-term work:

- Add an evidence schema so claims can point to source URL, query, and confidence. Done.
- Track trace, token, and latency metrics for agent evaluation. Done.
- Add end-to-end demos and benchmark scenarios. Done.
- Make the MCP adapter more standards-aligned and easier to extend.

See [ROADMAP.md](ROADMAP.md).

## Why This Matters

This project is suitable as an AI agent engineering portfolio piece. It gives you concrete topics to discuss in interviews:

- designing controllable agent workflows with LangGraph;
- reducing execution risk through human-in-the-loop planning;
- grounding generated reports in retrieved sources;
- testing and maintaining LLM applications with mocks and CI.

## Contributing

Issues and pull requests are welcome. See [CONTRIBUTING.md](CONTRIBUTING.md).

## License

MIT. See [LICENSE](LICENSE).
