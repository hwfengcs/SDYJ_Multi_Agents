# SDYJ Multi Agents

English | [中文](README.md)

[![CI](https://github.com/hwfengcs/SDYJ_Multi_Agents/actions/workflows/ci.yml/badge.svg)](https://github.com/hwfengcs/SDYJ_Multi_Agents/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)

SDYJ Multi Agents is a LangGraph-based multi-agent research framework. It turns an open-ended request into a controllable workflow: intent classification, planning, human review, multi-source retrieval, and synthesized Markdown, HTML, or JSON reports.

The project focuses on practical agent operations: Trace v2 observability, deterministic replay, durable checkpoints with resume, evidence-grounded reporting with citation validation, graceful degradation, benchmark gates, provider-agnostic LLM integration, tests, and CI.

## Highlights

- **Multi-agent workflow**: Coordinator, Planner, Researcher, and Rapporteur cooperate through a shared state.
- **Human-in-the-loop planning**: generated plans can be approved or revised before execution.
- **Provider-agnostic LLM layer**: DeepSeek, OpenAI, Claude, and Gemini share one interface.
- **Multi-source retrieval**: Tavily, arXiv, and an MCP adapter for external tools.
- **Evidence-grounded reports**: results are normalized into `E1/E2/...` evidence items; prompt evidence is budgeted by relevance; the LLM is instructed to cite `[E#]` inline (heuristics only backfill), and fabricated ids are stripped by post-generation validation.
- **Graceful degradation**: transient LLM errors retry with exponential backoff; a failed task or report section degrades visibly instead of killing the run; every degradation is recorded in state, trace, and report metrics.
- **Durable checkpoints & resume**: research runs write a per-run SQLite checkpoint; `resume <run-id>` continues a crashed or interrupted run — including a pending human-review gate.
- **Trace v2 observability**: every run records an event timeline with nodes, LLM calls (incl. retry counts), tool calls, routing decisions, latency, errors, report metrics, and replay cache.
- **Deterministic replay**: recorded LLM responses are matched by prompt hash first (sequential fallback for legacy traces), so replay survives call-order changes.
- **Practical benchmarks**: hard built-in scenarios (incl. an LLM-failure-injection scenario) support thresholds, `--fail-under`, summary comparison, trace completeness, offline determinism checks, and a dual-track LLM-as-judge faithfulness score.
- **Engineering-first repo**: Python packaging, CLI entry point, tests, CI, docs, and examples.

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

### 4. Trace / Replay / Resume

Runs write a bundle under `outputs/runs/<run-id>/` (trace, events, final or partial state, per-run sqlite checkpoint, report) and keep a backward-compatible copy under `outputs/traces/<run-id>.json`:

```bash
python main.py inspect-run
python main.py inspect-run <run-id> --timeline
python main.py runs list
python main.py replay <run-id>
python main.py resume <run-id>       # continue a crashed/interrupted run
python main.py diff-runs <run-a> <run-b>
```

Crashed research runs persist `state.partial.json` and exit with code 4 (eval gates keep 2/3). See [docs/trace-replay.md](docs/trace-replay.md).

### 5. Benchmark

Offline benchmarks use deterministic hard scenarios and canned evidence, so no real API key is required:

```bash
python main.py list-scenarios
python main.py benchmark run --max-scenarios 1 --max-iterations 2
python main.py benchmark run --fail-under 0.75
python main.py benchmark run --determinism-repeats 2
```

Live DeepSeek evaluation uses `DEEPSEEK_API_KEY` from `.env` while still defaulting to canned evidence for reproducibility:

```bash
python main.py benchmark run \
  --live \
  --provider deepseek \
  --model deepseek-v4-flash \
  --scenario agent_reliability_hard \
  --max-iterations 2
```

Add `--live-search` if you also want to evaluate real retrieval.

The older `python main.py eval ...` command remains supported. See [docs/benchmark.md](docs/benchmark.md).

## Project Layout

```text
SDYJ_Agents/
  agents/       # Coordinator / Planner / Researcher / Rapporteur
  cli/          # argparse CLI and interactive menu
  llm/          # provider-agnostic LLM wrappers
  prompts/      # Jinja prompt templates (with stable [PROMPT_ID] markers)
  tools/        # Tavily, arXiv, MCP adapters
  workflow/     # LangGraph graph, state, durable checkpoint, resume
  utils/        # config, logging, evidence, tracing, retry policy
  evaluation/   # benchmark scenarios, metrics, runner, LLM-as-judge
docs/           # architecture, trace/replay, benchmark, roadmap
examples/       # small reproducible examples
tests/          # unit tests with fake LLM/search
```

## Output

- Markdown: best for version control, editing, and report drafts.
- HTML: best for browser-based sharing and demos.
- JSON: best for downstream automation, regression checks, and integration.

Generated artifacts are written to `outputs/`, which is ignored by git:

- `outputs/research_report_*.md|html|json`: research reports.
- `outputs/runs/<run-id>/`: Trace v2 run bundles.
- `outputs/traces/*.json`: backward-compatible run traces.
- `outputs/eval_reports/`: evaluation reports and summaries.

Repository examples are available in [examples/sample_report.md](examples/sample_report.md), [examples/sample_trace.json](examples/sample_trace.json), and [examples/eval_summary.json](examples/eval_summary.json).

## Test

```bash
pytest
ruff check SDYJ_Agents tests
```

Unit tests use fake LLM and fake search implementations, so real API keys are not required.

## Roadmap

v0.6 includes:

- Citation-integrity pipeline: instructed citations + post-generation validation + validity metrics. Done.
- Graceful degradation: transient-error retry, node/section-level fallbacks, partial-state persistence, nonzero crash exit codes. Done.
- Durable SqliteSaver checkpoints and the `resume` command. Done.
- Prompt-hash replay matching (legacy traces fall back to order). Done.
- Dual-track LLM-as-judge faithfulness plus an LLM-failure-injection benchmark scenario. Done.

v0.5 includes:

- Trace v2 event timeline and run bundles. Done.
- Deterministic replay. Done.
- Benchmark gates, thresholds, summary comparison, and trace completeness. Done.
- JSON output. Done.

Next work: partial replay, external benchmark suites, a stronger tool registry, OpenTelemetry export, and a run-inspection web UI.

See [ROADMAP.md](ROADMAP.md).

## Contributing

Issues and pull requests are welcome. See [CONTRIBUTING.md](CONTRIBUTING.md).

## License

MIT. See [LICENSE](LICENSE).
