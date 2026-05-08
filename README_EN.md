# SDYJ Multi Agents

**A self-verifying, replayable, benchmarkable multi-agent research framework.**

English | [中文](README.md)

[![CI](https://github.com/hwfengcs/SDYJ_Multi_Agents/actions/workflows/ci.yml/badge.svg)](https://github.com/hwfengcs/SDYJ_Multi_Agents/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/sdyj-multi-agents?color=blue)](https://pypi.org/project/sdyj-multi-agents/)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)
[![Stars](https://img.shields.io/github/stars/hwfengcs/SDYJ_Multi_Agents?style=social)](https://github.com/hwfengcs/SDYJ_Multi_Agents)

<!-- Hosted demo badges are prepared but hidden until the URLs are verified:
[![Open in Spaces](https://img.shields.io/badge/Spaces-Try_demo-blue)](https://huggingface.co/spaces/<owner>/sdyj-multi-agents)
[![Trace Viewer](https://img.shields.io/badge/GitHub_Pages-Trace_Viewer-blue)](https://hwfengcs.github.io/SDYJ_Multi_Agents/trace-viewer-demo.html)
-->

> SDYJ turns an open-ended research request into a controllable LangGraph workflow — plan, human review, multi-source retrieval, evidence-grounded synthesis — with a Trace v2 event timeline, deterministic replay, per-call cost tracking, and benchmark-gated CI built in. The goal is **agent operations**, not another LangGraph hello-world demo.

## Why SDYJ

Most open-source agent frameworks ship a happy-path workflow and stop there. The moment something goes wrong — a tool returning empty, an LLM hallucinating a citation, a plan drifting off-topic — there is no good way to investigate, no replay, no regression test. SDYJ is built around the assumption that *agent operations are an evaluation problem*, not a prompting problem:

- Every node, LLM call, tool call, routing decision, and report metric becomes a structured trace event.
- Every run is replayable from recorded I/O — no real API calls needed to reproduce a failure.
- Every release is gated on hard scenario benchmarks with thresholds, so quality regressions break CI before they ship.
- Every retrieval result is normalized into deduplicated `E1/E2/...` evidence items linked back to the report claims.

## Comparison

| Capability                        | SDYJ Multi Agents | GPT Researcher | AutoGen | LangGraph examples |
|-----------------------------------|:-:|:-:|:-:|:-:|
| LangGraph state machine           | ✅ | ❌ | ⚠️ | ✅ |
| Human-in-the-loop plan approval   | ✅ | ❌ | ⚠️ | ⚠️ |
| Evidence-grounded `E1/E2/...` IDs | ✅ | ⚠️ | ❌ | ❌ |
| Trace v2 event timeline           | ✅ | ❌ | ⚠️ | ❌ |
| Deterministic replay from trace   | ✅ | ❌ | ❌ | ❌ |
| Per-call token + USD cost in trace| ✅ | ⚠️ | ⚠️ | ❌ |
| Benchmark gates with `--fail-under` | ✅ | ❌ | ❌ | ❌ |
| 4 LLM providers w/ unified abstraction | ✅ | ✅ | ✅ | ⚠️ |
| Self-verifying revise loop        | ✅ v0.6 alpha | ❌ | ❌ | ❌ |
| Public benchmark numbers (GAIA / AssistantBench) | 🚧 v0.6 | ⚠️ | ⚠️ | ❌ |

✅ first-class · ⚠️ partial / requires custom code · ❌ not provided · 🚧 in progress

## Quick start (60 seconds)

```bash
conda env create -f environment.yml
conda activate sdyj
cp .env.example .env                    # Windows PowerShell: copy .env.example .env
sdyj research "How should RAG agents be evaluated for reliability?"
```

Fill `DEEPSEEK_API_KEY` and `TAVILY_API_KEY` in `.env` before running live research. See [docs/conda-setup.md](docs/conda-setup.md) for the full environment guide.

Run without a query to enter the interactive menu:

```bash
sdyj
```

Container quick start:

```bash
docker build -t sdyj:0.6 .
docker run --rm sdyj:0.6 sdyj --help
docker compose up --build
```

See [docs/docker.md](docs/docker.md) for Docker and Compose deployment notes.

## Architecture

```text
User Query
    │
    ▼
Coordinator ─ classify intent / initialize state
    │
    ▼
Planner ─ build structured research plan ─────────┐
    │                                              │
    ▼                                              │ revise
Human Review ─ approve or request changes ────────┘
    │ approve
    ▼
Researcher ─ Tavily / arXiv / MCP retrieval (loop)
    │
    ▼
Rapporteur ─ Markdown / HTML / JSON report
    │
    ▼
Verifier ─ critique + revise loop
    │
    ▼
Trace v2 bundle → outputs/runs/<run-id>/
```

See [docs/architecture.md](docs/architecture.md) for the full design notes.

## What's new in v0.6 (in progress)

- **Per-call token + USD cost tracking** in every LLM call — see [`SDYJ_Agents/utils/cost.py`](SDYJ_Agents/utils/cost.py). The CLI `inspect-run` shows a per-call cost table and the trace `metrics` block aggregates totals.
- **Provider-agnostic usage capture**: OpenAI, Claude, DeepSeek, and Gemini now expose `last_usage` so cost estimation works regardless of provider.
- **Verifier loop** — a critic agent re-reads the report against evidence and can trigger bounded Rapporteur revisions when claims are unsupported.
- **Reflexive Researcher** — empty, failing, or low-relevance query batches now trigger one query-rewrite retry.
- **Mid-flight plan refinement** — after enough subtasks complete, the Planner can revise the remaining plan based on collected evidence.
- **Parallel tool execution** — each task can run its `(query, source)` lookups concurrently with a bounded concurrency limit.
- **Structured output path** — Planner, Rapporteur organization, Researcher reflection, and Verifier use provider-native JSON mode when available.
- **Streamlit Web UI MVP** — run it locally with `streamlit run streamlit_app.py` or `streamlit run SDYJ_Agents/web/app.py`.
- **PyPI release pipeline** with Trusted Publishers — see [docs/release-process.md](docs/release-process.md).
- **Public benchmark scores** — GAIA Level 1 subset and AssistantBench results, including v0.5-vs-v0.6 ablations. *Coming soon.*
- **Hugging Face Spaces deployment** for the Streamlit app. *Coming soon.*
- **Real MCP integration** via the official `mcp` Python SDK for stdio and streamable HTTP, with the legacy HTTP shim preserved as fallback.

The full v0.6 plan lives in [`docs/release-notes/v0.6.md`](docs/release-notes/v0.6.md) and [ROADMAP.md](ROADMAP.md).

## v0.6 release-readiness snapshot (2026-05-08)

The locally verifiable release work is mostly closed down:

- `python -m pytest`: `134 passed, 1 xfailed`.
- `python -m ruff check SDYJ_Agents tests examples`: passed.
- `python -m build` and `python -m twine check dist/*`: passed.
- GitHub Pages, Hugging Face Spaces, TestPyPI/PyPI, Docker, and GHCR now have
  local docs, workflows, or static gates. Real public URLs, packages, and
  images still require platform-side setup.
- The public benchmark harness now commits synthetic GAIA-style smoke
  `summary`, `manifest`, `predictions`, and `graded` artifacts. See
  [docs/benchmark-results-public.md](docs/benchmark-results-public.md). This is
  not a GAIA public score; it only proves the runner, grader, and artifact
  layout are reproducible.
- The MCP filesystem demo no-secret `--check` passes locally. The GitHub MCP
  `--check` fails safely without `GITHUB_PERSONAL_ACCESS_TOKEN` and prints only
  boolean status.

External conditions still required:

- Configure a real `TAVILY_API_KEY`, then rerun the DeepSeek + Tavily + arXiv
  live smoke and validate it with `inspect-run`, `replay`, and `diff-runs`.
- Run Docker build/run/compose smoke on a Docker-enabled host.
- Enable GitHub Pages, create the Hugging Face Space, configure Trusted
  Publishers, and only then unhide the README badges with verified URLs.
- Get Hugging Face GAIA dataset access plus real predictions before claiming a
  GAIA Level 1 slice result.

## Trace, replay, and inspection

Every run writes a bundle under `outputs/runs/<run-id>/` and a backward-compatible copy at `outputs/traces/<run-id>.json`:

```bash
sdyj inspect-run                       # latest run, summary + tool calls + LLM cost table
sdyj inspect-run <run-id> --timeline   # full event timeline
sdyj runs list
sdyj replay <run-id>                   # rebuild from recorded LLM/tool I/O, no real calls
sdyj diff-runs <run-a> <run-b>
sdyj doctor                            # local/deployment preflight, no API calls
```

Open `SDYJ_Agents/web/trace_viewer.html` in a browser to inspect `trace.json`
or `events.jsonl` with client-side filters and event details.

See [docs/trace-replay.md](docs/trace-replay.md).

## Benchmarks

Offline benchmarks use deterministic hard scenarios with canned evidence, so they require no real API keys and run in CI:

```bash
sdyj list-scenarios
sdyj benchmark run --max-scenarios 1 --max-iterations 2
sdyj benchmark run --fail-under 0.75            # gate for CI regression blocking
sdyj benchmark run --determinism-repeats 2      # offline determinism check
```

Live DeepSeek evaluation isolates model quality while keeping retrieval canned for reproducibility:

```bash
sdyj benchmark run \
  --live \
  --provider deepseek \
  --model deepseek-v4-flash \
  --scenario agent_reliability_hard \
  --max-iterations 2
```

Add `--live-search` for real retrieval. See [docs/benchmark.md](docs/benchmark.md).

The public benchmark harness now has a reproducible entrypoint:

```bash
sdyj benchmark external --suite gaia --source local --limit 3 --output-dir outputs/public_benchmarks
```

The local source is a synthetic GAIA-style smoke fixture; it only proves the
runner, grader, and artifact layout. Real GAIA Level 1 runs should use
`--source hf` after Hugging Face login and dataset access are confirmed. Results
are tracked in [docs/benchmark-results-public.md](docs/benchmark-results-public.md).

## Project layout

```text
SDYJ_Agents/
  agents/       # Coordinator / Planner / Researcher / Rapporteur / Verifier
  cli/          # argparse CLI and interactive menu
  llm/          # provider-agnostic LLM wrappers (OpenAI / Claude / Gemini / DeepSeek)
  prompts/      # Jinja prompt templates
  tools/        # Tavily, arXiv, MCP adapters
  workflow/     # LangGraph graph, state, nodes
  utils/        # config, logging, evidence, tracing, cost
  evaluation/   # benchmark scenarios, metrics, runner
docs/           # architecture, trace/replay, benchmark, release process
examples/       # small reproducible examples
tests/          # unit tests with fake LLM/search
```

## Configuration

```bash
copy .env.example .env
```

Fill at least one LLM API key. DeepSeek is the recommended first provider (cheapest):

```bash
LLM_PROVIDER=deepseek
LLM_MODEL=deepseek-v4-flash
DEEPSEEK_API_KEY=sk-...
TAVILY_API_KEY=tvly-...
```

Other providers use their official environment variable names — `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, `GOOGLE_API_KEY`. Legacy aliases `CLAUDE_API_KEY` and `GEMINI_API_KEY` remain supported.

MCP setup is optional. See [docs/mcp.md](docs/mcp.md) for stdio,
streamable HTTP, and legacy HTTP configuration.

## Output formats

- Markdown — best for version control, editing, and report drafts.
- HTML — best for browser-based sharing and demos.
- JSON — best for downstream automation, regression checks, and integration.

Sample artifacts: [examples/sample_report.md](examples/sample_report.md) · [examples/sample_trace.json](examples/sample_trace.json) · [examples/eval_summary.json](examples/eval_summary.json).

## Develop

```bash
conda env update -n sdyj -f environment.yml --prune
conda activate sdyj
pytest
ruff check SDYJ_Agents tests
```

Conda is the default development environment. Unit tests use fake LLM and fake search implementations, so real API keys are not required.

## Roadmap

| Milestone | Status |
|-----------|--------|
| v0.1 — engineering baseline (CI, tests, license, docs) | ✅ |
| v0.2 — evidence-grounded reports with deduplicated source IDs | ✅ |
| v0.3 — agent observability (Trace v2, latency, error rate) | ✅ |
| v0.4 — evaluation suite with hard scenarios and report quality metrics | ✅ |
| v0.5 — Trace v2, deterministic replay, benchmark gates, JSON output | ✅ |
| **v0.6 — self-verifying loop, cost tracking, public benchmarks, Web UI, MCP** | 🚧 |
| v0.7 — partial replay, OpenTelemetry export, plugin retrieval registry | ⏳ |

See [ROADMAP.md](ROADMAP.md) for the full plan.

## Contributing

Issues and pull requests are welcome. See [CONTRIBUTING.md](CONTRIBUTING.md).

## License

MIT. See [LICENSE](LICENSE).
