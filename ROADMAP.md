# Roadmap

This roadmap turns SDYJ Multi Agents into a practical multi-agent research
framework with observability, replay, evidence grounding, and regression
benchmarks.

## v0.1 - Engineering Baseline

- [x] Python package metadata and CLI entry point.
- [x] Unit tests with fake LLM/search.
- [x] GitHub Actions CI.
- [x] Clear README, license, examples, and contribution guide.
- [x] Consistent API key environment variable handling.

## v0.2 - Evidence-Grounded Reports

- [x] Add an explicit evidence schema for retrieved results.
- [x] Link major report claims to stable evidence IDs and source URLs.
- [x] Deduplicate repeated URLs across search tools.
- [x] Add source quality metadata such as domain, publication date, query, and score.

## v0.3 - Agent Observability

- [x] Persist a JSON trace per run.
- [x] Track tool-call latency, result counts, and error rate.
- [x] Track LLM latency, prompt/response size, and provider token usage where exposed.
- [x] Add `sdyj inspect-run <run-id>` for debugging.

## v0.4 - Evaluation Suite

- [x] Add canned scenario tests for hard agent engineering tasks.
- [x] Add live smoke tests behind an opt-in flag.
- [x] Add report quality checks for coverage, citation density, source deduplication, and tool reliability.
- [x] Publish benchmark results in `docs/evaluation.md`.

## v0.5 - Trace, Replay, and Benchmark Gates

- [x] Add Trace v2 event timeline while preserving legacy trace fields.
- [x] Store run bundles under `outputs/runs/<run-id>/`.
- [x] Add deterministic replay from recorded LLM/tool I/O.
- [x] Add `diff-runs` and `runs list` CLI utilities.
- [x] Add benchmark thresholds, `--fail-under`, summary comparison, and determinism checks.
- [x] Add trace completeness as a benchmark metric.
- [x] Add JSON report output for downstream automation.

## v0.6 - Self-Verifying Deep Research Agent

The v0.6 thesis is that *agent operations are an evaluation problem, not a
prompting problem*. The release pivots SDYJ from a "well-engineered LangGraph
demo" into a differentiated open-source product. See
[docs/release-notes/v0.6.md](docs/release-notes/v0.6.md) for the full plan.

Shipped in 0.6.0a1 and the current v0.6 branch:

- [x] Per-call token + USD cost estimation in `SDYJ_Agents/utils/cost.py`.
- [x] All four LLM providers expose `last_usage` for cost/token capture.
- [x] `trace.metrics` aggregates `total_prompt_tokens`, `total_completion_tokens`, `total_tokens`, `total_cost_usd`, with separate priced / unpriced counters.
- [x] CLI `inspect-run` shows a per-call LLM cost table.
- [x] `summarize_trace` and `diff-runs` include cost deltas.
- [x] PyPI Trusted-Publishers-based release pipeline with separate TestPyPI / PyPI tracks.
- [x] Packaging extras: `[web]`, `[mcp]`, `[benchmarks]`, `[all]`.
- [x] English-first README with comparison table and v0.6 status.
- [x] Verifier agent + bounded revise loop (5th LangGraph node).
- [x] Reflexive Researcher: query rewrite + retry on empty / low-relevance batches.
- [x] Plan refinement: Planner sees collected evidence after N tasks and adapts the rest of the plan.
- [x] Parallel tool execution within a task (asyncio.gather + concurrency limit).
- [x] Prompt-engineering pass for the new JSON-producing paths, using native JSON mode where providers support it.
- [x] Streamlit Web UI MVP with trace, plan, evidence, report, and cost tabs.
- [x] Offline v0.5-vs-v0.6 algorithm ablation tracked in `docs/benchmark-results.md`.
- [x] Static trace viewer for `trace.json` / `events.jsonl` inspection.
- [x] Real MCP client adapter via the official `mcp` Python SDK, with legacy HTTP fallback.
- [x] Hugging Face Spaces-compatible root `app.py` entrypoint.
- [x] `sdyj doctor` no-network environment preflight for local and hosted runs.

In progress on `feat/v0.6-self-verifying`:

- [ ] Harden live end-to-end runs across DeepSeek/OpenAI/Claude/Gemini and document known provider quirks.
- [ ] Live Hugging Face Spaces deployment for the Streamlit UI.
- [ ] Public benchmark scores: GAIA Level 1 subset + AssistantBench, including v0.5-vs-v0.6 ablation.
- [ ] MCP demo scripts for filesystem and GitHub servers.
- [ ] Publish the static trace viewer on GitHub Pages.

## v0.7 - Extensibility and Runtime Reliability

- [ ] Make the MCP adapter closer to the official MCP tool model.
- [ ] Add plugin-style registration for retrieval tools.
- [ ] Add provider capability metadata such as context window and structured output support.
- [ ] Add retry/timeout policy objects for each retrieval source.
- [ ] Add partial replay from a selected workflow node.
- [ ] Add external benchmark suite loading from JSON/YAML files.
- [ ] Add OpenTelemetry export for trace events.
