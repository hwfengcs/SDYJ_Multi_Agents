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

## v0.6 - Extensibility and Runtime Reliability

- [ ] Make the MCP adapter closer to the official MCP tool model.
- [ ] Add plugin-style registration for retrieval tools.
- [ ] Add provider capability metadata such as context window and structured output support.
- [ ] Add retry/timeout policy objects for each retrieval source.
- [ ] Add partial replay from a selected workflow node.
- [ ] Add external benchmark suite loading from JSON/YAML files.
- [ ] Add OpenTelemetry export for trace events.
