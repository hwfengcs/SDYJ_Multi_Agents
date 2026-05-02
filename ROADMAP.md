# Roadmap

This roadmap turns SDYJ Multi Agents from an MVP into a stronger open-source
portfolio project for AI agent engineering roles.

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

## v0.5 - Extensibility

- [ ] Make the MCP adapter closer to the official MCP tool model.
- [ ] Add plugin-style registration for retrieval tools.
- [ ] Add provider capability metadata such as context window and structured output support.
- [ ] Add JSON output for downstream automation.
