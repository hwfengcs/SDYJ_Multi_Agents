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

- [ ] Add an explicit evidence schema for retrieved results.
- [ ] Link every major report claim to source URLs.
- [ ] Deduplicate repeated URLs across search tools.
- [ ] Add source quality metadata such as domain, publication date, and score.

## v0.3 - Agent Observability

- [ ] Persist a JSON trace per run.
- [ ] Track tool-call latency and error rate.
- [ ] Track token usage and estimated cost where providers expose it.
- [ ] Add `sdyj inspect-run <run-id>` for debugging.

## v0.4 - Evaluation Suite

- [ ] Add canned scenario tests for academic, industry, and product research.
- [ ] Add live smoke tests behind an opt-in flag.
- [ ] Add report quality checks for coverage, citation density, and redundancy.
- [ ] Publish benchmark results in `docs/evaluation.md`.

## v0.5 - Extensibility

- [ ] Make the MCP adapter closer to the official MCP tool model.
- [ ] Add plugin-style registration for retrieval tools.
- [ ] Add provider capability metadata such as context window and structured output support.
- [ ] Add JSON output for downstream automation.
