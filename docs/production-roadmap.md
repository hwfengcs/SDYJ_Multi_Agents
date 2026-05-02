# Production Roadmap

SDYJ is moving toward a practical multi-agent research framework. The core
design principle is that agent behavior should be controllable, observable,
replayable, and benchmarked.

## Implemented in v0.5

- Trace v2 event timeline.
- Run bundles under `outputs/runs/<run-id>/`.
- Deterministic replay from recorded LLM and tool I/O.
- Trace diff for operational comparison.
- Benchmark gates with thresholds and `--fail-under`.
- Benchmark summary comparison.
- Trace completeness metric.
- Offline determinism checks.
- Markdown, HTML, and JSON report outputs.

## Near-Term Work

- Partial replay from a selected node.
- External benchmark suite loading from JSON/YAML files.
- Stronger tool registry with plugin-style retrieval adapters.
- Retry and timeout policy objects for each tool.
- Cost estimation per provider.
- Dataset-backed citation precision checks.
- OpenTelemetry export for traces.
- Web UI for run inspection.

## Deployment Considerations

- Store traces in a private artifact store if prompts or tool outputs contain
  sensitive data.
- Keep live-search benchmark results separate from deterministic offline gates.
- Use `benchmark run --fail-under` in CI for regression blocking.
- Use `replay` and `diff-runs` when a benchmark regression is detected.
