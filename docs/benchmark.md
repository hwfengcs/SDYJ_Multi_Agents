# Benchmark Suite

SDYJ benchmarks are designed as regression checks for agent behavior, not as
one-off demos. A benchmark run should answer three questions:

1. Did the agent produce a useful answer?
2. Was the answer grounded in traceable evidence?
3. Can failures be diagnosed from the trace?

## Run

Offline benchmark with canned evidence and fake LLM:

```bash
python main.py benchmark run --max-scenarios 1 --max-iterations 2
```

Use it as a CI gate:

```bash
python main.py benchmark run --fail-under 0.75
```

Override individual metric thresholds:

```bash
python main.py benchmark run \
  --threshold trace_completeness=0.9 \
  --threshold citation_id_coverage=0.8
```

Check deterministic behavior by repeating offline runs:

```bash
python main.py benchmark run --determinism-repeats 2
```

Compare against a saved summary:

```bash
python main.py benchmark run --compare-summary outputs/eval_reports/eval_summary_YYYYMMDD_HHMMSS.json
python main.py benchmark compare baseline.json candidate.json
```

The older command remains supported:

```bash
python main.py eval --max-scenarios 1
```

## Current Scenarios

| Scenario | What it tests |
| --- | --- |
| `agent_reliability_hard` | RAG, web search, MCP-style tools, human approval, ablation, latency/cost thresholds |
| `tool_failure_recovery_hard` | Timeout, empty result, duplicate URLs, fallback, source quality, review strategy |
| `mcp_rag_ops_hard` | Customer-support MCP+RAG with privacy, refusal, safety boundaries, SLO, and cost budget |

## Metrics

| Metric | Meaning |
| --- | --- |
| `plan_coverage` | Required concepts covered by plan/report |
| `section_completeness` | Expected report sections present |
| `citation_id_coverage` | Evidence IDs cited by the report |
| `citation_density_per_1k_chars` | Citation density normalized by length |
| `tool_success_rate` | Retrieval batches without errors |
| `grounded_key_finding_rate` | Key findings that include evidence IDs |
| `trace_completeness` | Trace has required fields, nodes, events, tool/LLM details, and replay cache |
| `overall_score` | Compact dashboard score |

Each scenario can define thresholds. A benchmark summary includes:

- `passed`
- `failed_scenarios`
- `failed_thresholds`
- `determinism`
- `comparison`
- paths to generated report and trace artifacts

## Live Mode

Run a real model while keeping canned evidence stable:

```bash
python main.py benchmark run \
  --live \
  --provider deepseek \
  --model deepseek-v4-flash \
  --scenario agent_reliability_hard
```

Also evaluate real retrieval:

```bash
python main.py benchmark run --live --live-search
```

Live-search scores can drift because external search results change. Use them
as smoke tests, not as deterministic regression gates.
