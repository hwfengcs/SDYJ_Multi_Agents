# Benchmark Suite

SDYJ benchmarks are designed as regression checks for agent behavior, not as
one-off demos. A benchmark run should answer three questions:

1. Did the agent produce a useful answer?
2. Was the answer grounded in traceable evidence?
3. Can failures be diagnosed from the trace?

## Run

Offline benchmark with canned evidence and fake LLM:

```bash
sdyj benchmark run --max-scenarios 1 --max-iterations 2
```

Use it as a CI gate:

```bash
sdyj benchmark run --fail-under 0.75
```

Override individual metric thresholds:

```bash
sdyj benchmark run \
  --threshold trace_completeness=0.9 \
  --threshold citation_id_coverage=0.8
```

Check deterministic behavior by repeating offline runs:

```bash
sdyj benchmark run --determinism-repeats 2
```

Compare against a saved summary:

```bash
sdyj benchmark run --compare-summary outputs/eval_reports/eval_summary_YYYYMMDD_HHMMSS.json
sdyj benchmark compare baseline.json candidate.json
```

Comparison is metric-aware. It checks the dashboard score and the key
per-metric slices (`plan_coverage`, citation coverage/density, tool success,
grounded findings, and trace completeness). A metric delta below `-0.02` is
treated as a regression even if the aggregate score stayed flat. The comparison
payload also records:

- new or missing scenarios;
- feature/context changes such as enabling verifier, reflection, refinement, or
  parallel tools;
- mean per-metric deltas across compared scenarios;
- root-cause rollups such as `planner_gap`, `citation_gap`, `tool_error`, and
  `trace_gap`;
- the top metric regressions and improvements.

The older command remains supported:

```bash
sdyj eval --max-scenarios 1
```

The current v0.5-vs-v0.6 algorithm ablation table is tracked in
[`docs/benchmark-results.md`](benchmark-results.md).

## External Public Benchmark Slices

`sdyj benchmark external` is the reproducible harness for public benchmark
slices such as GAIA. It separates three things that are easy to conflate:

- loading examples,
- collecting or importing predictions,
- grading predictions and writing artifacts.

Dry-run the pipeline without API keys or Hugging Face login:

```bash
sdyj benchmark external --suite gaia --source local --limit 3 --output-dir outputs/public_benchmarks
```

This uses a tiny synthetic GAIA-style fixture. It is not a public score; it
only proves the command, output layout, and grader are working.

Grade a predictions file:

```bash
sdyj benchmark external \
  --suite gaia \
  --source jsonl \
  --data-path data/gaia_sample.jsonl \
  --predictions outputs/my_gaia_predictions.jsonl \
  --limit 5 \
  --output-dir outputs/public_benchmarks
```

Prediction rows are JSONL:

```jsonl
{"task_id":"example-id","prediction":"final short answer"}
```

Attempt a real Hugging Face GAIA slice after logging in and accepting dataset
terms:

```bash
huggingface-cli login
sdyj benchmark external \
  --suite gaia \
  --source hf \
  --hf-dataset gaia-benchmark/GAIA \
  --hf-config 2023_level1 \
  --split validation \
  --limit 5 \
  --predictions outputs/gaia_predictions.jsonl \
  --output-dir outputs/public_benchmarks
```

Each run writes:

- `manifest.jsonl` with task ids/questions and `has_expected_answer`;
- `predictions.jsonl`, either imported from `--predictions` or generated from
  fixture baseline predictions;
- `graded.jsonl` with per-task correctness;
- `summary.json` with accuracy, prediction coverage, missing prediction count,
  missing expected-answer count, incorrect task ids, fail-under delta,
  pass/fail status, failure-analysis rollups, and artifact paths;
- `failure_analysis.json` and `failure_analysis.md` with per-task root causes
  (`missing_prediction`, `wrong_answer`, `missing_expected_answer`) and
  grouped task ids for audit.

Real GAIA validation/test data may be gated. If Hugging Face access is missing,
record that blocker in `docs/live-run-notes.md` and use `--source local` or
`--source jsonl` to keep the harness itself tested.

When a public or external run fails, fill out
[`docs/benchmark-failure-analysis-template.md`](benchmark-failure-analysis-template.md)
and link it from the result notes. The goal is to make missing predictions,
dataset access issues, retrieval drift, and actual model/agent failures
distinguishable.

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
- `failure_analysis` with failed metric counts and root-cause buckets such as
  `planner_gap`, `citation_gap`, `tool_error`, `trace_gap`, and
  `verifier_gap`
- `determinism`
- `comparison`, including `regression_analysis` with context changes,
  metric-delta summary, root-cause counts, missing/new scenarios, and top
  regressions/improvements
- paths to generated report and trace artifacts

## Live Mode

Run a real model while keeping canned evidence stable:

```bash
sdyj benchmark run \
  --live \
  --provider deepseek \
  --model deepseek-v4-flash \
  --scenario agent_reliability_hard
```

Also evaluate real retrieval:

```bash
sdyj benchmark run --live --live-search
```

Live-search scores can drift because external search results change. Use them
as smoke tests, not as deterministic regression gates.
