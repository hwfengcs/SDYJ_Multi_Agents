# Public Benchmark Results

This page is reserved for externally recognizable benchmark slices such as GAIA
Level 1 and AssistantBench. It intentionally separates real public scores from
local smoke tests.

## Current Status

| Suite | Source | Limit | Status | Result |
| --- | --- | ---: | --- | --- |
| GAIA-style smoke | bundled synthetic fixture | 3 | reproducible harness pass | accuracy `1.0000` |
| GAIA Level 1 | Hugging Face `gaia-benchmark/GAIA` | 5 | blocked until HF login / dataset access is confirmed | not claimed |
| AssistantBench | public slice TBD | TBD | not started | not claimed |

Latest local smoke run: `gaia_20260508_120004`.

Committed smoke artifacts:

- [`summary.json`](public-benchmark-artifacts/gaia-smoke-summary.json)
- [`manifest.jsonl`](public-benchmark-artifacts/gaia-smoke-manifest.jsonl)
- [`predictions.jsonl`](public-benchmark-artifacts/gaia-smoke-predictions.jsonl)
- [`graded.jsonl`](public-benchmark-artifacts/gaia-smoke-graded.jsonl)
- [`failure_analysis.json`](public-benchmark-artifacts/gaia-smoke-failure-analysis.json)
- [`failure_analysis.md`](public-benchmark-artifacts/gaia-smoke-failure-analysis.md)

Local full artifact path: `outputs/public_benchmarks/external_benchmarks/gaia_20260508_120004/summary.json`.

## Reproducible Smoke Command

```bash
sdyj benchmark external --suite gaia --source local --limit 3 --output-dir outputs/public_benchmarks --fail-under 1.0
```

Expected artifact layout:

```text
outputs/public_benchmarks/external_benchmarks/<run-id>/
  manifest.jsonl
  predictions.jsonl
  graded.jsonl
  failure_analysis.json
  failure_analysis.md
  summary.json
```

The bundled fixture is synthetic and exists only to verify the benchmark
harness. Its predictions come from each fixture row's `baseline_prediction`
field (`prediction_source=example_metadata_baseline`), so it should never be
presented as a GAIA score.

Latest local gate check:

```bash
sdyj benchmark run --max-scenarios 1 --max-iterations 2 --fail-under 0.75 --output-dir outputs/verify_benchmark_gate
```

Result: average score `1.0000`, passed. Summary:
`outputs/verify_benchmark_gate/eval_reports/eval_summary_20260508_115841.json`.

## Real GAIA Slice

After Hugging Face access is available:

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

Acceptance criteria before claiming a public score:

- predictions are produced by SDYJ or by a documented SDYJ-compatible runner;
- `summary.json`, `graded.jsonl`, and run configuration are archived;
- failure cases are summarized, not hidden;
- dataset access constraints are documented in `docs/live-run-notes.md`.

Use [`docs/benchmark-failure-analysis-template.md`](benchmark-failure-analysis-template.md)
for every failed threshold, incomplete prediction file, or external blocker.
The summary fields `prediction_coverage`, `missing_prediction_count`,
`missing_expected_answer_count`, `incorrect_task_ids`, and `fail_under_delta`
exist so public benchmark claims can be audited without rerunning the suite.
Each run also writes machine-readable and Markdown failure analysis files so
missing predictions, ungradeable examples, and wrong answers are grouped by
root cause instead of being buried in aggregate accuracy.
