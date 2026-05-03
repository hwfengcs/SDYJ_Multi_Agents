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

Latest local smoke artifact: `outputs/public_benchmarks/external_benchmarks/gaia_20260503_113601/summary.json`.

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
  summary.json
```

The bundled fixture is synthetic and exists only to verify the benchmark
harness. It should never be presented as a GAIA score.

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
