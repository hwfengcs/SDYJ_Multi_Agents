# Benchmark Failure Analysis Template

Use this template whenever a benchmark run misses a threshold, a public slice
has incomplete predictions, or an external dependency blocks a claimed score.
Keep the analysis next to the run artifacts or link it from
`docs/benchmark-results-public.md`.

## Run Metadata

- Run id:
- Date:
- Git commit:
- Command:
- Suite / source / split / limit:
- Provider / model:
- Search mode:
- Output directory:

## Result

- Passed:
- Accuracy / score:
- Fail-under threshold:
- Fail-under delta:
- Prediction coverage:
- Missing prediction count:
- Missing expected-answer count:
- Incorrect task ids:
- Failed scenarios:
- Failed thresholds:

## Artifacts

- Summary:
- Manifest:
- Predictions:
- Graded rows:
- Report:
- Trace bundle:
- Replay run:
- Diff run:

## Root Cause

Choose one or more:

- `missing_prediction`
- `wrong_answer`
- `missing_expected_answer`
- `retrieval_empty`
- `retrieval_drift`
- `tool_error`
- `provider_error`
- `planner_gap`
- `citation_gap`
- `verifier_gap`
- `external_blocker`

Notes:

```text
What happened, where it is visible in artifacts, and why the result should or
should not count as an SDYJ quality regression.
```

## Follow-Up

- Owner:
- Next action:
- Blocker:
- Date to revisit:
