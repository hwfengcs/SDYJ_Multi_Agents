# SDYJ v0.6 Algorithm Benchmark Results

Generated on 2026-05-03 with the built-in offline hard scenarios and canned evidence.
These runs are deterministic smoke/ablation benchmarks, not public GAIA/AssistantBench
scores. They are useful for verifying that v0.6 algorithm features preserve the
existing benchmark gate while adding verifier/reflection/refinement/parallel trace
coverage.

## Commands

```bash
# v0.5-compatible baseline: verifier/reflection/refinement/parallel all off
sdyj benchmark run --max-scenarios 3 --max-iterations 2 --output-dir outputs\phase2_v05

# v0.6 algorithm stack: verifier + reflection + plan refinement + parallel tools
sdyj benchmark run --max-scenarios 3 --max-iterations 2 --output-dir outputs\phase2_v06 --enable-verify --enable-reflect --enable-refine-plan --enable-parallel-tools
```

## Summary

| Configuration | Avg overall | Passed | Verification | Reflection | Plan refinement | Parallel tools |
| --- | ---: | --- | --- | --- | --- | --- |
| v0.5-compatible | 0.9693 | Yes | Off | Off | Off | Off |
| v0.6 algorithm stack | 0.9693 | Yes | On | On | On | On |

The offline canned scenarios already score near the ceiling, so v0.6 does not
move the aggregate score in this deterministic run. The value of this benchmark
is regression protection: the added verifier JSON call and trace events do not
break plan coverage, citation coverage, grounded finding rate, tool success, or
trace completeness.

## Scenario Comparison

| Scenario | v0.5 overall | v0.6 overall | Citation coverage | Grounded finding rate | Tool success | Trace completeness |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `agent_reliability_hard` | 1.0000 | 1.0000 | 1.00 | 1.00 | 1.00 | 1.00 |
| `tool_failure_recovery_hard` | 0.9824 | 0.9824 | 1.00 | 1.00 | 1.00 | 1.00 |
| `mcp_rag_ops_hard` | 0.9256 | 0.9256 | 1.00 | 1.00 | 1.00 | 1.00 |

## Trace and Cost-Relevant Operational Data

| Configuration | Avg node latency (offline ms) | Avg LLM calls | Avg tool calls | Avg Trace v2 events | Verifier quality |
| --- | ---: | ---: | ---: | ---: | --- |
| v0.5-compatible | 5.00 | 7 | 2 | 28 | n/a |
| v0.6 algorithm stack | 8.67 | 8 | 2 | 30 | 0.91 on all scenarios |

Latency numbers here are from fake offline LLM/search adapters and should not be
used as user-facing speed claims. Live runs are expected to show parallel tool
execution benefits when each task has multiple real network-bound `(query,
source)` calls.

## Interpretation

- v0.6 keeps the existing hard-scenario gate green while enabling the full
  algorithm stack.
- Verifier metrics now appear in benchmark summaries and traces:
  `verifier_overall_quality=0.91`, `verifier_revision_count=0` in these canned
  passing scenarios.
- Reflection did not fire in these runs because canned evidence was already
  strong. Dedicated unit tests cover weak-result query rewrite and retry.
- Parallel execution does not change offline scores because canned retrieval is
  effectively instantaneous; dedicated tests cover result collection, trace
  completeness, source error isolation, retry compatibility, and flag opt-out.

Next public credibility step: run the Phase 3 external benchmark slice
(GAIA Level 1 or AssistantBench/BrowseComp Mini) with a real provider and live
retrieval budget.
