# Evaluation

Deep research agents need evaluation beyond final-answer checks. SDYJ evaluates
planning quality, retrieval/tool reliability, evidence grounding, synthesis
quality, efficiency, trace completeness, replayability, and human-control
behavior. The operational benchmark guide is now maintained in
[benchmark.md](benchmark.md).

## Evaluation Dimensions

| Dimension | What to Measure | Example Signal |
| --- | --- | --- |
| Plan quality | Whether subtasks cover the user goal | Required scenario concepts appear in the plan/report |
| Tool reliability | Whether tool calls succeed and degrade gracefully | Tool success rate, errors, result counts, latency |
| Evidence quality | Whether claims are grounded in retrieved sources | Evidence IDs, duplicate URL ratio, citation coverage |
| Synthesis quality | Whether the report is coherent and complete | Section completeness, grounded key findings |
| Efficiency | Runtime budget and cost behavior | Iterations, latency, token usage when exposed |
| Human control | Whether approval gates reduce risk | Plan approval before expensive retrieval |
| Trace/replay | Whether failures can be diagnosed and replayed | Trace completeness, replay cache, event timeline |

## Implemented Test Layers

1. **Unit tests**
   - Mock LLM responses.
   - Mock Tavily, arXiv, and MCP search results.
   - Verify config, parsing, routing, evidence normalization, tracing, and evaluation.

2. **Scenario tests**
   - Fixed hard prompts with canned evidence.
   - Compare generated plans, report sections, evidence IDs, citation density,
     duplicate URL handling, and tool-call reliability.

3. **Live smoke tests**
   - Opt in with `python main.py eval --live ...`.
   - Use a real provider for planning and synthesis.
   - Keep canned evidence by default so model behavior can be evaluated without
     search-result drift.

4. **Benchmark set**
   - `agent_reliability_hard`: evaluation design for a RAG + web + MCP research agent.
   - `tool_failure_recovery_hard`: timeout, empty result, duplicate URL, and fallback behavior.
   - `mcp_rag_ops_hard`: customer-support MCP + RAG evaluation with privacy, refusal, SLO, and cost constraints.

## Commands

List scenarios:

```bash
python main.py list-scenarios
python main.py benchmark run --max-scenarios 1 --max-iterations 2
python main.py benchmark run --fail-under 0.75
```

The legacy command remains supported:

```bash
python main.py eval --max-scenarios 1 --max-iterations 2
```

Run a real DeepSeek eval:

```bash
python main.py eval \
  --live \
  --provider deepseek \
  --model deepseek-v4-flash \
  --scenario agent_reliability_hard \
  --max-iterations 2
```

Inspect a run:

```bash
python main.py inspect-run <run-id>
```

## Metrics

The evaluation runner writes a summary JSON under `outputs/eval_reports/` and a
run trace under `outputs/traces/`. Current metrics include:

| Metric | Meaning |
| --- | --- |
| `plan_coverage` | Required scenario concepts covered by the plan/report |
| `section_completeness` | Expected report sections present |
| `citation_id_coverage` | Share of evidence IDs cited in the report |
| `citation_density_per_1k_chars` | Evidence citation density normalized by report length |
| `duplicate_url_ratio` | Duplicate raw URLs before evidence deduplication |
| `tool_success_rate` | Retrieval batches without error |
| `grounded_key_finding_rate` | Key finding bullets that include evidence IDs |
| `overall_score` | Weighted dashboard score for quick comparison |

## Latest Local Benchmark

Run date: 2026-05-02
Mode: live DeepSeek model with canned evidence
Command:

```bash
python main.py eval --live --provider deepseek --model deepseek-v4-flash --scenario agent_reliability_hard --max-iterations 2
```

Result:

| Scenario | Model | Score | Plan coverage | Citation coverage | Tool success | Evidence | Duplicate URL ratio |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `agent_reliability_hard` | `deepseek-v4-flash` | 0.8254 | 0.8750 | 1.0000 | 1.0000 | 5 | 0.7917 |

Trace excerpt:

```json
{
  "run_id": "20260502_025454_73b37862",
  "provider": "deepseek",
  "model": "deepseek-v4-flash",
  "mode": "eval",
  "scenario_id": "agent_reliability_hard",
  "tool_calls": [
    {
      "source": "tavily",
      "result_count": 4,
      "latency_ms": 0,
      "error": null
    }
  ],
  "report": {
    "format": "markdown",
    "evidence_count": 5,
    "citation_count": 5
  }
}
```

This benchmark is intentionally difficult: it asks the model to design a
reproducible evaluation plan for a RAG + web + MCP research agent with
human-in-the-loop control, ablations, latency/cost constraints, and rollout
thresholds. The trace makes failure analysis possible even when the final report
looks plausible.
