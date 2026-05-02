# Evaluation Plan

Deep research agents need evaluation beyond final-answer style checks. SDYJ
should be evaluated across planning quality, retrieval quality, synthesis
quality, and operational reliability.

## Evaluation Dimensions

| Dimension | What to Measure | Example Signal |
| --- | --- | --- |
| Plan quality | Whether subtasks cover the user goal | Each required aspect appears in `research_plan.sub_tasks` |
| Tool reliability | Whether tool calls succeed and degrade gracefully | Error rate per source, retries, empty-result handling |
| Evidence quality | Whether claims are grounded in retrieved sources | Claim-to-source coverage, duplicate URL ratio |
| Synthesis quality | Whether the report is coherent and non-redundant | Section completeness, contradiction checks |
| Efficiency | Runtime budget and cost behavior | Iterations, latency, token usage, API cost |
| Human control | Whether user feedback changes the plan correctly | Plan diff after rejection and modification |

## Recommended Test Layers

1. **Unit tests**
   - Mock LLM responses.
   - Mock Tavily, arXiv, and MCP search results.
   - Verify config, parsing, routing, and report saving.

2. **Scenario tests**
   - Fixed research prompts with canned search results.
   - Compare generated plans and citations against expected patterns.

3. **Live smoke tests**
   - Run one short query against a real provider.
   - Assert the workflow produces a report and at least one source.

4. **Benchmark set**
   - Academic survey query.
   - Product comparison query.
   - Technical debugging query.
   - Ambiguous query that requires a clarifying plan.

## Metrics to Add Next

The next implementation milestone should persist a trace object such as:

```json
{
  "run_id": "20260502_001",
  "provider": "deepseek",
  "model": "deepseek-chat",
  "iterations": 3,
  "tool_calls": [
    {
      "source": "tavily",
      "query": "AI agent evaluation benchmarks",
      "latency_ms": 830,
      "result_count": 5,
      "error": null
    }
  ],
  "report": {
    "format": "markdown",
    "source_count": 12,
    "citation_count": 10
  }
}
```

This makes the project stronger for interviews because it shows you can reason
about agent behavior, not just generate text.
