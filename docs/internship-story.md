# Internship Story

Use this page as a concise narrative when discussing the project in an AI agent
algorithm engineer internship interview.

## One-Sentence Pitch

I built a LangGraph-based multi-agent deep research assistant that combines
human-in-the-loop planning, multi-source retrieval, provider-agnostic LLM
integration, evidence-grounded reporting, JSON tracing, and reproducible
evaluation.

## Technical Highlights

- Designed a controllable state machine instead of a single monolithic prompt.
- Split responsibilities across Coordinator, Planner, Researcher, and Rapporteur.
- Added a human approval gate before expensive or broad retrieval runs.
- Normalized Tavily, arXiv, and MCP-style results into one retrieval shape.
- Normalized retrieved sources into deduplicated `E1/E2/...` evidence items.
- Persisted run traces with node latency, LLM calls, tool calls, errors, and report metrics.
- Added hard evaluation scenarios with offline canned evidence and live DeepSeek evaluation.
- Added tests and CI so core behavior can be verified without real API keys.

## Interview Talking Points

1. **Workflow design**
   - Why the graph has an explicit planning node and approval node.
   - How `max_iterations` limits agent loops.
   - How simple requests avoid unnecessary research flow.

2. **Tool-use reliability**
   - Why tools return normalized result dictionaries.
   - How empty results and tool errors should be represented.
   - How future work can add retries, source scoring, and deduplication.
   - How evidence IDs connect final claims back to tool, query, URL, and metadata.

3. **Evaluation**
   - Why final answer quality is not enough.
   - How to measure plan coverage, evidence quality, latency, and cost.
   - How mock tests differ from live smoke tests.
   - How `sdyj eval --live --provider deepseek` isolates model behavior while canned evidence keeps benchmarks reproducible.

## Resume Bullet

Built a LangGraph-based multi-agent deep research system with human-in-the-loop
planning, multi-source retrieval, evidence-grounded reports, JSON run tracing,
DeepSeek-backed evaluation scenarios, tests, and CI.
