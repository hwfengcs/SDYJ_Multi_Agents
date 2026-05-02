# Internship Story

Use this page as a concise narrative when discussing the project in an AI agent
algorithm engineer internship interview.

## One-Sentence Pitch

I built a LangGraph-based multi-agent deep research assistant that combines
human-in-the-loop planning, multi-source retrieval, provider-agnostic LLM
integration, and structured report synthesis.

## Technical Highlights

- Designed a controllable state machine instead of a single monolithic prompt.
- Split responsibilities across Coordinator, Planner, Researcher, and Rapporteur.
- Added a human approval gate before expensive or broad retrieval runs.
- Normalized Tavily, arXiv, and MCP-style results into one retrieval shape.
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

3. **Evaluation**
   - Why final answer quality is not enough.
   - How to measure plan coverage, evidence quality, latency, and cost.
   - How mock tests differ from live smoke tests.

## Resume Bullet

Built a LangGraph-based multi-agent deep research system with human-in-the-loop
planning, multi-source retrieval, provider-agnostic LLM abstraction, Markdown/HTML
report generation, tests, and CI.
