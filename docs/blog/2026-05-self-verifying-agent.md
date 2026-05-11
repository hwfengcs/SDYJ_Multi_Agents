# Why your multi-agent framework needs traces, replay, and a verifier — not better prompts

*Published: 2026-05-02 · Tags: ai-agent, langgraph, observability, agent-evaluation*

There is a frustrating pattern in open-source agent frameworks. The README ships with a happy path: define your agents, wire up some tools, run a query, marvel at the markdown report. Then you put it in front of a real user. They paste a real question. The agent runs for two minutes, returns three confident paragraphs that cite a URL that does not exist, and you have *no idea* what went wrong.

So you tweak the prompt. You add a "Be sure to cite real sources!" line. You re-run. It happens again, slightly differently. You add another line. The prompt grows. The behavior gets less predictable, not more.

This is not a prompting problem. **It's an evaluation problem.** And the only way out is to make every agent run controllable, observable, replayable, and benchmarked — *as a first-class engineering concern*, not an afterthought.

That's the thesis behind v0.6 of [SDYJ Multi Agents](https://github.com/hwfengcs/SDYJ_Multi_Agents), the LangGraph-based multi-agent research framework I'm building. In this post I'll walk through the four pillars that make agent operations actually tractable — and what we just shipped in the v0.6.0a1 alpha to nail the first three of them.

## What's broken about the typical "agent demo"

Let's name the failure modes you actually hit in production:

1. **Hallucinated citations.** The model fills in a plausible-looking URL that returns 404. There is no quality gate.
2. **Tool failures invisible to the user.** Tavily times out, the agent silently moves on, the report claims "based on extensive web search". The trace, if any, is a few `print` lines.
3. **No way to reproduce.** A user reports "the agent gave a bad answer 3 hours ago" and you have nothing to go on. The LLM was non-deterministic. The web changed.
4. **Cost surprise.** You write `temperature=0.7, max_tokens=4000` and run a 5-iteration loop. Twenty queries later your DeepSeek bill is $40 and you have no idea which step is hot.
5. **Quality regression.** You change one prompt, ship to main. Three days later someone notices the citation rate dropped 30%. You can't tell *which* of the seven prompt edits caused it.

Each of these is solvable. None of them are solved by writing better prompts.

## Pillar 1: trace everything

Every meaningful step in an agent run should produce a structured event. Not a log line — a typed event with timestamp, parent ID, latency, status, and inputs/outputs hashed for replay.

In SDYJ this is the v2 trace schema. Each run produces a bundle:

```text
outputs/runs/<run-id>/
  trace.json        # full structured trace
  events.jsonl      # event-per-line, easy to grep
  state.final.json  # the workflow state at completion
  report.md         # the actual deliverable
```

The trace records:

- Every node entry/exit with latency and metadata.
- Every LLM call with prompt hash, response hash, token counts, latency, and cost (more on cost in pillar 3).
- Every tool call with source, query, result count, latency, error.
- Every routing decision with the reason and the metadata that drove it.

This is roughly equivalent to what LangSmith, Langfuse, or Weave do — but it's free, local, and shipped in the box. No external SaaS, no extra signup. The trace is the ground truth for everything else.

## Pillar 2: deterministic replay

Tracing is only half the picture. The other half is being able to **rerun a failed query without calling real APIs**.

SDYJ's replay reads the recorded LLM responses and tool results from the source trace, then re-runs the workflow with mocked I/O substituted in call order:

```bash
sdyj replay <run-id>
```

The replay produces a fresh trace bundle. You can `diff-runs` the original and the replay to confirm they took the same code path. When they don't — say, a routing decision changed because you tightened a threshold — the diff makes the regression obvious.

This is huge for two reasons:

1. **Bug investigation costs nothing.** When a user reports an issue, you replay locally with no API spend. You can step through the events with `sdyj inspect-run --timeline`. You can edit a prompt and replay to see if the new prompt would have fixed it.
2. **Determinism gates in CI.** Run the same scenario twice with `--determinism-repeats 2`. If the recorded trace doesn't replay identically, that's a flag that something in the workflow is non-deterministic or stateful in a way you didn't expect.

## Pillar 3: per-call cost in the trace

Cost surprise is solved with one rule: **every LLM call records its dollar cost at the time it happens**.

In v0.6.0a1 we wired this end-to-end:

- Every provider wrapper (OpenAI, Claude, DeepSeek, Gemini) now exposes `last_usage` after each call.
- A new `SDYJ_Agents/utils/cost.py` module ships a transparent `PRICING_TABLE` that's just a hardcoded `{(provider, model): (input_per_million, output_per_million)}` dict — no surprises, no third-party service calls. Update it when prices change, with a release note.
- The trace's `InstrumentedLLM` wrapper writes `prompt_tokens_actual`, `completion_tokens_actual`, and `cost_usd` into every `llm_calls` entry.
- `finalize_trace` rolls them up into `trace.metrics`: total tokens, total dollars, separate priced and unpriced call counts.
- The CLI `sdyj inspect-run` shows a per-call cost table.

Critically: when a (provider, model) pair isn't in `PRICING_TABLE`, we return `None`, not `0.00`. A `0.00` cell would be a *lie*. A `—` cell with the message "add this entry to PRICING_TABLE" tells the user the truth.

```text
                LLM Calls (per call cost estimates)
+-----------------------------------------------------------------+
| Call | Model            | Prompt tok | Completion tok | Cost    |
|------+------------------+------------+----------------+---------|
| L1   | deepseek-v4-flash|        450 |            120 | $0.00016|
| L2   | deepseek-v4-flash|       2104 |            812 | $0.00104|
| L3   | gpt-4o           |        980 |            340 | $0.00585|
+-----------------------------------------------------------------+
```

You can now look at a 5-iteration research run and see *exactly* which call is the cost driver. Usually it's the synthesized analysis call, not the planner. Now you have data to make a decision instead of a feeling.

## Pillar 4: a verifier loop (in progress)

This is the one I'm most excited about and it's coming next on the v0.6 branch.

Most agent frameworks treat the report as the final output. SDYJ v0.6 adds a fifth agent — the **Verifier** — that re-reads the report against the collected evidence and asks four questions:

1. **claim_evidence_alignment**: does every key finding actually trace back to an `E1/E2/...` evidence item? Or is the model "summarizing" things that came from nowhere?
2. **citation_completeness**: are all the evidence items used? Or did we collect 12 sources and cite 3?
3. **factual_consistency**: does the report contradict itself across sections?
4. **plan_coverage**: did the report actually cover the sub-tasks the Planner committed to, or did it drift?

The verifier outputs structured JSON with a `should_revise` flag and revision hints. If `should_revise` is true, the workflow loops back to the Rapporteur with the hints in context. The revise loop has a hard cap (default 2) so it can't run forever.

This is a plain ReAct/Reflexion-style design — the contribution isn't novelty, it's that the verifier becomes a **traced node** in the graph just like every other agent. Its critique is in the trace. Its revisions are in the trace. You can replay them. You can A/B with `--no-verify` to measure the lift.

That's how you turn "self-verification" from a marketing word into something you can put on a benchmark scoreboard.

## What's next

The full v0.6 plan is in [docs/release-notes/v0.6.md](https://github.com/hwfengcs/SDYJ_Multi_Agents/blob/main/docs/release-notes/v0.6.md). Headlines for the next 6 weeks:

- Verifier agent + revise loop (pillar 4 above).
- Reflexive Researcher: when a query batch returns empty or low-relevance results, the agent rewrites the query and retries before giving up.
- Plan refinement: after N tasks, the Planner sees collected evidence and prunes / adds remaining tasks.
- Parallel tool execution within a task (asyncio.gather + concurrency limit).
- Public benchmark numbers on **GAIA Level 1** and **AssistantBench**, including a v0.5-vs-v0.6 ablation. This is the data that decides whether the verifier loop is real or vibes.
- Streamlit Web UI (already shipped MVP in v0.6.0a1) and a Hugging Face Spaces demo.
- Real MCP integration via the official `mcp` Python SDK, replacing the current HTTP placeholder.

If you're building agents and you've ever felt the pain in the failure-mode list at the top of this post, I'd love your feedback — especially what's missing from the trace schema for *your* use case. The repo is at [github.com/hwfengcs/SDYJ_Multi_Agents](https://github.com/hwfengcs/SDYJ_Multi_Agents) and PRs / issues are welcome.

Star the repo if you want to follow along; v0.6 stable is targeted for end of June 2026 with the GAIA scores attached.
