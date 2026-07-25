# Trace and Replay

SDYJ records each run as a debuggable run bundle. The goal is not only to know
that a report was produced, but to understand how the agent reached it and to
reproduce the run when behavior changes.

## Run Bundle

Each completed research, replay, or benchmark run is written to:

```text
outputs/runs/<run-id>/
  trace.json
  events.jsonl
  state.final.json        # state.partial.json when a run crashed or was interrupted
  checkpoint.sqlite       # durable LangGraph checkpoint (research runs)
  report.md|html|json
```

For backward compatibility, the same trace is also written to:

```text
outputs/traces/<run-id>.json
```

## Trace v2

Trace v2 keeps the original summary arrays and adds an event timeline:

- `nodes`: workflow node latency and metadata.
- `llm_calls`: model latency, prompt/response sizes, hashes, previews, usage.
- `tool_calls`: source, query, task, result count, latency, and errors.
- `events`: normalized timeline events for nodes, LLM calls, tools, routing
  decisions, human approval, and artifacts.
- `replay_cache`: recorded LLM responses and tool results for deterministic
  replay.
- `artifacts`: paths to the run bundle files.

Sensitive fields such as API keys, tokens, passwords, and credentials are
redacted from snapshots.

## Inspect

Show the latest run:

```bash
python main.py inspect-run
```

Show a specific run with the event timeline:

```bash
python main.py inspect-run <run-id> --timeline
```

Show one event:

```bash
python main.py inspect-run <run-id> --timeline --event evt_000012
```

List recent run bundles:

```bash
python main.py runs list
```

## Replay

Deterministic replay uses the recorded LLM and tool outputs from Trace v2. It
does not call real model or search APIs.

```bash
python main.py replay <run-id>
```

Recorded LLM responses are matched by **prompt hash first** (each
`replay_cache.llm_calls` entry stores a sha256 of its prompt with the injected
`CURRENT_TIME` line normalized away, so timestamps never break matching),
falling back to sequential order for hash-less legacy traces or unmatched
prompts. Tool results are matched by `(source, query)` first, then by recorded
order. This keeps traces replayable even when the number or order of calls
shifts between code versions, and retries never change the call count: one
logical call produces exactly one `llm_calls` entry with `retries` /
`attempt_errors` fields.

Replay creates a new run with `mode=replay` and stores its own trace. This makes
it possible to compare the original and replayed executions:

```bash
python main.py diff-runs <original-run-id> <replay-run-id>
```

## Resume

Research runs default to a durable per-run sqlite checkpoint
(`outputs/runs/<run-id>/checkpoint.sqlite`, disable with
`SDYJ_DURABLE_CHECKPOINT=0`). A crashed or interrupted run can be continued
from its last completed super-step — including a pending human-review
interrupt:

```bash
python main.py resume <run-id>
python main.py resume <run-id> --auto-approve
```

Resume rebuilds the agents against the run's checkpoint database, reattaches
the original trace (marked with `resumed_at`), and continues streaming. Crash
paths persist `state.partial.json` plus the merged trace, and the CLI exits
with code 4 so automation can distinguish a crashed research run from success
(eval keeps exit codes 2/3).

## What Replay Is For

- Reproduce a failed or surprising agent path.
- Debug prompt, parsing, routing, and report synthesis changes.
- Keep benchmark regressions explainable instead of only reporting a score.

## Current Limits

- Replay is deterministic only for Trace v2 runs that include recorded LLM and
  tool I/O.
- It replays the whole workflow. Partial replay from a specific node is planned
  as a future extension.
- Prompt and response capture defaults are designed for local debugging. For
  sensitive deployments, store traces in a private location and tighten capture
  policy before sharing artifacts.
