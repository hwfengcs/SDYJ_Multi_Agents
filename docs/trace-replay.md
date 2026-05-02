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
  state.final.json
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

Replay creates a new run with `mode=replay` and stores its own trace. This makes
it possible to compare the original and replayed executions:

```bash
python main.py diff-runs <original-run-id> <replay-run-id>
```

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
