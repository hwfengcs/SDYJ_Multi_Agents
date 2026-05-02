"""Run tracing helpers for agent workflow observability and replay."""

from __future__ import annotations

import copy
import hashlib
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterator, Optional
from uuid import uuid4

from ..llm.base import BaseLLM


def utc_now_iso() -> str:
    """Return an ISO timestamp with millisecond-level readability."""
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def create_run_trace(
    query: str,
    provider: str,
    model: str | None,
    mode: str = "research",
    scenario_id: str | None = None,
) -> Dict[str, Any]:
    """Create a JSON-serializable trace object."""
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    suffix = uuid4().hex[:8]
    run_id = f"{timestamp}_{suffix}"
    return {
        "schema_version": "2.0",
        "run_id": run_id,
        "mode": mode,
        "scenario_id": scenario_id,
        "created_at": utc_now_iso(),
        "completed_at": None,
        "query": query,
        "provider": provider,
        "model": model,
        "config": {},
        "nodes": [],
        "llm_calls": [],
        "tool_calls": [],
        "events": [],
        "replay_cache": {
            "llm_calls": [],
            "tool_calls": [],
        },
        "artifacts": {},
        "capture": {
            "prompt": "preview+hash",
            "response": "full",
            "tool_result": "full",
            "snapshots": "safe-preview",
        },
        "report": {},
        "metrics": {},
        "errors": [],
    }


SENSITIVE_KEY_PARTS = (
    "api_key",
    "apikey",
    "authorization",
    "bearer",
    "password",
    "secret",
    "token",
    "credential",
)


def _json_default(value: Any) -> str:
    return str(value)


def _stable_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, ensure_ascii=False, default=_json_default)


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _safe_text(text: str, limit: int = 1000) -> str:
    text = text.replace("\x00", "")
    if len(text) <= limit:
        return text
    return f"{text[:limit]}...<truncated {len(text) - limit} chars>"


def safe_snapshot(value: Any, max_depth: int = 3, max_items: int = 20, text_limit: int = 1000) -> Any:
    """Create a JSON-safe, redacted snapshot for trace events."""
    if max_depth < 0:
        return "<max-depth>"

    if value is None or isinstance(value, (bool, int, float)):
        return value

    if isinstance(value, str):
        return _safe_text(value, text_limit)

    if isinstance(value, Path):
        return str(value)

    if isinstance(value, dict):
        output = {}
        for index, (key, item) in enumerate(value.items()):
            if index >= max_items:
                output["<truncated>"] = f"{len(value) - max_items} more keys"
                break
            key_str = str(key)
            key_lower = key_str.lower()
            if any(part in key_lower for part in SENSITIVE_KEY_PARTS):
                output[key_str] = "[REDACTED]"
            elif key_str == "trace":
                output[key_str] = {
                    "run_id": item.get("run_id") if isinstance(item, dict) else None,
                    "schema_version": item.get("schema_version") if isinstance(item, dict) else None,
                }
            else:
                output[key_str] = safe_snapshot(item, max_depth - 1, max_items, text_limit)
        return output

    if isinstance(value, (list, tuple, set)):
        items = list(value)
        output = [
            safe_snapshot(item, max_depth - 1, max_items, text_limit)
            for item in items[:max_items]
        ]
        if len(items) > max_items:
            output.append(f"<truncated {len(items) - max_items} items>")
        return output

    return _safe_text(str(value), text_limit)


def _next_event_id(trace: Dict[str, Any]) -> str:
    return f"evt_{len(trace.setdefault('events', [])) + 1:06d}"


def record_trace_event(
    trace: Optional[Dict[str, Any]],
    event_type: str,
    name: str,
    node: str | None = None,
    status: str = "ok",
    latency_ms: int | None = None,
    input_snapshot: Any = None,
    output_snapshot: Any = None,
    metadata: Optional[Dict[str, Any]] = None,
    parent_event_id: str | None = None,
    error: str | None = None,
) -> Optional[str]:
    """Append one structured event and return its event ID."""
    if not trace:
        return None

    seq = len(trace.setdefault("events", [])) + 1
    event = {
        "event_id": f"evt_{seq:06d}",
        "seq": seq,
        "parent_event_id": parent_event_id,
        "event_type": event_type,
        "name": name,
        "node": node,
        "status": status,
        "timestamp": utc_now_iso(),
        "latency_ms": latency_ms,
        "input_snapshot": safe_snapshot(input_snapshot) if input_snapshot is not None else None,
        "output_snapshot": safe_snapshot(output_snapshot) if output_snapshot is not None else None,
        "metadata": safe_snapshot(metadata or {}),
        "error": error,
    }
    trace.setdefault("events", []).append(event)
    if error:
        trace.setdefault("errors", []).append({"where": f"event:{name}", "error": error})
    return event["event_id"]


def record_decision(
    trace: Optional[Dict[str, Any]],
    node: str,
    decision: str,
    reason: str | None = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> None:
    """Record a routing or control-flow decision."""
    record_trace_event(
        trace,
        event_type="decision",
        name=decision,
        node=node,
        output_snapshot={"decision": decision, "reason": reason},
        metadata=metadata,
    )


def _safe_round_ms(seconds: float) -> int:
    return int(round(seconds * 1000))


def record_node_event(
    trace: Optional[Dict[str, Any]],
    node: str,
    latency_ms: int,
    status: str = "ok",
    metadata: Optional[Dict[str, Any]] = None,
    error: str | None = None,
) -> None:
    """Record one workflow node execution."""
    if not trace:
        return
    event = {
        "node": node,
        "latency_ms": latency_ms,
        "status": status,
        "metadata": metadata or {},
        "timestamp": utc_now_iso(),
    }
    if error:
        event["error"] = error
        trace.setdefault("errors", []).append({"where": f"node:{node}", "error": error})
    trace.setdefault("nodes", []).append(event)
    record_trace_event(
        trace,
        event_type="node_end",
        name=node,
        node=node,
        status=status,
        latency_ms=latency_ms,
        output_snapshot=metadata or {},
        metadata=metadata or {},
        error=error,
    )


def record_tool_call(
    trace: Optional[Dict[str, Any]],
    source: str,
    query: str,
    latency_ms: int,
    result_count: int,
    task_id: int | None = None,
    error: str | None = None,
    result: Optional[Dict[str, Any]] = None,
) -> None:
    """Record one retrieval/tool call."""
    if not trace:
        return
    tool_call_id = f"T{len(trace.get('tool_calls', [])) + 1}"
    event = {
        "tool_call_id": tool_call_id,
        "source": source,
        "query": query,
        "task_id": task_id,
        "latency_ms": latency_ms,
        "result_count": result_count,
        "error": error,
        "timestamp": utc_now_iso(),
    }
    trace.setdefault("tool_calls", []).append(event)
    if result is not None:
        trace.setdefault("replay_cache", {}).setdefault("tool_calls", []).append(
            {
                "tool_call_id": tool_call_id,
                "source": source,
                "query": query,
                "task_id": task_id,
                "result": copy.deepcopy(result),
            }
        )
    record_trace_event(
        trace,
        event_type="tool_call",
        name=source,
        node="researcher",
        status="error" if error else "ok",
        latency_ms=latency_ms,
        input_snapshot={"query": query, "source": source, "task_id": task_id},
        output_snapshot={
            "result_count": result_count,
            "error": error,
            "result_preview": result,
        },
        metadata={"tool_call_id": tool_call_id},
        error=error,
    )
    if error:
        trace.setdefault("errors", []).append({"where": f"tool:{source}", "error": error})


def record_report_summary(
    trace: Optional[Dict[str, Any]],
    report_format: str,
    source_count: int,
    evidence_count: int,
    citation_count: int,
) -> None:
    """Record final report-level observability data."""
    if not trace:
        return
    trace["report"] = {
        "format": report_format,
        "source_count": source_count,
        "evidence_count": evidence_count,
        "citation_count": citation_count,
    }
    record_trace_event(
        trace,
        event_type="artifact",
        name="report_summary",
        node="rapporteur",
        output_snapshot=trace["report"],
    )


def finalize_trace(trace: Optional[Dict[str, Any]], metrics: Optional[Dict[str, Any]] = None) -> None:
    """Mark a trace complete and attach final metrics."""
    if not trace:
        return
    trace["completed_at"] = utc_now_iso()
    if metrics:
        trace.setdefault("metrics", {}).update(metrics)


def merge_trace_state(
    base_trace: Optional[Dict[str, Any]],
    state_trace: Optional[Dict[str, Any]],
) -> Optional[Dict[str, Any]]:
    """Merge trace data that may have diverged across LangGraph state copies."""
    if not base_trace:
        return state_trace
    if not state_trace or state_trace is base_trace:
        return base_trace

    merged = dict(base_trace)
    for key in ("nodes", "llm_calls", "tool_calls", "errors"):
        combined = []
        seen = set()
        for source in (state_trace.get(key, []), base_trace.get(key, [])):
            for item in source:
                identity = json.dumps(item, sort_keys=True, ensure_ascii=False)
                if identity in seen:
                    continue
                seen.add(identity)
                combined.append(item)
        merged[key] = combined

    for key in ("events",):
        combined = []
        seen = set()
        for source in (state_trace.get(key, []), base_trace.get(key, [])):
            for item in source:
                identity = json.dumps(item, sort_keys=True, ensure_ascii=False, default=_json_default)
                if identity in seen:
                    continue
                seen.add(identity)
                combined.append(item)
        combined.sort(key=lambda item: (str(item.get("timestamp", "")), int(item.get("seq") or 0)))
        for index, item in enumerate(combined, 1):
            item["seq"] = index
            item["event_id"] = f"evt_{index:06d}"
        merged[key] = combined

    replay_cache = {
        "llm_calls": [],
        "tool_calls": [],
    }
    for cache in (base_trace.get("replay_cache") or {}, state_trace.get("replay_cache") or {}):
        for key in ("llm_calls", "tool_calls"):
            replay_cache[key].extend(cache.get(key, []))
    for key in ("llm_calls", "tool_calls"):
        unique = []
        seen = set()
        for item in replay_cache[key]:
            identity = json.dumps(item, sort_keys=True, ensure_ascii=False, default=_json_default)
            if identity in seen:
                continue
            seen.add(identity)
            unique.append(item)
        replay_cache[key] = unique
    merged["replay_cache"] = replay_cache

    for key in ("report", "metrics", "artifacts"):
        merged[key] = {
            **(base_trace.get(key) or {}),
            **(state_trace.get(key) or {}),
        }

    for key, value in state_trace.items():
        if key not in merged or merged.get(key) in (None, [], {}):
            merged[key] = value
    return merged


def run_dir_for(output_dir: str | Path, run_id: str) -> Path:
    """Return the run bundle directory for a run ID."""
    return Path(output_dir) / "runs" / run_id


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False, default=_json_default)


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)


def _state_for_artifact(final_state: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    if not final_state:
        return None
    state = {}
    for key, value in final_state.items():
        if key == "trace":
            continue
        state[key] = safe_snapshot(value, max_depth=8, max_items=200, text_limit=20000)
    return state


def save_trace(
    trace: Optional[Dict[str, Any]],
    output_dir: str | Path,
    final_state: Optional[Dict[str, Any]] = None,
    report: str | None = None,
    report_extension: str | None = None,
) -> Optional[Path]:
    """Persist trace JSON plus a run bundle under `<output_dir>/runs/<run-id>/`."""
    if not trace:
        return None
    finalize_trace(trace)
    run_id = trace["run_id"]

    run_dir = run_dir_for(output_dir, run_id)
    run_dir.mkdir(parents=True, exist_ok=True)

    trace["artifacts"].update(
        {
            "run_dir": str(run_dir),
            "trace": str(run_dir / "trace.json"),
            "events": str(run_dir / "events.jsonl"),
        }
    )

    state_artifact = _state_for_artifact(final_state)
    if state_artifact is not None:
        state_path = run_dir / "state.final.json"
        _write_json(state_path, state_artifact)
        trace["artifacts"]["final_state"] = str(state_path)

    if report is None and final_state:
        report = final_state.get("final_report")
    if report:
        extension = report_extension or ("html" if (final_state or {}).get("output_format") == "html" else "md")
        report_path = run_dir / f"report.{extension}"
        _write_text(report_path, report)
        trace["artifacts"]["report"] = str(report_path)

    events_path = run_dir / "events.jsonl"
    with open(events_path, "w", encoding="utf-8") as f:
        for event in trace.get("events", []):
            f.write(json.dumps(event, ensure_ascii=False, default=_json_default) + "\n")

    _write_json(run_dir / "trace.json", trace)

    # Backward compatibility for existing commands and docs.
    trace_dir = Path(output_dir) / "traces"
    trace_dir.mkdir(parents=True, exist_ok=True)
    path = trace_dir / f"{run_id}.json"
    _write_json(path, trace)
    return path


def load_trace(path_or_run_id: str, output_dir: str | Path = "./outputs") -> Dict[str, Any]:
    """Load a trace by file path or run ID."""
    candidate = Path(path_or_run_id)
    if not candidate.exists():
        candidate = Path(output_dir) / "traces" / f"{path_or_run_id}.json"
    if not candidate.exists():
        candidate = run_dir_for(output_dir, path_or_run_id) / "trace.json"
    with open(candidate, "r", encoding="utf-8") as f:
        return json.load(f)


def latest_trace_path(output_dir: str | Path = "./outputs") -> Optional[Path]:
    """Return the newest trace file, if present."""
    output_root = Path(output_dir)
    traces = []
    trace_dir = output_root / "traces"
    if trace_dir.exists():
        traces.extend(trace_dir.glob("*.json"))
    runs_dir = output_root / "runs"
    if runs_dir.exists():
        traces.extend(runs_dir.glob("*/trace.json"))
    traces = sorted(traces, key=lambda p: p.stat().st_mtime, reverse=True)
    return traces[0] if traces else None


def iter_timeline_events(trace: Dict[str, Any]) -> list[Dict[str, Any]]:
    """Return normalized timeline events for both v1 and v2 traces."""
    events = trace.get("events") or []
    if events:
        return sorted(events, key=lambda item: item.get("seq") or item.get("event_id") or "")

    synthesized = []
    seq = 1
    for node in trace.get("nodes", []):
        synthesized.append(
            {
                "event_id": f"legacy_{seq:06d}",
                "seq": seq,
                "event_type": "node_end",
                "name": node.get("node"),
                "node": node.get("node"),
                "status": node.get("status", "ok"),
                "timestamp": node.get("timestamp"),
                "latency_ms": node.get("latency_ms"),
                "metadata": node.get("metadata") or {},
                "error": node.get("error"),
            }
        )
        seq += 1
    for call in trace.get("llm_calls", []):
        synthesized.append(
            {
                "event_id": f"legacy_{seq:06d}",
                "seq": seq,
                "event_type": "llm_call",
                "name": call.get("call_id") or "llm",
                "node": None,
                "status": "error" if call.get("error") else "ok",
                "timestamp": call.get("timestamp"),
                "latency_ms": call.get("latency_ms"),
                "metadata": {
                    "model": call.get("model"),
                    "prompt_chars": call.get("prompt_chars"),
                    "response_chars": call.get("response_chars"),
                },
                "error": call.get("error"),
            }
        )
        seq += 1
    for call in trace.get("tool_calls", []):
        synthesized.append(
            {
                "event_id": f"legacy_{seq:06d}",
                "seq": seq,
                "event_type": "tool_call",
                "name": call.get("source"),
                "node": "researcher",
                "status": "error" if call.get("error") else "ok",
                "timestamp": call.get("timestamp"),
                "latency_ms": call.get("latency_ms"),
                "metadata": {
                    "query": call.get("query"),
                    "result_count": call.get("result_count"),
                },
                "error": call.get("error"),
            }
        )
        seq += 1
    return synthesized


def summarize_trace(trace: Dict[str, Any]) -> Dict[str, Any]:
    """Build a compact trace summary for CLI diff and benchmark reports."""
    node_latency = sum(item.get("latency_ms") or 0 for item in trace.get("nodes", []))
    llm_latency = sum(item.get("latency_ms") or 0 for item in trace.get("llm_calls", []))
    tool_latency = sum(item.get("latency_ms") or 0 for item in trace.get("tool_calls", []))
    metrics = trace.get("metrics") or {}
    return {
        "run_id": trace.get("run_id"),
        "mode": trace.get("mode"),
        "scenario_id": trace.get("scenario_id"),
        "provider": trace.get("provider"),
        "model": trace.get("model"),
        "nodes": len(trace.get("nodes", [])),
        "events": len(iter_timeline_events(trace)),
        "llm_calls": len(trace.get("llm_calls", [])),
        "tool_calls": len(trace.get("tool_calls", [])),
        "errors": len(trace.get("errors", [])),
        "node_latency_ms": node_latency,
        "llm_latency_ms": llm_latency,
        "tool_latency_ms": tool_latency,
        "evidence_count": metrics.get("evidence_count") or (trace.get("report") or {}).get("evidence_count"),
        "citation_count": metrics.get("citation_count") or (trace.get("report") or {}).get("citation_count"),
        "overall_score": metrics.get("overall_score"),
        "tool_success_rate": metrics.get("tool_success_rate"),
        "trace_completeness": metrics.get("trace_completeness"),
    }


def diff_traces(left: Dict[str, Any], right: Dict[str, Any]) -> Dict[str, Any]:
    """Compare two traces by stable operational summary fields."""
    left_summary = summarize_trace(left)
    right_summary = summarize_trace(right)
    rows = []
    for key in sorted(set(left_summary) | set(right_summary)):
        left_value = left_summary.get(key)
        right_value = right_summary.get(key)
        if left_value == right_value:
            delta = 0 if isinstance(left_value, (int, float)) else ""
        elif isinstance(left_value, (int, float)) and isinstance(right_value, (int, float)):
            delta = right_value - left_value
        else:
            delta = "changed"
        rows.append(
            {
                "metric": key,
                "left": left_value,
                "right": right_value,
                "delta": delta,
                "changed": left_value != right_value,
            }
        )
    return {
        "left_run_id": left.get("run_id"),
        "right_run_id": right.get("run_id"),
        "left": left_summary,
        "right": right_summary,
        "rows": rows,
    }


class InstrumentedLLM(BaseLLM):
    """Wrap an LLM and record latency, rough sizes, and provider usage metadata."""

    def __init__(self, inner: BaseLLM, trace: Optional[Dict[str, Any]]):
        super().__init__(
            api_key=getattr(inner, "api_key", ""),
            model=getattr(inner, "model", "unknown"),
        )
        self.inner = inner
        self.trace = trace

    def generate(self, prompt: str, **kwargs) -> str:
        started = time.perf_counter()
        call_id = f"L{len(self.trace.get('llm_calls', [])) + 1}" if self.trace else None
        try:
            response = self.inner.generate(prompt, **kwargs)
            latency_ms = _safe_round_ms(time.perf_counter() - started)
            self._record_call(
                call_id=call_id,
                prompt=prompt,
                response=response,
                kwargs=kwargs,
                latency_ms=latency_ms,
                error=None,
            )
            return response
        except Exception as exc:
            latency_ms = _safe_round_ms(time.perf_counter() - started)
            self._record_call(
                call_id=call_id,
                prompt=prompt,
                response="",
                kwargs=kwargs,
                latency_ms=latency_ms,
                error=str(exc),
            )
            raise

    def stream_generate(self, prompt: str, **kwargs) -> Iterator[str]:
        started = time.perf_counter()
        chunks = []
        call_id = f"L{len(self.trace.get('llm_calls', [])) + 1}" if self.trace else None
        try:
            for chunk in self.inner.stream_generate(prompt, **kwargs):
                chunks.append(chunk)
                yield chunk
            latency_ms = _safe_round_ms(time.perf_counter() - started)
            self._record_call(
                call_id=call_id,
                prompt=prompt,
                response="".join(chunks),
                kwargs=kwargs,
                latency_ms=latency_ms,
                error=None,
            )
        except Exception as exc:
            latency_ms = _safe_round_ms(time.perf_counter() - started)
            self._record_call(
                call_id=call_id,
                prompt=prompt,
                response="".join(chunks),
                kwargs=kwargs,
                latency_ms=latency_ms,
                error=str(exc),
            )
            raise

    def _record_call(
        self,
        call_id: str | None,
        prompt: str,
        response: str,
        kwargs: Dict[str, Any],
        latency_ms: int,
        error: str | None,
    ) -> None:
        if not self.trace:
            return
        event = {
            "call_id": call_id,
            "model": self.model,
            "latency_ms": latency_ms,
            "prompt_chars": len(prompt),
            "response_chars": len(response),
            "prompt_hash": _sha256_text(prompt),
            "response_hash": _sha256_text(response),
            "temperature": kwargs.get("temperature"),
            "max_tokens": kwargs.get("max_tokens"),
            "prompt_preview": prompt[:160].replace("\n", " "),
            "response_preview": response[:240].replace("\n", " "),
            "usage": getattr(self.inner, "last_usage", None),
            "error": error,
            "timestamp": utc_now_iso(),
        }
        self.trace.setdefault("llm_calls", []).append(event)
        self.trace.setdefault("replay_cache", {}).setdefault("llm_calls", []).append(
            {
                "call_id": call_id,
                "model": self.model,
                "response": response,
                "error": error,
            }
        )
        record_trace_event(
            self.trace,
            event_type="llm_call",
            name=call_id or "llm",
            node=None,
            status="error" if error else "ok",
            latency_ms=latency_ms,
            input_snapshot={
                "model": self.model,
                "prompt_hash": event["prompt_hash"],
                "prompt_preview": event["prompt_preview"],
                "temperature": event["temperature"],
                "max_tokens": event["max_tokens"],
            },
            output_snapshot={
                "response_hash": event["response_hash"],
                "response_preview": event["response_preview"],
                "usage": event["usage"],
            },
            metadata={"call_id": call_id},
            error=error,
        )
        if error:
            self.trace.setdefault("errors", []).append({"where": "llm", "error": error})

    def __repr__(self) -> str:
        return f"InstrumentedLLM(inner={self.inner!r})"
