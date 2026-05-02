"""Run tracing helpers for agent workflow observability."""

from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterator, Optional
from uuid import uuid4

from ..llm.base import BaseLLM


def utc_now_iso() -> str:
    """Return an ISO timestamp with second-level readability."""
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


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
        "schema_version": "1.0",
        "run_id": run_id,
        "mode": mode,
        "scenario_id": scenario_id,
        "created_at": utc_now_iso(),
        "completed_at": None,
        "query": query,
        "provider": provider,
        "model": model,
        "nodes": [],
        "llm_calls": [],
        "tool_calls": [],
        "report": {},
        "metrics": {},
        "errors": [],
    }


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


def record_tool_call(
    trace: Optional[Dict[str, Any]],
    source: str,
    query: str,
    latency_ms: int,
    result_count: int,
    task_id: int | None = None,
    error: str | None = None,
) -> None:
    """Record one retrieval/tool call."""
    if not trace:
        return
    event = {
        "source": source,
        "query": query,
        "task_id": task_id,
        "latency_ms": latency_ms,
        "result_count": result_count,
        "error": error,
        "timestamp": utc_now_iso(),
    }
    trace.setdefault("tool_calls", []).append(event)
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

    for key in ("report", "metrics"):
        merged[key] = {
            **(base_trace.get(key) or {}),
            **(state_trace.get(key) or {}),
        }

    for key, value in state_trace.items():
        if key not in merged or merged.get(key) in (None, [], {}):
            merged[key] = value
    return merged


def save_trace(trace: Optional[Dict[str, Any]], output_dir: str | Path) -> Optional[Path]:
    """Persist a trace under `<output_dir>/traces/<run_id>.json`."""
    if not trace:
        return None
    finalize_trace(trace)
    trace_dir = Path(output_dir) / "traces"
    trace_dir.mkdir(parents=True, exist_ok=True)
    path = trace_dir / f"{trace['run_id']}.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(trace, f, indent=2, ensure_ascii=False)
    return path


def load_trace(path_or_run_id: str, output_dir: str | Path = "./outputs") -> Dict[str, Any]:
    """Load a trace by file path or run ID."""
    candidate = Path(path_or_run_id)
    if not candidate.exists():
        candidate = Path(output_dir) / "traces" / f"{path_or_run_id}.json"
    with open(candidate, "r", encoding="utf-8") as f:
        return json.load(f)


def latest_trace_path(output_dir: str | Path = "./outputs") -> Optional[Path]:
    """Return the newest trace file, if present."""
    trace_dir = Path(output_dir) / "traces"
    if not trace_dir.exists():
        return None
    traces = sorted(trace_dir.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    return traces[0] if traces else None


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
            "temperature": kwargs.get("temperature"),
            "max_tokens": kwargs.get("max_tokens"),
            "prompt_preview": prompt[:160].replace("\n", " "),
            "usage": getattr(self.inner, "last_usage", None),
            "error": error,
            "timestamp": utc_now_iso(),
        }
        self.trace.setdefault("llm_calls", []).append(event)
        if error:
            self.trace.setdefault("errors", []).append({"where": "llm", "error": error})

    def __repr__(self) -> str:
        return f"InstrumentedLLM(inner={self.inner!r})"
