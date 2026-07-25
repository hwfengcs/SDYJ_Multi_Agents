"""Deterministic replay helpers for recorded SDYJ runs."""

from __future__ import annotations

import copy
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterator

from .agents.coordinator import Coordinator
from .agents.planner import Planner
from .agents.rapporteur import Rapporteur
from .agents.researcher import Researcher
from .evaluation.metrics import evaluate_state
from .evaluation.scenarios import get_scenario
from .llm.base import BaseLLM
from .utils.tracing import (
    InstrumentedLLM,
    create_run_trace,
    merge_trace_state,
    record_decision,
    replay_prompt_hash,
    save_trace,
)
from .workflow.graph import ResearchWorkflow


class ReplayLLM(BaseLLM):
    """LLM that replays recorded responses, matching by prompt hash first.

    Hash matching (on the CURRENT_TIME-normalized prompt) keeps traces
    replayable even when the number or order of LLM calls shifts between code
    versions; hash-less legacy entries and unmatched prompts fall back to the
    original sequential-order behavior.
    """

    def __init__(self, recorded_calls: list[Dict[str, Any]], model: str = "replay-llm"):
        super().__init__(api_key="replay", model=model)
        self.recorded_calls = [dict(call) for call in recorded_calls]
        self._consumed = [False] * len(self.recorded_calls)
        self._by_hash: Dict[str, deque] = {}
        for index, call in enumerate(self.recorded_calls):
            prompt_hash = call.get("prompt_hash")
            if prompt_hash:
                self._by_hash.setdefault(prompt_hash, deque()).append(index)
        self.last_usage = None

    def _next_sequential(self) -> int | None:
        for index, used in enumerate(self._consumed):
            if not used:
                return index
        return None

    def generate(self, prompt: str, **kwargs) -> str:
        index = None
        queue = self._by_hash.get(replay_prompt_hash(prompt))
        while queue:
            candidate = queue.popleft()
            if not self._consumed[candidate]:
                index = candidate
                break
        if index is None:
            index = self._next_sequential()
        if index is None:
            raise RuntimeError("ReplayLLM exhausted recorded LLM calls")
        self._consumed[index] = True
        call = self.recorded_calls[index]
        if call.get("error"):
            raise RuntimeError(call["error"])
        return call.get("response", "")

    def stream_generate(self, prompt: str, **kwargs) -> Iterator[str]:
        yield self.generate(prompt, **kwargs)


@dataclass
class ReplaySearchTool:
    """Search adapter that replays recorded tool results.

    Entries are matched by exact query first, then by recorded order within
    this tool's source, so reordered searches still find their results.
    """

    source: str
    recorded_calls: list[Dict[str, Any]]

    def __post_init__(self) -> None:
        self._entries = [
            dict(call)
            for call in self.recorded_calls
            if str(call.get("source", "")).lower() == self.source.lower()
        ]
        self._consumed = [False] * len(self._entries)
        self._by_query: Dict[str, deque] = {}
        for index, call in enumerate(self._entries):
            self._by_query.setdefault(str(call.get("query", "")), deque()).append(index)

    def _next_sequential(self) -> int | None:
        for index, used in enumerate(self._consumed):
            if not used:
                return index
        return None

    def search(self, query: str, **kwargs) -> Dict[str, Any]:
        index = None
        queue = self._by_query.get(str(query))
        while queue:
            candidate = queue.popleft()
            if not self._consumed[candidate]:
                index = candidate
                break
        if index is None:
            index = self._next_sequential()
        if index is None:
            return {
                "query": query,
                "source": self.source,
                "results": [],
                "error": f"ReplaySearchTool exhausted recorded calls for {self.source}",
            }
        self._consumed[index] = True
        result = copy.deepcopy(self._entries[index].get("result") or {})
        if not result:
            result = {
                "query": query,
                "source": self.source,
                "results": [],
                "error": "recorded tool result missing",
            }
        return result


def can_deterministically_replay(trace: Dict[str, Any]) -> tuple[bool, str]:
    """Return whether a trace has enough recorded I/O for deterministic replay."""
    cache = trace.get("replay_cache") or {}
    if not cache.get("llm_calls"):
        return False, "trace does not contain recorded LLM responses"
    if trace.get("tool_calls") and not cache.get("tool_calls"):
        return False, "trace does not contain recorded tool results"
    return True, "ok"


def run_deterministic_replay(
    source_trace: Dict[str, Any],
    output_dir: str | Path = "./outputs",
) -> Dict[str, Any]:
    """Replay a run using recorded LLM and tool outputs."""
    ok, reason = can_deterministically_replay(source_trace)
    if not ok:
        raise ValueError(reason)

    cache = source_trace.get("replay_cache") or {}
    replay_trace = create_run_trace(
        query=source_trace.get("query", ""),
        provider="replay",
        model=source_trace.get("model") or "replay-llm",
        mode="replay",
        scenario_id=source_trace.get("scenario_id"),
    )
    replay_trace["replayed_from"] = source_trace.get("run_id")
    record_decision(
        replay_trace,
        node="replay",
        decision="deterministic_replay_started",
        reason=f"source_run_id={source_trace.get('run_id')}",
    )

    llm = InstrumentedLLM(
        ReplayLLM(cache.get("llm_calls", []), model=source_trace.get("model") or "replay-llm"),
        replay_trace,
    )
    coordinator = Coordinator(llm)
    planner = Planner(llm)
    researcher = Researcher(llm)
    tool_calls = cache.get("tool_calls", [])
    researcher.tavily = ReplaySearchTool("tavily", tool_calls)
    researcher.arxiv = ReplaySearchTool("arxiv", tool_calls)
    researcher.mcp = ReplaySearchTool("mcp", tool_calls)
    rapporteur = Rapporteur(llm)
    workflow = ResearchWorkflow(coordinator, planner, researcher, rapporteur)

    final_state: Dict[str, Any] = {}
    for update in workflow.stream_interactive(
        source_trace.get("query", ""),
        max_iterations=(source_trace.get("config") or {}).get("max_iterations") or 5,
        auto_approve=True,
        output_format=(source_trace.get("report") or {}).get("format", "markdown"),
        trace=replay_trace,
    ):
        for value in update.values():
            if isinstance(value, dict):
                final_state = value

    replay_trace = merge_trace_state(replay_trace, final_state.get("trace"))
    replay_trace.setdefault("metrics", {})["replay_source_run_id"] = source_trace.get("run_id")
    if source_trace.get("scenario_id"):
        try:
            scenario = get_scenario(source_trace["scenario_id"])
            replay_metrics = evaluate_state(final_state, scenario, trace=replay_trace)
            replay_trace.setdefault("metrics", {}).update(replay_metrics)
        except Exception as exc:
            replay_trace.setdefault("errors", []).append(
                {"where": "replay:evaluate_state", "error": str(exc)}
            )
    trace_path = save_trace(replay_trace, output_dir, final_state=final_state)
    return {
        "source_run_id": source_trace.get("run_id"),
        "replay_run_id": replay_trace.get("run_id"),
        "trace_path": str(trace_path) if trace_path else None,
        "final_state": final_state,
        "trace": replay_trace,
    }
