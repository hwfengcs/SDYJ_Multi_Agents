"""Deterministic replay helpers for recorded SDYJ runs."""

from __future__ import annotations

import copy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterator

from .agents.coordinator import Coordinator
from .agents.planner import Planner
from .agents.rapporteur import Rapporteur
from .agents.researcher import Researcher
from .agents.verifier import Verifier
from .evaluation.metrics import evaluate_state
from .evaluation.scenarios import get_scenario
from .llm.base import BaseLLM
from .utils.tracing import (
    InstrumentedLLM,
    create_run_trace,
    merge_trace_state,
    record_decision,
    save_trace,
)
from .workflow.graph import ResearchWorkflow


class ReplayLLM(BaseLLM):
    """LLM that replays recorded responses in call order."""

    def __init__(self, recorded_calls: list[Dict[str, Any]], model: str = "replay-llm"):
        super().__init__(api_key="replay", model=model)
        self.recorded_calls = list(recorded_calls)
        self.index = 0
        self.last_usage = None

    def generate(self, prompt: str, **kwargs) -> str:
        if self.index >= len(self.recorded_calls):
            raise RuntimeError("ReplayLLM exhausted recorded LLM calls")
        call = self.recorded_calls[self.index]
        self.index += 1
        if call.get("error"):
            raise RuntimeError(call["error"])
        return call.get("response", "")

    def stream_generate(self, prompt: str, **kwargs) -> Iterator[str]:
        yield self.generate(prompt, **kwargs)


@dataclass
class ReplaySearchTool:
    """Search adapter that replays recorded tool results."""

    source: str
    recorded_calls: list[Dict[str, Any]]

    def __post_init__(self) -> None:
        self.index = 0

    def search(self, query: str, **kwargs) -> Dict[str, Any]:
        while self.index < len(self.recorded_calls):
            call = self.recorded_calls[self.index]
            self.index += 1
            if str(call.get("source", "")).lower() != self.source.lower():
                continue
            result = copy.deepcopy(call.get("result") or {})
            if not result:
                result = {
                    "query": query,
                    "source": self.source,
                    "results": [],
                    "error": "recorded tool result missing",
                }
            return result
        return {
            "query": query,
            "source": self.source,
            "results": [],
            "error": f"ReplaySearchTool exhausted recorded calls for {self.source}",
        }


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
    # Replay should reuse whatever verification config the source run used.
    # That keeps the recorded LLM call order intact: the verifier may have
    # made one or more LLM calls during the original run, and the
    # ReplayLLM hands those back in order. Forcing skip=True on a trace
    # that *did* run the verifier would leave the verifier's recorded calls
    # consumed by other nodes and break replay.
    source_config = source_trace.get("config") or {}
    skip_verification = bool(source_config.get("skip_verification", True))
    max_revisions = int(source_config.get("max_revisions") or 0)
    verifier = None if skip_verification else Verifier(llm)
    workflow = ResearchWorkflow(coordinator, planner, researcher, rapporteur, verifier)

    final_state: Dict[str, Any] = {}
    for update in workflow.stream_interactive(
        source_trace.get("query", ""),
        max_iterations=(source_trace.get("config") or {}).get("max_iterations") or 5,
        auto_approve=True,
        output_format=(source_trace.get("report") or {}).get("format", "markdown"),
        trace=replay_trace,
        skip_verification=skip_verification,
        max_revisions=max_revisions,
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
