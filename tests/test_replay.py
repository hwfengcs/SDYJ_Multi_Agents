from pathlib import Path

from SDYJ_Agents import replay as replay_module
from SDYJ_Agents.replay import _recorded_tool_calls_for_replay


class CapturingPlanner:
    instances = []

    def __init__(self, llm, enable_plan_refinement=True):
        self.llm = llm
        self.enable_plan_refinement = enable_plan_refinement
        self.__class__.instances.append(self)


class CapturingResearcher:
    instances = []

    def __init__(
        self,
        llm,
        enable_reflection=True,
        enable_parallel_tool_execution=True,
    ):
        self.llm = llm
        self.enable_reflection = enable_reflection
        self.enable_parallel_tool_execution = enable_parallel_tool_execution
        self.__class__.instances.append(self)


class FakeWorkflow:
    def __init__(self, coordinator, planner, researcher, rapporteur, verifier):
        self.planner = planner

    def stream_interactive(self, query, **kwargs):
        trace = kwargs["trace"]
        yield {
            "final": {
                "query": query,
                "trace": trace,
                "final_report": "replayed report",
                "research_plan": {"sub_tasks": []},
            }
        }


def _source_trace(config=None):
    return {
        "run_id": "source-run",
        "query": "How do agent traces help debugging?",
        "model": "fake-model",
        "config": config or {},
        "replay_cache": {"llm_calls": [{"response": "{}"}], "tool_calls": []},
        "tool_calls": [],
        "report": {"format": "markdown"},
    }


def _patch_replay_runtime(monkeypatch):
    CapturingPlanner.instances = []
    CapturingResearcher.instances = []
    monkeypatch.setattr(replay_module, "Planner", CapturingPlanner)
    monkeypatch.setattr(replay_module, "Researcher", CapturingResearcher)
    monkeypatch.setattr(replay_module, "ResearchWorkflow", FakeWorkflow)
    monkeypatch.setattr(
        replay_module,
        "save_trace",
        lambda trace, output_dir, final_state=None: Path(output_dir) / "trace.json",
    )


def test_replay_defaults_plan_refinement_off_for_legacy_traces(monkeypatch, tmp_path):
    _patch_replay_runtime(monkeypatch)

    result = replay_module.run_deterministic_replay(_source_trace(), output_dir=tmp_path)

    assert CapturingPlanner.instances[-1].enable_plan_refinement is False
    assert result["trace"]["config"]["enable_plan_refinement"] is False
    assert CapturingResearcher.instances[-1].enable_parallel_tool_execution is False
    assert result["trace"]["config"]["enable_parallel_tool_execution"] is False


def test_replay_restores_plan_refinement_flag_from_source_trace(monkeypatch, tmp_path):
    _patch_replay_runtime(monkeypatch)

    result = replay_module.run_deterministic_replay(
        _source_trace({"enable_plan_refinement": True}),
        output_dir=tmp_path,
    )

    assert CapturingPlanner.instances[-1].enable_plan_refinement is True
    assert result["trace"]["config"]["enable_plan_refinement"] is True


def test_replay_restores_parallel_tool_execution_flag_from_source_trace(monkeypatch, tmp_path):
    _patch_replay_runtime(monkeypatch)

    result = replay_module.run_deterministic_replay(
        _source_trace({"enable_parallel_tool_execution": True}),
        output_dir=tmp_path,
    )

    assert CapturingResearcher.instances[-1].enable_parallel_tool_execution is True
    assert result["trace"]["config"]["enable_parallel_tool_execution"] is True


def test_replay_synthesizes_recorded_tool_errors():
    source_trace = _source_trace()
    source_trace["tool_calls"] = [
        {
            "tool_call_id": "T1",
            "source": "tavily",
            "query": "missing key query",
            "task_id": 1,
            "result_count": 0,
            "error": "source unavailable or unsupported",
        },
        {
            "tool_call_id": "T2",
            "source": "arxiv",
            "query": "cached query",
            "task_id": 1,
            "result_count": 1,
            "error": None,
        },
    ]
    source_trace["replay_cache"]["tool_calls"] = [
        {
            "tool_call_id": "T2",
            "source": "arxiv",
            "query": "cached query",
            "task_id": 1,
            "result": {"query": "cached query", "source": "arxiv", "results": [{"title": "ok"}]},
        }
    ]

    recorded = _recorded_tool_calls_for_replay(source_trace)

    tavily = next(call for call in recorded if call["tool_call_id"] == "T1")
    assert tavily["result"] is None
    assert len(recorded) == 2
