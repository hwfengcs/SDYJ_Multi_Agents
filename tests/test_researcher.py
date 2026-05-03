import threading
import time

from SDYJ_Agents.agents.researcher import Researcher


class FakeLLM:
    def generate(self, prompt: str, **kwargs) -> str:
        return "summary"


def test_execute_task_aggregates_results_and_marks_task_completed(monkeypatch):
    researcher = Researcher(FakeLLM())

    def fake_search(query, source):
        return {
            "query": query,
            "source": source,
            "results": [{"title": "Result", "url": "https://example.com", "snippet": "Body"}],
            "timestamp": "2026-01-01T00:00:00",
        }

    monkeypatch.setattr(researcher, "_search", fake_search)
    state = {
        "research_results": [],
        "research_plan": {
            "sub_tasks": [
                {
                    "task_id": 1,
                    "description": "Search docs",
                    "search_queries": ["agent workflow"],
                    "sources": ["tavily"],
                    "status": "pending",
                }
            ]
        },
    }

    updated = researcher.execute_task(state, state["research_plan"]["sub_tasks"][0])

    assert len(updated["research_results"]) == 1
    assert updated["research_results"][0]["task_id"] == 1
    assert updated["research_plan"]["sub_tasks"][0]["status"] == "completed"


def test_researcher_initializes_mcp_from_stdio_config():
    researcher = Researcher(
        FakeLLM(),
        mcp_transport="stdio",
        mcp_command="python",
        mcp_args=["server.py"],
        mcp_tool_name="search_docs",
    )

    assert researcher.mcp is not None
    assert researcher.mcp.transport == "stdio"
    assert researcher.mcp.command == "python"
    assert researcher.mcp.args == ["server.py"]
    assert researcher.mcp.default_tool_name == "search_docs"


def test_execute_task_collects_mcp_source_and_trace():
    class FakeMCP:
        def search(self, query):
            return {
                "query": query,
                "source": "mcp",
                "tool": "search_docs",
                "results": [
                    {
                        "title": "MCP docs",
                        "url": "file:///docs/mcp.md",
                        "snippet": "MCP trace details",
                        "relevance_score": 0.9,
                    }
                ],
                "timestamp": "2026-01-01T00:00:00",
            }

    researcher = Researcher(FakeLLM(), enable_reflection=False)
    researcher.mcp = FakeMCP()
    task = {
        "task_id": 2,
        "description": "Search MCP docs",
        "search_queries": ["trace"],
        "sources": ["mcp"],
        "status": "pending",
    }
    state = _parallel_state(task)

    updated = researcher.execute_task(state, task)

    assert updated["research_results"][0]["source"] == "mcp"
    assert updated["evidence_items"][0]["source"] == "mcp"
    assert updated["trace"]["tool_calls"][0]["source"] == "mcp"
    assert updated["trace"]["tool_calls"][0]["result_count"] == 1


class _Probe:
    def __init__(self):
        self.active = 0
        self.max_active = 0
        self.calls = []
        self.lock = threading.Lock()

    def enter(self, source: str, query: str) -> None:
        with self.lock:
            self.active += 1
            self.max_active = max(self.max_active, self.active)
            self.calls.append((source, query))

    def exit(self) -> None:
        with self.lock:
            self.active -= 1


class _DelayedSearch:
    def __init__(self, source: str, probe: _Probe, delay: float = 0.03):
        self.source = source
        self.probe = probe
        self.delay = delay

    def search(self, query: str, **kwargs):
        self.probe.enter(self.source, query)
        try:
            time.sleep(self.delay)
            return {
                "query": query,
                "source": self.source,
                "results": [
                    {
                        "title": f"{self.source}:{query}",
                        "url": f"https://example.com/{self.source}/{query}",
                        "snippet": "Body",
                        "relevance_score": 0.9,
                    }
                ],
                "timestamp": "2026-01-01T00:00:00",
            }
        finally:
            self.probe.exit()


def _parallel_state(task):
    return {
        "research_results": [],
        "evidence_items": [],
        "trace": {"events": [], "tool_calls": [], "replay_cache": {"tool_calls": []}, "errors": []},
        "research_plan": {"sub_tasks": [task]},
    }


def _parallel_task(queries, sources):
    return {
        "task_id": 7,
        "description": "Search docs",
        "search_queries": list(queries),
        "sources": list(sources),
        "status": "pending",
    }


def test_parallel_execution_collects_all_query_source_results_and_trace_entries():
    probe = _Probe()
    researcher = Researcher(FakeLLM(), enable_reflection=False)
    researcher.tavily = _DelayedSearch("tavily", probe)
    researcher.arxiv = _DelayedSearch("arxiv", probe)
    researcher.mcp = None

    task = _parallel_task(["q1", "q2"], ["tavily", "arxiv"])
    state = _parallel_state(task)

    researcher.execute_task(state, task)

    assert probe.max_active > 1
    assert len(state["research_results"]) == 4
    assert {(r["query"], r["source"]) for r in state["research_results"]} == {
        ("q1", "tavily"),
        ("q1", "arxiv"),
        ("q2", "tavily"),
        ("q2", "arxiv"),
    }
    assert len(state["trace"]["tool_calls"]) == 4
    assert len([e for e in state["trace"]["events"] if e["event_type"] == "tool_call"]) == 4
    assert len(state["trace"]["replay_cache"]["tool_calls"]) == 4


def test_parallel_execution_respects_concurrency_limit():
    probe = _Probe()
    researcher = Researcher(
        FakeLLM(),
        enable_reflection=False,
        parallel_concurrency_limit=2,
    )
    researcher.tavily = _DelayedSearch("tavily", probe)
    researcher.arxiv = _DelayedSearch("arxiv", probe)
    researcher.mcp = None

    task = _parallel_task(["q1", "q2", "q3"], ["tavily", "arxiv"])
    state = _parallel_state(task)

    researcher.execute_task(state, task)

    assert probe.max_active == 2
    assert len(state["research_results"]) == 6


class _BoomSearch:
    source = "arxiv"

    def search(self, query: str, **kwargs):
        raise RuntimeError("source exploded")


def test_parallel_execution_records_source_errors_without_failing_task():
    probe = _Probe()
    researcher = Researcher(FakeLLM(), enable_reflection=False)
    researcher.tavily = _DelayedSearch("tavily", probe, delay=0)
    researcher.arxiv = _BoomSearch()
    researcher.mcp = None

    task = _parallel_task(["q"], ["tavily", "arxiv"])
    state = _parallel_state(task)

    researcher.execute_task(state, task)

    assert len(state["research_results"]) == 2
    errors = [r for r in state["research_results"] if r.get("error")]
    assert len(errors) == 1
    assert errors[0]["source"] == "arxiv"
    assert "source exploded" in errors[0]["error"]
    assert state["research_plan"]["sub_tasks"][0]["status"] == "completed"
    assert len(state["trace"]["tool_calls"]) == 2
    assert any(call.get("error") for call in state["trace"]["tool_calls"])


def test_parallel_feature_flag_off_uses_legacy_sequential_order():
    probe = _Probe()
    researcher = Researcher(
        FakeLLM(),
        enable_reflection=False,
        enable_parallel_tool_execution=False,
    )
    researcher.tavily = _DelayedSearch("tavily", probe, delay=0)
    researcher.arxiv = _DelayedSearch("arxiv", probe, delay=0)
    researcher.mcp = None

    task = _parallel_task(["q1", "q2"], ["tavily", "arxiv"])
    state = _parallel_state(task)

    researcher.execute_task(state, task)

    assert probe.max_active == 1
    assert probe.calls == [
        ("tavily", "q1"),
        ("arxiv", "q1"),
        ("tavily", "q2"),
        ("arxiv", "q2"),
    ]
