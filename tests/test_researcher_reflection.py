"""Tests for the Reflexive Researcher (v0.6).

The reflection step is a behavioural test: we want to know that *given
weak first-pass results*, the researcher rewrites the queries and runs
them again, and that none of the safety rails (single-shot reflection,
graceful failure on parse errors, opt-out flag) regress.
"""

from __future__ import annotations

import json
from typing import Any, Dict, Iterator, List

from SDYJ_Agents.agents.researcher import (
    HIGH_ERROR_RATE_THRESHOLD,
    WEAK_RELEVANCE_THRESHOLD,
    Researcher,
)


class _ScriptedLLM:
    """Returns canned responses in order, captures every prompt."""

    def __init__(self, responses: List[str]):
        self.responses = list(responses)
        self.calls: List[str] = []

    def generate(self, prompt: str, **kwargs) -> str:
        self.calls.append(prompt)
        if not self.responses:
            return ""
        return self.responses.pop(0)

    def stream_generate(self, prompt: str, **kwargs) -> Iterator[str]:
        yield self.generate(prompt, **kwargs)


class _ScriptedSearch:
    """Search adapter that returns scripted batches per query.

    Each call consumes one entry from ``queue``. If ``queue`` runs out we
    return an empty batch — mirroring how a real search would behave when
    rate-limited or empty.
    """

    def __init__(self, source: str, queue: List[Dict[str, Any]]):
        self.source = source
        self.queue = list(queue)
        self.queries: List[str] = []

    def search(self, query: str, **kwargs) -> Dict[str, Any]:
        self.queries.append(query)
        if self.queue:
            payload = self.queue.pop(0)
        else:
            payload = {"results": []}
        return {
            "query": query,
            "source": self.source,
            "results": payload.get("results", []),
            "error": payload.get("error"),
            "timestamp": "now",
        }


def _state(plan_tasks: List[Dict[str, Any]]) -> Dict[str, Any]:
    return {
        "query": "agent observability",
        "research_plan": {"sub_tasks": plan_tasks},
        "research_results": [],
        "evidence_items": [],
        "trace": {"events": [], "nodes": [], "metrics": {}},
    }


def _task(queries: List[str], sources: List[str]) -> Dict[str, Any]:
    return {
        "task_id": 1,
        "description": "investigate observability",
        "search_queries": list(queries),
        "sources": list(sources),
        "status": "pending",
    }


# ----- _should_reflect -----


def test_should_reflect_when_no_batches():
    assert Researcher._should_reflect([]) is True


def test_should_reflect_when_all_results_empty():
    batches = [{"source": "tavily", "results": []}, {"source": "arxiv", "results": []}]
    assert Researcher._should_reflect(batches) is True


def test_should_reflect_when_error_rate_at_threshold():
    batches = [
        {"source": "tavily", "results": [], "error": "timeout"},
        {"source": "arxiv", "results": [{"title": "ok", "relevance_score": 0.9}]},
    ]
    # 1 error / 2 batches = 0.5 = HIGH_ERROR_RATE_THRESHOLD -> reflect.
    assert HIGH_ERROR_RATE_THRESHOLD == 0.5
    assert Researcher._should_reflect(batches) is True


def test_should_reflect_when_average_relevance_below_threshold():
    batches = [
        {
            "source": "tavily",
            "results": [
                {"title": "noise 1", "relevance_score": 0.2},
                {"title": "noise 2", "relevance_score": 0.3},
            ],
        },
    ]
    avg = Researcher._average_relevance(batches)
    assert avg is not None and avg < WEAK_RELEVANCE_THRESHOLD
    assert Researcher._should_reflect(batches) is True


def test_should_not_reflect_when_results_are_strong():
    batches = [
        {
            "source": "tavily",
            "results": [
                {"title": "exact match", "relevance_score": 0.92},
                {"title": "good", "relevance_score": 0.85},
            ],
        }
    ]
    assert Researcher._should_reflect(batches) is False


def test_average_relevance_neutral_for_unscored_results():
    """arXiv-style results without a score should not bias the average to 0."""
    batches = [
        {
            "source": "tavily",
            "results": [{"title": "good", "relevance_score": 0.9}],
        },
        {
            "source": "arxiv",
            "results": [{"title": "no score"}, {"title": "no score 2"}],
        },
    ]
    avg = Researcher._average_relevance(batches)
    # 0.9 + 0.5 + 0.5 / 3 = ~0.633
    assert avg is not None and 0.6 < avg < 0.7


# ----- _parse_reflection_response -----


def test_parse_reflection_extracts_rewritten_queries():
    payload = {
        "diagnosis": "too jargon-heavy",
        "rewritten_queries": ["LangGraph observability", "agent trace tools"],
    }
    queries = Researcher._parse_reflection_response(json.dumps(payload))
    assert queries == ["LangGraph observability", "agent trace tools"]


def test_parse_reflection_handles_fenced_json():
    raw = "```json\n" + json.dumps({"diagnosis": "x", "rewritten_queries": ["a"]}) + "\n```"
    queries = Researcher._parse_reflection_response(raw)
    assert queries == ["a"]


def test_parse_reflection_returns_empty_on_garbage():
    assert Researcher._parse_reflection_response("the query was fine") == []
    assert Researcher._parse_reflection_response("") == []
    # Wrong shape (rewritten_queries is a string, not a list).
    bad = json.dumps({"diagnosis": "x", "rewritten_queries": "agent trace"})
    assert Researcher._parse_reflection_response(bad) == []


def test_parse_reflection_dedups_and_caps_at_two():
    payload = {
        "rewritten_queries": [
            "agent trace tools",
            "Agent Trace Tools",  # case-only duplicate
            "  ",  # blank
            "different query",
            "third query that should be dropped",
        ],
    }
    queries = Researcher._parse_reflection_response(json.dumps(payload))
    # Dedup is case-insensitive, blanks stripped, then capped at 2.
    assert len(queries) == 2
    assert queries[0] == "agent trace tools"
    assert queries[1] == "different query"


# ----- execute_task end-to-end -----


def test_execute_task_retries_with_rewritten_queries_when_first_pass_is_weak():
    # First pass: the only configured query returns one low-relevance hit so
    # the average score is below WEAK_RELEVANCE_THRESHOLD.
    weak_payload = {
        "results": [{"title": "off-topic", "url": "https://x", "relevance_score": 0.1}]
    }
    strong_payload = {
        "results": [
            {"title": "agent trace tools survey", "url": "https://y", "relevance_score": 0.93}
        ]
    }
    search = _ScriptedSearch("tavily", [weak_payload, strong_payload])

    llm = _ScriptedLLM(
        [
            json.dumps(
                {
                    "diagnosis": "too jargon",
                    "rewritten_queries": ["agent trace survey"],
                }
            )
        ]
    )

    researcher = Researcher(llm)
    researcher.tavily = search
    researcher.arxiv = None
    researcher.mcp = None

    state = _state([_task(["complex jargon"], ["tavily"])])
    task = state["research_plan"]["sub_tasks"][0]

    researcher.execute_task(state, task)

    # Two calls: original + rewritten.
    assert search.queries == ["complex jargon", "agent trace survey"]
    assert task.get("_reflected") is True
    # State should have both batches accumulated.
    assert len(state["research_results"]) == 2
    # Reflection counter should land in the trace metrics.
    assert state["trace"]["metrics"]["reflection_count"] == 1


def test_execute_task_does_not_reflect_when_first_pass_is_strong():
    strong = {
        "results": [{"title": "perfect", "url": "https://y", "relevance_score": 0.95}]
    }
    search = _ScriptedSearch("tavily", [strong])
    llm = _ScriptedLLM([])  # would be consumed if reflection ran
    researcher = Researcher(llm)
    researcher.tavily = search
    researcher.arxiv = None
    researcher.mcp = None

    state = _state([_task(["good query"], ["tavily"])])
    task = state["research_plan"]["sub_tasks"][0]

    researcher.execute_task(state, task)

    assert search.queries == ["good query"]
    assert llm.calls == [], "reflection LLM must not be invoked when first pass is strong"
    assert task.get("_reflected") is None
    assert "reflection_count" not in state["trace"]["metrics"]


def test_execute_task_skips_reflection_when_disabled():
    """``enable_reflection=False`` must restore v0.5 single-pass behavior."""
    weak = {"results": []}
    search = _ScriptedSearch("tavily", [weak])
    llm = _ScriptedLLM([])  # would crash test if reflection LLM fired
    researcher = Researcher(llm, enable_reflection=False)
    researcher.tavily = search
    researcher.arxiv = None
    researcher.mcp = None

    state = _state([_task(["empty result"], ["tavily"])])
    task = state["research_plan"]["sub_tasks"][0]

    researcher.execute_task(state, task)

    assert search.queries == ["empty result"]
    assert llm.calls == []
    assert task.get("_reflected") is None


def test_execute_task_only_reflects_once_per_task():
    """If the rewritten queries also fail, the researcher must NOT reflect
    again on the same task — that would risk an unbounded loop."""
    empty1 = {"results": []}
    empty2 = {"results": []}
    search = _ScriptedSearch("tavily", [empty1, empty2])
    llm = _ScriptedLLM(
        [
            json.dumps({"diagnosis": "x", "rewritten_queries": ["second try"]})
        ]
    )

    researcher = Researcher(llm)
    researcher.tavily = search
    researcher.arxiv = None
    researcher.mcp = None

    state = _state([_task(["first"], ["tavily"])])
    task = state["research_plan"]["sub_tasks"][0]

    researcher.execute_task(state, task)

    assert search.queries == ["first", "second try"]
    assert task.get("_reflected") is True
    # Only one reflection LLM call even though the second pass also failed.
    assert len(llm.calls) == 1


def test_execute_task_handles_reflection_llm_failure_gracefully():
    """A broken LLM must not crash the workflow — we just skip the retry."""

    class _BoomLLM:
        def generate(self, prompt: str, **kwargs):
            raise RuntimeError("model is down")

        def stream_generate(self, prompt: str, **kwargs):
            yield ""

    search = _ScriptedSearch("tavily", [{"results": []}])
    researcher = Researcher(_BoomLLM())
    researcher.tavily = search
    researcher.arxiv = None
    researcher.mcp = None

    state = _state([_task(["only query"], ["tavily"])])
    task = state["research_plan"]["sub_tasks"][0]

    # Should complete without raising.
    researcher.execute_task(state, task)
    assert task.get("_reflected") is True
    assert search.queries == ["only query"]  # no retry happened


def test_reflection_retry_runs_rewritten_query_through_parallel_sources():
    tavily = _ScriptedSearch(
        "tavily",
        [
            {"results": [{"title": "noise", "url": "https://x", "relevance_score": 0.1}]},
            {"results": [{"title": "good", "url": "https://y", "relevance_score": 0.95}]},
        ],
    )
    arxiv = _ScriptedSearch(
        "arxiv",
        [
            {"results": [{"title": "paper", "url": "https://arxiv.org/a", "relevance_score": 0.1}]},
            {"results": [{"title": "better paper", "url": "https://arxiv.org/b", "relevance_score": 0.95}]},
        ],
    )
    llm = _ScriptedLLM(
        [
            json.dumps(
                {
                    "diagnosis": "too vague",
                    "rewritten_queries": ["agent observability survey"],
                }
            )
        ]
    )

    researcher = Researcher(llm)
    researcher.tavily = tavily
    researcher.arxiv = arxiv
    researcher.mcp = None

    state = _state([_task(["vague agent traces"], ["tavily", "arxiv"])])
    task = state["research_plan"]["sub_tasks"][0]

    researcher.execute_task(state, task)

    assert tavily.queries == ["vague agent traces", "agent observability survey"]
    assert arxiv.queries == ["vague agent traces", "agent observability survey"]
    assert task.get("_reflected") is True
    assert task["search_queries"] == ["vague agent traces", "agent observability survey"]
    assert len(state["research_results"]) == 4
    assert len(state["trace"]["tool_calls"]) == 4
    assert state["trace"]["metrics"]["reflection_count"] == 1
