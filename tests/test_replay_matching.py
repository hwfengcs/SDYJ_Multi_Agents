"""Tests for hash-first replay matching with legacy index fallback."""

import pytest

from SDYJ_Agents.replay import (
    ReplayLLM,
    ReplaySearchTool,
    can_deterministically_replay,
    run_deterministic_replay,
)
from SDYJ_Agents.utils.tracing import replay_prompt_hash


def test_replay_llm_matches_by_prompt_hash_out_of_order():
    calls = [
        {"call_id": "L1", "prompt_hash": replay_prompt_hash("prompt A"), "response": "resp A"},
        {"call_id": "L2", "prompt_hash": replay_prompt_hash("prompt B"), "response": "resp B"},
    ]
    llm = ReplayLLM(calls)

    assert llm.generate("prompt B") == "resp B"
    assert llm.generate("prompt A") == "resp A"


def test_replay_hash_ignores_injected_current_time():
    recorded_prompt = "---\nCURRENT_TIME: 2026-07-25 10:00:00\n---\nsame body"
    replayed_prompt = "---\nCURRENT_TIME: 2027-01-01 00:00:00\n---\nsame body"

    assert replay_prompt_hash(recorded_prompt) == replay_prompt_hash(replayed_prompt)

    llm = ReplayLLM(
        [{"prompt_hash": replay_prompt_hash(recorded_prompt), "response": "matched"}]
    )
    assert llm.generate(replayed_prompt) == "matched"


def test_replay_llm_index_fallback_for_legacy_hashless_traces():
    calls = [
        {"call_id": "L1", "response": "first"},
        {"call_id": "L2", "response": "second"},
    ]
    llm = ReplayLLM(calls)

    assert llm.generate("anything") == "first"
    assert llm.generate("else") == "second"


def test_replay_llm_unmatched_prompt_falls_back_to_sequence():
    calls = [
        {"prompt_hash": replay_prompt_hash("known"), "response": "known-resp"},
        {"prompt_hash": replay_prompt_hash("other"), "response": "other-resp"},
    ]
    llm = ReplayLLM(calls)

    # A prompt that hashes to nothing recorded consumes the next unconsumed
    # entry in order, matching the legacy behavior.
    assert llm.generate("brand new prompt") == "known-resp"
    assert llm.generate("other") == "other-resp"


def test_replay_llm_replays_recorded_errors_and_exhaustion():
    llm = ReplayLLM([{"response": "", "error": "recorded failure"}])
    with pytest.raises(RuntimeError, match="recorded failure"):
        llm.generate("x")
    with pytest.raises(RuntimeError, match="exhausted"):
        llm.generate("x")


def test_replay_search_tool_matches_query_first_then_order():
    calls = [
        {
            "source": "tavily",
            "query": "q1",
            "result": {"query": "q1", "source": "tavily", "results": [{"title": "one"}]},
        },
        {
            "source": "tavily",
            "query": "q2",
            "result": {"query": "q2", "source": "tavily", "results": [{"title": "two"}]},
        },
        {
            "source": "arxiv",
            "query": "q1",
            "result": {"query": "q1", "source": "arxiv", "results": []},
        },
    ]
    tool = ReplaySearchTool("tavily", calls)

    assert tool.search("q2")["results"][0]["title"] == "two"  # out of order, by query
    assert tool.search("q1")["results"][0]["title"] == "one"
    assert tool.search("q3")["error"]  # exhausted for this source


def test_fresh_trace_round_trips_through_replay(tmp_path):
    from SDYJ_Agents.evaluation.runner import run_evaluation
    from SDYJ_Agents.utils.tracing import load_trace

    summary = run_evaluation(live=False, max_scenarios=1, output_dir=str(tmp_path))
    run_id = summary["results"][0]["run_id"]
    trace = load_trace(run_id, str(tmp_path))

    ok, reason = can_deterministically_replay(trace)
    assert ok, reason

    result = run_deterministic_replay(trace, output_dir=str(tmp_path))
    final_state = result["final_state"]

    assert final_state.get("final_report")
    assert result["trace"]["metrics"]["evidence_count"] == (
        summary["results"][0]["metrics"]["evidence_count"]
    )
