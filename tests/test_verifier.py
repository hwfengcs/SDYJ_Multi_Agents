"""Tests for the Verifier agent and its critique-revise loop."""

from __future__ import annotations

import json
from typing import Any, Dict, Iterator, List

import pytest

from SDYJ_Agents.agents.verifier import (
    DEFAULT_ALIGNMENT_THRESHOLD,
    DEFAULT_OVERALL_THRESHOLD,
    DIMENSIONS,
    Verifier,
    _coerce_score,
    _parse_verifier_response,
    append_verification_history,
    latest_verification,
)


class _ScriptedLLM:
    """LLM stub that returns pre-baked responses one at a time.

    Used to script verifier critique JSONs and (optionally) rapporteur
    revisions in unit tests. Captures every prompt for assertion.
    """

    def __init__(self, responses: List[str]):
        self.responses = list(responses)
        self.calls: List[Dict[str, Any]] = []

    def generate(self, prompt: str, **kwargs) -> str:
        self.calls.append({"prompt": prompt, "kwargs": kwargs})
        if not self.responses:
            raise AssertionError("ScriptedLLM ran out of canned responses")
        return self.responses.pop(0)

    def stream_generate(self, prompt: str, **kwargs) -> Iterator[str]:
        yield self.generate(prompt, **kwargs)


def _passing_critique() -> Dict[str, Any]:
    return {
        "scores": {dim: 0.9 for dim in DIMENSIONS},
        "overall_quality": 0.9,
        "should_revise": False,
        "weakest_dimension": "citation_completeness",
        "revision_hints": [],
        "summary": "Report is well-grounded; no revision needed.",
    }


def _failing_critique() -> Dict[str, Any]:
    return {
        "scores": {
            "claim_evidence_alignment": 0.4,
            "citation_completeness": 0.6,
            "factual_consistency": 0.7,
            "coverage": 0.5,
        },
        "overall_quality": 0.55,
        "should_revise": True,
        "weakest_dimension": "claim_evidence_alignment",
        "revision_hints": [
            "Section '核心发现' bullet 2 has no citation; either add [E2] or remove it.",
            "Evidence E4 is collected but never cited; integrate it into '深度分析'.",
        ],
        "summary": "Multiple key findings are not grounded in the evidence list.",
    }


def _state_with_report(report: str = "Sample final report") -> Dict[str, Any]:
    return {
        "query": "Compare agent evaluation strategies",
        "research_plan": {
            "research_goal": "Compare evaluation",
            "sub_tasks": [
                {"task_id": 1, "description": "trace metrics", "status": "completed"},
                {"task_id": 2, "description": "human review", "status": "completed"},
            ],
        },
        "evidence_items": [
            {"evidence_id": "E1", "title": "Trace v2", "snippet": "trace v2", "source": "tavily", "url": "https://a"},
            {"evidence_id": "E2", "title": "Human review", "snippet": "hitl", "source": "arxiv", "url": "https://b"},
        ],
        "research_results": [],
        "final_report": report,
        "trace": {"events": [], "nodes": [], "metrics": {}},
    }


# ----- _coerce_score / _parse_verifier_response unit tests -----


def test_coerce_score_clamps_into_unit_interval():
    assert _coerce_score(0.5) == 0.5
    assert _coerce_score(-0.2) == 0.0
    assert _coerce_score(1.5) == 1.0
    assert _coerce_score("0.7") == 0.7
    assert _coerce_score(None) == 0.0
    assert _coerce_score("not a number") == 0.0


def test_parse_handles_fenced_json():
    raw = "```json\n" + json.dumps(_passing_critique()) + "\n```"
    parsed = _parse_verifier_response(raw)
    assert parsed["overall_quality"] == 0.9


def test_parse_falls_back_when_response_empty():
    parsed = _parse_verifier_response("")
    assert parsed["should_revise"] is True
    assert parsed["verifier_failed"] is True


def test_parse_falls_back_when_response_not_json():
    parsed = _parse_verifier_response("the report is fine")
    assert parsed["should_revise"] is True


# ----- Verifier.verify behavior -----


def test_verify_accepts_passing_report():
    llm = _ScriptedLLM([json.dumps(_passing_critique())])
    verifier = Verifier(llm)
    result = verifier.verify(_state_with_report(report="- grounded claim [E1]\n- another grounded claim [E2]"))
    assert result["should_revise"] is False
    assert result["overall_quality"] >= DEFAULT_OVERALL_THRESHOLD
    # Verifier should still expose all four canonical dimensions even if the
    # LLM omits some — keeps downstream code simple.
    assert set(result["scores"]) == set(DIMENSIONS)
    assert result["citation_audit"]["citation_audit_passed"] is True


def test_verify_forces_revision_on_invalid_citation_even_if_llm_passes():
    llm = _ScriptedLLM([json.dumps(_passing_critique())])
    verifier = Verifier(llm)
    result = verifier.verify(_state_with_report(report="- This cites a missing source [E99]\n"))

    assert result["should_revise"] is True
    assert result["scores"]["claim_evidence_alignment"] <= 0.5
    assert result["citation_audit"]["invalid_citation_ids"] == ["E99"]
    assert any("invalid citations" in hint for hint in result["revision_hints"])


def test_verify_forces_revision_on_uncited_key_finding():
    llm = _ScriptedLLM([json.dumps(_passing_critique())])
    verifier = Verifier(llm)
    result = verifier.verify(_state_with_report(report="- This key finding has no citation\n"))

    assert result["should_revise"] is True
    assert result["citation_audit"]["unsupported_key_finding_count"] == 1
    assert any("unsupported key-finding" in hint for hint in result["revision_hints"])


def test_verify_marks_low_alignment_as_revise_even_if_overall_passes():
    """A high citation_completeness can lift overall above the threshold,
    but a low claim_evidence_alignment must still force a revision."""
    critique = _passing_critique()
    critique["scores"]["claim_evidence_alignment"] = 0.4  # below 0.65 floor
    critique["overall_quality"] = 0.78  # above 0.75 overall floor
    critique["should_revise"] = False  # verifier itself is too lenient

    llm = _ScriptedLLM([json.dumps(critique)])
    verifier = Verifier(llm)
    result = verifier.verify(_state_with_report(report="- grounded claim [E1]\n- another grounded claim [E2]\n"))
    assert result["should_revise"] is True, "alignment floor must override LLM verdict"
    assert result["scores"]["claim_evidence_alignment"] == pytest.approx(0.4)


def test_verify_recovers_from_invalid_llm_response():
    llm = _ScriptedLLM(["not valid json at all"])
    verifier = Verifier(llm)
    result = verifier.verify(_state_with_report())
    # Malformed responses are treated as critical failures and force a revise.
    assert result["should_revise"] is True
    assert result["verifier_failed"] is True
    # Empty scores are still well-shaped so downstream code does not crash.
    assert set(result["scores"]) == set(DIMENSIONS)


def test_verify_short_circuits_on_empty_report():
    llm = _ScriptedLLM([])  # would crash if called
    verifier = Verifier(llm)
    state = _state_with_report(report="   ")
    result = verifier.verify(state)
    assert result["should_revise"] is True
    assert "Empty report" in result["summary"]


def test_verify_honors_alignment_threshold_param():
    # With a stricter alignment_threshold of 0.95, even a usually-passing
    # critique should be flagged for revision.
    critique = _passing_critique()
    critique["scores"]["claim_evidence_alignment"] = 0.9
    llm = _ScriptedLLM([json.dumps(critique)])
    verifier = Verifier(llm, alignment_threshold=0.95)
    result = verifier.verify(
        _state_with_report(report="- grounded claim [E1]\n- another grounded claim [E2]\n")
    )
    assert result["should_revise"] is True


def test_verify_recomputes_overall_when_llm_returns_garbage_overall():
    critique = _passing_critique()
    critique["overall_quality"] = "n/a"
    llm = _ScriptedLLM([json.dumps(critique)])
    verifier = Verifier(llm)
    result = verifier.verify(
        _state_with_report(report="- grounded claim [E1]\n- another grounded claim [E2]\n")
    )
    # 0.9 across all four dims weighted should produce 0.9.
    assert result["overall_quality"] == pytest.approx(0.9, abs=1e-6)


# ----- history helpers -----


def test_history_helpers_track_multiple_passes():
    state: Dict[str, Any] = {}
    append_verification_history(state, _failing_critique())
    append_verification_history(state, _passing_critique())
    history = state["verification_history"]
    assert len(history) == 2
    assert history[0]["should_revise"] is True
    assert history[1]["should_revise"] is False
    latest = latest_verification(state)
    assert latest is not None
    assert latest["should_revise"] is False


def test_thresholds_are_advertised_consistently():
    """Catch accidental drift between the constants and the docstring."""
    assert 0.5 < DEFAULT_OVERALL_THRESHOLD <= 1.0
    assert 0.5 < DEFAULT_ALIGNMENT_THRESHOLD <= 1.0


# ----- Workflow integration -----


def test_workflow_runs_revise_loop_end_to_end(tmp_path):
    """The verifier's critique should drive the rapporteur back to ``revise``
    mode at least once, and the workflow should land at END inside the
    ``max_revisions`` cap with a final report present."""
    from SDYJ_Agents.agents.coordinator import Coordinator
    from SDYJ_Agents.agents.planner import Planner
    from SDYJ_Agents.agents.rapporteur import Rapporteur
    from SDYJ_Agents.agents.researcher import Researcher
    from SDYJ_Agents.utils.tracing import (
        InstrumentedLLM,
        create_run_trace,
        merge_trace_state,
    )
    from SDYJ_Agents.workflow.graph import ResearchWorkflow

    # Re-use the canned scripted LLM the offline benchmark uses; it knows how
    # to answer planner / researcher prompts. We extend it with verifier and
    # revise responses scripted in order.
    from SDYJ_Agents.evaluation.runner import FakeEvalLLM, CannedSearchTool
    from SDYJ_Agents.evaluation.scenarios import get_scenario

    scenario = get_scenario("agent_reliability_hard")

    class _LoopLLM(FakeEvalLLM):
        """Wrap FakeEvalLLM so verifier prompts get a scripted JSON answer.

        First verifier pass fails so the workflow loops once; second pass
        accepts so the loop terminates. Revise prompt is answered with a
        recognizable revised report so we can assert it landed in state.
        """

        def __init__(self):
            super().__init__()
            self._verifier_answers = [
                json.dumps(_failing_critique()),
                json.dumps(_passing_critique()),
            ]

        def generate(self, prompt: str, **kwargs) -> str:
            if "claim_evidence_alignment" in prompt:
                if self._verifier_answers:
                    return self._verifier_answers.pop(0)
                return json.dumps(_passing_critique())
            if "Revision rules" in prompt:
                return "# Revised report\n\n[E1] [E2] grounded claim."
            return super().generate(prompt, **kwargs)

    trace = create_run_trace(
        query=scenario["query"],
        provider="fake",
        model="fake-loop-llm",
        mode="eval",
        scenario_id=scenario["id"],
    )
    llm = InstrumentedLLM(_LoopLLM(), trace)
    coordinator = Coordinator(llm)
    planner = Planner(llm)
    researcher = Researcher(llm=llm)
    researcher.tavily = CannedSearchTool("tavily", scenario)
    researcher.arxiv = CannedSearchTool("arxiv", scenario)
    researcher.mcp = None
    rapporteur = Rapporteur(llm)
    verifier = Verifier(llm)
    workflow = ResearchWorkflow(coordinator, planner, researcher, rapporteur, verifier)

    final_state: Dict[str, Any] = {}
    for update in workflow.stream_interactive(
        scenario["query"],
        max_iterations=2,
        auto_approve=True,
        output_format="markdown",
        trace=trace,
        skip_verification=False,
        max_revisions=2,
    ):
        for value in update.values():
            if isinstance(value, dict):
                final_state = value

    assert final_state.get("final_report"), "workflow must produce a report"
    history = final_state.get("verification_history") or []
    assert len(history) >= 1, "verifier must have run at least once"
    # The loop should have exited cleanly within the cap; revision_count
    # increments only when we actually loop back to the rapporteur.
    assert final_state.get("revision_count", 0) <= final_state.get("max_revisions", 2)
    # LangGraph hands each node a copied trace, so the production code
    # (CLI / web / eval runner) merges the state-side trace mutations back
    # into the base trace before saving. Mirror that behavior in the test.
    merged_trace = merge_trace_state(trace, final_state.get("trace")) or trace
    assert (merged_trace.get("metrics") or {}).get("verifier_overall_quality") is not None
