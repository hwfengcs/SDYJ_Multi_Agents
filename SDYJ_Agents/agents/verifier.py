"""
Verifier Agent

The Verifier is the 5th agent in the SDYJ workflow. It re-reads the generated
research report against the collected evidence and grades it on four
dimensions: claim_evidence_alignment, citation_completeness,
factual_consistency, and coverage. When the overall quality dips below the
configured threshold, the workflow loops back to the Rapporteur with concrete
revision hints, up to ``max_revisions`` times.

This module is intentionally narrow: ``Verifier.verify`` produces a
JSON-shaped dict, and the routing decision lives in ``workflow/nodes.py``
where it can read the rest of the state. Keeping the agent dumb makes it
easy to swap implementations (e.g. a small local model verifier) without
changing the graph wiring.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

from ..llm.base import BaseLLM
from ..prompts.loader import PromptLoader
from ..utils.evidence import build_evidence_from_results, format_evidence_for_prompt
from ..workflow.state import ResearchState


# Default thresholds. Mirror these in CLI flags rather than hard-coding alt
# values across the codebase. Lowering the bar here means future trace data
# is no longer comparable to old runs, so adjust deliberately.
DEFAULT_OVERALL_THRESHOLD = 0.75
DEFAULT_ALIGNMENT_THRESHOLD = 0.65
DEFAULT_MAX_REVISIONS = 2

DIMENSIONS = (
    "claim_evidence_alignment",
    "citation_completeness",
    "factual_consistency",
    "coverage",
)

# Weights applied to compute ``overall_quality`` when the LLM's answer is
# malformed and we have to fall back to local computation. Must sum to 1.
DIMENSION_WEIGHTS = {
    "claim_evidence_alignment": 0.35,
    "citation_completeness": 0.20,
    "factual_consistency": 0.20,
    "coverage": 0.25,
}


class Verifier:
    """LLM-driven critic for the final report."""

    def __init__(
        self,
        llm: BaseLLM,
        overall_threshold: float = DEFAULT_OVERALL_THRESHOLD,
        alignment_threshold: float = DEFAULT_ALIGNMENT_THRESHOLD,
    ):
        """
        Args:
            llm: Language model used to score the report.
            overall_threshold: Below this, ``should_revise`` is forced True.
            alignment_threshold: Below this on claim_evidence_alignment,
                ``should_revise`` is forced True even if overall passes.
        """
        self.llm = llm
        self.prompt_loader = PromptLoader()
        self.overall_threshold = overall_threshold
        self.alignment_threshold = alignment_threshold

    def verify(self, state: ResearchState) -> Dict[str, Any]:
        """Score the report stored in ``state['final_report']`` against evidence.

        Always returns a dict shaped like the prompt schema, even when the
        LLM call or its parser fails — callers (the workflow) should never
        have to defensive-code this.
        """
        report = state.get("final_report") or ""
        if not report.strip():
            return _default_failure_result(
                summary="Empty report passed to verifier; nothing to grade."
            )

        plan = state.get("research_plan") or {}
        evidence_items = state.get("evidence_items") or build_evidence_from_results(
            state.get("research_results") or []
        )
        evidence_text = (
            format_evidence_for_prompt(evidence_items, limit=30)
            if evidence_items
            else "(no evidence collected)"
        )

        prompt = self.prompt_loader.load(
            "verifier_critique",
            query=state.get("query", ""),
            research_goal=plan.get("research_goal", state.get("query", "")),
            plan_subtasks_summary=_summarize_subtasks(plan.get("sub_tasks") or []),
            evidence=evidence_text,
            report=report,
        )

        try:
            response = self.llm.generate(prompt, temperature=0.1, max_tokens=1500)
        except Exception as exc:
            return _default_failure_result(
                summary=f"Verifier LLM call failed: {exc}",
            )

        parsed = _parse_verifier_response(response)
        return self._enforce_thresholds(parsed)

    def _enforce_thresholds(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """Enforce SDYJ's own pass/fail rules even if the LLM disagrees.

        The model can be too lenient — especially when graded against its
        own report. We treat the LLM's `should_revise` as advisory and apply
        the configured thresholds here.
        """
        scores = result.get("scores") or {}
        normalized_scores = {dim: _coerce_score(scores.get(dim)) for dim in DIMENSIONS}
        result["scores"] = normalized_scores

        overall = result.get("overall_quality")
        if not isinstance(overall, (int, float)) or overall < 0 or overall > 1:
            overall = sum(
                normalized_scores[dim] * DIMENSION_WEIGHTS[dim] for dim in DIMENSIONS
            )
            result["overall_quality"] = round(overall, 4)
        else:
            result["overall_quality"] = round(float(overall), 4)

        forced_revise = (
            result["overall_quality"] < self.overall_threshold
            or normalized_scores["claim_evidence_alignment"] < self.alignment_threshold
        )
        if forced_revise:
            result["should_revise"] = True

        if "weakest_dimension" not in result or result["weakest_dimension"] not in DIMENSIONS:
            result["weakest_dimension"] = min(DIMENSIONS, key=lambda d: normalized_scores[d])

        hints = result.get("revision_hints") or []
        if not isinstance(hints, list):
            hints = [str(hints)]
        result["revision_hints"] = [str(h).strip() for h in hints if str(h).strip()]

        if "summary" not in result or not isinstance(result["summary"], str):
            result["summary"] = ""

        return result


def _coerce_score(value: Any) -> float:
    """Map raw LLM output to a [0, 1] float, defaulting low when unparseable.

    Defaulting low is conservative — an unparseable score should bias toward
    revision, not toward shipping a bad report.
    """
    try:
        score = float(value)
    except (TypeError, ValueError):
        return 0.0
    if score != score:  # NaN
        return 0.0
    return max(0.0, min(1.0, score))


def _summarize_subtasks(sub_tasks: List[Dict[str, Any]]) -> str:
    """Render the plan's sub-tasks as a compact bullet list for the prompt."""
    if not sub_tasks:
        return "(plan had no sub-tasks)"
    lines = []
    for task in sub_tasks:
        task_id = task.get("task_id", "?")
        description = task.get("description", "")
        status = task.get("status", "pending")
        lines.append(f"- [{task_id}] ({status}) {description}")
    return "\n".join(lines)


def _parse_verifier_response(response: str) -> Dict[str, Any]:
    """Extract the JSON object the prompt asked for.

    The prompt asks for JSON only, but real models prefix things like
    ```json fences. We look for the first balanced ``{...}`` block and
    fall back to a structured failure if anything goes wrong.
    """
    if not response:
        return _default_failure_result(summary="Verifier returned empty response.")
    text = response.strip()
    # Strip Markdown code fences if present.
    if text.startswith("```"):
        text = text.strip("`")
        if text.lower().startswith("json"):
            text = text[4:]
    start = text.find("{")
    end = text.rfind("}") + 1
    if start == -1 or end <= start:
        return _default_failure_result(
            summary="Verifier response was not JSON; defaulting to revise."
        )
    try:
        return json.loads(text[start:end])
    except json.JSONDecodeError as exc:
        return _default_failure_result(
            summary=f"Verifier JSON parse error: {exc}",
        )


def _default_failure_result(summary: str) -> Dict[str, Any]:
    """Return a conservative result when the verifier itself failed.

    Conservative = recommend revision so a broken verifier never silently
    rubber-stamps a bad report.
    """
    return {
        "scores": {dim: 0.0 for dim in DIMENSIONS},
        "overall_quality": 0.0,
        "should_revise": True,
        "weakest_dimension": "claim_evidence_alignment",
        "revision_hints": [
            "Verifier failed to produce a structured critique. Re-run the report "
            "with stricter evidence grounding and add explicit [Ek] citations on "
            "every major claim."
        ],
        "summary": summary,
        "verifier_failed": True,
    }


def append_verification_history(
    state: Dict[str, Any],
    verification_result: Dict[str, Any],
) -> None:
    """Persist one verifier pass into ``state['verification_history']``.

    Kept as a free function so it can be called from both the workflow node
    and from tests without instantiating an agent.
    """
    history: List[Dict[str, Any]] = state.setdefault("verification_history", [])
    history.append(verification_result)


def latest_verification(state: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Return the most recent verifier result on the state, if any."""
    history = state.get("verification_history") or []
    if history:
        return history[-1]
    return state.get("verification_result")
