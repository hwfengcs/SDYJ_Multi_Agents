"""Scenario evaluation metrics."""

from __future__ import annotations

import json
import re
from typing import Any, Dict

from ..utils.evidence import calculate_evidence_metrics


def _contains_term(text: str, term: str) -> bool:
    return term.lower() in text.lower()


def _section_completeness(report: str, expected_sections: list[str]) -> float:
    if not expected_sections:
        return 1.0
    hits = 0
    for section in expected_sections:
        if f"## {section}" in report or section in report:
            hits += 1
    return hits / len(expected_sections)


def _plan_coverage(plan: Dict[str, Any], report: str, required_terms: list[str]) -> float:
    if not required_terms:
        return 1.0
    plan_text = json.dumps(plan or {}, ensure_ascii=False)
    combined = f"{plan_text}\n{report}"
    hits = sum(1 for term in required_terms if _contains_term(combined, term))
    return hits / len(required_terms)


def _citation_id_coverage(report: str, evidence_count: int) -> float:
    if evidence_count <= 0:
        return 0.0
    cited = set(re.findall(r"\[E\d+\]", report or ""))
    return len(cited) / evidence_count


def evaluate_state(state: Dict[str, Any], scenario: Dict[str, Any]) -> Dict[str, Any]:
    """Evaluate a completed workflow state against a scenario."""
    report = state.get("final_report") or ""
    plan = state.get("research_plan") or {}
    research_results = state.get("research_results") or []
    evidence_items = state.get("evidence_items") or []

    evidence_metrics = calculate_evidence_metrics(research_results, evidence_items, report)
    metrics = {
        "scenario_id": scenario["id"],
        "plan_coverage": _plan_coverage(plan, report, scenario.get("required_terms", [])),
        "section_completeness": _section_completeness(
            report,
            scenario.get("expected_sections", []),
        ),
        "citation_id_coverage": _citation_id_coverage(report, len(evidence_items)),
        "iteration_count": state.get("iteration_count", 0),
        **evidence_metrics,
    }

    # A compact score for dashboards. Keep the raw metrics visible for real review.
    metrics["overall_score"] = round(
        0.25 * metrics["plan_coverage"]
        + 0.2 * metrics["section_completeness"]
        + 0.2 * min(metrics["citation_id_coverage"], 1.0)
        + 0.15 * min(metrics["citation_density_per_1k_chars"] / 2.0, 1.0)
        + 0.1 * metrics["tool_success_rate"]
        + 0.1 * metrics["grounded_key_finding_rate"],
        4,
    )
    return metrics
