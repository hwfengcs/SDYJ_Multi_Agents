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


def _field_coverage(items: list[Dict[str, Any]], required_fields: list[str]) -> float:
    if not required_fields:
        return 1.0
    if not items:
        return 0.0
    hits = 0
    total = len(items) * len(required_fields)
    for item in items:
        for field in required_fields:
            if field in item:
                hits += 1
    return hits / total


def trace_completeness(trace: Dict[str, Any] | None, scenario: Dict[str, Any]) -> float:
    """Score whether a trace is useful for debugging and replay."""
    if not trace:
        return 0.0

    expected = scenario.get("expected_trace") or {}
    required_top = expected.get("required_top_level_fields") or [
        "run_id",
        "mode",
        "created_at",
        "completed_at",
        "query",
        "provider",
        "model",
        "nodes",
        "llm_calls",
        "tool_calls",
        "report",
        "metrics",
        "errors",
    ]
    required_nodes = expected.get("required_nodes") or ["planner", "researcher", "rapporteur"]
    required_tool_fields = expected.get("required_tool_fields") or [
        "source",
        "query",
        "latency_ms",
        "result_count",
        "error",
    ]
    required_llm_fields = expected.get("required_llm_fields") or [
        "call_id",
        "model",
        "latency_ms",
        "prompt_chars",
        "response_chars",
        "error",
    ]

    top_score = sum(1 for field in required_top if field in trace) / len(required_top)
    observed_nodes = {item.get("node") for item in trace.get("nodes", [])}
    node_score = (
        sum(1 for node in required_nodes if node in observed_nodes) / len(required_nodes)
        if required_nodes else 1.0
    )
    tool_score = _field_coverage(trace.get("tool_calls", []), required_tool_fields)
    llm_score = _field_coverage(trace.get("llm_calls", []), required_llm_fields)
    event_score = 1.0 if trace.get("events") else 0.0
    replay_cache = trace.get("replay_cache") or {}
    replay_score = 0.0
    if replay_cache.get("llm_calls"):
        replay_score += 0.5
    if not trace.get("tool_calls") or replay_cache.get("tool_calls"):
        replay_score += 0.5

    return round(
        0.2 * top_score
        + 0.2 * node_score
        + 0.15 * tool_score
        + 0.15 * llm_score
        + 0.15 * event_score
        + 0.15 * replay_score,
        4,
    )


def apply_thresholds(
    metrics: Dict[str, Any],
    scenario: Dict[str, Any],
    overrides: Dict[str, float] | None = None,
) -> Dict[str, Any]:
    """Apply scenario and caller-supplied metric thresholds."""
    thresholds = dict(scenario.get("thresholds") or {})
    if overrides:
        thresholds.update(overrides)

    failed = []
    for metric, threshold in thresholds.items():
        actual = metrics.get(metric)
        if actual is None or float(actual) < float(threshold):
            failed.append(
                {
                    "metric": metric,
                    "actual": actual,
                    "threshold": threshold,
                }
            )
    return {
        "passed": not failed,
        "thresholds": thresholds,
        "failed_thresholds": failed,
    }


def evaluate_state(
    state: Dict[str, Any],
    scenario: Dict[str, Any],
    trace: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
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
        "trace_completeness": trace_completeness(trace, scenario),
        **evidence_metrics,
    }

    # Surface v0.6 verifier-loop metrics whenever they are present so they
    # show up alongside the existing scenario metrics in benchmark summaries
    # and CI gates. They are pulled from trace.metrics (the source of truth)
    # rather than recomputed.
    trace_metrics = (trace or {}).get("metrics") or {}
    for key in (
        "verifier_overall_quality",
        "verifier_should_revise",
        "verifier_weakest_dimension",
        "verifier_revision_count",
    ):
        if key in trace_metrics:
            metrics[key] = trace_metrics[key]
    # Also expose the rapporteur revision count when set by the workflow.
    if "revision_count" in trace_metrics:
        metrics["revision_count"] = trace_metrics["revision_count"]

    # A compact score for dashboards. Keep the raw metrics visible for real review.
    metrics["overall_score"] = round(
        0.2 * metrics["plan_coverage"]
        + 0.16 * metrics["section_completeness"]
        + 0.18 * min(metrics["citation_id_coverage"], 1.0)
        + 0.14 * min(metrics["citation_density_per_1k_chars"] / 2.0, 1.0)
        + 0.12 * metrics["tool_success_rate"]
        + 0.1 * metrics["grounded_key_finding_rate"]
        + 0.1 * metrics["trace_completeness"],
        4,
    )
    return metrics
