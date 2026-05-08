"""Evaluation runner for canned and live-LLM scenarios."""

from __future__ import annotations

import json
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from ..agents.coordinator import Coordinator
from ..agents.planner import Planner
from ..agents.rapporteur import Rapporteur
from ..agents.researcher import Researcher
from ..agents.verifier import DEFAULT_MAX_REVISIONS, Verifier
from ..llm.base import BaseLLM
from ..llm.factory import LLMFactory
from ..utils.config import load_config_from_env
from ..utils.tracing import (
    InstrumentedLLM,
    create_run_trace,
    finalize_trace,
    merge_trace_state,
    save_trace,
)
from ..workflow.graph import ResearchWorkflow
from .metrics import apply_thresholds, evaluate_state
from .scenarios import HARD_SCENARIOS, get_scenario


REGRESSION_METRIC_KEYS = [
    "overall_score",
    "plan_coverage",
    "section_completeness",
    "citation_id_coverage",
    "citation_density_per_1k_chars",
    "tool_success_rate",
    "grounded_key_finding_rate",
    "trace_completeness",
]


def _metric_root_cause(metric: str) -> str:
    if metric == "plan_coverage":
        return "planner_gap"
    if metric == "section_completeness":
        return "report_structure_gap"
    if metric in {
        "citation_id_coverage",
        "citation_density_per_1k_chars",
        "grounded_key_finding_rate",
    }:
        return "citation_gap"
    if metric == "tool_success_rate":
        return "tool_error"
    if metric == "trace_completeness":
        return "trace_gap"
    if metric.startswith("verifier_") or metric == "revision_count":
        return "verifier_gap"
    return "aggregate_quality_gap"


def _scenario_failure_analysis(result: Dict[str, Any]) -> Dict[str, Any]:
    failed_metrics = []
    root_cause_counts: Counter[str] = Counter()
    for failure in result.get("failed_thresholds", []):
        metric = str(failure.get("metric"))
        root_cause = _metric_root_cause(metric)
        root_cause_counts[root_cause] += 1
        actual = failure.get("actual")
        threshold = failure.get("threshold")
        delta = (
            round(float(threshold) - float(actual), 4)
            if isinstance(actual, (int, float)) and isinstance(threshold, (int, float))
            else None
        )
        failed_metrics.append(
            {
                "metric": metric,
                "actual": actual,
                "threshold": threshold,
                "threshold_delta": delta,
                "root_cause": root_cause,
            }
        )
    return {
        "failed_metric_count": len(failed_metrics),
        "root_cause_counts": dict(root_cause_counts),
        "failed_metrics": failed_metrics,
    }


def _aggregate_failure_analysis(results: list[Dict[str, Any]]) -> Dict[str, Any]:
    root_cause_counts: Counter[str] = Counter()
    scenarios = []
    failed_metric_count = 0
    for result in results:
        analysis = result.get("failure_analysis") or {}
        failed_metrics = analysis.get("failed_metrics") or []
        if not failed_metrics:
            continue
        failed_metric_count += len(failed_metrics)
        root_cause_counts.update(analysis.get("root_cause_counts") or {})
        scenarios.append(
            {
                "scenario_id": result.get("scenario_id"),
                "passed": result.get("passed"),
                "failed_metrics": failed_metrics,
            }
        )
    return {
        "failed_scenario_count": len(scenarios),
        "failed_metric_count": failed_metric_count,
        "root_cause_counts": dict(root_cause_counts),
        "scenarios": scenarios,
    }


class CannedSearchTool:
    """A deterministic retrieval tool for reproducible evaluation."""

    def __init__(self, source: str, scenario: Dict[str, Any]):
        self.source = source
        self.scenario = scenario
        self.call_count = 0

    def search(self, query: str, **kwargs) -> Dict[str, Any]:
        self.call_count += 1
        for forced in self.scenario.get("forced_errors", {}).get(self.source, []):
            if forced.get("query_contains", "").lower() in query.lower():
                return {
                    "query": query,
                    "source": self.source,
                    "results": [],
                    "timestamp": datetime.now().isoformat(),
                    "error": forced.get("error", "forced retrieval error"),
                }

        results = self.scenario.get("canned_results", {}).get(self.source, [])
        return {
            "query": query,
            "source": self.source,
            "results": [dict(item) for item in results],
            "timestamp": datetime.now().isoformat(),
            "total_results": len(results),
        }


class FakeEvalLLM(BaseLLM):
    """Deterministic LLM used for offline eval smoke tests."""

    def __init__(self):
        super().__init__(api_key="fake", model="fake-eval-llm")
        self.last_usage = None

    def generate(self, prompt: str, **kwargs) -> str:
        if "请将查询分类为" in prompt:
            return "RESEARCH"
        if "Planner mid-flight" in prompt:
            return json.dumps(
                {
                    "research_goal": "Evaluate a complex agent system with traceable evidence",
                    "sub_tasks": [
                        {
                            "task_id": 1,
                            "description": "Collect evidence for agent evaluation metrics",
                            "search_queries": ["agent evaluation trace evidence citation latency cost"],
                            "sources": ["tavily", "arxiv"],
                            "priority": 1,
                            "status": "completed",
                        },
                        {
                            "task_id": 2,
                            "description": "Tighten analysis of failure recovery and human approval",
                            "search_queries": ["agent tool failure recovery human approval trace"],
                            "sources": ["tavily", "arxiv"],
                            "priority": 2,
                            "status": "pending",
                        },
                    ],
                    "completion_criteria": (
                        "Report must cover evidence, citation grounding, trace, latency, cost, "
                        "tool reliability, ablation, and human control."
                    ),
                    "estimated_iterations": 2,
                    "refinement_rationale": "The first task already surfaced the core metric vocabulary.",
                }
            )
        if (
            "Create a concrete, tool-aware research plan" in prompt
            or ("<output_schema>" in prompt and "research_goal" in prompt)
        ):
            return json.dumps(
                {
                    "research_goal": "Evaluate a complex agent system with traceable evidence",
                    "sub_tasks": [
                        {
                            "task_id": 1,
                            "description": "Collect evidence for agent evaluation metrics",
                            "search_queries": ["agent evaluation trace evidence citation latency cost"],
                            "sources": ["tavily", "arxiv"],
                            "priority": 1,
                        },
                        {
                            "task_id": 2,
                            "description": "Analyze tool failure, retry, fallback, and human approval",
                            "search_queries": ["tool timeout retry duplicate dedup fallback human review"],
                            "sources": ["tavily", "arxiv"],
                            "priority": 2,
                        },
                    ],
                    "completion_criteria": (
                        "Report must cover evidence, citation grounding, trace, latency, cost, "
                        "tool reliability, ablation, and human control."
                    ),
                    "estimated_iterations": 2,
                }
            )
        if "Evaluate whether the gathered research context is sufficient" in prompt:
            return "YES"
        if "claim_evidence_alignment" in prompt and "Output schema" in prompt:
            # Verifier critique prompt. The fake LLM has no judgment, so we
            # return a passing critique to keep the offline benchmark
            # deterministic. Tests that exercise the failing/revising path
            # use a dedicated scripted LLM instead (see tests/test_verifier.py).
            return json.dumps(
                {
                    "scores": {
                        "claim_evidence_alignment": 0.92,
                        "citation_completeness": 0.9,
                        "factual_consistency": 0.91,
                        "coverage": 0.9,
                    },
                    "overall_quality": 0.91,
                    "should_revise": False,
                    "weakest_dimension": "citation_completeness",
                    "revision_hints": [],
                    "summary": "Report is well-grounded; no revision needed.",
                }
            )
        if "Revision rules" in prompt:
            # Rapporteur revise prompt. If the offline benchmark ever lands
            # here, we return a placeholder so the workflow finishes; in
            # practice the verifier branch above should keep us out of revise.
            return "# Revised report\n\nKey claims are now grounded. [E1] [E2]"
        if "必须严格按照以下JSON格式输出" in prompt:
            return json.dumps(
                {
                    "themes": [
                        {
                            "name": "评测指标体系",
                            "key_points": [
                                "Agent 评测应覆盖 evidence、citation、trace、latency、cost 与 tool success，而不是只看最终答案。",
                                "Ablation study 可以比较人工审批、去重、重试和证据引用对整体可靠性的贡献。",
                            ],
                        },
                        {
                            "name": "工具可靠性治理",
                            "key_points": [
                                "工具 timeout、duplicate URL、空结果和低质量来源都应进入 trace，并由 fallback 策略处理。",
                                "Human review 适合放在高成本检索或不可逆工具调用之前，用来降低错误计划的执行成本。",
                            ],
                        },
                    ]
                }
            )
        if "执行摘要" in prompt:
            return (
                "本评测方案将研究型 Agent 拆解为计划、检索、证据归一化、报告合成和人工审批五个环节。"
                "核心指标包括 evidence coverage、citation density、tool success rate、latency、cost、trace completeness "
                "和 ablation gain。"
            )
        if "深度整合分析" in prompt or "深度分析框架" in prompt:
            return (
                "### 评测设计\n"
                "应以 trace 为主线，把每个 LLM 调用、工具调用、错误和证据项串起来。"
                "Ablation study 分别关闭人工审批、URL 去重、失败恢复和引用约束，比较成功率、成本与延迟变化。\n\n"
                "### 上线门槛\n"
                "建议设置 citation coverage、tool success rate、latency SLO 和人工复核通过率阈值。"
            )
        if "结论框架" in prompt:
            return (
                "Agent 上线前需要同时证明答案质量、证据可靠性和工程可观测性。"
                "建议以 trace 驱动评测，用 ablation 验证关键机制，并把人工审批作为高风险任务的控制阀。"
            )
        return "YES"

    def stream_generate(self, prompt: str, **kwargs):
        yield self.generate(prompt, **kwargs)


def _select_scenarios(scenario_ids: Optional[Iterable[str]], max_scenarios: Optional[int]) -> List[Dict[str, Any]]:
    if scenario_ids:
        selected = [get_scenario(scenario_id) for scenario_id in scenario_ids]
    else:
        selected = list(HARD_SCENARIOS)
    if max_scenarios:
        selected = selected[:max_scenarios]
    return selected


def _create_llm(live: bool, provider: str, model: Optional[str], trace: Dict[str, Any]) -> BaseLLM:
    if not live:
        return InstrumentedLLM(FakeEvalLLM(), trace)

    env_cfg = load_config_from_env()
    provider = provider or env_cfg.llm.provider
    model = model or env_cfg.llm.model
    llm = LLMFactory.create_llm(
        provider=provider,
        api_key=env_cfg.llm.api_key,
        model=model,
    )
    return InstrumentedLLM(llm, trace)


def _run_one_scenario(
    scenario: Dict[str, Any],
    live: bool,
    provider: str,
    model: Optional[str],
    live_search: bool,
    max_iterations: int,
    output_format: str,
    output_dir: str,
    threshold_overrides: Optional[Dict[str, float]] = None,
    skip_verification: bool = True,
    max_revisions: int = DEFAULT_MAX_REVISIONS,
    enable_reflection: bool = False,
    enable_plan_refinement: bool = False,
    enable_parallel_tool_execution: bool = False,
) -> Dict[str, Any]:
    trace = create_run_trace(
        query=scenario["query"],
        provider=provider if live else "fake",
        model=model if live else "fake-eval-llm",
        mode="eval",
        scenario_id=scenario["id"],
    )
    # Track which v0.6 features were active so deterministic replay can
    # reproduce the same workflow path.
    trace.setdefault("config", {}).update(
        {
            "enable_reflection": enable_reflection,
            "enable_plan_refinement": enable_plan_refinement,
            "enable_parallel_tool_execution": enable_parallel_tool_execution,
            "skip_verification": skip_verification,
            "max_revisions": max_revisions,
        }
    )
    llm = _create_llm(live=live, provider=provider, model=model, trace=trace)

    coordinator = Coordinator(llm)
    planner = Planner(llm, enable_plan_refinement=enable_plan_refinement)
    researcher = Researcher(
        llm,
        enable_reflection=enable_reflection,
        enable_parallel_tool_execution=enable_parallel_tool_execution,
    )
    if not live_search:
        researcher.tavily = CannedSearchTool("tavily", scenario)
        researcher.arxiv = CannedSearchTool("arxiv", scenario)
        researcher.mcp = None
    else:
        env_cfg = load_config_from_env()
        researcher = Researcher(
            llm=llm,
            tavily_api_key=env_cfg.search.tavily_api_key,
            mcp_server_url=env_cfg.search.mcp_server_url,
            mcp_api_key=env_cfg.search.mcp_api_key,
            mcp_transport=env_cfg.search.mcp_transport,
            mcp_tool_name=env_cfg.search.mcp_tool_name,
            mcp_config_path=env_cfg.search.mcp_config_path,
            mcp_server_name=env_cfg.search.mcp_server_name,
            mcp_command=env_cfg.search.mcp_command,
            mcp_args=env_cfg.search.mcp_args,
            mcp_env=env_cfg.search.mcp_env,
            mcp_query_arg=env_cfg.search.mcp_query_arg,
            mcp_tool_args=env_cfg.search.mcp_tool_args,
            enable_reflection=enable_reflection,
            enable_parallel_tool_execution=enable_parallel_tool_execution,
        )

    rapporteur = Rapporteur(llm)
    verifier = None if skip_verification else Verifier(llm)
    workflow = ResearchWorkflow(coordinator, planner, researcher, rapporteur, verifier)

    final_state: Dict[str, Any] = {}
    for update in workflow.stream_interactive(
        scenario["query"],
        max_iterations=max_iterations,
        auto_approve=True,
        output_format=output_format,
        trace=trace,
        skip_verification=skip_verification,
        max_revisions=max_revisions,
    ):
        for value in update.values():
            if isinstance(value, dict):
                final_state = value

    trace = merge_trace_state(trace, final_state.get("trace"))
    metrics = evaluate_state(final_state, scenario, trace=trace)
    threshold_result = apply_thresholds(metrics, scenario, threshold_overrides)
    trace.setdefault("metrics", {}).update(metrics)
    trace.setdefault("metrics", {})["passed"] = threshold_result["passed"]
    finalize_trace(trace, metrics)
    trace_path = save_trace(trace, output_dir, final_state=final_state)

    scenario_dir = Path(output_dir) / "eval_reports"
    scenario_dir.mkdir(parents=True, exist_ok=True)
    report_extension = "html" if output_format == "html" else "json" if output_format == "json" else "md"
    report_path = scenario_dir / f"{scenario['id']}_{trace['run_id']}.{report_extension}"
    if final_state.get("final_report"):
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(final_state["final_report"])

    result = {
        "scenario_id": scenario["id"],
        "title": scenario["title"],
        "query": scenario["query"],
        "metrics": metrics,
        "passed": threshold_result["passed"],
        "thresholds": threshold_result["thresholds"],
        "failed_thresholds": threshold_result["failed_thresholds"],
        "trace_path": str(trace_path) if trace_path else None,
        "report_path": str(report_path) if final_state.get("final_report") else None,
        "run_id": trace["run_id"],
    }
    result["failure_analysis"] = _scenario_failure_analysis(result)
    return result


def _stable_result_fingerprint(result: Dict[str, Any]) -> Dict[str, Any]:
    """Build a deterministic fingerprint for repeated offline benchmark runs."""
    metrics = result.get("metrics") or {}
    stable_metric_keys = [
        "scenario_id",
        "plan_coverage",
        "section_completeness",
        "citation_id_coverage",
        "evidence_count",
        "citation_count",
        "tool_success_rate",
        "grounded_key_finding_rate",
        "trace_completeness",
        "overall_score",
    ]
    return {
        "scenario_id": result.get("scenario_id"),
        "passed": result.get("passed"),
        "metrics": {key: metrics.get(key) for key in stable_metric_keys},
        "failed_thresholds": result.get("failed_thresholds", []),
    }


def _compare_summaries(current: Dict[str, Any], baseline: Dict[str, Any]) -> Dict[str, Any]:
    """Compare current benchmark summary against a saved baseline summary."""
    baseline_by_id = {item["scenario_id"]: item for item in baseline.get("results", [])}
    rows = []
    regressions = []
    for item in current.get("results", []):
        scenario_id = item["scenario_id"]
        old = baseline_by_id.get(scenario_id)
        if not old:
            rows.append({"scenario_id": scenario_id, "status": "new"})
            continue
        old_score = old.get("metrics", {}).get("overall_score", 0.0)
        new_score = item.get("metrics", {}).get("overall_score", 0.0)
        delta = round(new_score - old_score, 4)
        metric_regressions = []
        for metric in REGRESSION_METRIC_KEYS:
            old_metric = old.get("metrics", {}).get(metric)
            new_metric = item.get("metrics", {}).get(metric)
            if not isinstance(old_metric, (int, float)) or not isinstance(new_metric, (int, float)):
                continue
            metric_delta = round(new_metric - old_metric, 4)
            if metric_delta < -0.02:
                metric_regressions.append(
                    {
                        "metric": metric,
                        "baseline": old_metric,
                        "current": new_metric,
                        "delta": metric_delta,
                        "root_cause": _metric_root_cause(metric),
                    }
                )
        row = {
            "scenario_id": scenario_id,
            "baseline_score": old_score,
            "current_score": new_score,
            "delta": delta,
            "metric_regressions": metric_regressions,
            "regressed": delta < -0.02 or bool(metric_regressions),
        }
        rows.append(row)
        if row["regressed"]:
            regressions.append(row)
    return {
        "baseline_path": baseline.get("summary_path"),
        "rows": rows,
        "regressions": regressions,
        "metric_regression_count": sum(len(row.get("metric_regressions", [])) for row in regressions),
        "passed": not regressions,
    }


def compare_evaluation_summaries(current: Dict[str, Any], baseline: Dict[str, Any]) -> Dict[str, Any]:
    """Public wrapper for comparing benchmark summaries with metric-level regressions."""
    return _compare_summaries(current, baseline)


def run_evaluation(
    live: bool = False,
    provider: str = "deepseek",
    model: Optional[str] = None,
    scenario_ids: Optional[Iterable[str]] = None,
    max_scenarios: Optional[int] = None,
    live_search: bool = False,
    max_iterations: int = 3,
    output_format: str = "markdown",
    output_dir: str = "./outputs",
    fail_under: Optional[float] = None,
    threshold_overrides: Optional[Dict[str, float]] = None,
    compare_summary_path: Optional[str] = None,
    determinism_repeats: int = 1,
    enable_verification: bool = False,
    max_revisions: int = DEFAULT_MAX_REVISIONS,
    enable_reflection: bool = False,
    enable_plan_refinement: bool = False,
    enable_parallel_tool_execution: bool = False,
) -> Dict[str, Any]:
    """Run the evaluation suite and persist a JSON summary.

    v0.6 feature flags default to False so the v0.5 benchmark gates keep
    working unchanged. Flip them on to exercise the self-verifying,
    self-reflecting, plan-refining, and parallel retrieval paths. The flags
    are recorded in each run's trace.config so downstream comparisons are not
    confused.
    """
    skip_verification = not enable_verification
    selected = _select_scenarios(scenario_ids, max_scenarios)
    results = [
        _run_one_scenario(
            scenario=scenario,
            live=live,
            provider=provider,
            model=model,
            live_search=live_search,
            max_iterations=max_iterations,
            output_format=output_format,
            output_dir=output_dir,
            threshold_overrides=threshold_overrides,
            skip_verification=skip_verification,
            max_revisions=max_revisions,
            enable_reflection=enable_reflection,
            enable_plan_refinement=enable_plan_refinement,
            enable_parallel_tool_execution=enable_parallel_tool_execution,
        )
        for scenario in selected
    ]

    determinism = {
        "enabled": determinism_repeats > 1 and not live,
        "repeats": determinism_repeats,
        "passed": True,
        "results": [],
    }
    if determinism["enabled"]:
        for scenario in selected:
            repeated = [
                _stable_result_fingerprint(
                    _run_one_scenario(
                        scenario=scenario,
                        live=live,
                        provider=provider,
                        model=model,
                        live_search=live_search,
                        max_iterations=max_iterations,
                        output_format=output_format,
                        output_dir=output_dir,
                        threshold_overrides=threshold_overrides,
                        skip_verification=skip_verification,
                        max_revisions=max_revisions,
                        enable_reflection=enable_reflection,
                        enable_plan_refinement=enable_plan_refinement,
                        enable_parallel_tool_execution=enable_parallel_tool_execution,
                    )
                )
                for _ in range(determinism_repeats)
            ]
            passed = all(item == repeated[0] for item in repeated[1:])
            determinism["results"].append(
                {
                    "scenario_id": scenario["id"],
                    "passed": passed,
                    "fingerprints": repeated,
                }
            )
            determinism["passed"] = determinism["passed"] and passed

    failed_scenarios = [item for item in results if not item.get("passed")]
    average_score = (
        sum(item["metrics"]["overall_score"] for item in results) / len(results)
        if results else 0.0
    )
    passed = not failed_scenarios and determinism["passed"]
    if fail_under is not None and average_score < fail_under:
        passed = False

    summary = {
        "created_at": datetime.now().isoformat(),
        "mode": "live" if live else "offline",
        "provider": provider if live else "fake",
        "model": model if live else "fake-eval-llm",
        "live_search": live_search,
        "enable_verification": enable_verification,
        "enable_reflection": enable_reflection,
        "enable_plan_refinement": enable_plan_refinement,
        "enable_parallel_tool_execution": enable_parallel_tool_execution,
        "scenario_count": len(results),
        "average_score": average_score,
        "fail_under": fail_under,
        "passed": passed,
        "failed_scenarios": [
            {
                "scenario_id": item["scenario_id"],
                "failed_thresholds": item.get("failed_thresholds", []),
            }
            for item in failed_scenarios
        ],
        "determinism": determinism,
        "failure_analysis": _aggregate_failure_analysis(results),
        "results": results,
    }

    if compare_summary_path:
        with open(compare_summary_path, "r", encoding="utf-8") as f:
            baseline = json.load(f)
        comparison = _compare_summaries(summary, baseline)
        summary["comparison"] = comparison
        summary["passed"] = summary["passed"] and comparison["passed"]

    output = Path(output_dir) / "eval_reports"
    output.mkdir(parents=True, exist_ok=True)
    summary_path = output / f"eval_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    summary["summary_path"] = str(summary_path)
    return summary
