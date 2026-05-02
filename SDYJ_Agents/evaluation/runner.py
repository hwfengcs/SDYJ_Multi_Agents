"""Evaluation runner for canned and live-LLM scenarios."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from ..agents.coordinator import Coordinator
from ..agents.planner import Planner
from ..agents.rapporteur import Rapporteur
from ..agents.researcher import Researcher
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
from .metrics import evaluate_state
from .scenarios import HARD_SCENARIOS, get_scenario


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
        if "Create a structured research plan" in prompt:
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
) -> Dict[str, Any]:
    trace = create_run_trace(
        query=scenario["query"],
        provider=provider if live else "fake",
        model=model if live else "fake-eval-llm",
        mode="eval",
        scenario_id=scenario["id"],
    )
    llm = _create_llm(live=live, provider=provider, model=model, trace=trace)

    coordinator = Coordinator(llm)
    planner = Planner(llm)
    researcher = Researcher(llm)
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
        )

    rapporteur = Rapporteur(llm)
    workflow = ResearchWorkflow(coordinator, planner, researcher, rapporteur)

    final_state: Dict[str, Any] = {}
    for update in workflow.stream_interactive(
        scenario["query"],
        max_iterations=max_iterations,
        auto_approve=True,
        output_format=output_format,
        trace=trace,
    ):
        for value in update.values():
            if isinstance(value, dict):
                final_state = value

    metrics = evaluate_state(final_state, scenario)
    trace = merge_trace_state(trace, final_state.get("trace"))
    trace.setdefault("metrics", {}).update(metrics)
    finalize_trace(trace, metrics)
    trace_path = save_trace(trace, output_dir)

    scenario_dir = Path(output_dir) / "eval_reports"
    scenario_dir.mkdir(parents=True, exist_ok=True)
    report_extension = "html" if output_format == "html" else "md"
    report_path = scenario_dir / f"{scenario['id']}_{trace['run_id']}.{report_extension}"
    if final_state.get("final_report"):
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(final_state["final_report"])

    return {
        "scenario_id": scenario["id"],
        "title": scenario["title"],
        "query": scenario["query"],
        "metrics": metrics,
        "trace_path": str(trace_path) if trace_path else None,
        "report_path": str(report_path) if final_state.get("final_report") else None,
        "run_id": trace["run_id"],
    }


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
) -> Dict[str, Any]:
    """Run the evaluation suite and persist a JSON summary."""
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
        )
        for scenario in selected
    ]

    summary = {
        "created_at": datetime.now().isoformat(),
        "mode": "live" if live else "offline",
        "provider": provider if live else "fake",
        "model": model if live else "fake-eval-llm",
        "live_search": live_search,
        "scenario_count": len(results),
        "average_score": (
            sum(item["metrics"]["overall_score"] for item in results) / len(results)
            if results else 0.0
        ),
        "results": results,
    }

    output = Path(output_dir) / "eval_reports"
    output.mkdir(parents=True, exist_ok=True)
    summary_path = output / f"eval_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    summary["summary_path"] = str(summary_path)
    return summary
