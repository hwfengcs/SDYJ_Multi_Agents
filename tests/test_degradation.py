"""Tests for graceful degradation: node guards, section placeholders, fallbacks."""

import json

from SDYJ_Agents.agents.coordinator import Coordinator
from SDYJ_Agents.agents.planner import Planner
from SDYJ_Agents.agents.rapporteur import Rapporteur, SECTION_FAILURE_PLACEHOLDER
from SDYJ_Agents.agents.researcher import Researcher
from SDYJ_Agents.workflow.graph import ResearchWorkflow


class ScriptedLLM:
    """Dispatches on the [PROMPT_ID: ...] marker; a script value that is an
    Exception instance is raised instead of returned."""

    def __init__(self, script):
        self.script = script

    def generate(self, prompt: str, **kwargs) -> str:
        for marker, action in self.script.items():
            if marker in prompt:
                if isinstance(action, Exception):
                    raise action
                return action
        raise AssertionError(f"Unexpected LLM call: {prompt[:120]}")

    def stream_generate(self, prompt: str, **kwargs):
        yield self.generate(prompt, **kwargs)


class FakeSearchTool:
    def search(self, query: str, **kwargs):
        return {
            "query": query,
            "source": "tavily",
            "results": [
                {
                    "title": "Doc",
                    "url": "https://example.com/doc",
                    "snippet": "A relevant snippet.",
                    "relevance_score": 0.9,
                }
            ],
            "total_results": 1,
        }


def _plan_json(task_count=1):
    return json.dumps(
        {
            "research_goal": "goal",
            "sub_tasks": [
                {
                    "task_id": index + 1,
                    "description": f"task {index + 1}",
                    "search_queries": [f"query {index + 1}"],
                    "sources": ["tavily"],
                    "priority": 1,
                }
                for index in range(task_count)
            ],
            "completion_criteria": "done",
            "estimated_iterations": task_count,
        }
    )


THEMES_JSON = json.dumps(
    {"themes": [{"name": "主题", "key_points": ["发现一 [E1]"]}]},
    ensure_ascii=False,
)


def test_planner_llm_failure_degrades_to_fallback_plan():
    llm = ScriptedLLM({"[PROMPT_ID: planner_create_plan]": RuntimeError("llm down")})
    planner = Planner(llm)
    state = {"query": "q", "user_feedback": None, "degraded_events": []}

    state = planner.create_research_plan(state)

    assert state["research_plan"]["sub_tasks"]
    assert state["degraded_events"][0]["where"] == "planner_create_plan"


def test_sufficiency_check_failure_defaults_to_report():
    llm = ScriptedLLM({"[PROMPT_ID: planner_evaluate_context]": RuntimeError("llm down")})
    planner = Planner(llm)
    state = {
        "query": "q",
        "research_plan": {"research_goal": "g", "completion_criteria": "c"},
        "research_results": [{"source": "tavily", "results": [{"title": "t"}]}],
        "iteration_count": 1,
        "max_iterations": 3,
        "degraded_events": [],
    }

    assert planner.evaluate_context_sufficiency(state) is True
    assert state["degraded_events"][0]["where"] == "planner_evaluate_context"


def test_rapporteur_section_failure_yields_placeholder_not_crash():
    llm = ScriptedLLM(
        {
            "[PROMPT_ID: rapporteur_summarize]": "摘要内容 [E1]",
            "[PROMPT_ID: rapporteur_organize_info]": THEMES_JSON,
            "[PROMPT_ID: rapporteur_synthesized_analysis]": RuntimeError("llm down"),
            "[PROMPT_ID: rapporteur_conclusion]": "结论内容",
        }
    )
    rapporteur = Rapporteur(llm)
    state = {
        "query": "q",
        "research_plan": {"research_goal": "g"},
        "research_results": [
            {
                "source": "tavily",
                "query": "q",
                "results": [
                    {"title": "Doc", "url": "https://example.com/doc", "snippet": "s"}
                ],
            }
        ],
        "evidence_items": [],
        "output_format": "markdown",
        "degraded_events": [],
    }

    state = rapporteur.generate_report(state)

    report = state["final_report"]
    assert "## 深度分析" in report
    assert SECTION_FAILURE_PLACEHOLDER in report
    assert "结论内容" in report
    assert state["report_metrics"]["degraded"] is True
    assert state["report_metrics"]["degraded_event_count"] == 1
    assert state["degraded_events"][0]["where"] == "rapporteur_analysis"


def test_failed_research_task_degrades_and_run_still_completes():
    llm = ScriptedLLM(
        {
            "[PROMPT_ID: coordinator_classify_query]": "RESEARCH",
            "[PROMPT_ID: planner_create_plan]": _plan_json(task_count=1),
            "[PROMPT_ID: rapporteur_summarize]": "摘要",
            "[PROMPT_ID: rapporteur_organize_info]": THEMES_JSON,
        }
    )
    researcher = Researcher(llm)
    researcher.tavily = FakeSearchTool()

    def exploding_task(state, task):
        raise RuntimeError("task explode")

    researcher.execute_task = exploding_task

    workflow = ResearchWorkflow(Coordinator(llm), Planner(llm), researcher, Rapporteur(llm))
    final_state = {}
    for update in workflow.stream_interactive(
        "研究一个问题",
        max_iterations=1,
        auto_approve=True,
        output_format="json",
    ):
        for value in update.values():
            if isinstance(value, dict):
                final_state = value

    assert final_state["final_report"]
    wheres = [event["where"] for event in final_state["degraded_events"]]
    assert "researcher_task" in wheres
    task_status = final_state["research_plan"]["sub_tasks"][0]["status"]
    assert task_status == "failed"
    assert final_state["report_metrics"]["degraded"] is True
