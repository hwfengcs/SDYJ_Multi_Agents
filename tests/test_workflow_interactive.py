import json

from SDYJ_Agents.agents.coordinator import Coordinator
from SDYJ_Agents.agents.planner import Planner
from SDYJ_Agents.agents.rapporteur import Rapporteur
from SDYJ_Agents.agents.researcher import Researcher
from SDYJ_Agents.workflow.graph import ResearchWorkflow


class QueueLLM:
    def __init__(self, responses):
        self.responses = list(responses)

    def generate(self, prompt: str, **kwargs) -> str:
        if not self.responses:
            raise AssertionError(f"Unexpected LLM call: {prompt[:120]}")
        return self.responses.pop(0)

    def stream_generate(self, prompt: str, **kwargs):
        yield self.generate(prompt, **kwargs)


class FakeSearchTool:
    def search(self, query: str, **kwargs):
        return {
            "query": query,
            "source": "tavily",
            "results": [
                {
                    "title": "Peking University history",
                    "url": "https://example.com/pku-history",
                    "snippet": "Peking University was founded in 1898.",
                    "relevance_score": 0.9,
                    "metadata": {"published_date": "2026-01-01"},
                }
            ],
            "total_results": 1,
        }


def _plan(goal: str, description: str) -> str:
    return json.dumps(
        {
            "research_goal": goal,
            "sub_tasks": [
                {
                    "task_id": 1,
                    "description": description,
                    "search_queries": ["Peking University history"],
                    "sources": ["tavily"],
                    "priority": 1,
                }
            ],
            "completion_criteria": "Answer the focused question",
            "estimated_iterations": 1,
        }
    )


def test_stream_interactive_allows_reject_modify_then_approve():
    llm = QueueLLM(
        [
            "RESEARCH",
            _plan("Broad Peking University research", "Research broad PKU context"),
            _plan("Peking University history only", "Research PKU history"),
            "PKU history summary [E1]",
            json.dumps(
                {
                    "themes": [
                        {
                            "name": "历史",
                            "key_points": ["北京大学的历史可追溯至 1898 年。"],
                        }
                    ]
                },
                ensure_ascii=False,
            ),
        ]
    )

    researcher = Researcher(llm)
    researcher.tavily = FakeSearchTool()
    workflow = ResearchWorkflow(
        Coordinator(llm),
        Planner(llm),
        researcher,
        Rapporteur(llm),
    )
    decisions = [(False, "专注于历史"), (True, None)]
    reviewed_goals = []

    def approve_after_revision(state):
        reviewed_goals.append(state["research_plan"]["research_goal"])
        return decisions.pop(0)

    final_state = {}
    for update in workflow.stream_interactive(
        "调研北京大学",
        max_iterations=1,
        auto_approve=False,
        human_approval_callback=approve_after_revision,
        output_format="json",
    ):
        for value in update.values():
            if isinstance(value, dict):
                final_state = value

    assert decisions == []
    assert reviewed_goals == ["Broad Peking University research", "Peking University history only"]
    assert final_state["current_step"] == "completed"
    assert final_state["research_plan"]["research_goal"] == "Peking University history only"
    assert final_state["iteration_count"] == 1
    assert final_state["final_report"]
