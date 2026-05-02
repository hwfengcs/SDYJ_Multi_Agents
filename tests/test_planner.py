import json

from SDYJ_Agents.agents.planner import Planner


class FakeLLM:
    def __init__(self, response: str):
        self.response = response

    def generate(self, prompt: str, **kwargs) -> str:
        return self.response


def test_create_research_plan_parses_json_and_preserves_runtime_limit():
    response = json.dumps(
        {
            "research_goal": "Understand agent evaluation",
            "sub_tasks": [
                {
                    "task_id": 1,
                    "description": "Find evaluation methods",
                    "search_queries": ["agent evaluation"],
                    "sources": ["tavily", "arxiv"],
                    "priority": 1,
                }
            ],
            "completion_criteria": "Summarize practical metrics",
            "estimated_iterations": 2,
        }
    )
    planner = Planner(FakeLLM(response))
    state = {
        "query": "How should we evaluate AI agents?",
        "max_iterations": 7,
        "research_results": [],
    }

    updated = planner.create_research_plan(state)

    assert updated["research_plan"]["research_goal"] == "Understand agent evaluation"
    assert updated["research_plan"]["sub_tasks"][0]["status"] == "pending"
    assert updated["estimated_iterations"] == 2
    assert updated["max_iterations"] == 7


def test_format_plan_for_display_uses_readable_labels():
    planner = Planner(FakeLLM("{}"))
    plan = {
        "research_goal": "Map the project architecture",
        "estimated_iterations": 3,
        "completion_criteria": "All modules covered",
        "sub_tasks": [
            {
                "task_id": 1,
                "description": "Inspect workflow",
                "search_queries": ["workflow"],
                "sources": ["tavily"],
                "priority": 1,
                "status": "pending",
            }
        ],
    }

    display = planner.format_plan_for_display(plan)

    assert "Research Goal" in display
    assert "Estimated Iterations" in display
    assert "Subtasks" in display
    assert "=Ë" not in display
