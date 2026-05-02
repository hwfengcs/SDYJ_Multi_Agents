import copy
import json

from SDYJ_Agents.agents.planner import Planner


class FakeLLM:
    def __init__(self, response: str):
        self.response = response
        self.calls = []

    def generate(self, prompt: str, **kwargs) -> str:
        self.calls.append({"prompt": prompt, "kwargs": kwargs})
        return self.response


class CountingLLM:
    def __init__(self, response: str = "{}"):
        self.response = response
        self.calls = 0

    def generate(self, prompt: str, **kwargs) -> str:
        self.calls += 1
        return self.response


class FailingLLM:
    def generate(self, prompt: str, **kwargs) -> str:
        raise RuntimeError("model down")


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


def test_refine_plan_preserves_completed_tasks_and_merges_pending_updates():
    response = json.dumps(
        {
            "research_goal": "Refined goal",
            "sub_tasks": [
                {
                    "task_id": 1,
                    "description": "LLM tried to edit completed task",
                    "search_queries": ["bad edit"],
                    "sources": ["tavily"],
                    "status": "completed",
                    "priority": 99,
                },
                {
                    "task_id": 2,
                    "description": "Use tighter remaining query",
                    "search_queries": ["tighter query"],
                    "sources": ["tavily"],
                    "priority": 2,
                    "_reflected": True,
                },
                {
                    "description": "Follow up on evidence gap",
                    "search_queries": ["evidence gap"],
                    "sources": ["arxiv"],
                    "priority": 3,
                },
            ],
            "completion_criteria": "Answer with updated evidence",
            "estimated_iterations": 4,
            "refinement_rationale": "The first two tasks narrowed the scope.",
        }
    )
    planner = Planner(FakeLLM(response))
    completed_task = {
        "task_id": 1,
        "description": "Original completed task",
        "search_queries": ["original query"],
        "sources": ["tavily"],
        "status": "completed",
        "priority": 1,
        "runtime_note": "keep me",
    }
    state = {
        "query": "agent evaluation",
        "research_plan": {
            "research_goal": "Original goal",
            "sub_tasks": [
                completed_task,
                {
                    "task_id": 2,
                    "description": "Original pending task",
                    "search_queries": ["broad query"],
                    "sources": ["tavily"],
                    "status": "pending",
                    "priority": 2,
                },
            ],
            "completion_criteria": "Original criteria",
            "estimated_iterations": 3,
        },
        "evidence_items": [{"title": "Evidence", "snippet": "Useful finding"}],
    }

    updated = planner.refine_plan(state)

    tasks = updated["research_plan"]["sub_tasks"]
    assert updated["plan_refined"] is True
    assert tasks[0] == completed_task
    assert [task["task_id"] for task in tasks] == [1, 2, 3]
    assert [task["description"] for task in tasks] == [
        "Original completed task",
        "Use tighter remaining query",
        "Follow up on evidence gap",
    ]
    assert tasks[1]["status"] == "pending"
    assert "_reflected" not in tasks[1]
    assert updated["research_plan"]["research_goal"] == "Refined goal"
    assert updated["research_plan"]["completion_criteria"] == "Answer with updated evidence"
    assert updated["research_plan"]["estimated_iterations"] == 4
    assert updated["research_plan"]["history"][-1] == {
        "event": "plan_refined",
        "rationale": "The first two tasks narrowed the scope.",
        "completed_count": 1,
    }


def test_refine_plan_rekeys_duplicate_missing_and_non_numeric_pending_task_ids():
    response = json.dumps(
        {
            "sub_tasks": [
                {"task_id": 3, "description": "Keep explicit id"},
                {"task_id": 3, "description": "Duplicate id"},
                {"task_id": None, "description": "Missing id"},
                {"task_id": 0, "description": "Zero id"},
                {"task_id": "", "description": "Blank id"},
                {"task_id": "task-new", "description": "Non-numeric id"},
                {"task_id": "9", "description": "Numeric string id"},
            ]
        }
    )
    planner = Planner(FakeLLM(response))
    state = {
        "query": "agent evaluation",
        "research_plan": {
            "sub_tasks": [
                {"task_id": 1, "description": "Completed one", "status": "completed"},
                {"task_id": 2, "description": "Completed two", "status": "completed"},
                {"task_id": 3, "description": "Original pending", "status": "pending"},
            ]
        },
    }

    updated = planner.refine_plan(state)

    tasks = updated["research_plan"]["sub_tasks"]
    assert [task["task_id"] for task in tasks] == [1, 2, 3, 4, 5, 6, 7, 8, 9]
    assert [task["description"] for task in tasks[2:]] == [
        "Keep explicit id",
        "Duplicate id",
        "Missing id",
        "Zero id",
        "Blank id",
        "Non-numeric id",
        "Numeric string id",
    ]
    assert all(task["status"] == "pending" for task in tasks[2:])
    assert updated["plan_refined"] is True


def test_refine_plan_marks_refined_and_keeps_original_plan_when_response_is_invalid():
    planner = Planner(FakeLLM("not json"))
    state = {
        "query": "agent evaluation",
        "research_plan": {
            "research_goal": "Original goal",
            "sub_tasks": [
                {"task_id": 1, "description": "Completed", "status": "completed"},
                {"task_id": 2, "description": "Pending", "status": "pending"},
            ],
            "completion_criteria": "Original criteria",
            "estimated_iterations": 3,
        },
    }
    original_plan = copy.deepcopy(state["research_plan"])

    updated = planner.refine_plan(state)

    assert updated["plan_refined"] is True
    assert updated["research_plan"] == original_plan


def test_refine_plan_marks_refined_and_keeps_original_plan_when_llm_fails():
    planner = Planner(FailingLLM())
    state = {
        "query": "agent evaluation",
        "research_plan": {
            "research_goal": "Original goal",
            "sub_tasks": [
                {"task_id": 1, "description": "Completed", "status": "completed"},
                {"task_id": 2, "description": "Pending", "status": "pending"},
            ],
            "completion_criteria": "Original criteria",
            "estimated_iterations": 3,
        },
    }
    original_plan = copy.deepcopy(state["research_plan"])

    updated = planner.refine_plan(state)

    assert updated["plan_refined"] is True
    assert updated["research_plan"] == original_plan


def test_refine_plan_skips_llm_when_no_remaining_tasks():
    llm = CountingLLM()
    planner = Planner(llm)
    state = {
        "query": "agent evaluation",
        "research_plan": {
            "sub_tasks": [
                {"task_id": 1, "description": "Completed one", "status": "completed"},
                {"task_id": 2, "description": "Completed two", "status": "completed"},
            ]
        },
    }
    original_plan = copy.deepcopy(state["research_plan"])

    updated = planner.refine_plan(state)

    assert updated["plan_refined"] is True
    assert updated["research_plan"] == original_plan
    assert llm.calls == 0
