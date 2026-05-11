from SDYJ_Agents.workflow.nodes import WorkflowNodes


class FakePlanner:
    enable_plan_refinement = True
    refine_after_n_tasks = 2

    def __init__(self):
        self.calls = 0

    def refine_plan(self, state):
        self.calls += 1
        state["plan_refined"] = True
        return state


def test_maybe_refine_plan_triggers_only_once_after_threshold_with_remaining_tasks():
    planner = FakePlanner()
    nodes = WorkflowNodes(
        coordinator=None,
        planner=planner,
        researcher=None,
        rapporteur=None,
        verifier=None,
    )
    state = {
        "research_plan": {
            "sub_tasks": [
                {"task_id": 1, "status": "completed"},
                {"task_id": 2, "status": "completed"},
                {"task_id": 3, "status": "pending"},
            ]
        },
        "trace": {"events": [], "nodes": [], "metrics": {}},
    }

    nodes._maybe_refine_plan(state)
    nodes._maybe_refine_plan(state)

    assert planner.calls == 1
    assert state["plan_refined"] is True
    assert state["trace"]["metrics"]["plan_refinement_count"] == 1
    refinement_events = [
        event
        for event in state["trace"]["events"]
        if event["event_type"] == "plan_refinement"
    ]
    assert len(refinement_events) == 1
    assert refinement_events[0]["input_snapshot"] == {
        "completed_subtasks": 2,
        "remaining_subtasks": 1,
    }
    assert refinement_events[0]["metadata"]["plan_refined"] is True
