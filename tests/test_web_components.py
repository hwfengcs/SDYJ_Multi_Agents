from SDYJ_Agents.web.components import _timeline_event_details


def test_timeline_event_details_summarizes_plan_refinement():
    details = _timeline_event_details(
        {
            "event_type": "plan_refinement",
            "input_snapshot": {
                "completed_subtasks": 2,
                "remaining_subtasks": 1,
            },
            "output_snapshot": {
                "added_task_ids": [4],
                "removed_task_ids": [3],
                "rationale": "Drop stale task and follow the new evidence.",
            },
            "metadata": {"plan_refined": True},
        }
    )

    assert "completed=2, remaining=1" in details
    assert "added=[4]" in details
    assert "removed=[3]" in details
    assert "rationale=Drop stale task" in details


def test_timeline_event_details_keeps_generic_metadata_compact():
    details = _timeline_event_details(
        {
            "event_type": "decision",
            "metadata": {"task_id": 7, "reason": "pending task available"},
        }
    )

    assert '"task_id": 7' in details
    assert "pending task available" in details
