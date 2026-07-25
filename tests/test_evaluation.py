from SDYJ_Agents.evaluation.runner import run_evaluation
from SDYJ_Agents.evaluation.metrics import apply_thresholds, trace_completeness


def test_offline_evaluation_runs_one_scenario(tmp_path):
    summary = run_evaluation(
        live=False,
        max_scenarios=1,
        output_dir=str(tmp_path),
        max_iterations=2,
    )

    assert summary["scenario_count"] == 1
    result = summary["results"][0]
    assert result["metrics"]["evidence_count"] >= 1
    assert result["metrics"]["overall_score"] > 0
    assert result["metrics"]["trace_completeness"] > 0
    assert "passed" in summary
    assert result["trace_path"]
    assert result["report_path"]


def test_evaluation_applies_fail_under(tmp_path):
    summary = run_evaluation(
        live=False,
        max_scenarios=1,
        output_dir=str(tmp_path),
        max_iterations=2,
        fail_under=1.1,
    )

    assert summary["passed"] is False


def test_threshold_overrides_report_failed_metric(tmp_path):
    summary = run_evaluation(
        live=False,
        max_scenarios=1,
        output_dir=str(tmp_path),
        max_iterations=2,
        threshold_overrides={"tool_success_rate": 1.1},
    )

    result = summary["results"][0]
    assert result["passed"] is False
    assert result["failed_thresholds"][0]["metric"] == "tool_success_rate"


def test_trace_completeness_scores_required_fields():
    scenario = {
        "expected_trace": {
            "required_nodes": ["planner"],
            "required_top_level_fields": ["run_id", "nodes", "llm_calls", "tool_calls", "events", "replay_cache"],
            "required_tool_fields": ["source", "query"],
            "required_llm_fields": ["call_id", "model"],
        }
    }
    trace = {
        "run_id": "r1",
        "nodes": [{"node": "planner"}],
        "llm_calls": [{"call_id": "L1", "model": "fake"}],
        "tool_calls": [{"source": "tavily", "query": "q"}],
        "events": [{"event_id": "evt_000001"}],
        "replay_cache": {"llm_calls": [{"response": "ok"}], "tool_calls": [{"result": {}}]},
    }

    assert trace_completeness(trace, scenario) == 1.0


def test_apply_thresholds_marks_failure():
    result = apply_thresholds({"overall_score": 0.5}, {"thresholds": {"overall_score": 0.8}})

    assert result["passed"] is False
    assert result["failed_thresholds"][0]["actual"] == 0.5


def test_offline_evaluation_reports_faithfulness_dimension(tmp_path):
    summary = run_evaluation(
        live=False,
        max_scenarios=1,
        output_dir=str(tmp_path),
        max_iterations=2,
    )

    metrics = summary["results"][0]["metrics"]
    assert metrics["judged_claim_count"] > 0
    assert metrics["faithfulness_score"] >= 0.70
    assert metrics["citation_precision"] > 0
    assert metrics["citation_validity_rate"] == 1.0
    assert metrics["invalid_citation_count"] == 0


def test_llm_failure_scenario_survives_with_retry_and_degradation(tmp_path):
    summary = run_evaluation(
        live=False,
        scenario_ids=["llm_failure_recovery_hard"],
        output_dir=str(tmp_path),
        max_iterations=2,
    )

    result = summary["results"][0]
    metrics = result["metrics"]
    assert result["passed"], result["failed_thresholds"]
    assert result["report_path"]  # the degraded report still shipped
    assert metrics["retries_total"] >= 1  # transient summarize failure retried
    assert metrics["degraded_event_count"] >= 1  # permanent conclusion failure degraded
    assert metrics["tool_success_rate"] >= 0.5  # forced arxiv outage isolated

