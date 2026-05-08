from SDYJ_Agents.evaluation import compare_evaluation_summaries, run_evaluation
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
    assert summary["enable_plan_refinement"] is False
    assert summary["enable_parallel_tool_execution"] is False


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
    assert result["failure_analysis"]["failed_metrics"][0]["root_cause"] == "tool_error"
    assert summary["failure_analysis"]["root_cause_counts"] == {"tool_error": 1}


def test_compare_summary_flags_metric_regression():
    comparison = compare_evaluation_summaries(
        current={
            "results": [
                {
                    "scenario_id": "agent_reliability_hard",
                    "metrics": {"overall_score": 1.0, "citation_id_coverage": 0.5},
                }
            ]
        },
        baseline={
            "summary_path": "baseline.json",
            "results": [
                {
                    "scenario_id": "agent_reliability_hard",
                    "metrics": {"overall_score": 1.0, "citation_id_coverage": 1.0},
                }
            ],
        },
    )

    assert comparison["passed"] is False
    assert comparison["metric_regression_count"] == 1
    regression = comparison["rows"][0]["metric_regressions"][0]
    assert regression["metric"] == "citation_id_coverage"
    assert regression["root_cause"] == "citation_gap"


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
