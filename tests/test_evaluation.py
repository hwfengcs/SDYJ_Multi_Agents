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
            "enable_verification": True,
            "results": [
                {
                    "scenario_id": "agent_reliability_hard",
                    "metrics": {
                        "overall_score": 1.0,
                        "citation_id_coverage": 0.5,
                        "tool_success_rate": 1.0,
                    },
                }
            ]
        },
        baseline={
            "summary_path": "baseline.json",
            "enable_verification": False,
            "results": [
                {
                    "scenario_id": "agent_reliability_hard",
                    "metrics": {
                        "overall_score": 1.0,
                        "citation_id_coverage": 1.0,
                        "tool_success_rate": 0.5,
                    },
                }
            ],
        },
    )

    assert comparison["passed"] is False
    assert comparison["metric_regression_count"] == 1
    regression = comparison["rows"][0]["metric_regressions"][0]
    assert regression["metric"] == "citation_id_coverage"
    assert regression["root_cause"] == "citation_gap"
    analysis = comparison["regression_analysis"]
    assert analysis["context_changes"] == [
        {"key": "enable_verification", "baseline": False, "current": True}
    ]
    assert analysis["root_cause_counts"] == {"citation_gap": 1}
    assert analysis["metric_delta_summary"]["citation_id_coverage"]["mean_delta"] == -0.5
    assert analysis["metric_delta_summary"]["tool_success_rate"]["improvement_count"] == 1
    assert analysis["top_metric_regressions"][0]["scenario_id"] == "agent_reliability_hard"
    assert analysis["top_metric_improvements"][0]["metric"] == "tool_success_rate"


def test_compare_summary_flags_missing_scenario():
    comparison = compare_evaluation_summaries(
        current={"results": []},
        baseline={
            "results": [
                {
                    "scenario_id": "agent_reliability_hard",
                    "metrics": {"overall_score": 1.0},
                }
            ],
        },
    )

    assert comparison["passed"] is False
    assert comparison["missing_scenario_count"] == 1
    assert comparison["rows"][0]["status"] == "missing"
    assert comparison["regression_analysis"]["root_cause_counts"] == {"missing_scenario": 1}


def test_run_evaluation_writes_comparison_analysis(tmp_path):
    baseline = tmp_path / "baseline.json"
    baseline.write_text(
        '{"results":[{"scenario_id":"agent_reliability_hard","metrics":{"overall_score":1.0,"trace_completeness":1.0}}]}',
        encoding="utf-8",
    )

    summary = run_evaluation(
        live=False,
        scenario_ids=["agent_reliability_hard"],
        output_dir=str(tmp_path),
        max_iterations=2,
        compare_summary_path=str(baseline),
    )

    assert summary["comparison"]["regression_analysis"]["compared_scenario_count"] == 1
    assert "trace_completeness" in summary["comparison"]["regression_analysis"]["metric_delta_summary"]


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
