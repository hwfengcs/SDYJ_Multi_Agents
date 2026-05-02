from SDYJ_Agents.evaluation.runner import run_evaluation


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
    assert result["trace_path"]
    assert result["report_path"]
