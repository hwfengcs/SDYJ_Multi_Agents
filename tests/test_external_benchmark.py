import json
from pathlib import Path

import pytest

from SDYJ_Agents.benchmarks.external_runner import run_external_benchmark
from SDYJ_Agents.benchmarks.gaia_loader import load_gaia_examples
from SDYJ_Agents.benchmarks.grader import is_correct_prediction


def test_gaia_local_fixture_loads_limit():
    examples = load_gaia_examples(source="local", limit=2)

    assert len(examples) == 2
    assert examples[0].task_id == "gaia-mini-001"


def test_external_benchmark_writes_artifacts(tmp_path):
    summary = run_external_benchmark(
        suite="gaia",
        source="local",
        limit=2,
        output_dir=tmp_path,
        fail_under=1.0,
    )

    assert summary["passed"] is True
    assert summary["example_count"] == 2
    assert summary["accuracy"] == 1.0
    assert summary["prediction_coverage"] == 1.0
    assert summary["missing_prediction_count"] == 0
    assert summary["incorrect_task_ids"] == []
    assert summary["prediction_source"] == "example_metadata_baseline"
    summary_path = tmp_path / "external_benchmarks" / summary["run_id"] / "summary.json"
    manifest_path = tmp_path / "external_benchmarks" / summary["run_id"] / "manifest.jsonl"
    assert summary_path.exists()
    assert manifest_path.exists()
    assert json.loads(summary_path.read_text(encoding="utf-8"))["accuracy"] == 1.0
    failure_analysis_path = tmp_path / "external_benchmarks" / summary["run_id"] / "failure_analysis.json"
    assert failure_analysis_path.exists()
    assert summary["failure_analysis"]["failure_row_count"] == 0
    assert summary["failure_analysis"]["root_cause_counts"] == {}
    assert Path(summary["artifacts"]["failure_analysis_md"]).exists()


def test_external_benchmark_accepts_prediction_file(tmp_path):
    predictions = tmp_path / "predictions.jsonl"
    predictions.write_text(
        '{"task_id":"gaia-mini-001","prediction":"Mars"}\n'
        '{"task_id":"gaia-mini-002","prediction":"wrong"}\n',
        encoding="utf-8",
    )

    summary = run_external_benchmark(
        suite="gaia",
        source="local",
        limit=2,
        output_dir=tmp_path,
        predictions_path=predictions,
    )

    assert summary["correct"] == 1
    assert summary["accuracy"] == 0.5
    assert summary["prediction_source"] == "predictions_file"
    assert summary["prediction_coverage"] == 1.0
    assert summary["incorrect_task_ids"] == ["gaia-mini-002"]
    assert summary["failure_analysis"]["root_cause_counts"] == {"wrong_answer": 1}


def test_external_benchmark_records_missing_predictions_and_failed_threshold(tmp_path):
    predictions = tmp_path / "predictions.jsonl"
    predictions.write_text('{"task_id":"gaia-mini-001","prediction":"Mars"}\n', encoding="utf-8")

    summary = run_external_benchmark(
        suite="gaia",
        source="local",
        limit=3,
        output_dir=tmp_path,
        predictions_path=predictions,
        fail_under=0.9,
    )

    assert summary["passed"] is False
    assert summary["accuracy"] == pytest.approx(1 / 3)
    assert summary["prediction_coverage"] == pytest.approx(1 / 3)
    assert summary["missing_prediction_count"] == 2
    assert summary["incorrect_task_ids"] == ["gaia-mini-002", "gaia-mini-003"]
    assert summary["fail_under_delta"] == pytest.approx(0.9 - (1 / 3))
    assert summary["failure_analysis"]["root_cause_counts"] == {"missing_prediction": 2}
    assert summary["failure_analysis"]["task_ids_by_root_cause"]["missing_prediction"] == [
        "gaia-mini-002",
        "gaia-mini-003",
    ]


def test_external_benchmark_source_jsonl_without_expected_answer_is_auditable(tmp_path):
    data = tmp_path / "gaia_sample.jsonl"
    data.write_text(
        '{"task_id":"custom-001","question":"Question with no answer"}\n',
        encoding="utf-8",
    )
    predictions = tmp_path / "predictions.jsonl"
    predictions.write_text(
        '{"task_id":"custom-001","prediction":"anything"}\n',
        encoding="utf-8",
    )

    summary = run_external_benchmark(
        suite="gaia",
        source="jsonl",
        data_path=data,
        predictions_path=predictions,
        output_dir=tmp_path,
    )

    run_dir = tmp_path / "external_benchmarks" / summary["run_id"]
    graded_row = json.loads((run_dir / "graded.jsonl").read_text(encoding="utf-8"))

    assert summary["missing_expected_answer_count"] == 1
    assert summary["prediction_coverage"] == 1.0
    assert summary["failure_analysis"]["root_cause_counts"] == {"missing_expected_answer": 1}
    assert graded_row["has_expected_answer"] is False
    assert graded_row["correct"] is False


def test_external_benchmark_rejects_unsupported_suite_and_source(tmp_path):
    with pytest.raises(ValueError, match="unsupported external benchmark suite"):
        run_external_benchmark(suite="unknown", output_dir=tmp_path)

    with pytest.raises(ValueError, match="unsupported GAIA source"):
        run_external_benchmark(suite="gaia", source="unknown", output_dir=tmp_path)


def test_grader_normalizes_answer_variants():
    assert is_correct_prediction("the mars.", "Mars")
    assert is_correct_prediction("H2O", "water|H2O")


def test_committed_public_benchmark_smoke_artifacts_are_complete():
    docs_artifacts = Path(__file__).resolve().parents[1] / "docs" / "public-benchmark-artifacts"

    for filename in [
        "gaia-smoke-summary.json",
        "gaia-smoke-manifest.jsonl",
        "gaia-smoke-predictions.jsonl",
        "gaia-smoke-graded.jsonl",
        "gaia-smoke-failure-analysis.json",
        "gaia-smoke-failure-analysis.md",
    ]:
        assert (docs_artifacts / filename).is_file()

    summary = json.loads((docs_artifacts / "gaia-smoke-summary.json").read_text(encoding="utf-8"))
    assert summary["failure_analysis"]["failure_row_count"] == 0
    assert summary["artifacts"]["failure_analysis_json"].endswith("failure_analysis.json")
