import json

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
    summary_path = tmp_path / "external_benchmarks" / summary["run_id"] / "summary.json"
    manifest_path = tmp_path / "external_benchmarks" / summary["run_id"] / "manifest.jsonl"
    assert summary_path.exists()
    assert manifest_path.exists()
    assert json.loads(summary_path.read_text(encoding="utf-8"))["accuracy"] == 1.0


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


def test_grader_normalizes_answer_variants():
    assert is_correct_prediction("the mars.", "Mars")
    assert is_correct_prediction("H2O", "water|H2O")
