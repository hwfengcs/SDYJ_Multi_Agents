"""Runner for public/external benchmark slices."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

from .gaia_loader import DEFAULT_HF_CONFIG, DEFAULT_HF_DATASET, DEFAULT_SPLIT, load_gaia_examples
from .grader import ExternalExample, Prediction, grade_predictions


def _json_default(value: Any) -> Any:
    if hasattr(value, "item"):
        return value.item()
    return str(value)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=_json_default), encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, default=_json_default) + "\n")


def _load_predictions(path: str | Path) -> list[Prediction]:
    predictions = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            predictions.append(
                Prediction(
                    task_id=str(row.get("task_id") or row.get("id") or row.get("Task ID")),
                    prediction=str(row.get("prediction") or row.get("answer") or ""),
                )
            )
    return predictions


def _baseline_predictions(examples: list[ExternalExample]) -> list[Prediction]:
    predictions = []
    for example in examples:
        metadata = example.metadata or {}
        predictions.append(
            Prediction(
                task_id=example.task_id,
                prediction=str(metadata.get("baseline_prediction", "")),
            )
        )
    return predictions


def _manifest_rows(examples: list[ExternalExample]) -> list[dict[str, Any]]:
    rows = []
    for example in examples:
        rows.append(
            {
                "task_id": example.task_id,
                "level": example.level,
                "question": example.question,
                "file_name": example.file_name,
                "has_expected_answer": bool(example.final_answer),
            }
        )
    return rows


def _prediction_rows(predictions: list[Prediction]) -> list[dict[str, str]]:
    return [
        {
            "task_id": prediction.task_id,
            "prediction": prediction.prediction,
        }
        for prediction in predictions
    ]


def run_external_benchmark(
    suite: str = "gaia",
    source: str = "local",
    split: str = DEFAULT_SPLIT,
    limit: int | None = None,
    output_dir: str | Path = "./outputs",
    predictions_path: str | Path | None = None,
    data_path: str | Path | None = None,
    hf_dataset: str = DEFAULT_HF_DATASET,
    hf_config: str = DEFAULT_HF_CONFIG,
    fail_under: float | None = None,
) -> dict[str, Any]:
    """Load examples, grade predictions, and persist reproducible artifacts."""
    suite = suite.lower().strip()
    if suite != "gaia":
        raise ValueError(f"unsupported external benchmark suite: {suite}")

    examples = load_gaia_examples(
        source=source,
        split=split,
        limit=limit,
        data_path=data_path,
        hf_dataset=hf_dataset,
        hf_config=hf_config,
    )
    prediction_source = "predictions_file" if predictions_path else "example_metadata_baseline"
    predictions = _load_predictions(predictions_path) if predictions_path else _baseline_predictions(examples)
    graded = grade_predictions(examples, predictions)
    passed = True
    fail_under_delta = None
    if fail_under is not None and graded["accuracy"] < fail_under:
        passed = False
        fail_under_delta = fail_under - graded["accuracy"]

    run_id = f"{suite}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    run_dir = Path(output_dir) / "external_benchmarks" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = run_dir / "manifest.jsonl"
    predictions_out = run_dir / "predictions.jsonl"
    graded_path = run_dir / "graded.jsonl"
    summary_path = run_dir / "summary.json"

    _write_jsonl(manifest_path, _manifest_rows(examples))
    _write_jsonl(predictions_out, _prediction_rows(predictions))
    _write_jsonl(graded_path, graded["rows"])

    summary = {
        "run_id": run_id,
        "created_at": datetime.now().isoformat(),
        "suite": suite,
        "source": source,
        "split": split,
        "limit": limit,
        "hf_dataset": hf_dataset if source == "hf" else None,
        "hf_config": hf_config if source == "hf" else None,
        "data_path": str(data_path) if data_path else None,
        "predictions_input": str(predictions_path) if predictions_path else None,
        "prediction_source": prediction_source,
        "example_count": graded["total"],
        "correct": graded["correct"],
        "accuracy": graded["accuracy"],
        "prediction_coverage": graded["prediction_coverage"],
        "missing_prediction_count": graded["missing_prediction_count"],
        "missing_expected_answer_count": graded["missing_expected_answer_count"],
        "incorrect_task_ids": graded["incorrect_task_ids"],
        "fail_under": fail_under,
        "fail_under_delta": fail_under_delta,
        "passed": passed,
        "artifacts": {
            "run_dir": str(run_dir),
            "manifest": str(manifest_path),
            "predictions": str(predictions_out),
            "graded": str(graded_path),
            "summary": str(summary_path),
        },
    }
    _write_json(summary_path, summary)
    return summary
