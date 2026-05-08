"""Runner for public/external benchmark slices."""

from __future__ import annotations

import json
from collections import Counter, defaultdict
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


_FAILURE_ROOT_CAUSES = (
    "missing_prediction",
    "wrong_answer",
    "missing_expected_answer",
)


def _failure_root_causes(row: dict[str, Any]) -> list[str]:
    causes = []
    if not row.get("has_prediction"):
        causes.append("missing_prediction")
    if not row.get("has_expected_answer"):
        causes.append("missing_expected_answer")
    if row.get("has_prediction") and row.get("has_expected_answer") and not row.get("correct"):
        causes.append("wrong_answer")
    if not causes and not row.get("correct"):
        causes.append("wrong_answer")
    return causes


def _failure_note(root_causes: list[str]) -> str:
    notes = []
    if "missing_prediction" in root_causes:
        notes.append("Prediction file omitted this task.")
    if "missing_expected_answer" in root_causes:
        notes.append("Source row has no expected answer, so the example is auditable but not fully gradeable.")
    if "wrong_answer" in root_causes:
        notes.append("Prediction did not normalize to the expected answer.")
    return " ".join(notes)


def _render_failure_analysis_markdown(analysis: dict[str, Any]) -> str:
    lines = [
        "# External Benchmark Failure Analysis",
        "",
        "## Run Metadata",
        f"- Run id: `{analysis['run_id']}`",
        f"- Suite: `{analysis['suite']}`",
        f"- Source: `{analysis['source']}`",
        f"- Split: `{analysis['split']}`",
        f"- Limit: `{analysis['limit']}`",
        f"- Prediction source: `{analysis['prediction_source']}`",
        "",
        "## Result",
        f"- Passed: `{analysis['passed']}`",
        f"- Accuracy: `{analysis['accuracy']:.4f}`",
        f"- Prediction coverage: `{analysis['prediction_coverage']:.4f}`",
        f"- Missing prediction count: `{analysis['missing_prediction_count']}`",
        f"- Missing expected-answer count: `{analysis['missing_expected_answer_count']}`",
        f"- Incorrect task ids: `{', '.join(analysis['incorrect_task_ids']) or 'none'}`",
        f"- Failure row count: `{analysis['failure_row_count']}`",
        "",
        "## Root Cause Counts",
    ]
    counts = analysis.get("root_cause_counts") or {}
    if counts:
        for cause in _FAILURE_ROOT_CAUSES:
            if cause in counts:
                lines.append(f"- `{cause}`: `{counts[cause]}`")
    else:
        lines.append("- None")

    lines.extend(["", "## Failed Rows"])
    rows = analysis.get("failure_rows") or []
    if not rows:
        lines.append("- None")
    else:
        for row in rows:
            causes = ", ".join(row.get("root_causes") or []) or "none"
            prediction = (row.get("prediction") or "").strip()
            prediction_excerpt = prediction if len(prediction) <= 120 else f"{prediction[:117]}..."
            note = row.get("note") or "No additional note."
            lines.append(
                f"- `{row['task_id']}` | `{causes}` | `{prediction_excerpt or 'n/a'}` | {note}"
            )
    lines.extend(
        [
            "",
            "## Artifacts",
            f"- Summary: `{analysis['artifacts']['summary']}`",
            f"- Manifest: `{analysis['artifacts']['manifest']}`",
            f"- Predictions: `{analysis['artifacts']['predictions']}`",
            f"- Graded rows: `{analysis['artifacts']['graded']}`",
        ]
    )
    return "\n".join(lines)


def _build_failure_analysis(
    *,
    run_id: str,
    suite: str,
    source: str,
    split: str,
    limit: int | None,
    prediction_source: str,
    fail_under: float | None,
    passed: bool,
    graded: dict[str, Any],
    examples: list[ExternalExample],
    artifacts: dict[str, str],
) -> dict[str, Any]:
    failure_rows = []
    root_cause_counts: Counter[str] = Counter()
    task_ids_by_root_cause: dict[str, list[str]] = defaultdict(list)

    for example, row in zip(examples, graded.get("rows", []), strict=True):
        if row.get("correct"):
            continue
        root_causes = _failure_root_causes(row)
        for cause in root_causes:
            root_cause_counts[cause] += 1
            task_ids_by_root_cause[cause].append(example.task_id)
        failure_rows.append(
            {
                "task_id": example.task_id,
                "level": example.level,
                "file_name": example.file_name,
                "prediction": row.get("prediction", ""),
                "has_prediction": row.get("has_prediction", False),
                "has_expected_answer": row.get("has_expected_answer", False),
                "correct": row.get("correct", False),
                "root_causes": root_causes,
                "note": _failure_note(root_causes),
            }
        )

    return {
        "run_id": run_id,
        "created_at": datetime.now().isoformat(),
        "suite": suite,
        "source": source,
        "split": split,
        "limit": limit,
        "prediction_source": prediction_source,
        "fail_under": fail_under,
        "passed": passed,
        "accuracy": graded["accuracy"],
        "prediction_coverage": graded["prediction_coverage"],
        "missing_prediction_count": graded["missing_prediction_count"],
        "missing_expected_answer_count": graded["missing_expected_answer_count"],
        "incorrect_task_ids": graded["incorrect_task_ids"],
        "failure_row_count": len(failure_rows),
        "root_cause_counts": {
            cause: root_cause_counts[cause]
            for cause in _FAILURE_ROOT_CAUSES
            if root_cause_counts[cause]
        },
        "task_ids_by_root_cause": {
            cause: task_ids_by_root_cause[cause]
            for cause in _FAILURE_ROOT_CAUSES
            if task_ids_by_root_cause[cause]
        },
        "failure_rows": failure_rows,
        "artifacts": artifacts,
    }


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
    failure_analysis_json_path = run_dir / "failure_analysis.json"
    failure_analysis_md_path = run_dir / "failure_analysis.md"

    artifact_paths = {
        "run_dir": str(run_dir),
        "manifest": str(manifest_path),
        "predictions": str(predictions_out),
        "graded": str(graded_path),
        "summary": str(summary_path),
        "failure_analysis_json": str(failure_analysis_json_path),
        "failure_analysis_md": str(failure_analysis_md_path),
    }

    _write_jsonl(manifest_path, _manifest_rows(examples))
    _write_jsonl(predictions_out, _prediction_rows(predictions))
    _write_jsonl(graded_path, graded["rows"])

    failure_analysis = _build_failure_analysis(
        run_id=run_id,
        suite=suite,
        source=source,
        split=split,
        limit=limit,
        prediction_source=prediction_source,
        fail_under=fail_under,
        passed=passed,
        graded=graded,
        examples=examples,
        artifacts=artifact_paths,
    )
    failure_analysis_md_path.write_text(
        _render_failure_analysis_markdown(failure_analysis),
        encoding="utf-8",
    )
    _write_json(failure_analysis_json_path, failure_analysis)

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
            **artifact_paths,
        },
        "failure_analysis": {
            "failure_row_count": failure_analysis["failure_row_count"],
            "root_cause_counts": failure_analysis["root_cause_counts"],
            "task_ids_by_root_cause": failure_analysis["task_ids_by_root_cause"],
            "json": failure_analysis["artifacts"]["failure_analysis_json"],
            "markdown": failure_analysis["artifacts"]["failure_analysis_md"],
        },
    }
    _write_json(summary_path, summary)
    return summary
