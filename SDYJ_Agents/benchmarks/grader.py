"""Lightweight answer grading for external benchmark predictions."""

from __future__ import annotations

import re
import string
from dataclasses import dataclass
from typing import Any, Iterable


_ARTICLES_RE = re.compile(r"\b(a|an|the)\b", flags=re.IGNORECASE)


@dataclass(frozen=True)
class ExternalExample:
    task_id: str
    question: str
    final_answer: str
    level: str | int | None = None
    file_name: str | None = None
    metadata: dict[str, Any] | None = None


@dataclass(frozen=True)
class Prediction:
    task_id: str
    prediction: str


def normalize_answer(value: str) -> str:
    """Normalize short benchmark answers using common QA exact-match rules."""
    text = str(value or "").strip().lower()
    text = _ARTICLES_RE.sub(" ", text)
    text = "".join(ch for ch in text if ch not in string.punctuation)
    return " ".join(text.split())


def is_correct_prediction(prediction: str, expected: str) -> bool:
    """Return whether a prediction matches the expected answer."""
    expected_variants = [item.strip() for item in str(expected).split("|") if item.strip()]
    if not expected_variants:
        return False
    normalized_prediction = normalize_answer(prediction)
    return any(normalized_prediction == normalize_answer(item) for item in expected_variants)


def grade_predictions(
    examples: Iterable[ExternalExample],
    predictions: Iterable[Prediction],
) -> dict[str, Any]:
    prediction_by_id = {item.task_id: item.prediction for item in predictions}
    rows = []
    correct = 0
    total = 0
    missing_prediction_count = 0
    missing_expected_answer_count = 0
    incorrect_task_ids = []
    for example in examples:
        total += 1
        prediction = prediction_by_id.get(example.task_id, "")
        if not prediction:
            missing_prediction_count += 1
        if not example.final_answer:
            missing_expected_answer_count += 1
        is_correct = is_correct_prediction(prediction, example.final_answer)
        correct += int(is_correct)
        if not is_correct:
            incorrect_task_ids.append(example.task_id)
        rows.append(
            {
                "task_id": example.task_id,
                "level": example.level,
                "prediction": prediction,
                "has_prediction": bool(prediction),
                "has_expected_answer": bool(example.final_answer),
                "correct": is_correct,
            }
        )

    accuracy = correct / total if total else 0.0
    prediction_coverage = (total - missing_prediction_count) / total if total else 0.0
    return {
        "total": total,
        "correct": correct,
        "accuracy": accuracy,
        "prediction_coverage": prediction_coverage,
        "missing_prediction_count": missing_prediction_count,
        "missing_expected_answer_count": missing_expected_answer_count,
        "incorrect_task_ids": incorrect_task_ids,
        "rows": rows,
    }
