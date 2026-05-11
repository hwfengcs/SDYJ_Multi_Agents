"""GAIA-style benchmark loading.

The bundled fixture is synthetic and exists only to keep the external benchmark
pipeline testable without Hugging Face credentials. Use ``source="hf"`` for a
real GAIA slice.
"""

from __future__ import annotations

import json
from importlib import resources
from pathlib import Path
from typing import Any

from .grader import ExternalExample


DEFAULT_HF_DATASET = "gaia-benchmark/GAIA"
DEFAULT_HF_CONFIG = "2023_level1"
DEFAULT_SPLIT = "validation"


def _coerce_example(row: dict[str, Any], index: int) -> ExternalExample:
    task_id = str(
        row.get("task_id")
        or row.get("id")
        or row.get("Task ID")
        or row.get("task")
        or f"example-{index + 1}"
    )
    question = str(row.get("question") or row.get("Question") or row.get("query") or "")
    final_answer = str(
        row.get("final_answer")
        or row.get("Final answer")
        or row.get("answer")
        or row.get("Answer")
        or ""
    )
    level = row.get("level") or row.get("Level")
    file_name = row.get("file_name") or row.get("file") or row.get("File Name")
    metadata = {
        key: value
        for key, value in row.items()
        if key
        not in {
            "task_id",
            "id",
            "Task ID",
            "task",
            "question",
            "Question",
            "query",
            "final_answer",
            "Final answer",
            "answer",
            "Answer",
            "level",
            "Level",
            "file_name",
            "file",
            "File Name",
        }
    }
    return ExternalExample(
        task_id=task_id,
        question=question,
        final_answer=final_answer,
        level=level,
        file_name=file_name,
        metadata=metadata,
    )


def _read_jsonl(path: Path, limit: int | None = None) -> list[ExternalExample]:
    examples = []
    with path.open("r", encoding="utf-8") as handle:
        for index, line in enumerate(handle):
            if limit is not None and len(examples) >= limit:
                break
            line = line.strip()
            if not line:
                continue
            examples.append(_coerce_example(json.loads(line), index))
    return examples


def _fixture_path() -> Path:
    return Path(str(resources.files(__package__) / "fixtures" / "gaia_mini.jsonl"))


def load_gaia_examples(
    source: str = "local",
    split: str = DEFAULT_SPLIT,
    limit: int | None = None,
    data_path: str | Path | None = None,
    hf_dataset: str = DEFAULT_HF_DATASET,
    hf_config: str = DEFAULT_HF_CONFIG,
) -> list[ExternalExample]:
    """Load a GAIA-style slice from a local fixture, JSONL, or Hugging Face."""
    source = source.lower().strip()
    if source == "local":
        return _read_jsonl(_fixture_path(), limit=limit)
    if source == "jsonl":
        if not data_path:
            raise ValueError("--data-path is required when --source jsonl")
        return _read_jsonl(Path(data_path), limit=limit)
    if source == "hf":
        try:
            from datasets import load_dataset
        except ImportError as exc:
            raise RuntimeError(
                "Hugging Face dataset loading requires the [benchmarks] extra: "
                "pip install 'sdyj-multi-agents[benchmarks]'"
            ) from exc

        try:
            dataset = load_dataset(hf_dataset, hf_config, split=split)
        except Exception as exc:
            raise RuntimeError(
                "Unable to load the GAIA dataset from Hugging Face. "
                "If the dataset is gated, run `huggingface-cli login` and accept "
                "the dataset terms, or use --source local/--source jsonl for a dry run."
            ) from exc

        rows = []
        for index, row in enumerate(dataset):
            if limit is not None and len(rows) >= limit:
                break
            rows.append(_coerce_example(dict(row), index))
        return rows
    raise ValueError(f"unsupported GAIA source: {source}")
