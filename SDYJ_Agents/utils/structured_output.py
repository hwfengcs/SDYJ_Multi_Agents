"""Helpers for LLM structured JSON output."""

from __future__ import annotations

import json
from typing import Any, Dict, Mapping

from ..llm.base import parse_json_object


def generate_json_object(
    llm: Any,
    prompt: str,
    schema: Mapping[str, Any] | None = None,
    **kwargs,
) -> Dict[str, Any]:
    """Call native ``generate_json`` when available, otherwise parse text.

    This keeps agents compatible with simple test doubles that only implement
    ``generate`` while letting real providers use JSON mode through BaseLLM.
    """
    if hasattr(llm, "generate_json"):
        return llm.generate_json(prompt, schema=schema, **kwargs)

    schema_hint = ""
    if schema:
        schema_hint = (
            "\n\nReturn a single valid JSON object matching this schema. "
            "Do not include Markdown fences or prose.\n"
            f"{json.dumps(schema, ensure_ascii=False, indent=2)}"
        )
    response = llm.generate(prompt + schema_hint, **kwargs)
    return parse_json_object(response)
