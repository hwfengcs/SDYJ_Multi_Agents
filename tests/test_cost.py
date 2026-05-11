"""Tests for cost estimation utilities."""

from __future__ import annotations

import pytest

from SDYJ_Agents.utils.cost import (
    PRICING_TABLE,
    aggregate_trace_cost,
    estimate_call_cost_usd,
    normalize_usage,
)


def test_normalize_usage_handles_openai_style():
    usage = {"prompt_tokens": 100, "completion_tokens": 50}
    assert normalize_usage(usage) == (100, 50)


def test_normalize_usage_handles_anthropic_style():
    usage = {"input_tokens": 200, "output_tokens": 75}
    assert normalize_usage(usage) == (200, 75)


def test_normalize_usage_handles_gemini_style():
    usage = {
        "prompt_token_count": 300,
        "candidates_token_count": 120,
        "total_token_count": 420,
    }
    assert normalize_usage(usage) == (300, 120)


def test_normalize_usage_handles_missing_or_empty():
    assert normalize_usage(None) == (0, 0)
    assert normalize_usage({}) == (0, 0)
    # Defensive: nulls inside the dict must not raise.
    assert normalize_usage({"prompt_tokens": None, "completion_tokens": None}) == (0, 0)


def test_estimate_call_cost_for_known_model():
    # gpt-4o-mini: (0.15, 0.60) per 1M tokens.
    cost = estimate_call_cost_usd("openai", "gpt-4o-mini", 1_000_000, 1_000_000)
    assert cost == pytest.approx(0.75, rel=1e-6)


def test_estimate_call_cost_returns_none_for_unknown_pair():
    assert estimate_call_cost_usd("openai", "made-up-model", 100, 50) is None
    assert estimate_call_cost_usd(None, "gpt-4o", 100, 50) is None
    assert estimate_call_cost_usd("openai", None, 100, 50) is None


def test_estimate_call_cost_is_case_insensitive_on_provider():
    cost_lower = estimate_call_cost_usd("deepseek", "deepseek-v4-flash", 1000, 500)
    cost_upper = estimate_call_cost_usd("DEEPSEEK", "deepseek-v4-flash", 1000, 500)
    assert cost_lower == cost_upper
    assert cost_lower is not None and cost_lower > 0


def test_aggregate_trace_cost_sums_only_priced_calls():
    trace = {
        "llm_calls": [
            {
                "prompt_tokens_actual": 1000,
                "completion_tokens_actual": 500,
                "cost_usd": 0.001,
            },
            {
                "prompt_tokens_actual": 2000,
                "completion_tokens_actual": 1000,
                "cost_usd": None,  # provider/model not in PRICING_TABLE
            },
            {
                "prompt_tokens_actual": 1500,
                "completion_tokens_actual": 750,
                "cost_usd": 0.002,
            },
        ]
    }
    summary = aggregate_trace_cost(trace)
    assert summary["total_prompt_tokens"] == 4500
    assert summary["total_completion_tokens"] == 2250
    assert summary["total_tokens"] == 6750
    # Only the two priced calls contribute to the dollar total.
    assert summary["total_cost_usd"] == pytest.approx(0.003, rel=1e-6)
    assert summary["priced_call_count"] == 2
    assert summary["unpriced_call_count"] == 1


def test_aggregate_trace_cost_returns_none_when_all_unpriced():
    """An estimate of $0.00 when every call is unpriced is misleading; we
    surface ``None`` so the CLI can render an explicit "unknown"."""
    trace = {
        "llm_calls": [
            {"prompt_tokens_actual": 100, "completion_tokens_actual": 50, "cost_usd": None},
            {"prompt_tokens_actual": 200, "completion_tokens_actual": 100, "cost_usd": None},
        ]
    }
    summary = aggregate_trace_cost(trace)
    assert summary["total_cost_usd"] is None
    assert summary["priced_call_count"] == 0
    assert summary["unpriced_call_count"] == 2
    # Token totals still report because they come straight from usage data.
    assert summary["total_prompt_tokens"] == 300
    assert summary["total_completion_tokens"] == 150


def test_aggregate_trace_cost_handles_empty_trace():
    summary = aggregate_trace_cost({})
    assert summary["total_prompt_tokens"] == 0
    assert summary["total_completion_tokens"] == 0
    assert summary["total_tokens"] == 0
    assert summary["total_cost_usd"] is None
    assert summary["priced_call_count"] == 0
    assert summary["unpriced_call_count"] == 0


def test_pricing_table_has_entries_for_known_providers():
    """Smoke test so a future deletion is caught instead of silently producing
    None cost estimates everywhere."""
    providers = {key[0] for key in PRICING_TABLE}
    assert {"openai", "claude", "deepseek", "gemini"}.issubset(providers)
