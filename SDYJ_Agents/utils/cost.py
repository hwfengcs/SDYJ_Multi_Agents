"""Cost estimation for LLM calls.

Per-million-token pricing in USD, hard-coded for transparency. When provider
prices change, edit ``PRICING_TABLE`` directly and add a release note. The
helper deliberately returns ``None`` for unknown (provider, model) combinations
so callers can render an "unknown" cell rather than mislead users with a
fabricated number.

The trace pipeline (``SDYJ_Agents.utils.tracing.InstrumentedLLM``) feeds raw
provider ``usage`` dicts into ``normalize_usage`` and then ``estimate_call_cost_usd``,
so the same numbers show up in ``trace.metrics`` and the CLI ``inspect-run``
output.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple


# (provider, model) -> (input USD per 1M tokens, output USD per 1M tokens)
# Update with each pricing change. Sources are noted in the comments.
PRICING_TABLE: Dict[Tuple[str, str], Tuple[float, float]] = {
    # OpenAI - https://openai.com/api/pricing/
    ("openai", "gpt-4o"): (2.50, 10.00),
    ("openai", "gpt-4o-mini"): (0.15, 0.60),
    ("openai", "gpt-4.1"): (2.00, 8.00),
    ("openai", "gpt-4.1-mini"): (0.40, 1.60),
    ("openai", "o1-mini"): (1.10, 4.40),
    ("openai", "o3-mini"): (1.10, 4.40),
    # Anthropic - https://www.anthropic.com/pricing
    ("claude", "claude-3-5-sonnet-20241022"): (3.00, 15.00),
    ("claude", "claude-3-5-haiku-20241022"): (0.80, 4.00),
    ("claude", "claude-3-opus-20240229"): (15.00, 75.00),
    ("claude", "claude-3-haiku-20240307"): (0.25, 1.25),
    # DeepSeek - https://api-docs.deepseek.com/quick_start/pricing
    ("deepseek", "deepseek-v4-flash"): (0.07, 1.10),
    ("deepseek", "deepseek-v4-pro"): (0.27, 1.10),
    ("deepseek", "deepseek-chat"): (0.14, 0.28),
    ("deepseek", "deepseek-reasoner"): (0.55, 2.19),
    # Google - https://ai.google.dev/pricing
    ("gemini", "gemini-1.5-pro"): (1.25, 5.00),
    ("gemini", "gemini-1.5-flash"): (0.075, 0.30),
    ("gemini", "gemini-2.0-flash"): (0.10, 0.40),
    ("gemini", "gemini-pro"): (0.50, 1.50),
}


def normalize_usage(usage: Optional[Dict[str, Any]]) -> Tuple[int, int]:
    """Convert provider-specific ``usage`` dicts to ``(prompt_tokens, completion_tokens)``.

    Supports OpenAI/DeepSeek (``prompt_tokens`` / ``completion_tokens``),
    Anthropic (``input_tokens`` / ``output_tokens``), and Gemini
    (``prompt_token_count`` / ``candidates_token_count``).

    Returns ``(0, 0)`` when usage is missing — the caller must treat that as
    "unknown" rather than "free".
    """
    if not usage:
        return 0, 0
    if "prompt_tokens" in usage:
        return int(usage.get("prompt_tokens") or 0), int(usage.get("completion_tokens") or 0)
    if "input_tokens" in usage:
        return int(usage.get("input_tokens") or 0), int(usage.get("output_tokens") or 0)
    if "prompt_token_count" in usage:
        return (
            int(usage.get("prompt_token_count") or 0),
            int(usage.get("candidates_token_count") or 0),
        )
    return 0, 0


def estimate_call_cost_usd(
    provider: Optional[str],
    model: Optional[str],
    prompt_tokens: int,
    completion_tokens: int,
) -> Optional[float]:
    """Return USD cost for one call, or ``None`` if pricing is unknown.

    A ``None`` return is intentional and meaningful: it tells the caller the
    estimate is not trustworthy. Aggregators should skip ``None`` values rather
    than treat them as zero.
    """
    if not provider or not model:
        return None
    key = (provider.lower(), model.lower())
    pricing = PRICING_TABLE.get(key)
    if pricing is None:
        return None
    input_per_million, output_per_million = pricing
    cost = (
        prompt_tokens * input_per_million / 1_000_000
        + completion_tokens * output_per_million / 1_000_000
    )
    # 6 decimals = $0.000001 resolution, plenty for a single call estimate.
    return round(cost, 6)


def aggregate_trace_cost(trace: Dict[str, Any]) -> Dict[str, Any]:
    """Sum token counts and dollar cost across all LLM calls in a trace.

    Returns a small summary dict suitable for ``trace.metrics``::

        {
          "total_prompt_tokens": int,
          "total_completion_tokens": int,
          "total_tokens": int,
          "total_cost_usd": float | None,
          "priced_call_count": int,
          "unpriced_call_count": int,
        }

    ``total_cost_usd`` is ``None`` when *every* call has unknown pricing — that
    way callers can render a clear "unknown" instead of "$0.00".
    """
    total_prompt = 0
    total_completion = 0
    total_cost = 0.0
    priced = 0
    unpriced = 0
    for call in trace.get("llm_calls", []) or []:
        prompt_tokens = int(call.get("prompt_tokens_actual") or 0)
        completion_tokens = int(call.get("completion_tokens_actual") or 0)
        total_prompt += prompt_tokens
        total_completion += completion_tokens
        cost = call.get("cost_usd")
        if isinstance(cost, (int, float)):
            total_cost += float(cost)
            priced += 1
        else:
            unpriced += 1

    return {
        "total_prompt_tokens": total_prompt,
        "total_completion_tokens": total_completion,
        "total_tokens": total_prompt + total_completion,
        "total_cost_usd": round(total_cost, 6) if priced else None,
        "priced_call_count": priced,
        "unpriced_call_count": unpriced,
    }
