"""Retry policy for transient LLM API failures.

A single transient provider error (timeout, 429, 5xx) should never kill an
otherwise-complete research run. InstrumentedLLM consults this policy and
records retry counts on the one llm_call event it emits per logical call, so
retries never change the call count that deterministic replay depends on.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from typing import Callable


class TransientLLMError(Exception):
    """An LLM failure worth retrying (timeouts, rate limits, 5xx)."""


_NON_TRANSIENT_MARKERS = (
    "api key",
    "apikey",
    "authentication",
    "unauthorized",
    "forbidden",
    "permission",
    "invalid request",
    "invalid_request",
    "not found",
    "401",
    "403",
    "404",
)

_TRANSIENT_MARKERS = (
    "timeout",
    "timed out",
    "connection",
    "temporarily unavailable",
    "service unavailable",
    "rate limit",
    "rate_limit",
    "too many requests",
    "overloaded",
    "internal server error",
    "bad gateway",
    "429",
    "500",
    "502",
    "503",
    "504",
)


def is_transient(exc: BaseException) -> bool:
    """Classify an exception as retry-worthy.

    Auth/invalid-request failures are permanent by definition and must fail
    fast; anything that looks like a network or capacity hiccup is retried.
    """
    if isinstance(exc, TransientLLMError):
        return True
    text = f"{type(exc).__name__} {exc}".lower()
    if any(marker in text for marker in _NON_TRANSIENT_MARKERS):
        return False
    return any(marker in text for marker in _TRANSIENT_MARKERS)


@dataclass
class RetryPolicy:
    """Exponential-backoff retry settings for LLM calls."""

    max_retries: int = 2
    base_delay: float = 1.0
    sleeper: Callable[[float], None] = field(default=time.sleep, repr=False)

    @classmethod
    def from_env(cls) -> "RetryPolicy":
        return cls(
            max_retries=int(os.getenv("LLM_MAX_RETRIES", "2")),
            base_delay=float(os.getenv("LLM_RETRY_BASE_DELAY", "1.0")),
        )

    def delay_for_attempt(self, attempt: int) -> float:
        return self.base_delay * (2 ** attempt)
