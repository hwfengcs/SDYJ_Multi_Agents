"""Tests for transient-failure retry in InstrumentedLLM."""

import pytest

from SDYJ_Agents.utils.llm_retry import RetryPolicy, TransientLLMError, is_transient
from SDYJ_Agents.utils.tracing import InstrumentedLLM, create_run_trace


class FlakyLLM:
    """Fails N times with the given exception, then succeeds."""

    def __init__(self, failures, exception=None):
        self.failures = failures
        self.exception = exception or TransientLLMError("simulated timeout")
        self.calls = 0
        self.api_key = "fake"
        self.model = "flaky"
        self.last_usage = None

    def generate(self, prompt: str, **kwargs) -> str:
        self.calls += 1
        if self.calls <= self.failures:
            raise self.exception
        return "ok"

    def stream_generate(self, prompt: str, **kwargs):
        yield self.generate(prompt, **kwargs)


def _policy(max_retries=2):
    sleeps = []
    policy = RetryPolicy(max_retries=max_retries, base_delay=1.0, sleeper=sleeps.append)
    return policy, sleeps


def test_transient_error_is_retried_and_recorded_once():
    trace = create_run_trace("q", "fake", "flaky", mode="test")
    policy, sleeps = _policy()
    llm = InstrumentedLLM(FlakyLLM(failures=1), trace, retry_policy=policy)

    assert llm.generate("prompt") == "ok"
    assert len(trace["llm_calls"]) == 1  # one logical call, replay-stable
    call = trace["llm_calls"][0]
    assert call["retries"] == 1
    assert call["attempt_errors"] == ["simulated timeout"]
    assert call["error"] is None
    assert sleeps == [1.0]  # base_delay * 2**0


def test_backoff_grows_exponentially():
    trace = create_run_trace("q", "fake", "flaky", mode="test")
    policy, sleeps = _policy(max_retries=3)
    llm = InstrumentedLLM(FlakyLLM(failures=2), trace, retry_policy=policy)

    assert llm.generate("prompt") == "ok"
    assert sleeps == [1.0, 2.0]


def test_exhausted_retries_raise_with_attempts_recorded():
    trace = create_run_trace("q", "fake", "flaky", mode="test")
    policy, sleeps = _policy(max_retries=1)
    llm = InstrumentedLLM(FlakyLLM(failures=5), trace, retry_policy=policy)

    with pytest.raises(TransientLLMError):
        llm.generate("prompt")

    call = trace["llm_calls"][0]
    assert call["error"] == "simulated timeout"
    assert call["retries"] == 1
    assert len(call["attempt_errors"]) == 1
    assert trace["errors"]  # failure is observable in the trace


def test_non_transient_error_fails_fast():
    trace = create_run_trace("q", "fake", "flaky", mode="test")
    policy, sleeps = _policy()
    llm = InstrumentedLLM(
        FlakyLLM(failures=5, exception=RuntimeError("invalid request: bad api key")),
        trace,
        retry_policy=policy,
    )

    with pytest.raises(RuntimeError):
        llm.generate("prompt")

    assert sleeps == []  # no retry attempted
    assert trace["llm_calls"][0]["retries"] == 0


def test_no_policy_keeps_legacy_fail_fast_behavior():
    trace = create_run_trace("q", "fake", "flaky", mode="test")
    llm = InstrumentedLLM(FlakyLLM(failures=1), trace)

    with pytest.raises(TransientLLMError):
        llm.generate("prompt")


def test_is_transient_classification():
    assert is_transient(TransientLLMError("anything"))
    assert is_transient(RuntimeError("connection reset by peer"))
    assert is_transient(RuntimeError("HTTP 429 too many requests"))
    assert is_transient(RuntimeError("server overloaded, try again"))
    assert not is_transient(RuntimeError("401 unauthorized"))
    assert not is_transient(RuntimeError("invalid request payload"))
    assert not is_transient(ValueError("totally unrelated"))
