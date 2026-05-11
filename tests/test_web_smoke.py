"""Smoke tests for the Streamlit Web UI.

These tests run the app once with ``AppTest`` (Streamlit's built-in headless
test runner) so they catch import-time errors, page-config crashes, and the
empty-state render path. They deliberately do not trigger a real workflow run
because that needs API keys and would be flaky.

The tests are marked as optional via the import skip below so the core CI
matrix does not need Streamlit installed unless the ``[web]`` extra is used.
"""

from __future__ import annotations

from pathlib import Path

import pytest

streamlit_testing = pytest.importorskip("streamlit.testing.v1")


def _run_streamlit_app(app_path: Path):
    AppTest = streamlit_testing.AppTest  # noqa: N806 - Streamlit ships PascalCase
    at = AppTest.from_file(str(app_path))
    at.run(timeout=30)
    assert not at.exception, [str(e) for e in at.exception]
    return at


@pytest.fixture(autouse=True)
def _scrub_provider_env(monkeypatch):
    """Remove provider keys and stub ``load_dotenv``.

    This keeps the user's real ``.env`` from leaking keys back in during the
    test.
    """
    for var in (
        "DEEPSEEK_API_KEY",
        "OPENAI_API_KEY",
        "ANTHROPIC_API_KEY",
        "CLAUDE_API_KEY",
        "GOOGLE_API_KEY",
        "GEMINI_API_KEY",
        "TAVILY_API_KEY",
    ):
        monkeypatch.delenv(var, raising=False)
    # The Streamlit script imports ``load_dotenv`` and calls it at the top of
    # ``main``. We replace the bound name so the on-disk ``.env`` is ignored.
    monkeypatch.setattr("SDYJ_Agents.web.app.load_dotenv", lambda *a, **kw: None)


def test_app_runs_without_errors():
    app_path = Path(__file__).resolve().parents[1] / "SDYJ_Agents" / "web" / "app.py"
    _run_streamlit_app(app_path)


def test_huggingface_root_app_runs_without_errors():
    app_path = Path(__file__).resolve().parents[1] / "app.py"
    _run_streamlit_app(app_path)


@pytest.mark.xfail(
    reason=(
        "Streamlit AppTest's sidebar element traversal varies between "
        "releases; the missing-key warning renders correctly when the app is "
        "served, but the headless test framework does not expose with-sidebar "
        "widgets uniformly. Tracked for v0.6 follow-up."
    ),
    strict=False,
)
def test_app_shows_missing_key_warning_when_env_unset():
    """When no provider API key is set, the sidebar should name it."""
    app_path = Path(__file__).resolve().parents[1] / "SDYJ_Agents" / "web" / "app.py"
    at = _run_streamlit_app(app_path)
    sidebar_errors = [str(e.value) for e in at.sidebar.error]
    assert any("DEEPSEEK_API_KEY" in msg for msg in sidebar_errors), sidebar_errors


def test_app_seed_query_via_example_button():
    """Clicking an example chip should populate the query text area."""
    app_path = Path(__file__).resolve().parents[1] / "SDYJ_Agents" / "web" / "app.py"
    at = _run_streamlit_app(app_path)

    example_buttons = [b for b in at.button if b.key and b.key.startswith("example_")]
    assert example_buttons, "expected at least one example chip in the UI"
    example_buttons[0].click()
    at.run(timeout=30)
    assert not at.exception, [str(e) for e in at.exception]
    assert "query" in at.session_state, "example button did not seed the query box"
    assert at.session_state["query"], "example button seeded an empty query"
