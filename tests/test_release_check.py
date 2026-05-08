from pathlib import Path

from SDYJ_Agents.release_check import (
    CommandExecution,
    build_release_gates,
    classify_secret_state,
    collect_release_readiness,
    release_readiness_exit_code,
)


def _passing_runner(command, cwd, timeout_seconds, env):
    return CommandExecution(0, "ok", 0.01)


def test_release_check_plan_includes_core_gates():
    gates = build_release_gates(output_dir="outputs/release_readiness")
    names = [gate.name for gate in gates]

    assert "doctor" in names
    assert "pytest" in names
    assert "ruff" in names
    assert "external benchmark smoke" in names
    assert "offline benchmark gate" in names
    assert "MCP filesystem check" in names
    assert "build" in names
    assert "twine check" in names


def test_secret_state_classifier_never_needs_secret_value():
    assert classify_secret_state(None) == "missing"
    assert classify_secret_state("") == "missing"
    assert classify_secret_state("your_tavily_key") == "placeholder"
    assert classify_secret_state("<TAVILY_API_KEY>") == "placeholder"
    assert classify_secret_state("tvly-real-looking-token") == "usable"


def test_missing_github_mcp_token_is_blocked_not_failed(tmp_path, monkeypatch):
    monkeypatch.delenv("GITHUB_PERSONAL_ACCESS_TOKEN", raising=False)
    monkeypatch.delenv("TAVILY_API_KEY", raising=False)

    results = collect_release_readiness(
        include_build=False,
        include_benchmark=False,
        include_mcp=True,
        include_external_prereqs=False,
        repo_root=tmp_path,
        command_runner=_passing_runner,
    )

    github = next(result for result in results if result.name == "MCP GitHub check")
    assert github.status == "BLOCKED"
    assert github.required is False
    assert "GITHUB_PERSONAL_ACCESS_TOKEN=missing" in github.detail
    assert release_readiness_exit_code(results) == 0


def test_required_gate_failure_controls_exit_code(tmp_path):
    def runner(command, cwd, timeout_seconds, env):
        if "pytest" in command:
            return CommandExecution(1, "tests failed", 0.01)
        return CommandExecution(0, "ok", 0.01)

    results = collect_release_readiness(
        include_build=False,
        include_benchmark=False,
        include_mcp=False,
        include_external_prereqs=False,
        repo_root=tmp_path,
        command_runner=runner,
    )

    pytest_result = next(result for result in results if result.name == "pytest")
    assert pytest_result.status == "FAIL"
    assert release_readiness_exit_code(results) == 1


def test_script_wrapper_is_packaged():
    assert (Path("scripts") / "release_readiness.py").is_file()
