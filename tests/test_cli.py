import json

import pytest

from SDYJ_Agents import __version__
from SDYJ_Agents.cli.main import (
    _create_config_from_args,
    _configure_stream_for_safe_console,
    collect_doctor_checks,
    get_api_key_for_provider,
    parse_args,
)


def test_parse_research_subcommand():
    args = parse_args(["research", "agent eval", "--max-iterations", "2", "--auto-approve"])
    config = _create_config_from_args(args)

    assert args.command == "research"
    assert args.query == "agent eval"
    assert config.max_iterations == 2
    assert config.auto_approve is True


def test_parse_direct_query_backwards_compatible():
    args = parse_args(["agent eval"])

    assert args.command == "research"
    assert args.query == "agent eval"


def test_parse_utility_commands_without_api_key():
    models = parse_args(["list-models", "deepseek"])
    info = parse_args(["config-info"])
    inspect = parse_args(["inspect-run", "abc123"])
    inspect_timeline = parse_args(["inspect-run", "abc123", "--timeline", "--event", "evt_000001"])
    eval_args = parse_args([
        "eval",
        "--max-scenarios",
        "1",
        "--enable-refine-plan",
        "--enable-parallel-tools",
    ])
    benchmark_args = parse_args(
        [
            "benchmark",
            "run",
            "--fail-under",
            "0.7",
            "--determinism-repeats",
            "2",
            "--compare-summary",
            "baseline.json",
        ]
    )
    benchmark_external = parse_args(["benchmark", "external", "--suite", "gaia", "--limit", "2"])
    benchmark_compare = parse_args(["benchmark", "compare", "baseline.json", "candidate.json", "--json"])
    replay_args = parse_args(["replay", "abc123"])
    diff_args = parse_args(["diff-runs", "run-a", "run-b", "--json"])
    runs_args = parse_args(["runs", "list", "--limit", "5"])
    doctor_args = parse_args(["doctor", "--provider", "deepseek", "--strict"])
    release_check_args = parse_args(["release-check", "--skip-build", "--dry-run"])

    assert models.command == "list-models"
    assert models.provider == "deepseek"
    assert info.command == "config-info"
    assert inspect.command == "inspect-run"
    assert inspect.run_id == "abc123"
    assert inspect_timeline.timeline is True
    assert inspect_timeline.event_id == "evt_000001"
    assert eval_args.command == "eval"
    assert eval_args.max_scenarios == 1
    assert eval_args.enable_refine_plan is True
    assert eval_args.enable_parallel_tools is True
    assert benchmark_args.command == "eval"
    assert benchmark_args.fail_under == 0.7
    assert benchmark_args.determinism_repeats == 2
    assert benchmark_args.compare_summary == "baseline.json"
    assert benchmark_external.command == "benchmark-external"
    assert benchmark_external.suite == "gaia"
    assert benchmark_external.limit == 2
    assert benchmark_compare.command == "benchmark-compare"
    assert benchmark_compare.json is True
    assert replay_args.command == "replay"
    assert diff_args.command == "diff-runs"
    assert diff_args.json is True
    assert runs_args.command == "runs"
    assert runs_args.runs_command == "list"
    assert runs_args.limit == 5
    assert doctor_args.command == "doctor"
    assert doctor_args.provider == "deepseek"
    assert doctor_args.strict is True
    assert release_check_args.command == "release-check"
    assert release_check_args.skip_build is True
    assert release_check_args.dry_run is True


def test_get_api_key_accepts_provider_aliases(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.setenv("CLAUDE_API_KEY", "legacy-key")

    assert get_api_key_for_provider("claude") == "legacy-key"


def test_version_flag_matches_package_version(capsys):
    with pytest.raises(SystemExit) as exc_info:
        parse_args(["--version"])

    assert exc_info.value.code == 0
    assert __version__ in capsys.readouterr().out


def test_configure_stream_uses_replacement_errors():
    class DummyStream:
        def __init__(self):
            self.kwargs = None

        def reconfigure(self, **kwargs):
            self.kwargs = kwargs

    stream = DummyStream()

    _configure_stream_for_safe_console(stream)

    assert stream.kwargs == {"errors": "replace"}


def test_doctor_reports_missing_required_api_key(monkeypatch):
    monkeypatch.setattr("SDYJ_Agents.cli.main.load_dotenv", lambda *a, **kw: None)
    for var in (
        "DEEPSEEK_API_KEY",
        "TAVILY_API_KEY",
        "MCP_SERVER_URL",
        "MCP_CONFIG_PATH",
        "MCP_COMMAND",
        "MCP_TRANSPORT",
    ):
        monkeypatch.delenv(var, raising=False)

    checks = collect_doctor_checks(provider="deepseek")

    llm_check = next(check for check in checks if check.name == "LLM API key")
    mcp_check = next(check for check in checks if check.name == "MCP config")
    assert llm_check.status == "FAIL"
    assert llm_check.required is True
    assert mcp_check.status == "WARN"


def test_benchmark_external_cli_writes_artifacts(tmp_path, monkeypatch):
    from SDYJ_Agents.cli import main as cli_main

    monkeypatch.setattr(cli_main, "load_dotenv", lambda *a, **kw: None)

    exit_code = cli_main.main(
        [
            "benchmark",
            "external",
            "--suite",
            "gaia",
            "--source",
            "local",
            "--limit",
            "1",
            "--output-dir",
            str(tmp_path),
            "--fail-under",
            "1.0",
        ]
    )

    assert exit_code == 0
    summaries = list((tmp_path / "external_benchmarks").glob("*/summary.json"))
    predictions = list((tmp_path / "external_benchmarks").glob("*/predictions.jsonl"))
    failure_analysis = list((tmp_path / "external_benchmarks").glob("*/failure_analysis.json"))
    assert len(summaries) == 1
    assert len(predictions) == 1
    assert len(failure_analysis) == 1


def test_benchmark_compare_cli_flags_metric_regression(tmp_path):
    from SDYJ_Agents.cli import main as cli_main

    baseline = tmp_path / "baseline.json"
    candidate = tmp_path / "candidate.json"
    baseline.write_text(
        json.dumps(
            {
                "average_score": 1.0,
                "results": [
                    {
                        "scenario_id": "agent_reliability_hard",
                        "metrics": {"overall_score": 1.0, "citation_id_coverage": 1.0},
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    candidate.write_text(
        json.dumps(
            {
                "average_score": 1.0,
                "results": [
                    {
                        "scenario_id": "agent_reliability_hard",
                        "metrics": {"overall_score": 1.0, "citation_id_coverage": 0.5},
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    exit_code = cli_main.main(["benchmark", "compare", str(baseline), str(candidate), "--json"])

    assert exit_code == 3


def test_benchmark_compare_cli_payload_includes_regression_analysis(tmp_path, capsys):
    from SDYJ_Agents.cli import main as cli_main

    baseline = tmp_path / "baseline.json"
    candidate = tmp_path / "candidate.json"
    baseline.write_text(
        json.dumps(
            {
                "results": [
                    {
                        "scenario_id": "agent_reliability_hard",
                        "metrics": {"overall_score": 1.0, "trace_completeness": 1.0},
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    candidate.write_text(json.dumps({"results": []}), encoding="utf-8")

    exit_code = cli_main.main(["benchmark", "compare", str(baseline), str(candidate), "--json"])
    output = capsys.readouterr().out
    payload = json.loads(output)

    assert exit_code == 3
    assert payload["missing_scenario_count"] == 1
    assert payload["regression_analysis"]["root_cause_counts"] == {"missing_scenario": 1}


def test_release_check_cli_dry_run_does_not_require_api_key(monkeypatch):
    from SDYJ_Agents.cli import main as cli_main

    monkeypatch.setattr(cli_main, "load_dotenv", lambda *a, **kw: None)
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)

    exit_code = cli_main.main(
        [
            "release-check",
            "--dry-run",
            "--skip-build",
            "--skip-benchmark",
            "--skip-mcp",
            "--skip-external-prereqs",
        ]
    )

    assert exit_code == 0
