import pytest

from SDYJ_Agents import __version__
from SDYJ_Agents.cli.main import (
    _create_config_from_args,
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
    benchmark_args = parse_args(["benchmark", "run", "--fail-under", "0.7", "--determinism-repeats", "2"])
    replay_args = parse_args(["replay", "abc123"])
    diff_args = parse_args(["diff-runs", "run-a", "run-b", "--json"])
    runs_args = parse_args(["runs", "list", "--limit", "5"])
    doctor_args = parse_args(["doctor", "--provider", "deepseek", "--strict"])

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
    assert replay_args.command == "replay"
    assert diff_args.command == "diff-runs"
    assert diff_args.json is True
    assert runs_args.command == "runs"
    assert runs_args.runs_command == "list"
    assert runs_args.limit == 5
    assert doctor_args.command == "doctor"
    assert doctor_args.provider == "deepseek"
    assert doctor_args.strict is True


def test_get_api_key_accepts_provider_aliases(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.setenv("CLAUDE_API_KEY", "legacy-key")

    assert get_api_key_for_provider("claude") == "legacy-key"


def test_version_flag_matches_package_version(capsys):
    with pytest.raises(SystemExit) as exc_info:
        parse_args(["--version"])

    assert exc_info.value.code == 0
    assert __version__ in capsys.readouterr().out


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
