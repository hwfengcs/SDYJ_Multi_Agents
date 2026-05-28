from SDYJ_Agents.cli.main import (
    _create_config_from_args,
    get_api_key_for_provider,
    parse_args,
    summarize_report_for_cli,
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
    eval_args = parse_args(["eval", "--max-scenarios", "1"])
    benchmark_args = parse_args(["benchmark", "run", "--fail-under", "0.7", "--determinism-repeats", "2"])
    replay_args = parse_args(["replay", "abc123"])
    diff_args = parse_args(["diff-runs", "run-a", "run-b", "--json"])
    runs_args = parse_args(["runs", "list", "--limit", "5"])

    assert models.command == "list-models"
    assert models.provider == "deepseek"
    assert info.command == "config-info"
    assert inspect.command == "inspect-run"
    assert inspect.run_id == "abc123"
    assert inspect_timeline.timeline is True
    assert inspect_timeline.event_id == "evt_000001"
    assert eval_args.command == "eval"
    assert eval_args.max_scenarios == 1
    assert benchmark_args.command == "eval"
    assert benchmark_args.fail_under == 0.7
    assert benchmark_args.determinism_repeats == 2
    assert replay_args.command == "replay"
    assert diff_args.command == "diff-runs"
    assert diff_args.json is True
    assert runs_args.command == "runs"
    assert runs_args.runs_command == "list"
    assert runs_args.limit == 5


def test_get_api_key_accepts_provider_aliases(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.setenv("CLAUDE_API_KEY", "legacy-key")

    assert get_api_key_for_provider("claude") == "legacy-key"


def test_report_summary_for_cli_is_concise():
    report = """
# 研究报告

## 执行摘要
北京大学历史悠久，是中国现代高等教育的重要代表。[E1]

## 核心发现
- 学校创办于 1898 年，早期与京师大学堂密切相关。[E1]
- 五四运动等历史事件让北大拥有鲜明的思想文化传统。[E2]

## 参考资料
- [E1] source
"""

    summary = summarize_report_for_cli(report, "markdown", limit=3)

    assert len(summary) == 3
    assert "执行摘要" not in summary[0]
    assert "[E1]" not in summary[0]
    assert "北京大学历史悠久" in summary[0]
