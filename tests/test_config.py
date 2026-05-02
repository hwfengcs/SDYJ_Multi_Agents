from SDYJ_Agents.utils.config import load_config_from_env


def test_load_config_accepts_canonical_deepseek_key(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "deepseek")
    monkeypatch.setenv("DEEPSEEK_API_KEY", "test-key")
    monkeypatch.setenv("MAX_ITERATIONS", "3")

    config = load_config_from_env()

    assert config.llm.provider == "deepseek"
    assert config.llm.api_key == "test-key"
    assert config.workflow.max_iterations == 3


def test_load_config_accepts_legacy_claude_alias(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "claude")
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.setenv("CLAUDE_API_KEY", "legacy-claude-key")

    config = load_config_from_env()

    assert config.llm.provider == "claude"
    assert config.llm.api_key == "legacy-claude-key"
