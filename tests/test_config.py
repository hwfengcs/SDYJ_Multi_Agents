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


def test_load_config_parses_mcp_sdk_options(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "deepseek")
    monkeypatch.setenv("DEEPSEEK_API_KEY", "test-key")
    monkeypatch.setenv("MCP_TRANSPORT", "stdio")
    monkeypatch.setenv("MCP_COMMAND", "python")
    monkeypatch.setenv("MCP_ARGS", '["server.py", "--root", "."]')
    monkeypatch.setenv("MCP_ENV_JSON", '{"ROOT": ".", "MODE": "test"}')
    monkeypatch.setenv("MCP_TOOL_NAME", "search_docs")
    monkeypatch.setenv("MCP_QUERY_ARG", "pattern")
    monkeypatch.setenv("MCP_TOOL_ARGS_JSON", '{"path": ".", "pattern": "{query}", "limit": 5}')

    config = load_config_from_env()

    assert config.search.mcp_transport == "stdio"
    assert config.search.mcp_command == "python"
    assert config.search.mcp_args == ["server.py", "--root", "."]
    assert config.search.mcp_env == {"ROOT": ".", "MODE": "test"}
    assert config.search.mcp_tool_name == "search_docs"
    assert config.search.mcp_query_arg == "pattern"
    assert config.search.mcp_tool_args == {"path": ".", "pattern": "{query}", "limit": 5}
