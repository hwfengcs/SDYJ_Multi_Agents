"""
Configuration Management

This module handles configuration loading and management.
"""

import json
import os
import shlex
from typing import Optional, Dict, Any
from dotenv import load_dotenv
from pydantic import BaseModel, Field


class LLMConfig(BaseModel):
    """LLM configuration."""
    provider: str = Field(default="deepseek", description="LLM provider")
    model: Optional[str] = Field(default=None, description="Model name")
    api_key: str = Field(..., description="API key")
    temperature: float = Field(default=0.7, description="Temperature")
    max_tokens: Optional[int] = Field(default=None, description="Max tokens")


class SearchConfig(BaseModel):
    """Search tools configuration."""
    tavily_api_key: Optional[str] = Field(default=None, description="Tavily API key")
    mcp_server_url: Optional[str] = Field(default=None, description="MCP server URL")
    mcp_api_key: Optional[str] = Field(default=None, description="MCP API key")
    mcp_transport: Optional[str] = Field(default=None, description="MCP transport: legacy_http, streamable_http, or stdio")
    mcp_tool_name: str = Field(default="web_search", description="Default MCP search tool name")
    mcp_config_path: Optional[str] = Field(default=None, description="Path to Claude-style MCP config JSON")
    mcp_server_name: Optional[str] = Field(default=None, description="MCP server name inside config JSON")
    mcp_command: Optional[str] = Field(default=None, description="MCP stdio command")
    mcp_args: list[str] = Field(default_factory=list, description="MCP stdio command args")
    mcp_env: Dict[str, str] = Field(default_factory=dict, description="MCP stdio environment overrides")


class WorkflowConfig(BaseModel):
    """Workflow configuration."""
    max_iterations: int = Field(default=5, description="Maximum research iterations")
    auto_approve_plan: bool = Field(default=False, description="Auto-approve research plan")
    output_dir: str = Field(default="./outputs", description="Output directory for reports")


class Config(BaseModel):
    """Main configuration."""
    llm: LLMConfig
    search: SearchConfig
    workflow: WorkflowConfig


def _get_env_with_aliases(*names: str) -> Optional[str]:
    """Return the first non-empty environment variable value from aliases."""
    for name in names:
        value = os.getenv(name)
        if value:
            return value
    return None


def _parse_env_args(value: Optional[str]) -> list[str]:
    """Parse MCP_ARGS from JSON list or shell-style text."""
    if not value:
        return []
    try:
        parsed = json.loads(value)
        if isinstance(parsed, list):
            return [str(item) for item in parsed]
    except json.JSONDecodeError:
        pass
    return shlex.split(value)


def _parse_env_json_object(value: Optional[str]) -> Dict[str, str]:
    """Parse MCP_ENV_JSON into a string-to-string dict."""
    if not value:
        return {}
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError:
        return {}
    if not isinstance(parsed, dict):
        return {}
    return {str(key): str(item) for key, item in parsed.items()}


def load_config_from_env() -> Config:
    """
    Load configuration from environment variables.

    Returns:
        Config instance
    """
    # Load .env file
    load_dotenv()

    # Get LLM provider and determine API key
    llm_provider = os.getenv("LLM_PROVIDER", "deepseek").lower()

    # Map provider to API key environment variables. The first name is the
    # canonical one documented by the provider; later names are compatibility
    # aliases kept for older local .env files.
    api_key_map = {
        "openai": ("OPENAI_API_KEY",),
        "claude": ("ANTHROPIC_API_KEY", "CLAUDE_API_KEY"),
        "gemini": ("GOOGLE_API_KEY", "GEMINI_API_KEY"),
        "deepseek": ("DEEPSEEK_API_KEY",)
    }

    api_key_envs = api_key_map.get(llm_provider, ("OPENAI_API_KEY",))
    llm_api_key = _get_env_with_aliases(*api_key_envs)

    if not llm_api_key:
        expected = " or ".join(api_key_envs)
        raise ValueError(
            f"API key not found for {llm_provider}. Please set {expected} in .env file"
        )

    # Create LLM config
    llm_config = LLMConfig(
        provider=llm_provider,
        model=os.getenv("LLM_MODEL"),
        api_key=llm_api_key,
        temperature=float(os.getenv("LLM_TEMPERATURE", "0.7")),
        max_tokens=int(os.getenv("LLM_MAX_TOKENS")) if os.getenv("LLM_MAX_TOKENS") else None
    )

    # Create search config
    search_config = SearchConfig(
        tavily_api_key=os.getenv("TAVILY_API_KEY"),
        mcp_server_url=os.getenv("MCP_SERVER_URL"),
        mcp_api_key=os.getenv("MCP_API_KEY"),
        mcp_transport=os.getenv("MCP_TRANSPORT"),
        mcp_tool_name=os.getenv("MCP_TOOL_NAME", "web_search"),
        mcp_config_path=os.getenv("MCP_CONFIG_PATH"),
        mcp_server_name=os.getenv("MCP_SERVER_NAME"),
        mcp_command=os.getenv("MCP_COMMAND"),
        mcp_args=_parse_env_args(os.getenv("MCP_ARGS")),
        mcp_env=_parse_env_json_object(os.getenv("MCP_ENV_JSON")),
    )

    # Create workflow config
    workflow_config = WorkflowConfig(
        max_iterations=int(os.getenv("MAX_ITERATIONS", "5")),
        auto_approve_plan=os.getenv("AUTO_APPROVE_PLAN", "false").lower() == "true",
        output_dir=os.getenv("OUTPUT_DIR", "./outputs")
    )

    return Config(
        llm=llm_config,
        search=search_config,
        workflow=workflow_config
    )


def save_config_to_file(config: Config, filepath: str) -> bool:
    """
    Save configuration to a file.

    Args:
        config: Configuration instance
        filepath: Path to save the config

    Returns:
        True if successful
    """
    try:
        with open(filepath, 'w') as f:
            f.write(config.model_dump_json(indent=2))
        return True
    except Exception as e:
        print(f"Error saving config: {e}")
        return False


def load_config_from_file(filepath: str) -> Config:
    """
    Load configuration from a file.

    Args:
        filepath: Path to the config file

    Returns:
        Config instance
    """
    import json

    with open(filepath, 'r') as f:
        data = json.load(f)

    return Config(**data)


def get_default_config() -> Dict[str, Any]:
    """
    Get default configuration values.

    Returns:
        Dictionary of default config
    """
    return {
        "llm": {
            "provider": "deepseek",
            "model": "deepseek-v4-flash",
            "temperature": 0.7
        },
        "search": {
            "tavily_api_key": None,
            "mcp_server_url": None,
            "mcp_api_key": None,
            "mcp_transport": None,
            "mcp_tool_name": "web_search",
            "mcp_config_path": None,
            "mcp_server_name": None,
            "mcp_command": None,
            "mcp_args": [],
            "mcp_env": {},
        },
        "workflow": {
            "max_iterations": 5,
            "auto_approve_plan": False,
            "output_dir": "./outputs"
        }
    }
