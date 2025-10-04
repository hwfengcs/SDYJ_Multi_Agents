"""
Configuration Management

This module handles configuration loading and management.
"""

import os
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

    # Map provider to API key environment variable
    api_key_map = {
        "openai": "OPENAI_API_KEY",
        "claude": "ANTHROPIC_API_KEY",
        "gemini": "GOOGLE_API_KEY",
        "deepseek": "DEEPSEEK_API_KEY"
    }

    api_key_env = api_key_map.get(llm_provider, "OPENAI_API_KEY")
    llm_api_key = os.getenv(api_key_env)

    if not llm_api_key:
        raise ValueError(f"API key not found for {llm_provider}. Please set {api_key_env} in .env file")

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
        mcp_api_key=os.getenv("MCP_API_KEY")
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
            "model": "deepseek-chat",
            "temperature": 0.7
        },
        "search": {
            "tavily_api_key": None,
            "mcp_server_url": None,
            "mcp_api_key": None
        },
        "workflow": {
            "max_iterations": 5,
            "auto_approve_plan": False,
            "output_dir": "./outputs"
        }
    }
