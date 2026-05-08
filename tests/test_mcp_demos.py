import subprocess
import sys
import os
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_filesystem_demo_prints_env_without_launching_server():
    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "examples" / "mcp_demos" / "mcp_filesystem_demo.py"),
            "--root",
            ".",
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=True,
    )

    assert "MCP_TRANSPORT=stdio" in result.stdout
    assert "@modelcontextprotocol/server-filesystem" in result.stdout
    assert "MCP_TOOL_ARGS_JSON" in result.stdout


def test_github_demo_prints_env_without_launching_server():
    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "examples" / "mcp_demos" / "mcp_github_demo.py"),
            "--token-env",
            "GITHUB_TOKEN",
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=True,
    )

    assert "MCP_TRANSPORT=stdio" in result.stdout
    assert "GITHUB_TOKEN" in result.stdout
    assert "@modelcontextprotocol/server-github" in result.stdout


def test_filesystem_demo_check_does_not_launch_server():
    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "examples" / "mcp_demos" / "mcp_filesystem_demo.py"),
            "--check",
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert '"npx"' in result.stdout
    assert '"mcp_python_sdk"' in result.stdout


def test_github_demo_check_without_token_is_no_secret_failure():
    env = os.environ.copy()
    env.pop("GITHUB_PERSONAL_ACCESS_TOKEN", None)

    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "examples" / "mcp_demos" / "mcp_github_demo.py"),
            "--token-env",
            "GITHUB_PERSONAL_ACCESS_TOKEN",
            "--check",
        ],
        cwd=ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 1
    assert '"GITHUB_PERSONAL_ACCESS_TOKEN": false' in result.stdout
    assert "ghp_" not in result.stdout
    assert "github_pat_" not in result.stdout
