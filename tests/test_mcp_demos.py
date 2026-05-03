import subprocess
import sys
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
