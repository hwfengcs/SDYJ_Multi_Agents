"""Configure or probe the official MCP GitHub server for SDYJ.

Default mode prints the environment variables needed for SDYJ. Use
``--list-tools`` when Node.js, ``npx``, and the ``[mcp]`` extra are available.
"""

from __future__ import annotations

import argparse
import asyncio
import importlib.util
import json
import os
import shutil
import sys

from SDYJ_Agents.tools.mcp_client import MCPClient


def build_client(token_env: str, tool_name: str) -> MCPClient:
    token = os.getenv(token_env, "")
    return MCPClient(
        transport="stdio",
        command="npx",
        args=["-y", "@modelcontextprotocol/server-github"],
        env={token_env: token} if token else {},
        default_tool_name=tool_name,
    )


def print_env(token_env: str, tool_name: str) -> None:
    args_json = json.dumps(["-y", "@modelcontextprotocol/server-github"])
    env_json = json.dumps({token_env: f"${token_env}"})
    print("MCP_TRANSPORT=stdio")
    print("MCP_COMMAND=npx")
    print(f"MCP_ARGS={args_json}")
    print(f"MCP_ENV_JSON={env_json}")
    print(f"MCP_TOOL_NAME={tool_name}")


def dependency_status(token_env: str) -> dict[str, object]:
    return {
        "npx": bool(shutil.which("npx")),
        "mcp_python_sdk": importlib.util.find_spec("mcp") is not None,
        token_env: bool(os.getenv(token_env)),
    }


def print_check(token_env: str) -> int:
    status = dependency_status(token_env)
    print(json.dumps(status, indent=2))
    required = ["npx", "mcp_python_sdk", token_env]
    return 0 if all(status[item] for item in required) else 1


async def list_tools(token_env: str, tool_name: str) -> int:
    client = build_client(token_env, tool_name)
    tools = await client.list_tools()
    print(json.dumps(tools, indent=2, ensure_ascii=False))
    if not tools:
        print(
            "No tools returned. Check Node.js/npx, the mcp Python extra, network access, "
            f"and the {token_env} environment variable.",
            file=sys.stderr,
        )
        if client.last_error:
            print(f"MCP error: {client.last_error}", file=sys.stderr)
    return 0 if tools else 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="SDYJ MCP GitHub demo")
    parser.add_argument(
        "--token-env",
        default="GITHUB_PERSONAL_ACCESS_TOKEN",
        help="Environment variable holding the GitHub token",
    )
    parser.add_argument("--tool-name", default="search_repositories", help="Default MCP tool SDYJ should call")
    parser.add_argument("--check", action="store_true", help="Check local dependencies without launching the server")
    parser.add_argument("--dry-run", action="store_true", help="Print SDYJ .env values without launching the server")
    parser.add_argument("--list-tools", action="store_true", help="Launch the server and list MCP tools")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.check:
        return print_check(args.token_env)
    if args.list_tools:
        return asyncio.run(list_tools(args.token_env, args.tool_name))
    print_env(args.token_env, args.tool_name)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
