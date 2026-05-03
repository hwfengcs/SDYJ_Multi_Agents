"""Configure or probe the official MCP filesystem server for SDYJ.

Default mode prints the environment variables needed for SDYJ. Use
``--list-tools`` when Node.js, ``npx``, and the ``[mcp]`` extra are available.
"""

from __future__ import annotations

import argparse
import asyncio
import importlib.util
import json
import shutil
import sys
from pathlib import Path

from SDYJ_Agents.tools.mcp_client import MCPClient


def build_client(root: str, tool_name: str) -> MCPClient:
    return MCPClient(
        transport="stdio",
        command="npx",
        args=["-y", "@modelcontextprotocol/server-filesystem", root],
        default_tool_name=tool_name,
        tool_arguments={"path": root, "pattern": "{query}"},
    )


def print_env(root: str, tool_name: str) -> None:
    args_json = json.dumps(["-y", "@modelcontextprotocol/server-filesystem", root])
    tool_args_json = json.dumps({"path": root, "pattern": "{query}"})
    print("MCP_TRANSPORT=stdio")
    print("MCP_COMMAND=npx")
    print(f"MCP_ARGS={args_json}")
    print(f"MCP_TOOL_NAME={tool_name}")
    print(f"MCP_TOOL_ARGS_JSON={tool_args_json}")


def dependency_status() -> dict[str, object]:
    return {
        "npx": bool(shutil.which("npx")),
        "mcp_python_sdk": importlib.util.find_spec("mcp") is not None,
    }


def print_check() -> int:
    status = dependency_status()
    print(json.dumps(status, indent=2))
    return 0 if all(status.values()) else 1


async def list_tools(root: str, tool_name: str) -> int:
    client = build_client(root, tool_name)
    tools = await client.list_tools()
    print(json.dumps(tools, indent=2, ensure_ascii=False))
    if not tools:
        print(
            "No tools returned. Check Node.js/npx, the mcp Python extra, network access, "
            "and whether @modelcontextprotocol/server-filesystem can be installed.",
            file=sys.stderr,
        )
        if client.last_error:
            print(f"MCP error: {client.last_error}", file=sys.stderr)
    return 0 if tools else 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="SDYJ MCP filesystem demo")
    parser.add_argument("--root", default=".", help="Directory exposed to the MCP server")
    parser.add_argument("--tool-name", default="search_files", help="Default MCP tool SDYJ should call")
    parser.add_argument("--check", action="store_true", help="Check local dependencies without launching the server")
    parser.add_argument("--dry-run", action="store_true", help="Print SDYJ .env values without launching the server")
    parser.add_argument("--list-tools", action="store_true", help="Launch the server and list MCP tools")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = str(Path(args.root).resolve())
    if args.check:
        return print_check()
    if args.list_tools:
        return asyncio.run(list_tools(root, args.tool_name))
    print_env(root, args.tool_name)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
