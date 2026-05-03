"""Configure or probe the official MCP filesystem server for SDYJ.

Default mode prints the environment variables needed for SDYJ. Use
``--list-tools`` when Node.js, ``npx``, and the ``[mcp]`` extra are available.
"""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path

from SDYJ_Agents.tools.mcp_client import MCPClient


def build_client(root: str, tool_name: str) -> MCPClient:
    return MCPClient(
        transport="stdio",
        command="npx",
        args=["-y", "@modelcontextprotocol/server-filesystem", root],
        default_tool_name=tool_name,
    )


def print_env(root: str, tool_name: str) -> None:
    args_json = json.dumps(["-y", "@modelcontextprotocol/server-filesystem", root])
    print("MCP_TRANSPORT=stdio")
    print("MCP_COMMAND=npx")
    print(f"MCP_ARGS={args_json}")
    print(f"MCP_TOOL_NAME={tool_name}")


async def list_tools(root: str, tool_name: str) -> int:
    client = build_client(root, tool_name)
    tools = await client.list_tools()
    print(json.dumps(tools, indent=2, ensure_ascii=False))
    return 0 if tools else 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="SDYJ MCP filesystem demo")
    parser.add_argument("--root", default=".", help="Directory exposed to the MCP server")
    parser.add_argument("--tool-name", default="search", help="Default MCP tool SDYJ should call")
    parser.add_argument("--list-tools", action="store_true", help="Launch the server and list MCP tools")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = str(Path(args.root).resolve())
    if args.list_tools:
        return asyncio.run(list_tools(root, args.tool_name))
    print_env(root, args.tool_name)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
