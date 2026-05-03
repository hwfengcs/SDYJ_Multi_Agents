# MCP Demo Scripts

These scripts demonstrate how to configure SDYJ against common MCP servers.
They do not require LLM API keys. By default they print the environment
variables you can paste into `.env`; pass `--list-tools` to launch the MCP
server and inspect its exposed tools.

## Filesystem

```bash
python examples/mcp_demos/mcp_filesystem_demo.py --root .
python examples/mcp_demos/mcp_filesystem_demo.py --root . --list-tools
```

## GitHub

```bash
python examples/mcp_demos/mcp_github_demo.py --token-env GITHUB_PERSONAL_ACCESS_TOKEN
python examples/mcp_demos/mcp_github_demo.py --list-tools
```

Both demos use `npx` and official MCP server packages, so Node.js must be
available when you pass `--list-tools`.
