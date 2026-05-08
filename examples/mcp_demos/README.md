# MCP Demo Scripts

These scripts demonstrate how to configure SDYJ against common MCP servers.
They do not require LLM API keys. By default they print the environment
variables you can paste into `.env`; pass `--list-tools` to launch the MCP
server and inspect its exposed tools.

## Filesystem

```bash
python examples/mcp_demos/mcp_filesystem_demo.py --root .
python examples/mcp_demos/mcp_filesystem_demo.py --root . --check
python examples/mcp_demos/mcp_filesystem_demo.py --root . --list-tools
```

The filesystem demo defaults to `search_files` and prints
`MCP_TOOL_ARGS_JSON={"path":"...","pattern":"{query}"}` so SDYJ can call the
official server's tool schema rather than assuming every MCP tool takes
`{"query": "..."}`.

## GitHub

```bash
python examples/mcp_demos/mcp_github_demo.py --token-env GITHUB_PERSONAL_ACCESS_TOKEN
python examples/mcp_demos/mcp_github_demo.py --token-env GITHUB_PERSONAL_ACCESS_TOKEN --check
python examples/mcp_demos/mcp_github_demo.py --list-tools
```

Both demos use `npx` and official MCP server packages, so Node.js must be
available when you pass `--list-tools`. `--check` only verifies local
prerequisites and does not launch a server.

## No-secret check matrix

Run `--check` before `--list-tools`:

| Demo | Command | Expected without secrets |
| --- | --- | --- |
| Filesystem | `python examples/mcp_demos/mcp_filesystem_demo.py --root . --check` | Exits `0` when `npx` and the Python `mcp` package are available; prints only booleans. |
| GitHub | `python examples/mcp_demos/mcp_github_demo.py --token-env GITHUB_PERSONAL_ACCESS_TOKEN --check` | Exits non-zero until the token environment variable is present; prints only booleans such as `"GITHUB_PERSONAL_ACCESS_TOKEN": false`. |

Do not run GitHub `--list-tools` until `--check` is green. The scripts never
print token values; they only report whether the configured token variable is
present.
