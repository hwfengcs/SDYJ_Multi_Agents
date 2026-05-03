# MCP Integration

SDYJ supports two MCP paths:

- `legacy_http`: the original compatibility adapter that calls
  `POST /tools/<tool_name>` and `GET /tools`.
- `stdio` and `streamable_http`: official MCP Python SDK transports. Install
  them with `pip install -e .[mcp]` or use the Conda environment.

The Researcher treats MCP as another retrieval source named `mcp`. The default
tool is `web_search`, but you can change it with `MCP_TOOL_NAME`.
If a tool does not use a plain `query` argument, set `MCP_TOOL_ARGS_JSON` with
`{query}` placeholders.

## Stdio Server

Use a Claude-style config file:

```bash
copy mcp_config.json.example mcp_config.json
```

Then set:

```bash
MCP_TRANSPORT=stdio
MCP_CONFIG_PATH=./mcp_config.json
MCP_SERVER_NAME=filesystem
MCP_TOOL_NAME=search_files
MCP_TOOL_ARGS_JSON={"path":".","pattern":"{query}"}
```

You can also configure a stdio server directly:

```bash
MCP_TRANSPORT=stdio
MCP_COMMAND=npx
MCP_ARGS="-y @modelcontextprotocol/server-filesystem ."
MCP_TOOL_NAME=search_files
MCP_TOOL_ARGS_JSON={"path":".","pattern":"{query}"}
```

`MCP_ARGS` accepts either shell-style text or a JSON array:

```bash
MCP_ARGS=["-y","@modelcontextprotocol/server-filesystem","."]
```

The repository also ships two helper demos:

```bash
python examples/mcp_demos/mcp_filesystem_demo.py --root .
python examples/mcp_demos/mcp_filesystem_demo.py --root . --check
python examples/mcp_demos/mcp_github_demo.py --token-env GITHUB_PERSONAL_ACCESS_TOKEN
python examples/mcp_demos/mcp_github_demo.py --token-env GITHUB_PERSONAL_ACCESS_TOKEN --check
```

Pass `--list-tools` to either script to launch the server through `npx` and
print the tools exposed to SDYJ.

`--check` verifies local prerequisites without launching a server. For the
filesystem server, the demo prints:

```bash
MCP_TRANSPORT=stdio
MCP_COMMAND=npx
MCP_ARGS=["-y", "@modelcontextprotocol/server-filesystem", "..."]
MCP_TOOL_NAME=search_files
MCP_TOOL_ARGS_JSON={"path": "...", "pattern": "{query}"}
```

Verified local filesystem run on 2026-05-03:

```bash
python examples/mcp_demos/mcp_filesystem_demo.py --root . --list-tools
```

The official filesystem server returned tools including `read_text_file`,
`list_directory`, `directory_tree`, and `search_files`. A direct SDK-backed
`MCPClient.search("README.md")` call through `search_files` returned one
normalized result for the repository README.

GitHub MCP verification is still blocked locally until
`GITHUB_PERSONAL_ACCESS_TOKEN` is available.

## Streamable HTTP Server

For an SDK-backed HTTP MCP server:

```bash
MCP_TRANSPORT=streamable_http
MCP_SERVER_URL=http://localhost:8000/mcp
MCP_API_KEY=
MCP_TOOL_NAME=search
MCP_QUERY_ARG=query
```

You can also use an `mcp+http://` or `mcp+https://` URL to infer
`streamable_http` automatically.

## Legacy HTTP Fallback

Existing deployments that expose a simple REST shim can keep using:

```bash
MCP_TRANSPORT=legacy_http
MCP_SERVER_URL=http://localhost:9000
MCP_TOOL_NAME=web_search
```

The fallback expects:

```text
GET  /tools
POST /tools/<tool_name>
```

and normalizes a JSON response with a `results` list into SDYJ evidence items.
