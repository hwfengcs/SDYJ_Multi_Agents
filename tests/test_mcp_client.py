import asyncio
import json
from types import SimpleNamespace

from SDYJ_Agents.tools.mcp_client import MCPClient


def test_stdio_client_loads_claude_style_config(tmp_path):
    config_path = tmp_path / "mcp_config.json"
    config_path.write_text(
        json.dumps(
            {
                "mcpServers": {
                    "filesystem": {
                        "command": "python",
                        "args": ["server.py", "--root", "."],
                        "env": {"ROOT": "."},
                    }
                }
            }
        ),
        encoding="utf-8",
    )

    client = MCPClient(config_path=str(config_path), server_name="filesystem")

    assert client.transport == "stdio"
    assert client.server_name == "filesystem"
    assert client.command == "python"
    assert client.args == ["server.py", "--root", "."]
    assert client.env == {"ROOT": "."}


def test_mcp_prefixed_http_url_infers_streamable_transport():
    client = MCPClient(server_url="mcp+https://example.com/mcp")

    assert client.transport == "streamable_http"
    assert client.server_url == "https://example.com/mcp"


def test_sdk_result_text_json_is_normalized_to_search_results():
    result = SimpleNamespace(
        content=[
            SimpleNamespace(
                type="text",
                text=json.dumps(
                    {
                        "results": [
                            {
                                "title": "MCP result",
                                "url": "file:///doc.md",
                                "content": "Result body",
                                "score": 0.8,
                            }
                        ]
                    }
                ),
            )
        ]
    )

    payload = MCPClient._sdk_result_to_dict(result)
    normalized = MCPClient._normalize_search_results(payload, "search")

    assert normalized == [
        {
            "title": "MCP result",
            "url": "file:///doc.md",
            "snippet": "Result body",
            "relevance_score": 0.8,
            "metadata": {},
        }
    ]


def test_search_uses_sdk_call_tool_and_wraps_result(monkeypatch):
    client = MCPClient(command="python", args=["server.py"], default_tool_name="search_docs")

    async def fake_run_sdk_operation(operation):
        class Session:
            async def call_tool(self, tool_name, arguments):
                assert tool_name == "search_docs"
                assert arguments == {"query": "agent trace"}
                return {
                    "results": [
                        {
                            "title": "Trace docs",
                            "snippet": "Trace details",
                            "url": "https://example.com/trace",
                        }
                    ]
                }

        result = operation(Session())
        if hasattr(result, "__await__"):
            return await result
        return result

    monkeypatch.setattr(client, "_run_sdk_operation", fake_run_sdk_operation)

    result = asyncio.run(client.search("agent trace"))

    assert result["source"] == "mcp"
    assert result["transport"] == "stdio"
    assert result["tool"] == "search_docs"
    assert result["total_results"] == 1
    assert result["results"][0]["title"] == "Trace docs"


def test_list_tools_normalizes_sdk_tool_objects(monkeypatch):
    client = MCPClient(command="python", args=["server.py"])

    async def fake_run_sdk_operation(operation):
        class Session:
            async def list_tools(self):
                return SimpleNamespace(
                    tools=[
                        SimpleNamespace(
                            name="search_docs",
                            description="Search local docs",
                            inputSchema={"type": "object"},
                        )
                    ]
                )

        result = operation(Session())
        if hasattr(result, "__await__"):
            return await result
        return result

    monkeypatch.setattr(client, "_run_sdk_operation", fake_run_sdk_operation)

    tools = asyncio.run(client.list_tools())

    assert tools == [
        {
            "name": "search_docs",
            "description": "Search local docs",
            "input_schema": {"type": "object"},
        }
    ]
