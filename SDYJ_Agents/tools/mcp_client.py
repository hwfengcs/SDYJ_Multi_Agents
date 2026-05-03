"""MCP (Model Context Protocol) client adapters.

The project historically exposed an HTTP placeholder shaped like
``POST /tools/<tool_name>``. v0.6 keeps that mode as ``legacy_http`` and adds
optional official MCP SDK transports for stdio and streamable HTTP servers.
The SDK is imported lazily so the core package still runs without installing
the ``[mcp]`` extra.
"""

from __future__ import annotations

import json
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, AsyncIterator, Callable, Dict, List, Optional

import httpx


SUPPORTED_TRANSPORTS = {"legacy_http", "stdio", "streamable_http"}


class MCPClient:
    """Client for external MCP tools and resources.

    Args:
        server_url: HTTP URL for legacy or streamable HTTP transports.
        api_key: Optional bearer token used for HTTP transports.
        transport: ``legacy_http``, ``stdio``, ``streamable_http``, or ``auto``.
        default_tool_name: Tool used by :meth:`search`.
        config_path: Claude-style config JSON with an ``mcpServers`` object.
        server_name: Server key to select from ``config_path``.
        command: Direct stdio command when no config file is used.
        args: Direct stdio command args.
        env: Direct stdio environment overrides.
        query_argument: Argument name used when no explicit tool_arguments are configured.
        tool_arguments: Optional argument template for :meth:`search`; string values
            can contain ``{query}``.
    """

    def __init__(
        self,
        server_url: Optional[str] = None,
        api_key: Optional[str] = None,
        transport: Optional[str] = None,
        default_tool_name: str = "web_search",
        config_path: Optional[str] = None,
        server_name: Optional[str] = None,
        command: Optional[str] = None,
        args: Optional[List[str]] = None,
        env: Optional[Dict[str, str]] = None,
        query_argument: str = "query",
        tool_arguments: Optional[Dict[str, Any]] = None,
    ):
        self.server_url = server_url.rstrip("/") if server_url else None
        self.api_key = api_key
        self.headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
        self.default_tool_name = default_tool_name or "web_search"
        self.config_path = config_path
        self.server_name = server_name
        self.command = command
        self.args = list(args or [])
        self.env = dict(env or {})
        self.query_argument = query_argument or "query"
        self.tool_arguments = dict(tool_arguments or {})
        self.last_error: Optional[str] = None
        self._configured_transport: Optional[str] = None

        if config_path:
            self._load_stdio_config(config_path, server_name)

        self.transport = self._infer_transport(transport)

    def _infer_transport(self, transport: Optional[str]) -> str:
        requested = (transport or self._configured_transport or "auto").strip().lower()
        if requested in {"", "auto"}:
            if self.command:
                return "stdio"
            if self.server_url and self.server_url.startswith(("mcp+http://", "mcp+https://")):
                self.server_url = self.server_url.replace("mcp+", "", 1)
                return "streamable_http"
            return "legacy_http"
        if requested not in SUPPORTED_TRANSPORTS:
            raise ValueError(
                f"Unsupported MCP transport: {transport}. "
                f"Expected one of {', '.join(sorted(SUPPORTED_TRANSPORTS))}."
            )
        return requested

    def _load_stdio_config(self, config_path: str, server_name: Optional[str]) -> None:
        config = json.loads(Path(config_path).read_text(encoding="utf-8"))
        servers = config.get("mcpServers") or config.get("servers") or {}
        if not isinstance(servers, dict) or not servers:
            raise ValueError(f"No MCP servers found in {config_path}")
        chosen_name = server_name or next(iter(servers))
        if chosen_name not in servers:
            raise ValueError(f"MCP server {chosen_name!r} not found in {config_path}")
        server = servers[chosen_name]
        if not isinstance(server, dict):
            raise ValueError(f"MCP server {chosen_name!r} must be an object")
        self.server_name = chosen_name
        self.command = server.get("command") or self.command
        self.args = [str(arg) for arg in server.get("args", self.args) or []]
        server_env = server.get("env") or {}
        if isinstance(server_env, dict):
            self.env = {**{str(k): str(v) for k, v in server_env.items()}, **self.env}
        if "url" in server and not self.server_url:
            self.server_url = str(server["url"])
        if "transport" in server:
            self._configured_transport = str(server["transport"]).lower()

    async def search(
        self,
        query: str,
        tool_name: Optional[str] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """Perform a search-like MCP tool call and normalize the result."""
        selected_tool = tool_name or self.default_tool_name
        parameters = self._build_tool_parameters(query, kwargs)
        try:
            payload = await self.execute_tool(selected_tool, parameters)
            if payload.get("error"):
                raise RuntimeError(str(payload["error"]))
            results = self._normalize_search_results(payload, selected_tool)
            return {
                "query": query,
                "source": "mcp",
                "tool": selected_tool,
                "transport": self.transport,
                "results": results,
                "timestamp": datetime.now().isoformat(),
                "total_results": len(results),
            }
        except Exception as exc:
            return {
                "query": query,
                "source": "mcp",
                "tool": selected_tool,
                "transport": self.transport,
                "results": [],
                "timestamp": datetime.now().isoformat(),
                "error": str(exc),
            }

    def _build_tool_parameters(self, query: str, overrides: Dict[str, Any]) -> Dict[str, Any]:
        if self.tool_arguments:
            parameters = self._render_argument_template(self.tool_arguments, query)
            parameters.update(overrides)
            return parameters
        return {self.query_argument: query, **overrides}

    @classmethod
    def _render_argument_template(cls, value: Any, query: str) -> Any:
        if isinstance(value, str):
            return value.replace("{query}", query)
        if isinstance(value, list):
            return [cls._render_argument_template(item, query) for item in value]
        if isinstance(value, dict):
            return {
                str(key): cls._render_argument_template(item, query)
                for key, item in value.items()
            }
        return value

    async def list_tools(self) -> List[Dict[str, Any]]:
        """List tools exposed by the configured MCP server."""
        if self.transport == "legacy_http":
            return await self._legacy_list_tools()

        try:
            response = await self._run_sdk_operation(lambda session: session.list_tools())
            tools = getattr(response, "tools", response)
            self.last_error = None
            return [self._tool_to_dict(tool) for tool in tools or []]
        except Exception as exc:
            self.last_error = str(exc)
            return []

    async def list_resources(self) -> List[Dict[str, Any]]:
        """List resources exposed by the configured MCP server."""
        if self.transport == "legacy_http":
            return []

        try:
            response = await self._run_sdk_operation(lambda session: session.list_resources())
            resources = getattr(response, "resources", response)
            self.last_error = None
            return [self._resource_to_dict(resource) for resource in resources or []]
        except Exception as exc:
            self.last_error = str(exc)
            return []

    async def call_tool(self, tool_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Call a specific MCP tool."""
        return await self.execute_tool(tool_name, arguments)

    async def execute_tool(self, tool_name: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a tool through the configured transport."""
        if self.transport == "legacy_http":
            return await self._legacy_execute_tool(tool_name, parameters)

        try:
            response = await self._run_sdk_operation(
                lambda session: session.call_tool(tool_name, arguments=parameters)
            )
            self.last_error = None
            return self._sdk_result_to_dict(response)
        except Exception as exc:
            self.last_error = str(exc)
            return {"error": str(exc), "tool": tool_name}

    async def _legacy_list_tools(self) -> List[Dict[str, Any]]:
        if not self.server_url:
            self.last_error = "MCP server URL is not configured"
            return []
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(f"{self.server_url}/tools", headers=self.headers)
                response.raise_for_status()
                self.last_error = None
                return response.json().get("tools", [])
        except Exception as exc:
            self.last_error = str(exc)
            return []

    async def _legacy_execute_tool(self, tool_name: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
        if not self.server_url:
            return {"error": "MCP server URL is not configured", "tool": tool_name}
        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{self.server_url}/tools/{tool_name}",
                    json=parameters,
                    headers=self.headers,
                )
                response.raise_for_status()
                return response.json()
        except Exception as exc:
            return {"error": str(exc), "tool": tool_name}

    async def _run_sdk_operation(self, operation: Callable[[Any], Any]) -> Any:
        async with self._sdk_session() as session:
            result = operation(session)
            if hasattr(result, "__await__"):
                return await result
            return result

    @asynccontextmanager
    async def _sdk_session(self) -> AsyncIterator[Any]:
        """Open an official MCP SDK session for one operation."""
        if self.transport == "stdio":
            if not self.command:
                raise ValueError("MCP stdio transport requires command or config_path")
            try:
                from mcp import ClientSession, StdioServerParameters
                from mcp.client.stdio import stdio_client
            except ImportError as exc:
                raise RuntimeError("Install sdyj-multi-agents[mcp] to use MCP stdio") from exc

            server_params = StdioServerParameters(
                command=self.command,
                args=self.args,
                env=self.env or None,
            )
            async with stdio_client(server_params) as (read, write):
                async with ClientSession(read, write) as session:
                    await session.initialize()
                    yield session
            return

        if self.transport == "streamable_http":
            if not self.server_url:
                raise ValueError("MCP streamable_http transport requires server_url")
            try:
                from mcp import ClientSession
                from mcp.client.streamable_http import streamable_http_client
            except ImportError as exc:
                raise RuntimeError(
                    "Install sdyj-multi-agents[mcp] to use MCP streamable_http"
                ) from exc

            if self.headers:
                async with httpx.AsyncClient(headers=self.headers) as http_client:
                    async with streamable_http_client(
                        self.server_url,
                        http_client=http_client,
                    ) as streams:
                        read, write, *_ = streams
                        async with ClientSession(read, write) as session:
                            await session.initialize()
                            yield session
            else:
                async with streamable_http_client(self.server_url) as streams:
                    read, write, *_ = streams
                    async with ClientSession(read, write) as session:
                        await session.initialize()
                        yield session
            return

        raise ValueError(f"SDK session is not available for transport={self.transport}")

    @staticmethod
    def _tool_to_dict(tool: Any) -> Dict[str, Any]:
        if isinstance(tool, dict):
            return tool
        return {
            "name": getattr(tool, "name", ""),
            "description": getattr(tool, "description", ""),
            "input_schema": getattr(tool, "inputSchema", None)
            or getattr(tool, "input_schema", None),
        }

    @staticmethod
    def _resource_to_dict(resource: Any) -> Dict[str, Any]:
        if isinstance(resource, dict):
            return resource
        return {
            "uri": str(getattr(resource, "uri", "")),
            "name": getattr(resource, "name", ""),
            "description": getattr(resource, "description", ""),
            "mime_type": getattr(resource, "mimeType", None)
            or getattr(resource, "mime_type", None),
        }

    @classmethod
    def _sdk_result_to_dict(cls, result: Any) -> Dict[str, Any]:
        if isinstance(result, dict):
            return result

        structured = (
            getattr(result, "structured_content", None)
            or getattr(result, "structuredContent", None)
        )
        if isinstance(structured, dict):
            return structured

        content = getattr(result, "content", None)
        if content is not None:
            parsed_content = [cls._content_block_to_dict(block) for block in content]
            for block in parsed_content:
                text = block.get("text")
                if text:
                    try:
                        parsed = json.loads(text)
                    except json.JSONDecodeError:
                        continue
                    if isinstance(parsed, dict):
                        return parsed
            return {"content": parsed_content}

        return {"result": str(result)}

    @staticmethod
    def _content_block_to_dict(block: Any) -> Dict[str, Any]:
        if isinstance(block, dict):
            return block
        output: Dict[str, Any] = {"type": getattr(block, "type", block.__class__.__name__)}
        text = getattr(block, "text", None)
        if text is not None:
            output["text"] = text
        data = getattr(block, "data", None)
        if data is not None:
            output["data"] = data
        mime_type = getattr(block, "mimeType", None) or getattr(block, "mime_type", None)
        if mime_type is not None:
            output["mime_type"] = mime_type
        return output

    @classmethod
    def _normalize_search_results(
        cls,
        payload: Dict[str, Any],
        tool_name: str = "mcp",
    ) -> List[Dict[str, Any]]:
        raw_results = payload.get("results")
        if raw_results is None and isinstance(payload.get("content"), list):
            raw_results = payload["content"]
        if raw_results is None:
            raw_results = payload.get("items")
        if raw_results is None:
            raw_results = [payload] if payload else []
        if not isinstance(raw_results, list):
            raw_results = [raw_results]

        normalized = []
        for index, item in enumerate(raw_results, start=1):
            if not isinstance(item, dict):
                item = {"snippet": str(item)}
            title = item.get("title") or item.get("name") or f"{tool_name} result {index}"
            snippet = (
                item.get("snippet")
                or item.get("content")
                or item.get("text")
                or item.get("description")
                or json.dumps(item, ensure_ascii=False, default=str)
            )
            normalized.append(
                {
                    "title": title,
                    "url": item.get("url") or item.get("uri") or "",
                    "snippet": snippet,
                    "relevance_score": item.get("score") or item.get("relevance_score"),
                    "metadata": item.get("metadata", {}),
                }
            )
        return normalized
