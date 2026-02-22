"""
Tool testing utilities for ChukMCPServer.

Provides a lightweight test harness for invoking MCP tools without
a transport layer, making it easy to write unit tests.
"""

import uuid
from typing import Any


class ToolRunner:
    """Test harness for invoking MCP tools without transport.

    Usage:
        from chuk_mcp_server import tool, ToolRunner

        @tool
        def add(a: int, b: int) -> int:
            return a + b

        runner = ToolRunner()
        result = await runner.call_tool("add", {"a": 1, "b": 2})
        # result is the full JSON-RPC response dict

        text = await runner.call_tool_text("add", {"a": 1, "b": 2})
        # text is the extracted text content string
    """

    def __init__(self, server: Any | None = None):
        """Initialize ToolRunner.

        Args:
            server: Optional ChukMCPServer instance. If None, uses globally
                    registered tools from @tool decorator.
        """
        if server is not None:
            self.protocol = server.protocol
        else:
            from .core import ChukMCPServer

            self._server = ChukMCPServer(name="test-runner", version="0.0.0")
            self.protocol = self._server.protocol

        self._session_id: str | None = None

    async def _ensure_session(self) -> str:
        """Ensure a session exists, creating one via initialize if needed."""
        if self._session_id is not None:
            return self._session_id

        init_msg = {
            "jsonrpc": "2.0",
            "id": "init-1",
            "method": "initialize",
            "params": {
                "protocolVersion": "2025-06-18",
                "capabilities": {},
                "clientInfo": {"name": "test-runner", "version": "0.0.0"},
            },
        }
        _, session_id = await self.protocol.handle_request(init_msg)
        self._session_id = session_id
        return str(session_id)

    async def call_tool(self, name: str, arguments: dict[str, Any] | None = None) -> dict[str, Any]:
        """Call a tool and return the full JSON-RPC response.

        Args:
            name: Tool name
            arguments: Tool arguments dict

        Returns:
            Full JSON-RPC response dict with "jsonrpc", "id", and "result" keys.
            The result contains "content" with the tool output.
        """
        session_id = await self._ensure_session()
        request = {
            "jsonrpc": "2.0",
            "id": f"test-{uuid.uuid4().hex[:8]}",
            "method": "tools/call",
            "params": {"name": name, "arguments": arguments or {}},
        }
        response, _ = await self.protocol.handle_request(request, session_id)
        result: dict[str, Any] = response
        return result

    async def call_tool_text(self, name: str, arguments: dict[str, Any] | None = None) -> str:
        """Call a tool and return the text content.

        Args:
            name: Tool name
            arguments: Tool arguments dict

        Returns:
            Extracted text content from the tool result.

        Raises:
            RuntimeError: If the tool returns an error response.
        """
        response = await self.call_tool(name, arguments)

        if "error" in response:
            raise RuntimeError(f"Tool error: {response['error'].get('message', response['error'])}")

        content = response["result"]["content"]
        parts = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text" or isinstance(item, dict) and "text" in item:
                parts.append(item["text"])
        return "\n".join(parts)

    async def list_tools(self) -> list[dict[str, Any]]:
        """List all registered tools.

        Returns:
            List of tool info dicts with "name", "description", "inputSchema".
        """
        session_id = await self._ensure_session()
        request = {
            "jsonrpc": "2.0",
            "id": f"test-{uuid.uuid4().hex[:8]}",
            "method": "tools/list",
            "params": {},
        }
        response, _ = await self.protocol.handle_request(request, session_id)
        tools: list[dict[str, Any]] = response.get("result", {}).get("tools", [])
        return tools

    async def list_tool_names(self) -> list[str]:
        """List registered tool names.

        Returns:
            List of tool name strings.
        """
        tools = await self.list_tools()
        return [t["name"] for t in tools]
