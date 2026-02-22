"""Tests for ToolRunner test harness."""

import pytest

from chuk_mcp_server import ChukMCPServer, ToolRunner
from chuk_mcp_server.testing import ToolRunner as ToolRunnerDirect
from chuk_mcp_server.types import ToolHandler


def _make_server_with_tools():
    """Create a ChukMCPServer with test tools registered."""
    server = ChukMCPServer(name="test-server", version="1.0.0")

    def add(a: int, b: int) -> int:
        """Add two numbers."""
        return a + b

    def greet(name: str) -> str:
        """Greet someone."""
        return f"Hello, {name}!"

    server.protocol.register_tool(ToolHandler.from_function(add, name="add", description="Add two numbers."))
    server.protocol.register_tool(ToolHandler.from_function(greet, name="greet", description="Greet someone."))
    return server


class TestToolRunner:
    """Tests for ToolRunner."""

    @pytest.mark.asyncio
    async def test_call_tool_returns_jsonrpc_response(self):
        """call_tool returns a full JSON-RPC response dict."""
        server = _make_server_with_tools()
        runner = ToolRunner(server)
        response = await runner.call_tool("add", {"a": 3, "b": 4})

        assert response["jsonrpc"] == "2.0"
        assert "result" in response
        assert "content" in response["result"]

    @pytest.mark.asyncio
    async def test_call_tool_text_returns_string(self):
        """call_tool_text extracts text content."""
        server = _make_server_with_tools()
        runner = ToolRunner(server)
        text = await runner.call_tool_text("greet", {"name": "World"})

        assert text == "Hello, World!"

    @pytest.mark.asyncio
    async def test_call_tool_text_with_numeric_result(self):
        """call_tool_text works with numeric return values."""
        server = _make_server_with_tools()
        runner = ToolRunner(server)
        text = await runner.call_tool_text("add", {"a": 10, "b": 20})

        assert "30" in text

    @pytest.mark.asyncio
    async def test_call_tool_error_for_unknown_tool(self):
        """call_tool returns error for non-existent tool."""
        server = _make_server_with_tools()
        runner = ToolRunner(server)
        response = await runner.call_tool("nonexistent", {})

        assert "error" in response

    @pytest.mark.asyncio
    async def test_call_tool_text_raises_on_error(self):
        """call_tool_text raises RuntimeError for tool errors."""
        server = _make_server_with_tools()
        runner = ToolRunner(server)

        with pytest.raises(RuntimeError, match="Tool error"):
            await runner.call_tool_text("nonexistent", {})

    @pytest.mark.asyncio
    async def test_list_tools(self):
        """list_tools returns all registered tools."""
        server = _make_server_with_tools()
        runner = ToolRunner(server)
        tools = await runner.list_tools()

        names = [t["name"] for t in tools]
        assert "add" in names
        assert "greet" in names

    @pytest.mark.asyncio
    async def test_list_tool_names(self):
        """list_tool_names returns just the names."""
        server = _make_server_with_tools()
        runner = ToolRunner(server)
        names = await runner.list_tool_names()

        assert set(names) == {"add", "greet"}

    @pytest.mark.asyncio
    async def test_session_created_lazily(self):
        """Session is created on first call, not at init."""
        server = _make_server_with_tools()
        runner = ToolRunner(server)

        assert runner._session_id is None
        await runner.call_tool("add", {"a": 1, "b": 2})
        assert runner._session_id is not None

    @pytest.mark.asyncio
    async def test_session_reused_across_calls(self):
        """Same session is reused across multiple calls."""
        server = _make_server_with_tools()
        runner = ToolRunner(server)

        await runner.call_tool("add", {"a": 1, "b": 2})
        first_session = runner._session_id

        await runner.call_tool("greet", {"name": "test"})
        assert runner._session_id == first_session

    @pytest.mark.asyncio
    async def test_direct_import_works(self):
        """ToolRunner can be imported directly from testing module."""
        assert ToolRunnerDirect is ToolRunner
