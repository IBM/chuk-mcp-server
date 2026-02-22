"""Tests for improved error messages."""

import pytest

from chuk_mcp_server import ChukMCPServer
from chuk_mcp_server.errors import (
    MCPError,
    format_missing_argument_error,
    format_unknown_tool_error,
    suggest_tool_name,
)
from chuk_mcp_server.testing import ToolRunner
from chuk_mcp_server.types import ToolHandler


class TestSuggestToolName:
    """Tests for fuzzy tool name matching."""

    def test_exact_match_not_needed(self):
        """Close misspellings are suggested."""
        result = suggest_tool_name("ad", ["add", "greet", "multiply"])
        assert result == "add"

    def test_typo_suggestion(self):
        """Common typos produce suggestions."""
        result = suggest_tool_name("gret", ["add", "greet", "multiply"])
        assert result == "greet"

    def test_no_match_returns_none(self):
        """Completely different names return None."""
        result = suggest_tool_name("xyz_unknown", ["add", "greet", "multiply"])
        assert result is None

    def test_empty_list_returns_none(self):
        """Empty available tools returns None."""
        result = suggest_tool_name("add", [])
        assert result is None


class TestFormatUnknownToolError:
    """Tests for unknown tool error formatting."""

    def test_with_suggestion(self):
        """Error includes 'Did you mean' when close match exists."""
        msg = format_unknown_tool_error("ad", ["add", "greet"])
        assert "Did you mean 'add'" in msg
        assert "'ad'" in msg

    def test_without_suggestion_lists_tools(self):
        """Error lists available tools when no close match."""
        msg = format_unknown_tool_error("xyz", ["add", "greet"])
        assert "Available tools:" in msg
        assert "add" in msg
        assert "greet" in msg

    def test_no_tools_registered(self):
        """Error indicates no tools when list is empty."""
        msg = format_unknown_tool_error("anything", [])
        assert "No tools are registered" in msg

    def test_truncates_long_tool_list(self):
        """Tool list is truncated after 10 entries."""
        tools = [f"tool_{i}" for i in range(20)]
        msg = format_unknown_tool_error("xyz", tools)
        assert "..." in msg


class TestFormatMissingArgumentError:
    """Tests for missing argument error formatting."""

    def test_basic_message(self):
        """Basic missing argument message."""
        msg = format_missing_argument_error("add", "a")
        assert "add" in msg
        assert "'a'" in msg
        assert "missing required argument" in msg

    def test_with_schema(self):
        """Message includes type info from schema."""
        schema = {
            "properties": {
                "name": {"type": "string", "description": "Person's name"},
            }
        }
        msg = format_missing_argument_error("greet", "name", schema)
        assert "string" in msg
        assert "Person's name" in msg


class TestMCPError:
    """Tests for MCPError class."""

    def test_basic_error(self):
        """MCPError stores code and message."""
        err = MCPError("something broke", code=-32600)
        assert str(err) == "something broke"
        assert err.code == -32600

    def test_to_message_with_suggestion(self):
        """to_message includes suggestion."""
        err = MCPError("fail", suggestion="Try this instead")
        assert "Try this instead" in err.to_message()

    def test_to_message_plain(self):
        """to_message works without suggestion."""
        err = MCPError("fail")
        assert err.to_message() == "fail"


class TestProtocolErrorIntegration:
    """Test improved errors through the protocol handler."""

    @pytest.mark.asyncio
    async def test_unknown_tool_suggests_correction(self):
        """Calling a misspelled tool suggests the correct name."""
        server = ChukMCPServer(name="test", version="1.0.0")

        def add(a: int, b: int) -> int:
            return a + b

        server.protocol.register_tool(ToolHandler.from_function(add, name="add", description="Add numbers."))

        runner = ToolRunner(server)
        response = await runner.call_tool("ad", {"a": 1, "b": 2})

        assert "error" in response
        error_msg = response["error"]["message"]
        assert "Did you mean 'add'" in error_msg

    @pytest.mark.asyncio
    async def test_unknown_tool_lists_available(self):
        """Calling a completely wrong tool lists available tools."""
        server = ChukMCPServer(name="test", version="1.0.0")

        def add(a: int, b: int) -> int:
            return a + b

        server.protocol.register_tool(ToolHandler.from_function(add, name="add", description="Add numbers."))

        runner = ToolRunner(server)
        response = await runner.call_tool("xyz_totally_wrong", {})

        assert "error" in response
        error_msg = response["error"]["message"]
        assert "Available tools:" in error_msg
        assert "add" in error_msg
