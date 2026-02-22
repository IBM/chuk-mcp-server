#!/usr/bin/env python3
"""Tests for MCP resource links in tool results."""

import pytest

from chuk_mcp_server.context import (
    add_resource_link,
    get_resource_links,
    set_resource_links,
)
from chuk_mcp_server.protocol import MCPProtocolHandler
from chuk_mcp_server.types.content import create_resource_link
from chuk_mcp_server.types.tools import ToolHandler


@pytest.fixture
def handler():
    from chuk_mcp_server.types.base import ServerCapabilities, ServerInfo

    info = ServerInfo(name="test", version="1.0")
    caps = ServerCapabilities()
    return MCPProtocolHandler(info, caps)


class TestCreateResourceLink:
    """Test the create_resource_link helper in content.py."""

    def test_minimal(self):
        link = create_resource_link("file://readme.md")
        assert link == {"uri": "file://readme.md"}

    def test_full(self):
        link = create_resource_link(
            "docs://api-spec",
            name="API Specification",
            description="The full API spec",
            mime_type="application/json",
        )
        assert link["uri"] == "docs://api-spec"
        assert link["name"] == "API Specification"
        assert link["description"] == "The full API spec"
        assert link["mimeType"] == "application/json"

    def test_partial_fields(self):
        link = create_resource_link("test://x", name="Test")
        assert link == {"uri": "test://x", "name": "Test"}
        assert "description" not in link
        assert "mimeType" not in link


class TestResourceLinksContextFunctions:
    """Test the context-based resource link accumulation."""

    def test_initial_state_is_none(self):
        set_resource_links(None)
        assert get_resource_links() is None

    def test_add_creates_list(self):
        set_resource_links(None)
        add_resource_link("file://test.txt")
        links = get_resource_links()
        assert links is not None
        assert len(links) == 1
        assert links[0]["uri"] == "file://test.txt"
        set_resource_links(None)

    def test_add_multiple(self):
        set_resource_links(None)
        add_resource_link("file://a.txt", name="File A")
        add_resource_link("file://b.txt", name="File B", mime_type="text/plain")
        links = get_resource_links()
        assert len(links) == 2
        assert links[0]["uri"] == "file://a.txt"
        assert links[1]["mimeType"] == "text/plain"
        set_resource_links(None)

    def test_add_with_all_fields(self):
        set_resource_links(None)
        add_resource_link(
            "docs://spec",
            name="Spec",
            description="The specification",
            mime_type="application/json",
        )
        links = get_resource_links()
        assert len(links) == 1
        link = links[0]
        assert link["uri"] == "docs://spec"
        assert link["name"] == "Spec"
        assert link["description"] == "The specification"
        assert link["mimeType"] == "application/json"
        set_resource_links(None)

    def test_clear(self):
        set_resource_links(None)
        add_resource_link("file://x")
        assert get_resource_links() is not None
        set_resource_links(None)
        assert get_resource_links() is None


class TestResourceLinksInToolResults:
    """Test that resource links appear in tool call responses."""

    @pytest.mark.asyncio
    async def test_tool_with_resource_links(self, handler):
        """Tool that adds resource links should have them in response _meta."""

        def search_docs(query: str) -> str:
            add_resource_link("docs://result1", name="Result 1")
            add_resource_link("docs://result2", name="Result 2")
            return f"Found 2 results for '{query}'"

        handler.register_tool(ToolHandler.from_function(search_docs))

        params = {"name": "search_docs", "arguments": {"query": "MCP"}}
        response, _ = await handler._handle_tools_call(params, "req-1")
        result = response["result"]

        assert "content" in result
        assert "_meta" in result
        assert "links" in result["_meta"]
        links = result["_meta"]["links"]
        assert len(links) == 2
        assert links[0]["uri"] == "docs://result1"
        assert links[1]["uri"] == "docs://result2"

    @pytest.mark.asyncio
    async def test_tool_without_resource_links(self, handler):
        """Tool that doesn't add links should not have _meta.links."""

        def simple(x: str) -> str:
            return f"hello {x}"

        handler.register_tool(ToolHandler.from_function(simple))

        params = {"name": "simple", "arguments": {"x": "world"}}
        response, _ = await handler._handle_tools_call(params, "req-1")
        result = response["result"]

        assert "content" in result
        # Either no _meta at all or no links in _meta
        if "_meta" in result:
            assert "links" not in result["_meta"]

    @pytest.mark.asyncio
    async def test_resource_links_cleaned_between_calls(self, handler):
        """Resource links should be cleaned up between tool calls."""

        call_count = 0

        def tool_with_links(x: str) -> str:
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                add_resource_link("docs://first")
            return f"call {call_count}"

        handler.register_tool(ToolHandler.from_function(tool_with_links))

        # First call - should have links
        params = {"name": "tool_with_links", "arguments": {"x": "a"}}
        response, _ = await handler._handle_tools_call(params, "req-1")
        result = response["result"]
        assert "_meta" in result
        assert len(result["_meta"]["links"]) == 1

        # Second call - should NOT have links from first call
        params = {"name": "tool_with_links", "arguments": {"x": "b"}}
        response, _ = await handler._handle_tools_call(params, "req-2")
        result = response["result"]
        if "_meta" in result:
            assert "links" not in result["_meta"]
