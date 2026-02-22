#!/usr/bin/env python3
"""Tests for cursor-based pagination on list methods."""

import base64

import pytest

from chuk_mcp_server.constants import DEFAULT_PAGE_SIZE
from chuk_mcp_server.protocol import MCPProtocolHandler
from chuk_mcp_server.types.prompts import PromptHandler
from chuk_mcp_server.types.resources import ResourceHandler
from chuk_mcp_server.types.tools import ToolHandler


@pytest.fixture
def handler():
    from chuk_mcp_server.types.base import ServerCapabilities, ServerInfo

    info = ServerInfo(name="test", version="1.0")
    caps = ServerCapabilities()
    return MCPProtocolHandler(info, caps)


def _make_cursor(offset: int) -> str:
    return base64.b64encode(str(offset).encode()).decode()


class TestPaginateHelper:
    """Test the _paginate static method directly."""

    def test_no_cursor_returns_first_page(self, handler):
        items = [{"name": f"item_{i}"} for i in range(150)]
        result = handler._paginate(items, "tools", {})
        assert len(result["tools"]) == DEFAULT_PAGE_SIZE
        assert result["tools"][0]["name"] == "item_0"
        assert "nextCursor" in result

    def test_no_pagination_needed(self, handler):
        items = [{"name": f"item_{i}"} for i in range(5)]
        result = handler._paginate(items, "tools", {})
        assert len(result["tools"]) == 5
        assert "nextCursor" not in result

    def test_exact_page_size(self, handler):
        items = [{"name": f"item_{i}"} for i in range(DEFAULT_PAGE_SIZE)]
        result = handler._paginate(items, "tools", {})
        assert len(result["tools"]) == DEFAULT_PAGE_SIZE
        assert "nextCursor" not in result

    def test_cursor_fetches_second_page(self, handler):
        items = [{"name": f"item_{i}"} for i in range(150)]
        cursor = _make_cursor(DEFAULT_PAGE_SIZE)
        result = handler._paginate(items, "resources", {"cursor": cursor})
        assert len(result["resources"]) == 50
        assert result["resources"][0]["name"] == f"item_{DEFAULT_PAGE_SIZE}"
        assert "nextCursor" not in result

    def test_cursor_chain(self, handler):
        """Paginate through all items using cursor chain."""
        items = [{"name": f"item_{i}"} for i in range(250)]

        # First page
        result = handler._paginate(items, "tools", {})
        assert len(result["tools"]) == DEFAULT_PAGE_SIZE
        assert "nextCursor" in result
        cursor1 = result["nextCursor"]

        # Second page
        result = handler._paginate(items, "tools", {"cursor": cursor1})
        assert len(result["tools"]) == DEFAULT_PAGE_SIZE
        assert "nextCursor" in result
        cursor2 = result["nextCursor"]

        # Third page (last)
        result = handler._paginate(items, "tools", {"cursor": cursor2})
        assert len(result["tools"]) == 50
        assert "nextCursor" not in result

    def test_invalid_cursor_starts_from_beginning(self, handler):
        items = [{"name": f"item_{i}"} for i in range(5)]
        result = handler._paginate(items, "tools", {"cursor": "invalid!!"})
        assert len(result["tools"]) == 5

    def test_empty_list(self, handler):
        result = handler._paginate([], "prompts", {})
        assert result["prompts"] == []
        assert "nextCursor" not in result

    def test_cursor_beyond_end(self, handler):
        items = [{"name": "only"}]
        cursor = _make_cursor(999)
        result = handler._paginate(items, "tools", {"cursor": cursor})
        assert result["tools"] == []
        assert "nextCursor" not in result


class TestToolsListPagination:
    """Test pagination through the tools/list protocol handler."""

    @pytest.mark.asyncio
    async def test_tools_list_no_cursor(self, handler):
        for i in range(5):

            def make_fn(idx=i):
                def fn(x: str) -> str:
                    return x

                fn.__name__ = f"tool_{idx}"
                return fn

            handler.register_tool(ToolHandler.from_function(make_fn()))

        response, _ = await handler._handle_tools_list({}, "req-1")
        result = response["result"]
        assert len(result["tools"]) == 5
        assert "nextCursor" not in result

    @pytest.mark.asyncio
    async def test_tools_list_with_cursor(self, handler):
        # Register more tools than page size to trigger pagination
        for i in range(DEFAULT_PAGE_SIZE + 10):

            def make_fn(idx=i):
                def fn(x: str) -> str:
                    return x

                fn.__name__ = f"tool_{idx}"
                return fn

            handler.register_tool(ToolHandler.from_function(make_fn()))

        # First page
        response, _ = await handler._handle_tools_list({}, "req-1")
        result = response["result"]
        assert len(result["tools"]) == DEFAULT_PAGE_SIZE
        assert "nextCursor" in result

        # Second page
        response, _ = await handler._handle_tools_list({"cursor": result["nextCursor"]}, "req-2")
        result = response["result"]
        assert len(result["tools"]) == 10
        assert "nextCursor" not in result


class TestResourcesListPagination:
    """Test pagination through the resources/list protocol handler."""

    @pytest.mark.asyncio
    async def test_resources_list_paginated(self, handler):
        for i in range(DEFAULT_PAGE_SIZE + 5):

            def make_fn():
                def fn() -> str:
                    return "content"

                return fn

            handler.register_resource(ResourceHandler.from_function(f"res://item_{i}", make_fn(), name=f"item_{i}"))

        response, _ = await handler._handle_resources_list({}, "req-1")
        result = response["result"]
        assert len(result["resources"]) == DEFAULT_PAGE_SIZE
        assert "nextCursor" in result


class TestPromptsListPagination:
    """Test pagination through the prompts/list protocol handler."""

    @pytest.mark.asyncio
    async def test_prompts_list_paginated(self, handler):
        for i in range(DEFAULT_PAGE_SIZE + 3):

            def make_fn(idx=i):
                def fn(topic: str) -> str:
                    return f"About {topic}"

                fn.__name__ = f"prompt_{idx}"
                return fn

            handler.register_prompt(PromptHandler.from_function(make_fn()))

        response, _ = await handler._handle_prompts_list({}, "req-1")
        result = response["result"]
        assert len(result["prompts"]) == DEFAULT_PAGE_SIZE
        assert "nextCursor" in result
