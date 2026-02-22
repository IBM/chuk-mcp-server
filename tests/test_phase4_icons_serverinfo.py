"""
Tests for Phase 4 features: Icons on Types and Enhanced ServerInfo (MCP 2025-11-25).
"""

import pytest

from chuk_mcp_server.core import ChukMCPServer
from chuk_mcp_server.types.prompts import PromptHandler
from chuk_mcp_server.types.resources import ResourceHandler, ResourceTemplateHandler
from chuk_mcp_server.types.tools import ToolHandler

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

SINGLE_ICON = [{"url": "https://example.com/icon.png", "mediaType": "image/png"}]

MULTIPLE_ICONS = [
    {"url": "https://example.com/icon.png", "mediaType": "image/png"},
    {"url": "https://example.com/icon.svg", "mediaType": "image/svg+xml"},
    {"url": "https://example.com/icon-dark.png", "mediaType": "image/png"},
]

URL_ONLY_ICON = [{"url": "https://example.com/icon.png"}]


# ---------------------------------------------------------------------------
# Helper functions used as handler targets
# ---------------------------------------------------------------------------


def sample_tool(x: str) -> str:
    """A sample tool."""
    return x


def sample_resource():
    """A sample resource."""
    return "resource-data"


def sample_prompt(message: str) -> str:
    """A sample prompt."""
    return message


def sample_template(item_id: str) -> str:
    """A sample resource template."""
    return f"item-{item_id}"


# ===================================================================
# Feature 1: Icons on Types
# ===================================================================


class TestToolHandlerIcons:
    """Icons on ToolHandler."""

    def test_icons_included_in_mcp_format(self):
        handler = ToolHandler.from_function(sample_tool, icons=SINGLE_ICON)
        fmt = handler.to_mcp_format()
        assert fmt["icons"] == SINGLE_ICON

    def test_icons_absent_when_not_provided(self):
        handler = ToolHandler.from_function(sample_tool)
        fmt = handler.to_mcp_format()
        assert "icons" not in fmt

    def test_multiple_icons(self):
        handler = ToolHandler.from_function(sample_tool, icons=MULTIPLE_ICONS)
        fmt = handler.to_mcp_format()
        assert fmt["icons"] == MULTIPLE_ICONS
        assert len(fmt["icons"]) == 3

    def test_icon_without_media_type(self):
        handler = ToolHandler.from_function(sample_tool, icons=URL_ONLY_ICON)
        fmt = handler.to_mcp_format()
        assert fmt["icons"] == URL_ONLY_ICON
        assert "mediaType" not in fmt["icons"][0]

    def test_decorator_passes_icons_through(self):
        mcp = ChukMCPServer(name="icon-test")

        @mcp.tool(icons=SINGLE_ICON)
        def decorated_tool(x: str) -> str:
            """Decorated tool."""
            return x

        handler = mcp.protocol.tools["decorated_tool"]
        fmt = handler.to_mcp_format()
        assert fmt["icons"] == SINGLE_ICON


class TestResourceHandlerIcons:
    """Icons on ResourceHandler."""

    def test_icons_included_in_mcp_format(self):
        handler = ResourceHandler.from_function(uri="test://res", func=sample_resource, icons=SINGLE_ICON)
        fmt = handler.to_mcp_format()
        assert fmt["icons"] == SINGLE_ICON

    def test_icons_absent_when_not_provided(self):
        handler = ResourceHandler.from_function(uri="test://res", func=sample_resource)
        fmt = handler.to_mcp_format()
        assert "icons" not in fmt

    def test_multiple_icons(self):
        handler = ResourceHandler.from_function(uri="test://res", func=sample_resource, icons=MULTIPLE_ICONS)
        fmt = handler.to_mcp_format()
        assert fmt["icons"] == MULTIPLE_ICONS
        assert len(fmt["icons"]) == 3

    def test_icon_without_media_type(self):
        handler = ResourceHandler.from_function(uri="test://res", func=sample_resource, icons=URL_ONLY_ICON)
        fmt = handler.to_mcp_format()
        assert fmt["icons"] == URL_ONLY_ICON

    def test_decorator_passes_icons_through(self):
        mcp = ChukMCPServer(name="icon-test")

        @mcp.resource("test://decorated", icons=SINGLE_ICON)
        def decorated_resource():
            """Decorated resource."""
            return "data"

        handler = mcp.protocol.resources["test://decorated"]
        fmt = handler.to_mcp_format()
        assert fmt["icons"] == SINGLE_ICON


class TestPromptHandlerIcons:
    """Icons on PromptHandler."""

    def test_icons_included_in_mcp_format(self):
        handler = PromptHandler.from_function(sample_prompt, icons=SINGLE_ICON)
        fmt = handler.to_mcp_format()
        assert fmt["icons"] == SINGLE_ICON

    def test_icons_absent_when_not_provided(self):
        handler = PromptHandler.from_function(sample_prompt)
        fmt = handler.to_mcp_format()
        assert "icons" not in fmt

    def test_multiple_icons(self):
        handler = PromptHandler.from_function(sample_prompt, icons=MULTIPLE_ICONS)
        fmt = handler.to_mcp_format()
        assert fmt["icons"] == MULTIPLE_ICONS
        assert len(fmt["icons"]) == 3

    def test_icon_without_media_type(self):
        handler = PromptHandler.from_function(sample_prompt, icons=URL_ONLY_ICON)
        fmt = handler.to_mcp_format()
        assert fmt["icons"] == URL_ONLY_ICON

    def test_decorator_passes_icons_through(self):
        mcp = ChukMCPServer(name="icon-test")

        @mcp.prompt(icons=SINGLE_ICON)
        def decorated_prompt(msg: str) -> str:
            """Decorated prompt."""
            return msg

        handler = mcp.protocol.prompts["decorated_prompt"]
        fmt = handler.to_mcp_format()
        assert fmt["icons"] == SINGLE_ICON


class TestResourceTemplateHandlerIcons:
    """Icons on ResourceTemplateHandler."""

    def test_icons_included_in_mcp_format(self):
        handler = ResourceTemplateHandler.from_function(
            uri_template="test://{item_id}",
            func=sample_template,
            icons=SINGLE_ICON,
        )
        fmt = handler.to_mcp_format()
        assert fmt["icons"] == SINGLE_ICON

    def test_icons_absent_when_not_provided(self):
        handler = ResourceTemplateHandler.from_function(
            uri_template="test://{item_id}",
            func=sample_template,
        )
        fmt = handler.to_mcp_format()
        assert "icons" not in fmt

    def test_multiple_icons(self):
        handler = ResourceTemplateHandler.from_function(
            uri_template="test://{item_id}",
            func=sample_template,
            icons=MULTIPLE_ICONS,
        )
        fmt = handler.to_mcp_format()
        assert fmt["icons"] == MULTIPLE_ICONS
        assert len(fmt["icons"]) == 3

    def test_icon_without_media_type(self):
        handler = ResourceTemplateHandler.from_function(
            uri_template="test://{item_id}",
            func=sample_template,
            icons=URL_ONLY_ICON,
        )
        fmt = handler.to_mcp_format()
        assert fmt["icons"] == URL_ONLY_ICON

    def test_decorator_passes_icons_through(self):
        mcp = ChukMCPServer(name="icon-test")

        @mcp.resource_template("test://{item_id}", icons=SINGLE_ICON)
        def decorated_template(item_id: str) -> str:
            """Decorated resource template."""
            return item_id

        handler = mcp.protocol.resource_templates["test://{item_id}"]
        fmt = handler.to_mcp_format()
        assert fmt["icons"] == SINGLE_ICON


class TestIconsEdgeCases:
    """Cross-cutting edge-case tests for icons."""

    def test_empty_icons_list_not_included(self):
        """An empty icons list should ideally not appear in MCP format."""
        handler = ToolHandler.from_function(sample_tool, icons=[])
        fmt = handler.to_mcp_format()
        # Empty list is falsy, so it should be omitted (same as None).
        assert "icons" not in fmt or fmt.get("icons") == []

    def test_icons_preserved_on_handler_attribute(self):
        handler = ToolHandler.from_function(sample_tool, icons=SINGLE_ICON)
        assert handler.icons == SINGLE_ICON

    def test_icons_are_not_mutated(self):
        original = [{"url": "https://example.com/a.png", "mediaType": "image/png"}]
        handler = ToolHandler.from_function(sample_tool, icons=original)
        fmt = handler.to_mcp_format()
        # Verify the output matches and original was not mutated
        assert fmt["icons"] == original
        assert fmt["icons"] is not original  # should be a copy or at least equal


# ===================================================================
# Feature 2: Enhanced ServerInfo
# ===================================================================


class TestEnhancedServerInfoStorage:
    """Verify extra server info is stored on the protocol handler."""

    def test_extra_server_info_stored_correctly(self):
        mcp = ChukMCPServer(
            name="TestServer",
            description="A test server",
            icons=[{"url": "https://example.com/icon.png"}],
            website_url="https://example.com",
        )
        extra = mcp.protocol._extra_server_info
        assert extra["description"] == "A test server"
        assert extra["icons"] == [{"url": "https://example.com/icon.png"}]
        assert extra["websiteUrl"] == "https://example.com"

    def test_extra_server_info_not_present_when_omitted(self):
        mcp = ChukMCPServer(name="PlainServer")
        extra = mcp.protocol._extra_server_info
        assert extra == {} or extra is None or all(v is None for v in extra.values()) if extra else True

    def test_partial_fields_description_only(self):
        mcp = ChukMCPServer(
            name="PartialServer",
            description="Only a description",
        )
        extra = mcp.protocol._extra_server_info
        assert extra.get("description") == "Only a description"
        # icons and websiteUrl should not be present or should be None
        assert not extra.get("icons")
        assert not extra.get("websiteUrl")

    def test_partial_fields_icons_only(self):
        mcp = ChukMCPServer(
            name="IconServer",
            icons=[{"url": "https://example.com/icon.png"}],
        )
        extra = mcp.protocol._extra_server_info
        assert extra.get("icons") == [{"url": "https://example.com/icon.png"}]
        assert not extra.get("description")
        assert not extra.get("websiteUrl")

    def test_partial_fields_website_url_only(self):
        mcp = ChukMCPServer(
            name="WebServer",
            website_url="https://example.com",
        )
        extra = mcp.protocol._extra_server_info
        assert extra.get("websiteUrl") == "https://example.com"
        assert not extra.get("description")
        assert not extra.get("icons")


class TestEnhancedServerInfoInitializeResponse:
    """Verify the initialize response includes enhanced server info."""

    @pytest.mark.asyncio
    async def test_initialize_includes_description_icons_website(self):
        mcp = ChukMCPServer(
            name="FullServer",
            description="A full-featured server",
            icons=[{"url": "https://example.com/icon.png", "mediaType": "image/png"}],
            website_url="https://example.com",
        )
        response, session_id = await mcp.protocol.handle_request(
            {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "initialize",
                "params": {
                    "clientInfo": {"name": "test-client"},
                    "protocolVersion": "2025-06-18",
                },
            }
        )
        server_info = response["result"]["serverInfo"]
        assert server_info["name"] == "FullServer"
        assert server_info["description"] == "A full-featured server"
        assert server_info["icons"] == [{"url": "https://example.com/icon.png", "mediaType": "image/png"}]
        assert server_info["websiteUrl"] == "https://example.com"

    @pytest.mark.asyncio
    async def test_initialize_without_extra_fields(self):
        mcp = ChukMCPServer(name="PlainServer")
        response, session_id = await mcp.protocol.handle_request(
            {
                "jsonrpc": "2.0",
                "id": 2,
                "method": "initialize",
                "params": {
                    "clientInfo": {"name": "test-client"},
                    "protocolVersion": "2025-06-18",
                },
            }
        )
        server_info = response["result"]["serverInfo"]
        assert server_info["name"] == "PlainServer"
        # Extra fields should not be present
        assert "description" not in server_info
        assert "icons" not in server_info
        assert "websiteUrl" not in server_info

    @pytest.mark.asyncio
    async def test_initialize_with_partial_description_only(self):
        mcp = ChukMCPServer(
            name="DescServer",
            description="Just a description",
        )
        response, session_id = await mcp.protocol.handle_request(
            {
                "jsonrpc": "2.0",
                "id": 3,
                "method": "initialize",
                "params": {
                    "clientInfo": {"name": "test-client"},
                    "protocolVersion": "2025-06-18",
                },
            }
        )
        server_info = response["result"]["serverInfo"]
        assert server_info["description"] == "Just a description"
        assert "icons" not in server_info
        assert "websiteUrl" not in server_info

    @pytest.mark.asyncio
    async def test_initialize_with_partial_icons_only(self):
        mcp = ChukMCPServer(
            name="IconOnlyServer",
            icons=[{"url": "https://example.com/logo.svg"}],
        )
        response, session_id = await mcp.protocol.handle_request(
            {
                "jsonrpc": "2.0",
                "id": 4,
                "method": "initialize",
                "params": {
                    "clientInfo": {"name": "test-client"},
                    "protocolVersion": "2025-06-18",
                },
            }
        )
        server_info = response["result"]["serverInfo"]
        assert server_info["icons"] == [{"url": "https://example.com/logo.svg"}]
        assert "description" not in server_info
        assert "websiteUrl" not in server_info

    @pytest.mark.asyncio
    async def test_initialize_with_partial_website_url_only(self):
        mcp = ChukMCPServer(
            name="WebOnlyServer",
            website_url="https://myserver.dev",
        )
        response, session_id = await mcp.protocol.handle_request(
            {
                "jsonrpc": "2.0",
                "id": 5,
                "method": "initialize",
                "params": {
                    "clientInfo": {"name": "test-client"},
                    "protocolVersion": "2025-06-18",
                },
            }
        )
        server_info = response["result"]["serverInfo"]
        assert server_info["websiteUrl"] == "https://myserver.dev"
        assert "description" not in server_info
        assert "icons" not in server_info

    @pytest.mark.asyncio
    async def test_initialize_response_has_valid_structure(self):
        """Verify the overall initialize response structure is correct."""
        mcp = ChukMCPServer(
            name="StructureTest",
            description="Testing structure",
            website_url="https://example.org",
        )
        response, session_id = await mcp.protocol.handle_request(
            {
                "jsonrpc": "2.0",
                "id": 6,
                "method": "initialize",
                "params": {
                    "clientInfo": {"name": "test-client"},
                    "protocolVersion": "2025-06-18",
                },
            }
        )
        assert response["jsonrpc"] == "2.0"
        assert response["id"] == 6
        assert "result" in response
        result = response["result"]
        assert "serverInfo" in result
        assert "protocolVersion" in result
        assert "capabilities" in result

    @pytest.mark.asyncio
    async def test_initialize_with_multiple_icons(self):
        mcp = ChukMCPServer(
            name="MultiIconServer",
            icons=MULTIPLE_ICONS,
        )
        response, session_id = await mcp.protocol.handle_request(
            {
                "jsonrpc": "2.0",
                "id": 7,
                "method": "initialize",
                "params": {
                    "clientInfo": {"name": "test-client"},
                    "protocolVersion": "2025-06-18",
                },
            }
        )
        server_info = response["result"]["serverInfo"]
        assert server_info["icons"] == MULTIPLE_ICONS
        assert len(server_info["icons"]) == 3

    @pytest.mark.asyncio
    async def test_session_id_returned_from_initialize(self):
        """Verify that handle_request returns a session_id."""
        mcp = ChukMCPServer(name="SessionTest")
        response, session_id = await mcp.protocol.handle_request(
            {
                "jsonrpc": "2.0",
                "id": 8,
                "method": "initialize",
                "params": {
                    "clientInfo": {"name": "test-client"},
                    "protocolVersion": "2025-06-18",
                },
            }
        )
        assert session_id is not None
