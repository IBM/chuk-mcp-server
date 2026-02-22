"""
Tests for Phase 4 features (MCP 2025-11-25):
  1. Tool Calling in Sampling
  2. URL Mode Elicitation
"""

from unittest.mock import AsyncMock

import pytest

# ---------------------------------------------------------------------------
# Feature 1: Tool Calling in Sampling
# ---------------------------------------------------------------------------


class TestSamplingWithTools:
    """send_sampling_request includes tools / toolChoice when provided."""

    @pytest.fixture
    def handler(self):
        from chuk_mcp_server.protocol import MCPProtocolHandler
        from chuk_mcp_server.types import ServerInfo, create_server_capabilities

        return MCPProtocolHandler(
            ServerInfo(name="Test", version="1.0"),
            create_server_capabilities(tools=True),
        )

    # -- tools and tool_choice forwarded to the wire request ----------------

    @pytest.mark.asyncio
    async def test_send_sampling_request_includes_tools_and_tool_choice(self, handler):
        mock_send = AsyncMock(
            return_value={
                "result": {
                    "role": "assistant",
                    "content": {"type": "text", "text": "Hello"},
                    "model": "test",
                    "stopReason": "end_turn",
                }
            }
        )
        handler._send_to_client = mock_send

        tools = [
            {
                "name": "calculator",
                "description": "A calc",
                "inputSchema": {"type": "object"},
            }
        ]
        tool_choice = {"type": "auto"}

        await handler.send_sampling_request(
            messages=[{"role": "user", "content": {"type": "text", "text": "hi"}}],
            tools=tools,
            tool_choice=tool_choice,
        )

        # The mock was called exactly once
        mock_send.assert_called_once()

        call_args = mock_send.call_args[0][0]
        assert call_args["params"]["tools"] == tools
        assert call_args["params"]["toolChoice"] == tool_choice

    @pytest.mark.asyncio
    async def test_send_sampling_request_multiple_tools(self, handler):
        mock_send = AsyncMock(
            return_value={
                "result": {
                    "role": "assistant",
                    "content": {"type": "text", "text": "result"},
                    "model": "test",
                    "stopReason": "end_turn",
                }
            }
        )
        handler._send_to_client = mock_send

        tools = [
            {
                "name": "calculator",
                "description": "Math operations",
                "inputSchema": {"type": "object", "properties": {"expr": {"type": "string"}}},
            },
            {
                "name": "weather",
                "description": "Get weather",
                "inputSchema": {"type": "object", "properties": {"city": {"type": "string"}}},
            },
        ]
        tool_choice = {"type": "any"}

        await handler.send_sampling_request(
            messages=[{"role": "user", "content": {"type": "text", "text": "What is 2+2 and the weather?"}}],
            tools=tools,
            tool_choice=tool_choice,
        )

        call_args = mock_send.call_args[0][0]
        assert len(call_args["params"]["tools"]) == 2
        assert call_args["params"]["tools"][0]["name"] == "calculator"
        assert call_args["params"]["tools"][1]["name"] == "weather"
        assert call_args["params"]["toolChoice"] == {"type": "any"}

    @pytest.mark.asyncio
    async def test_send_sampling_request_tool_choice_none_type(self, handler):
        """tool_choice with type 'none' should still be forwarded."""
        mock_send = AsyncMock(
            return_value={
                "result": {
                    "role": "assistant",
                    "content": {"type": "text", "text": "no tools"},
                    "model": "test",
                    "stopReason": "end_turn",
                }
            }
        )
        handler._send_to_client = mock_send

        tools = [{"name": "calc", "description": "A calc", "inputSchema": {"type": "object"}}]
        tool_choice = {"type": "none"}

        await handler.send_sampling_request(
            messages=[{"role": "user", "content": {"type": "text", "text": "hi"}}],
            tools=tools,
            tool_choice=tool_choice,
        )

        call_args = mock_send.call_args[0][0]
        assert call_args["params"]["toolChoice"] == {"type": "none"}

    # -- tools provided, tool_choice omitted --------------------------------

    @pytest.mark.asyncio
    async def test_send_sampling_request_tools_without_tool_choice(self, handler):
        mock_send = AsyncMock(
            return_value={
                "result": {
                    "role": "assistant",
                    "content": {"type": "text", "text": "ok"},
                    "model": "test",
                    "stopReason": "end_turn",
                }
            }
        )
        handler._send_to_client = mock_send

        tools = [{"name": "calc", "description": "A calc", "inputSchema": {"type": "object"}}]

        await handler.send_sampling_request(
            messages=[{"role": "user", "content": {"type": "text", "text": "hi"}}],
            tools=tools,
        )

        call_args = mock_send.call_args[0][0]
        assert call_args["params"]["tools"] == tools
        # toolChoice should not be present when not supplied
        assert "toolChoice" not in call_args["params"]

    # -- backwards compatibility: no tools at all ---------------------------

    @pytest.mark.asyncio
    async def test_send_sampling_request_without_tools_backwards_compatible(self, handler):
        mock_send = AsyncMock(
            return_value={
                "result": {
                    "role": "assistant",
                    "content": {"type": "text", "text": "Hi"},
                    "model": "test",
                    "stopReason": "end_turn",
                }
            }
        )
        handler._send_to_client = mock_send

        await handler.send_sampling_request(
            messages=[{"role": "user", "content": {"type": "text", "text": "hi"}}],
        )

        call_args = mock_send.call_args[0][0]
        assert "tools" not in call_args["params"]
        assert "toolChoice" not in call_args["params"]

    # -- return value is propagated -----------------------------------------

    @pytest.mark.asyncio
    async def test_send_sampling_request_returns_result(self, handler):
        expected = {
            "role": "assistant",
            "content": {"type": "text", "text": "Hello"},
            "model": "test",
            "stopReason": "end_turn",
        }
        mock_send = AsyncMock(return_value={"result": expected})
        handler._send_to_client = mock_send

        result = await handler.send_sampling_request(
            messages=[{"role": "user", "content": {"type": "text", "text": "hi"}}],
            tools=[{"name": "t", "description": "d", "inputSchema": {"type": "object"}}],
            tool_choice={"type": "auto"},
        )

        assert result == expected


class TestCreateMessageWithTools:
    """context.create_message passes tools / tool_choice through."""

    @pytest.fixture(autouse=True)
    def _cleanup_sampling_fn(self):
        """Ensure set_sampling_fn is reset after every test."""
        yield
        from chuk_mcp_server.context import set_sampling_fn

        set_sampling_fn(None)

    @pytest.mark.asyncio
    async def test_create_message_forwards_tools_and_tool_choice(self):
        from chuk_mcp_server.context import create_message, set_sampling_fn

        mock_fn = AsyncMock(return_value={"role": "assistant", "content": {"type": "text", "text": "OK"}})
        set_sampling_fn(mock_fn)

        await create_message(
            messages=[{"role": "user", "content": {"type": "text", "text": "hi"}}],
            tools=[{"name": "calc"}],
            tool_choice="auto",
        )

        mock_fn.assert_called_once()
        call_kwargs = mock_fn.call_args[1]
        assert call_kwargs["tools"] == [{"name": "calc"}]
        assert call_kwargs["tool_choice"] == "auto"

    @pytest.mark.asyncio
    async def test_create_message_without_tools(self):
        from chuk_mcp_server.context import create_message, set_sampling_fn

        mock_fn = AsyncMock(return_value={"role": "assistant", "content": {"type": "text", "text": "OK"}})
        set_sampling_fn(mock_fn)

        await create_message(
            messages=[{"role": "user", "content": {"type": "text", "text": "hi"}}],
        )

        mock_fn.assert_called_once()
        call_kwargs = mock_fn.call_args[1]
        # tools / tool_choice should either be absent or None
        assert call_kwargs.get("tools") is None
        assert call_kwargs.get("tool_choice") is None

    @pytest.mark.asyncio
    async def test_create_message_returns_sampling_result(self):
        from chuk_mcp_server.context import create_message, set_sampling_fn

        expected = {"role": "assistant", "content": {"type": "text", "text": "Done"}}
        mock_fn = AsyncMock(return_value=expected)
        set_sampling_fn(mock_fn)

        result = await create_message(
            messages=[{"role": "user", "content": {"type": "text", "text": "hi"}}],
            tools=[{"name": "calc"}],
            tool_choice={"type": "auto"},
        )

        assert result == expected


# ---------------------------------------------------------------------------
# Feature 2: URL Mode Elicitation
# ---------------------------------------------------------------------------


class TestURLElicitationRequiredError:
    """URLElicitationRequiredError stores url, description, mime_type."""

    def test_all_fields(self):
        from chuk_mcp_server.types.errors import URLElicitationRequiredError

        err = URLElicitationRequiredError(
            url="https://auth.example.com/oauth",
            description="Please authenticate",
            mime_type="text/html",
        )
        assert err.url == "https://auth.example.com/oauth"
        assert err.description == "Please authenticate"
        assert err.mime_type == "text/html"
        assert "URL elicitation required" in str(err)

    def test_without_optional_fields(self):
        from chuk_mcp_server.types.errors import URLElicitationRequiredError

        err = URLElicitationRequiredError(url="https://example.com")
        assert err.url == "https://example.com"
        assert err.description is None
        assert err.mime_type is None

    def test_is_exception(self):
        from chuk_mcp_server.types.errors import URLElicitationRequiredError

        err = URLElicitationRequiredError(url="https://example.com")
        assert isinstance(err, Exception)

    def test_str_representation(self):
        from chuk_mcp_server.types.errors import URLElicitationRequiredError

        err = URLElicitationRequiredError(
            url="https://auth.example.com/oauth",
            description="Login needed",
        )
        s = str(err)
        assert "URL elicitation required" in s

    def test_url_only_str_representation(self):
        from chuk_mcp_server.types.errors import URLElicitationRequiredError

        err = URLElicitationRequiredError(url="https://example.com/login")
        s = str(err)
        assert "URL elicitation required" in s

    def test_with_description_no_mime(self):
        from chuk_mcp_server.types.errors import URLElicitationRequiredError

        err = URLElicitationRequiredError(
            url="https://example.com/auth",
            description="Authenticate first",
        )
        assert err.url == "https://example.com/auth"
        assert err.description == "Authenticate first"
        assert err.mime_type is None

    def test_with_mime_no_description(self):
        from chuk_mcp_server.types.errors import URLElicitationRequiredError

        err = URLElicitationRequiredError(
            url="https://example.com/auth",
            mime_type="application/json",
        )
        assert err.url == "https://example.com/auth"
        assert err.description is None
        assert err.mime_type == "application/json"


class TestURLElicitationProtocolHandling:
    """Protocol handler catches URLElicitationRequiredError -> error -32042."""

    @pytest.fixture
    def handler(self):
        from chuk_mcp_server.protocol import MCPProtocolHandler
        from chuk_mcp_server.types import ServerInfo, create_server_capabilities

        return MCPProtocolHandler(
            ServerInfo(name="Test", version="1.0"),
            create_server_capabilities(tools=True),
        )

    @pytest.mark.asyncio
    async def test_tool_raising_url_elicitation_returns_32042(self, handler):
        from chuk_mcp_server.types.errors import URLElicitationRequiredError
        from chuk_mcp_server.types.tools import ToolHandler

        async def oauth_tool():
            raise URLElicitationRequiredError(
                url="https://auth.example.com",
                description="Login needed",
            )

        tool = ToolHandler.from_function(oauth_tool, name="oauth_tool")
        handler.register_tool(tool)

        response, _ = await handler.handle_request(
            {
                "jsonrpc": "2.0",
                "id": "1",
                "method": "tools/call",
                "params": {"name": "oauth_tool", "arguments": {}},
            }
        )

        assert response["error"]["code"] == -32042
        assert response["error"]["data"]["url"] == "https://auth.example.com"
        assert response["error"]["data"]["description"] == "Login needed"

    @pytest.mark.asyncio
    async def test_tool_url_elicitation_all_fields_in_error_data(self, handler):
        from chuk_mcp_server.types.errors import URLElicitationRequiredError
        from chuk_mcp_server.types.tools import ToolHandler

        async def full_oauth_tool():
            raise URLElicitationRequiredError(
                url="https://auth.example.com/oauth2",
                description="OAuth required",
                mime_type="text/html",
            )

        tool = ToolHandler.from_function(full_oauth_tool, name="full_oauth_tool")
        handler.register_tool(tool)

        response, _ = await handler.handle_request(
            {
                "jsonrpc": "2.0",
                "id": "2",
                "method": "tools/call",
                "params": {"name": "full_oauth_tool", "arguments": {}},
            }
        )

        assert response["error"]["code"] == -32042
        data = response["error"]["data"]
        assert data["url"] == "https://auth.example.com/oauth2"
        assert data["description"] == "OAuth required"
        assert data["mimeType"] == "text/html"

    @pytest.mark.asyncio
    async def test_tool_url_elicitation_without_optional_fields(self, handler):
        from chuk_mcp_server.types.errors import URLElicitationRequiredError
        from chuk_mcp_server.types.tools import ToolHandler

        async def minimal_tool():
            raise URLElicitationRequiredError(url="https://example.com/login")

        tool = ToolHandler.from_function(minimal_tool, name="minimal_tool")
        handler.register_tool(tool)

        response, _ = await handler.handle_request(
            {
                "jsonrpc": "2.0",
                "id": "3",
                "method": "tools/call",
                "params": {"name": "minimal_tool", "arguments": {}},
            }
        )

        assert response["error"]["code"] == -32042
        data = response["error"]["data"]
        assert data["url"] == "https://example.com/login"

    @pytest.mark.asyncio
    async def test_normal_tool_still_works(self, handler):
        """Ensure tools that do NOT raise URLElicitationRequiredError work normally."""
        from chuk_mcp_server.types.tools import ToolHandler

        async def normal_tool():
            return "all good"

        tool = ToolHandler.from_function(normal_tool, name="normal_tool")
        handler.register_tool(tool)

        response, _ = await handler.handle_request(
            {
                "jsonrpc": "2.0",
                "id": "4",
                "method": "tools/call",
                "params": {"name": "normal_tool", "arguments": {}},
            }
        )

        # Should be a success response, no error key
        assert "error" not in response
        assert "result" in response


class TestURLElicitationImport:
    """URLElicitationRequiredError is importable from the top-level package."""

    def test_import_from_package(self):
        from chuk_mcp_server import URLElicitationRequiredError

        assert URLElicitationRequiredError is not None

    def test_import_from_errors_module(self):
        from chuk_mcp_server.types.errors import URLElicitationRequiredError

        assert URLElicitationRequiredError is not None

    def test_both_imports_are_same_class(self):
        from chuk_mcp_server import URLElicitationRequiredError as FromPkg
        from chuk_mcp_server.types.errors import URLElicitationRequiredError as FromMod

        assert FromPkg is FromMod
