#!/usr/bin/env python3
"""
Tests for WP2: Request Validation & Size Limits.

Covers:
- Constants are defined with expected values
- Body size limit rejection in HTTP endpoint
- Body size limit acceptance for normal-sized body
- STDIO message size rejection (async and sync transports)
- Argument type validation (not a dict -> INVALID_PARAMS)
- Argument key count validation (too many -> INVALID_PARAMS)
- Normal arguments pass validation
"""

from unittest.mock import AsyncMock, MagicMock

import pytest

from chuk_mcp_server.constants import (
    MAX_ARGUMENT_KEYS,
    MAX_REQUEST_BODY_BYTES,
    JsonRpcError,
)
from chuk_mcp_server.protocol import MCPProtocolHandler
from chuk_mcp_server.types import ServerInfo, ToolHandler, create_server_capabilities

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _create_handler() -> MCPProtocolHandler:
    """Create a protocol handler with tools capability for testing."""
    info = ServerInfo(name="test", version="0.1.0")
    caps = create_server_capabilities(tools=True)
    handler = MCPProtocolHandler(server_info=info, capabilities=caps)
    return handler


def _register_dummy_tool(handler: MCPProtocolHandler) -> None:
    """Register a mock tool on the handler that accepts any arguments."""
    tool = AsyncMock(spec=ToolHandler)
    tool.name = "test_tool"
    tool.requires_auth = False
    tool.output_schema = None
    tool.execute.return_value = "ok"
    handler.register_tool(tool)


# ============================================================================
# Constants tests
# ============================================================================


class TestConstants:
    """Verify request validation constants are defined with expected values."""

    def test_max_request_body_bytes_value(self):
        """MAX_REQUEST_BODY_BYTES should be 10 MB."""
        assert MAX_REQUEST_BODY_BYTES == 10 * 1024 * 1024

    def test_max_argument_keys_value(self):
        """MAX_ARGUMENT_KEYS should be 100."""
        assert MAX_ARGUMENT_KEYS == 100

    def test_constants_are_ints(self):
        """Both constants should be integers."""
        assert isinstance(MAX_REQUEST_BODY_BYTES, int)
        assert isinstance(MAX_ARGUMENT_KEYS, int)


# ============================================================================
# HTTP endpoint body size tests
# ============================================================================


class TestHTTPBodySizeLimit:
    """Test body size validation in the MCP HTTP endpoint."""

    @pytest.mark.asyncio
    async def test_oversized_body_rejected(self):
        """POST with a body exceeding MAX_REQUEST_BODY_BYTES returns error."""
        from chuk_mcp_server.endpoints.mcp import MCPEndpoint

        handler = _create_handler()
        endpoint = MCPEndpoint(handler)

        oversized_body = b"x" * (MAX_REQUEST_BODY_BYTES + 1)

        mock_request = MagicMock()
        mock_request.method = "POST"
        mock_request.headers = {"accept": "application/json"}

        # Make body() an async function returning oversized content
        async def _body():
            return oversized_body

        mock_request.body = _body

        response = await endpoint.handle_request(mock_request)

        assert response.status_code == 400
        import orjson

        body_data = orjson.loads(response.body)
        assert body_data["error"]["code"] == -32600  # INVALID_REQUEST
        assert "too large" in body_data["error"]["message"]

    @pytest.mark.asyncio
    async def test_normal_body_accepted(self):
        """POST with a normal-sized valid JSON-RPC body is processed."""
        from chuk_mcp_server.endpoints.mcp import MCPEndpoint

        handler = _create_handler()
        endpoint = MCPEndpoint(handler)

        import orjson

        normal_body = orjson.dumps(
            {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "initialize",
                "params": {
                    "clientInfo": {"name": "test"},
                    "protocolVersion": "2025-03-26",
                },
            }
        )

        mock_request = MagicMock()
        mock_request.method = "POST"
        mock_request.headers = {"accept": "application/json"}

        async def _body():
            return normal_body

        mock_request.body = _body

        response = await endpoint.handle_request(mock_request)

        # Should succeed (200) with a valid initialize response
        assert response.status_code == 200
        body_data = orjson.loads(response.body)
        assert "result" in body_data
        assert body_data["result"]["serverInfo"]["name"] == "test"


# ============================================================================
# STDIO transport message size tests
# ============================================================================


class TestStdioAsyncMessageSize:
    """Test message size validation in the async StdioTransport."""

    @pytest.mark.asyncio
    async def test_oversized_message_rejected(self):
        """StdioTransport._handle_message rejects oversized messages."""
        from chuk_mcp_server.stdio_transport import StdioTransport

        handler = _create_handler()
        transport = StdioTransport(handler)
        transport.writer = MagicMock()

        # Create a message larger than the limit
        oversized_msg = "x" * (MAX_REQUEST_BODY_BYTES + 1)

        # Patch _send_error to capture the call
        transport._send_error = AsyncMock()

        await transport._handle_message(oversized_msg)

        transport._send_error.assert_called_once()
        args = transport._send_error.call_args
        # _send_error(request_id, code, message)
        assert args[0][0] is None  # request_id
        assert args[0][1] == JsonRpcError.INVALID_REQUEST
        assert "too large" in args[0][2].lower() or "Message too large" in args[0][2]

    @pytest.mark.asyncio
    async def test_normal_message_processed(self):
        """StdioTransport._handle_message processes normal-sized messages."""
        from chuk_mcp_server.stdio_transport import StdioTransport

        handler = _create_handler()
        transport = StdioTransport(handler)
        transport.writer = MagicMock()

        import orjson

        normal_msg = orjson.dumps(
            {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "ping",
            }
        ).decode("utf-8")

        # Patch _send_response to capture
        transport._send_response = AsyncMock()

        await transport._handle_message(normal_msg)

        # Should have sent a response (ping -> pong)
        transport._send_response.assert_called_once()
        response = transport._send_response.call_args[0][0]
        assert response["result"] == {}


class TestStdioSyncMessageSize:
    """Test message size validation in the synchronous StdioSyncTransport."""

    @pytest.mark.asyncio
    async def test_oversized_message_rejected(self):
        """StdioSyncTransport._handle_message rejects oversized messages."""
        from chuk_mcp_server.stdio_transport import StdioSyncTransport

        handler = _create_handler()
        transport = StdioSyncTransport(handler)

        oversized_msg = "x" * (MAX_REQUEST_BODY_BYTES + 1)

        # Patch _send_error to capture the call
        transport._send_error = MagicMock()

        await transport._handle_message(oversized_msg)

        transport._send_error.assert_called_once()
        args = transport._send_error.call_args
        # _send_error(code, message)
        assert args[0][0] == JsonRpcError.INVALID_REQUEST
        assert "too large" in args[0][1].lower() or "Message too large" in args[0][1]


# ============================================================================
# Argument validation tests (protocol handler)
# ============================================================================


class TestArgumentValidation:
    """Test argument validation in _handle_tools_call."""

    @pytest.mark.asyncio
    async def test_arguments_not_dict_rejected(self):
        """Non-dict arguments return INVALID_PARAMS error."""
        handler = _create_handler()
        _register_dummy_tool(handler)

        message = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "tools/call",
            "params": {"name": "test_tool", "arguments": "not_a_dict"},
        }

        response, _ = await handler.handle_request(message)

        assert response is not None
        assert "error" in response
        assert response["error"]["code"] == JsonRpcError.INVALID_PARAMS
        assert "must be an object" in response["error"]["message"]
        assert "str" in response["error"]["message"]

    @pytest.mark.asyncio
    async def test_arguments_list_rejected(self):
        """List arguments return INVALID_PARAMS error."""
        handler = _create_handler()
        _register_dummy_tool(handler)

        message = {
            "jsonrpc": "2.0",
            "id": 2,
            "method": "tools/call",
            "params": {"name": "test_tool", "arguments": [1, 2, 3]},
        }

        response, _ = await handler.handle_request(message)

        assert response is not None
        assert "error" in response
        assert response["error"]["code"] == JsonRpcError.INVALID_PARAMS
        assert "must be an object" in response["error"]["message"]
        assert "list" in response["error"]["message"]

    @pytest.mark.asyncio
    async def test_too_many_argument_keys_rejected(self):
        """Arguments with more than MAX_ARGUMENT_KEYS keys return INVALID_PARAMS."""
        handler = _create_handler()
        _register_dummy_tool(handler)

        too_many_args = {f"key_{i}": i for i in range(MAX_ARGUMENT_KEYS + 1)}

        message = {
            "jsonrpc": "2.0",
            "id": 3,
            "method": "tools/call",
            "params": {"name": "test_tool", "arguments": too_many_args},
        }

        response, _ = await handler.handle_request(message)

        assert response is not None
        assert "error" in response
        assert response["error"]["code"] == JsonRpcError.INVALID_PARAMS
        assert "Too many argument keys" in response["error"]["message"]
        assert str(MAX_ARGUMENT_KEYS) in response["error"]["message"]

    @pytest.mark.asyncio
    async def test_normal_arguments_accepted(self):
        """Normal dict arguments within limits are accepted and tool is called."""
        handler = _create_handler()
        _register_dummy_tool(handler)

        message = {
            "jsonrpc": "2.0",
            "id": 4,
            "method": "tools/call",
            "params": {"name": "test_tool", "arguments": {"key": "value"}},
        }

        response, _ = await handler.handle_request(message)

        assert response is not None
        assert "result" in response
        assert "content" in response["result"]

    @pytest.mark.asyncio
    async def test_max_argument_keys_exactly_accepted(self):
        """Exactly MAX_ARGUMENT_KEYS keys should be accepted (boundary test)."""
        handler = _create_handler()
        _register_dummy_tool(handler)

        exact_args = {f"key_{i}": i for i in range(MAX_ARGUMENT_KEYS)}

        message = {
            "jsonrpc": "2.0",
            "id": 5,
            "method": "tools/call",
            "params": {"name": "test_tool", "arguments": exact_args},
        }

        response, _ = await handler.handle_request(message)

        assert response is not None
        assert "result" in response
        assert "content" in response["result"]

    @pytest.mark.asyncio
    async def test_empty_arguments_accepted(self):
        """Empty dict arguments are accepted."""
        handler = _create_handler()
        _register_dummy_tool(handler)

        message = {
            "jsonrpc": "2.0",
            "id": 6,
            "method": "tools/call",
            "params": {"name": "test_tool", "arguments": {}},
        }

        response, _ = await handler.handle_request(message)

        assert response is not None
        assert "result" in response
        assert "content" in response["result"]
