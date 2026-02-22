#!/usr/bin/env python3
"""
Tests for WP4: Narrow Exception Handling & Pending Request Cleanup.

Covers:
- handle_request re-raises asyncio.CancelledError
- handle_request catches ValueError/TypeError/KeyError as INVALID_PARAMS
- handle_request catches generic Exception as INTERNAL_ERROR with sanitized message
- StdioTransport.stop() cancels pending futures
- StdioTransport._send_and_receive rejects when at MAX_PENDING_REQUESTS
- Constants are defined correctly
"""

import asyncio
from unittest.mock import MagicMock

import pytest

from chuk_mcp_server.constants import (
    JSONRPC_KEY,
    KEY_ERROR,
    KEY_ID,
    KEY_METHOD,
    KEY_PARAMS,
    MAX_PENDING_REQUESTS,
    JsonRpcError,
)
from chuk_mcp_server.protocol import MCPProtocolHandler
from chuk_mcp_server.stdio_transport import StdioTransport
from chuk_mcp_server.types import ServerInfo, create_server_capabilities

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_handler() -> MCPProtocolHandler:
    """Create a minimal MCPProtocolHandler for testing."""
    info = ServerInfo(name="test", version="0.1.0")
    caps = create_server_capabilities(tools=True)
    return MCPProtocolHandler(server_info=info, capabilities=caps)


def _make_request(method: str, params: dict | None = None, msg_id: int = 1) -> dict:
    """Build a JSON-RPC request dict."""
    req: dict = {JSONRPC_KEY: "2.0", KEY_METHOD: method, KEY_ID: msg_id}
    if params is not None:
        req[KEY_PARAMS] = params
    return req


# ============================================================================
# Protocol handler – narrowed exception handling
# ============================================================================


class TestHandleRequestCancelledError:
    """handle_request must re-raise asyncio.CancelledError."""

    @pytest.mark.asyncio
    async def test_cancelled_error_propagates(self):
        handler = _make_handler()

        # Patch the internal initialize handler to raise CancelledError
        async def _raise_cancelled(*a, **kw):
            raise asyncio.CancelledError()

        handler._handle_initialize = _raise_cancelled  # type: ignore[assignment]

        with pytest.raises(asyncio.CancelledError):
            await handler.handle_request(
                _make_request("initialize", {"clientInfo": {"name": "c"}, "protocolVersion": "2025-06-18"})
            )


class TestHandleRequestValueError:
    """ValueError should be caught and returned as INVALID_PARAMS."""

    @pytest.mark.asyncio
    async def test_value_error_returns_invalid_params(self):
        handler = _make_handler()

        async def _raise_value_error(*a, **kw):
            raise ValueError("bad value")

        handler._handle_ping = _raise_value_error  # type: ignore[assignment]

        response, session_id = await handler.handle_request(_make_request("ping"))
        assert response is not None
        assert KEY_ERROR in response
        assert response[KEY_ERROR]["code"] == JsonRpcError.INVALID_PARAMS
        assert "bad value" in response[KEY_ERROR]["message"]


class TestHandleRequestTypeError:
    """TypeError should be caught and returned as INVALID_PARAMS."""

    @pytest.mark.asyncio
    async def test_type_error_returns_invalid_params(self):
        handler = _make_handler()

        async def _raise_type_error(*a, **kw):
            raise TypeError("wrong type")

        handler._handle_ping = _raise_type_error  # type: ignore[assignment]

        response, session_id = await handler.handle_request(_make_request("ping"))
        assert response is not None
        assert KEY_ERROR in response
        assert response[KEY_ERROR]["code"] == JsonRpcError.INVALID_PARAMS
        assert "wrong type" in response[KEY_ERROR]["message"]


class TestHandleRequestKeyError:
    """KeyError should be caught and returned as INVALID_PARAMS."""

    @pytest.mark.asyncio
    async def test_key_error_returns_invalid_params(self):
        handler = _make_handler()

        async def _raise_key_error(*a, **kw):
            raise KeyError("missing_key")

        handler._handle_ping = _raise_key_error  # type: ignore[assignment]

        response, session_id = await handler.handle_request(_make_request("ping"))
        assert response is not None
        assert KEY_ERROR in response
        assert response[KEY_ERROR]["code"] == JsonRpcError.INVALID_PARAMS
        assert "missing_key" in response[KEY_ERROR]["message"]


class TestHandleRequestGenericException:
    """Generic exceptions should be INTERNAL_ERROR with a sanitized message."""

    @pytest.mark.asyncio
    async def test_generic_exception_returns_internal_error_sanitized(self):
        handler = _make_handler()

        async def _raise_runtime(*a, **kw):
            raise RuntimeError("secret internal detail")

        handler._handle_ping = _raise_runtime  # type: ignore[assignment]

        response, session_id = await handler.handle_request(_make_request("ping"))
        assert response is not None
        assert KEY_ERROR in response
        assert response[KEY_ERROR]["code"] == JsonRpcError.INTERNAL_ERROR
        # The message must NOT leak the internal detail
        assert response[KEY_ERROR]["message"] == "Internal server error"
        assert "secret" not in response[KEY_ERROR]["message"]


# ============================================================================
# StdioTransport – pending request cleanup on stop
# ============================================================================


class TestStdioTransportStopCancelsPending:
    """StdioTransport.stop() must cancel all pending futures and clear the dict."""

    @pytest.mark.asyncio
    async def test_stop_cancels_pending_futures(self):
        handler = _make_handler()
        transport = StdioTransport(handler)

        loop = asyncio.get_event_loop()
        f1: asyncio.Future[dict] = loop.create_future()
        f2: asyncio.Future[dict] = loop.create_future()
        transport._pending_requests["req-1"] = f1
        transport._pending_requests["req-2"] = f2

        await transport.stop()

        assert f1.cancelled()
        assert f2.cancelled()
        assert len(transport._pending_requests) == 0
        assert transport.running is False

    @pytest.mark.asyncio
    async def test_stop_skips_already_done_futures(self):
        handler = _make_handler()
        transport = StdioTransport(handler)

        loop = asyncio.get_event_loop()
        done_future: asyncio.Future[dict] = loop.create_future()
        done_future.set_result({"ok": True})
        pending_future: asyncio.Future[dict] = loop.create_future()

        transport._pending_requests["done"] = done_future
        transport._pending_requests["pending"] = pending_future

        await transport.stop()

        # done_future was already done, so cancel() is a no-op but it should not raise
        assert done_future.done()
        assert pending_future.cancelled()
        assert len(transport._pending_requests) == 0


# ============================================================================
# StdioTransport – MAX_PENDING_REQUESTS enforcement
# ============================================================================


class TestSendAndReceivePendingLimit:
    """_send_and_receive must reject when at MAX_PENDING_REQUESTS."""

    @pytest.mark.asyncio
    async def test_rejects_when_at_limit(self):
        handler = _make_handler()
        transport = StdioTransport(handler)

        loop = asyncio.get_event_loop()
        # Fill up to the limit
        for i in range(MAX_PENDING_REQUESTS):
            transport._pending_requests[str(i)] = loop.create_future()

        request = {JSONRPC_KEY: "2.0", KEY_ID: "overflow-id", KEY_METHOD: "ping"}

        with pytest.raises(RuntimeError, match="Too many pending requests"):
            await transport._send_and_receive(request)

        # Cleanup futures to avoid warnings
        for f in transport._pending_requests.values():
            f.cancel()

    @pytest.mark.asyncio
    async def test_notifications_bypass_limit(self):
        """Notifications (no id) should bypass the pending request limit."""
        handler = _make_handler()
        transport = StdioTransport(handler)
        transport.writer = MagicMock()

        loop = asyncio.get_event_loop()
        # Fill up to the limit
        for i in range(MAX_PENDING_REQUESTS):
            transport._pending_requests[str(i)] = loop.create_future()

        # Notification has no id — should succeed even at limit
        notification = {JSONRPC_KEY: "2.0", KEY_METHOD: "notifications/progress", KEY_PARAMS: {}}
        result = await transport._send_and_receive(notification)
        assert result == {}

        # Cleanup futures to avoid warnings
        for f in transport._pending_requests.values():
            f.cancel()


# ============================================================================
# Constants validation
# ============================================================================


class TestConstants:
    """Verify the constants are defined and have sensible values."""

    def test_max_pending_requests_defined(self):
        assert MAX_PENDING_REQUESTS == 100

    def test_json_rpc_error_invalid_params(self):
        assert JsonRpcError.INVALID_PARAMS == -32602

    def test_json_rpc_error_internal_error(self):
        assert JsonRpcError.INTERNAL_ERROR == -32603

    def test_json_rpc_error_parse_error(self):
        assert JsonRpcError.PARSE_ERROR == -32700
