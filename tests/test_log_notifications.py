#!/usr/bin/env python3
"""Tests for MCP log message notifications (notifications/message)."""

from unittest.mock import AsyncMock

import pytest

from chuk_mcp_server.constants import McpMethod
from chuk_mcp_server.context import (
    clear_all,
    get_log_fn,
    send_log,
    set_log_fn,
)
from chuk_mcp_server.protocol import MCPProtocolHandler


@pytest.fixture(autouse=True)
def _clean_context():
    """Clean up context after each test."""
    yield
    clear_all()


class TestSendLogNotification:
    """Test MCPProtocolHandler.send_log_notification()."""

    @pytest.fixture
    def handler(self):
        from chuk_mcp_server.types.base import ServerCapabilities, ServerInfo

        info = ServerInfo(name="test", version="1.0")
        caps = ServerCapabilities()
        return MCPProtocolHandler(info, caps)

    @pytest.mark.asyncio
    async def test_sends_notification(self, handler):
        """Log notification should be sent via transport callback."""
        mock_send = AsyncMock()
        handler._send_to_client = mock_send

        await handler.send_log_notification(level="info", data="test message")

        mock_send.assert_called_once()
        notification = mock_send.call_args[0][0]
        assert notification["method"] == McpMethod.NOTIFICATIONS_MESSAGE
        assert notification["params"]["level"] == "info"
        assert notification["params"]["data"] == "test message"
        assert "id" not in notification  # Notifications have no id

    @pytest.mark.asyncio
    async def test_includes_logger_name(self, handler):
        """Logger name should be included when provided."""
        mock_send = AsyncMock()
        handler._send_to_client = mock_send

        await handler.send_log_notification(level="warning", data={"count": 5}, logger_name="myapp.db")

        notification = mock_send.call_args[0][0]
        assert notification["params"]["logger"] == "myapp.db"

    @pytest.mark.asyncio
    async def test_no_logger_name_omitted(self, handler):
        """Logger field should be omitted when not provided."""
        mock_send = AsyncMock()
        handler._send_to_client = mock_send

        await handler.send_log_notification(level="error", data="oops")

        notification = mock_send.call_args[0][0]
        assert "logger" not in notification["params"]

    @pytest.mark.asyncio
    async def test_noop_without_transport(self, handler):
        """Should silently no-op when no transport callback is set."""
        handler._send_to_client = None
        await handler.send_log_notification(level="info", data="ignored")
        # No error raised

    @pytest.mark.asyncio
    async def test_handles_transport_error(self, handler):
        """Should not raise when transport callback fails."""
        mock_send = AsyncMock(side_effect=Exception("transport error"))
        handler._send_to_client = mock_send

        await handler.send_log_notification(level="error", data="test")
        # No error raised

    @pytest.mark.asyncio
    async def test_structured_data(self, handler):
        """Data can be any JSON-serializable value."""
        mock_send = AsyncMock()
        handler._send_to_client = mock_send

        await handler.send_log_notification(
            level="debug",
            data={"query": "SELECT *", "duration_ms": 42},
        )

        notification = mock_send.call_args[0][0]
        assert notification["params"]["data"]["duration_ms"] == 42


class TestContextSendLog:
    """Test the context.send_log() function."""

    @pytest.mark.asyncio
    async def test_send_log_calls_fn(self):
        """send_log() should call the registered log function."""
        mock_fn = AsyncMock()
        set_log_fn(mock_fn)

        await send_log("info", "processing started")

        mock_fn.assert_called_once_with(level="info", data="processing started", logger_name=None)

    @pytest.mark.asyncio
    async def test_send_log_with_logger(self):
        """send_log() should pass logger_name through."""
        mock_fn = AsyncMock()
        set_log_fn(mock_fn)

        await send_log("warning", "slow query", logger_name="db")

        mock_fn.assert_called_once_with(level="warning", data="slow query", logger_name="db")

    @pytest.mark.asyncio
    async def test_send_log_noop_without_fn(self):
        """send_log() should silently no-op when no fn is registered."""
        set_log_fn(None)
        await send_log("error", "this should be ignored")
        # No error raised

    def test_get_set_log_fn(self):
        """get/set_log_fn should work correctly."""
        assert get_log_fn() is None
        mock = AsyncMock()
        set_log_fn(mock)
        assert get_log_fn() is mock
