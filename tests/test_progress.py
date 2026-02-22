#!/usr/bin/env python3
"""
Tests for progress notifications.

Tests cover progress context management, notification format, and integration
with the tool execution lifecycle.
"""

import pytest

from chuk_mcp_server.context import (
    clear_all,
    get_progress_notify_fn,
    send_progress,
    set_progress_notify_fn,
    set_progress_token,
    set_session_id,
)
from chuk_mcp_server.protocol import MCPProtocolHandler
from chuk_mcp_server.types import ServerInfo, ToolHandler, create_server_capabilities


@pytest.fixture(autouse=True)
def cleanup_context():
    """Ensure context is clean before and after each test."""
    clear_all()
    yield
    clear_all()


class TestProgress:
    """Test progress notification context management."""

    def test_get_progress_notify_fn_default(self):
        """get_progress_notify_fn returns None by default."""
        assert get_progress_notify_fn() is None

    def test_set_and_get_progress_notify_fn(self):
        """set_progress_notify_fn and get_progress_notify_fn round-trip."""

        async def mock_fn(**kwargs):
            pass

        set_progress_notify_fn(mock_fn)
        assert get_progress_notify_fn() is mock_fn

    def test_set_progress_notify_fn_none(self):
        """Setting progress notify fn to None clears it."""

        async def mock_fn(**kwargs):
            pass

        set_progress_notify_fn(mock_fn)
        set_progress_notify_fn(None)
        assert get_progress_notify_fn() is None

    @pytest.mark.asyncio
    async def test_send_progress_noop_when_no_fn(self):
        """send_progress is a no-op when fn is None (no error)."""
        # Should not raise
        await send_progress(progress=50, total=100, message="Processing")

    @pytest.mark.asyncio
    async def test_send_progress_noop_when_no_token(self):
        """send_progress is a no-op when progress_token is None (no error)."""

        async def mock_fn(**kwargs):
            raise RuntimeError("Should not be called")

        set_progress_notify_fn(mock_fn)
        set_progress_token(None)

        # Should not raise or call the fn
        await send_progress(progress=50, total=100)

    @pytest.mark.asyncio
    async def test_send_progress_calls_fn_with_correct_params(self):
        """send_progress calls fn with correct parameters."""
        captured_kwargs = {}

        async def mock_fn(**kwargs):
            captured_kwargs.update(kwargs)

        set_progress_notify_fn(mock_fn)
        set_progress_token("token123")

        await send_progress(progress=50, total=100, message="Halfway done")

        assert captured_kwargs["progress_token"] == "token123"
        assert captured_kwargs["progress"] == 50
        assert captured_kwargs["total"] == 100
        assert captured_kwargs["message"] == "Halfway done"

    @pytest.mark.asyncio
    async def test_multiple_progress_updates(self):
        """send_progress can be called multiple times."""
        call_count = {"count": 0}

        async def mock_fn(**kwargs):
            call_count["count"] += 1

        set_progress_notify_fn(mock_fn)
        set_progress_token("token456")

        await send_progress(progress=25, total=100)
        await send_progress(progress=50, total=100)
        await send_progress(progress=75, total=100)
        await send_progress(progress=100, total=100)

        assert call_count["count"] == 4

    def test_clear_all_resets_progress_notify_fn(self):
        """clear_all() resets the progress notify fn."""

        async def mock_fn(**kwargs):
            pass

        set_progress_notify_fn(mock_fn)
        clear_all()
        assert get_progress_notify_fn() is None


@pytest.mark.asyncio
class TestProtocolProgress:
    """Test progress notification support in protocol handler."""

    async def test_progress_notification_format(self):
        """send_progress_notification has no id, correct method and params."""
        server_info = ServerInfo(name="test", version="1.0.0")
        capabilities = create_server_capabilities()
        handler = MCPProtocolHandler(server_info, capabilities)

        captured_notification = {}

        async def mock_send(notification):
            captured_notification.update(notification)

        handler._send_to_client = mock_send

        await handler.send_progress_notification(
            progress_token="token123",
            progress=50,
            total=100,
            message="Processing",
        )

        # Verify it's a notification (no id field)
        assert "id" not in captured_notification
        assert captured_notification["jsonrpc"] == "2.0"
        assert captured_notification["method"] == "notifications/progress"
        assert captured_notification["params"]["progressToken"] == "token123"
        assert captured_notification["params"]["progress"] == 50
        assert captured_notification["params"]["total"] == 100
        assert captured_notification["params"]["message"] == "Processing"

    async def test_progress_context_cleanup(self):
        """verify fn cleared after tool execution."""
        from chuk_mcp_server.context import get_progress_notify_fn, get_progress_token

        server_info = ServerInfo(name="test", version="1.0.0")
        capabilities = create_server_capabilities()
        handler = MCPProtocolHandler(server_info, capabilities)

        # Track whether progress was available during tool execution
        progress_was_available = {}

        async def dummy_tool(x: int) -> str:
            progress_was_available["fn"] = get_progress_notify_fn() is not None
            progress_was_available["token"] = get_progress_token()
            return str(x)

        tool = ToolHandler.from_function(dummy_tool, name="dummy_tool", description="test")
        handler.register_tool(tool)

        # Initialize session
        params = {
            "clientInfo": {"name": "test-client"},
            "protocolVersion": "2025-03-26",
            "capabilities": {},
        }
        _, session_id = await handler._handle_initialize(params, 1)
        set_session_id(session_id)

        # Set up transport
        async def mock_send(request):
            return {"jsonrpc": "2.0", "id": request["id"], "result": {}}

        handler._send_to_client = mock_send

        # Call the tool with progress token in _meta
        await handler._handle_tools_call(
            {
                "name": "dummy_tool",
                "arguments": {"x": 42},
                "_meta": {"progressToken": "token789"},
            },
            2,
        )

        # Progress fn and token should have been available during tool execution
        assert progress_was_available["fn"] is True
        assert progress_was_available["token"] == "token789"

        # Progress fn and token should be cleared after tool execution
        assert get_progress_notify_fn() is None
        assert get_progress_token() is None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
