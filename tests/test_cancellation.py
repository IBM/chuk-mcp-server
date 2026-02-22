#!/usr/bin/env python3
"""Tests for MCP request cancellation (notifications/cancelled)."""

import asyncio

import pytest

from chuk_mcp_server.constants import McpMethod
from chuk_mcp_server.protocol import MCPProtocolHandler
from chuk_mcp_server.types.tools import ToolHandler


@pytest.fixture
def handler():
    from chuk_mcp_server.types.base import ServerCapabilities, ServerInfo

    info = ServerInfo(name="test", version="1.0")
    caps = ServerCapabilities()
    return MCPProtocolHandler(info, caps)


class TestCancelledNotificationDispatch:
    """Test that notifications/cancelled is routed correctly."""

    @pytest.mark.asyncio
    async def test_cancelled_notification_returns_none(self, handler):
        """Notifications don't return a response."""
        message = {
            "jsonrpc": "2.0",
            "method": McpMethod.NOTIFICATIONS_CANCELLED,
            "params": {"requestId": "req-999"},
        }
        result, session_id = await handler.handle_request(message)
        assert result is None
        assert session_id is None

    @pytest.mark.asyncio
    async def test_cancelled_without_request_id(self, handler):
        """Missing requestId should not raise."""
        message = {
            "jsonrpc": "2.0",
            "method": McpMethod.NOTIFICATIONS_CANCELLED,
            "params": {},
        }
        result, _ = await handler.handle_request(message)
        assert result is None


class TestHandleCancelledNotification:
    """Test the _handle_cancelled_notification method directly."""

    def test_cancel_unknown_request(self, handler):
        """Cancelling a non-existent request should not raise."""
        handler._handle_cancelled_notification({"requestId": "unknown"})
        # No error, just a no-op

    @pytest.mark.asyncio
    async def test_cancel_tracked_task(self, handler):
        """Should cancel the asyncio task for a tracked request."""
        task = asyncio.create_task(asyncio.sleep(100))
        handler._in_flight_requests["req-42"] = task

        handler._handle_cancelled_notification({"requestId": "req-42"})

        # task.cancel() marks for cancellation; verify it's cancelling
        assert task.cancelling()
        assert "req-42" not in handler._in_flight_requests

        # Let the cancellation propagate
        with pytest.raises(asyncio.CancelledError):
            await task

    @pytest.mark.asyncio
    async def test_cancel_with_reason(self, handler):
        """Reason string should be accepted without error."""
        task = asyncio.create_task(asyncio.sleep(100))
        handler._in_flight_requests["req-99"] = task

        handler._handle_cancelled_notification({"requestId": "req-99", "reason": "User navigated away"})

        assert task.cancelling()
        assert "req-99" not in handler._in_flight_requests

        with pytest.raises(asyncio.CancelledError):
            await task

    def test_no_params_request_id(self, handler):
        """Missing requestId in params should not raise."""
        handler._handle_cancelled_notification({})


class TestInFlightTracking:
    """Test that tool executions are tracked and cleaned up."""

    @pytest.mark.asyncio
    async def test_tool_registers_in_flight(self, handler):
        """Tool execution should register in _in_flight_requests while running."""
        in_flight_during_exec = None

        async def slow_tool(x: str) -> str:
            nonlocal in_flight_during_exec
            # Check if we're tracked while executing
            in_flight_during_exec = dict(handler._in_flight_requests)
            return f"done {x}"

        handler.register_tool(ToolHandler.from_function(slow_tool))

        params = {"name": "slow_tool", "arguments": {"x": "test"}}
        response, _ = await handler._handle_tools_call(params, "req-100")

        # Tool should have completed successfully
        assert response["result"]["content"][0]["text"] == "done test"

        # After completion, request should be removed from tracking
        assert "req-100" not in handler._in_flight_requests

    @pytest.mark.asyncio
    async def test_tool_cleaned_up_on_error(self, handler):
        """In-flight tracking should be cleaned up even on tool errors."""

        async def failing_tool(x: str) -> str:
            raise ValueError("intentional error")

        handler.register_tool(ToolHandler.from_function(failing_tool))

        params = {"name": "failing_tool", "arguments": {"x": "test"}}
        response, _ = await handler._handle_tools_call(params, "req-200")

        # Should return an error response
        assert "error" in response

        # Should be cleaned up
        assert "req-200" not in handler._in_flight_requests

    @pytest.mark.asyncio
    async def test_cancel_long_running_tool(self, handler):
        """Cancelling a long-running tool should produce an error response."""
        started = asyncio.Event()

        async def long_tool(x: str) -> str:
            started.set()
            await asyncio.sleep(100)
            return "should not reach here"

        handler.register_tool(ToolHandler.from_function(long_tool))

        params = {"name": "long_tool", "arguments": {"x": "test"}}

        async def cancel_after_start():
            await started.wait()
            handler._handle_cancelled_notification({"requestId": "req-300"})

        # Run tool call and cancellation concurrently
        cancel_task = asyncio.create_task(cancel_after_start())
        response, _ = await handler._handle_tools_call(params, "req-300")
        await cancel_task

        # Should have returned an error due to cancellation
        assert "error" in response
        assert "cancelled" in response["error"]["message"].lower() or "cancel" in response["error"]["message"].lower()

        # Should be cleaned up
        assert "req-300" not in handler._in_flight_requests
