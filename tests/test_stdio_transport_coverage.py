#!/usr/bin/env python3
"""
Additional tests for stdio_transport.py to achieve 90%+ coverage.
Focuses on error paths and edge cases not covered by existing tests.

Targets these missing lines:
- 76-96:   StdioTransport._send_and_receive (notification + request-response paths)
- 170-178: StdioTransport._handle_message (response routing to pending requests)
- 248->exit: StdioTransport.__exit__ context manager
- 335-367: StdioSyncTransport._send_and_receive (all branches)
- 388, 401-402: StdioSyncTransport.run() exception branches
- 421-426: StdioSyncTransport._handle_message (JSON decode error, general exception)
"""

import asyncio
import json
from io import StringIO
from unittest.mock import AsyncMock, MagicMock, Mock, patch

import orjson
import pytest

from chuk_mcp_server.protocol import MCPProtocolHandler
from chuk_mcp_server.stdio_transport import StdioSyncTransport, StdioTransport
from chuk_mcp_server.types.base import ServerCapabilities, ServerInfo

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_handler() -> MCPProtocolHandler:
    """Create a real MCPProtocolHandler for testing."""
    info = ServerInfo(name="test", version="1.0")
    caps = ServerCapabilities()
    return MCPProtocolHandler(info, caps)


def _make_mock_handler() -> Mock:
    """Create a mock MCPProtocolHandler with required attributes."""
    handler = Mock()
    handler.handle_request = AsyncMock(return_value=({}, None))
    handler.session_manager = Mock()
    handler.session_manager.create_session = Mock(return_value="session123")
    return handler


# ============================================================================
# StdioTransport async tests
# ============================================================================


class TestStdioTransportSendAndReceive:
    """Cover lines 76-96: _send_and_receive notification and request paths."""

    @pytest.fixture
    def transport(self):
        handler = _make_mock_handler()
        t = StdioTransport(handler)
        # Give the transport a writable writer so _send_response works
        t.writer = StringIO()
        return t

    @pytest.mark.asyncio
    async def test_notification_fire_and_forget(self, transport):
        """Notification (no id) sends the message and returns empty dict (lines 76-80)."""
        notification = {
            "jsonrpc": "2.0",
            "method": "notifications/progress",
            "params": {"progressToken": "tok", "progress": 0.5},
        }
        result = await transport._send_and_receive(notification)
        assert result == {}
        # Verify that the notification was written to stdout
        output = transport.writer.getvalue()
        assert "notifications/progress" in output

    @pytest.mark.asyncio
    async def test_request_response_with_future(self, transport):
        """Request (with id) creates a future, sends, and awaits response (lines 82-96)."""
        request = {
            "jsonrpc": "2.0",
            "id": "req-1",
            "method": "sampling/createMessage",
            "params": {"messages": []},
        }

        expected_response = {
            "jsonrpc": "2.0",
            "id": "req-1",
            "result": {"role": "assistant", "content": {"type": "text", "text": "hi"}},
        }

        async def _resolve_future():
            # Give _send_and_receive time to register the future
            await asyncio.sleep(0.05)
            future = transport._pending_requests.get("req-1")
            assert future is not None
            future.set_result(expected_response)

        result, _ = await asyncio.gather(
            transport._send_and_receive(request),
            _resolve_future(),
        )

        assert result == expected_response
        # The future should be cleaned up
        assert "req-1" not in transport._pending_requests

    @pytest.mark.asyncio
    async def test_request_response_timeout(self, transport):
        """Request that times out raises RuntimeError (lines 93-94)."""
        request = {
            "jsonrpc": "2.0",
            "id": "req-timeout",
            "method": "sampling/createMessage",
            "params": {},
        }

        # Use a very short timeout by patching asyncio.wait_for
        original_wait_for = asyncio.wait_for

        async def fast_wait_for(fut, timeout=None):
            return await original_wait_for(fut, timeout=0.01)

        with patch("asyncio.wait_for", side_effect=fast_wait_for):
            with pytest.raises(RuntimeError, match="Timeout waiting for client response"):
                await transport._send_and_receive(request)

        # Cleanup: future should be removed
        assert "req-timeout" not in transport._pending_requests


class TestStdioTransportHandleMessageResponseRouting:
    """Cover lines 170-178: routing responses to pending requests."""

    @pytest.fixture
    def transport(self):
        handler = _make_mock_handler()
        t = StdioTransport(handler)
        t.writer = StringIO()
        return t

    @pytest.mark.asyncio
    async def test_response_routed_to_pending_future(self, transport):
        """Response (no method, has id) resolves a pending future (lines 170-175)."""
        loop = asyncio.get_event_loop()
        future = loop.create_future()
        transport._pending_requests["42"] = future

        response_msg = json.dumps(
            {
                "jsonrpc": "2.0",
                "id": 42,
                "result": {"answer": "yes"},
            }
        )

        await transport._handle_message(response_msg)

        assert future.done()
        assert future.result() == {"jsonrpc": "2.0", "id": 42, "result": {"answer": "yes"}}

    @pytest.mark.asyncio
    async def test_response_for_unknown_request_id(self, transport):
        """Response for unknown request ID is ignored (lines 176-178)."""
        response_msg = json.dumps(
            {
                "jsonrpc": "2.0",
                "id": 999,
                "result": {},
            }
        )

        # Should not raise; just logs and returns
        await transport._handle_message(response_msg)
        # handle_request should NOT have been called
        transport.protocol.handle_request.assert_not_called()

    @pytest.mark.asyncio
    async def test_response_for_already_done_future(self, transport):
        """Response arriving after future is already done does not crash (line 173-174)."""
        loop = asyncio.get_event_loop()
        future = loop.create_future()
        future.set_result({"stale": True})  # Already resolved
        transport._pending_requests["7"] = future

        response_msg = json.dumps(
            {
                "jsonrpc": "2.0",
                "id": 7,
                "result": {"late": True},
            }
        )

        # Should not raise
        await transport._handle_message(response_msg)

    @pytest.mark.asyncio
    async def test_regular_request_still_routes_to_protocol(self, transport):
        """Message with method key routes normally through protocol handler."""
        transport.protocol.handle_request.return_value = (
            {"jsonrpc": "2.0", "id": 1, "result": {}},
            None,
        )

        request_msg = json.dumps(
            {
                "jsonrpc": "2.0",
                "method": "ping",
                "id": 1,
            }
        )

        await transport._handle_message(request_msg)
        transport.protocol.handle_request.assert_called_once()


class TestStdioTransportContextManager:
    """Cover line 248->exit: __exit__ with and without a running event loop."""

    def test_exit_with_running_loop(self):
        """__exit__ creates a stop task when a loop is running (line 259-260)."""
        handler = _make_mock_handler()
        transport = StdioTransport(handler)

        mock_loop = MagicMock()
        with patch("asyncio.get_running_loop", return_value=mock_loop):
            transport.__exit__(None, None, None)

        mock_loop.create_task.assert_called_once()

    def test_exit_without_running_loop(self):
        """__exit__ sets running=False when no event loop is running (lines 261-263)."""
        handler = _make_mock_handler()
        transport = StdioTransport(handler)
        transport.running = True

        with patch("asyncio.get_running_loop", side_effect=RuntimeError("No loop")):
            transport.__exit__(None, None, None)

        assert transport.running is False

    def test_context_manager_enter_and_exit(self):
        """Full context manager round-trip."""
        handler = _make_mock_handler()
        transport = StdioTransport(handler)

        with patch("asyncio.get_running_loop", side_effect=RuntimeError("No loop")):
            with transport as t:
                assert t is transport
            # After exiting, running should be False
            assert transport.running is False


# ============================================================================
# StdioSyncTransport tests
# ============================================================================


class TestStdioSyncTransportSendAndReceive:
    """Cover lines 335-367: _send_and_receive all branches."""

    @pytest.fixture
    def transport(self):
        handler = _make_mock_handler()
        return StdioSyncTransport(handler)

    @pytest.mark.asyncio
    async def test_notification_returns_empty_dict(self, transport):
        """Notification (no id) sends response and returns {} (lines 336-339)."""
        notification = {
            "jsonrpc": "2.0",
            "method": "notifications/progress",
            "params": {"progress": 1.0},
        }

        with patch.object(transport, "_send_response") as mock_send:
            result = await transport._send_and_receive(notification)

        assert result == {}
        mock_send.assert_called_once_with(notification)

    @pytest.mark.asyncio
    async def test_request_response_success(self, transport):
        """Request with id sends and reads matching response (lines 341-367)."""
        response_json = (
            orjson.dumps(
                {
                    "jsonrpc": "2.0",
                    "id": "req-42",
                    "result": {"model": "gpt-4"},
                }
            ).decode("utf-8")
            + "\n"
        )

        request = {
            "jsonrpc": "2.0",
            "id": "req-42",
            "method": "sampling/createMessage",
            "params": {"messages": []},
        }

        with patch.object(transport, "_send_response"):
            with patch("chuk_mcp_server.stdio_transport.sys.stdin") as mock_stdin:
                mock_stdin.readline = MagicMock(return_value=response_json)
                result = await transport._send_and_receive(request)

        assert result["id"] == "req-42"
        assert result["result"]["model"] == "gpt-4"

    @pytest.mark.asyncio
    async def test_request_response_timeout(self, transport):
        """Timeout raises RuntimeError (lines 350-351)."""
        request = {
            "jsonrpc": "2.0",
            "id": "req-slow",
            "method": "sampling/createMessage",
            "params": {},
        }

        with patch.object(transport, "_send_response"):
            with patch(
                "chuk_mcp_server.stdio_transport.asyncio.wait_for",
                side_effect=TimeoutError("timed out"),
            ):
                with pytest.raises(RuntimeError, match="Timeout waiting for client response"):
                    await transport._send_and_receive(request)

    @pytest.mark.asyncio
    async def test_request_response_stdin_closed(self, transport):
        """Empty line from stdin means client closed connection (lines 353-354)."""
        request = {
            "jsonrpc": "2.0",
            "id": "req-eof",
            "method": "sampling/createMessage",
            "params": {},
        }

        with patch.object(transport, "_send_response"):
            with patch("chuk_mcp_server.stdio_transport.sys.stdin") as mock_stdin:
                mock_stdin.readline = MagicMock(return_value="")
                with pytest.raises(RuntimeError, match="Client closed stdin"):
                    await transport._send_and_receive(request)

    @pytest.mark.asyncio
    async def test_request_response_empty_stripped_line(self, transport):
        """Whitespace-only response raises RuntimeError (lines 357-358)."""
        request = {
            "jsonrpc": "2.0",
            "id": "req-blank",
            "method": "sampling/createMessage",
            "params": {},
        }

        with patch.object(transport, "_send_response"):
            with patch("chuk_mcp_server.stdio_transport.sys.stdin") as mock_stdin:
                mock_stdin.readline = MagicMock(return_value="   \n")
                with pytest.raises(RuntimeError, match="Empty response from client"):
                    await transport._send_and_receive(request)

    @pytest.mark.asyncio
    async def test_request_response_id_mismatch(self, transport):
        """Response ID mismatch raises RuntimeError (lines 364-365)."""
        response_json = (
            orjson.dumps(
                {
                    "jsonrpc": "2.0",
                    "id": "wrong-id",
                    "result": {},
                }
            ).decode("utf-8")
            + "\n"
        )

        request = {
            "jsonrpc": "2.0",
            "id": "req-correct",
            "method": "sampling/createMessage",
            "params": {},
        }

        with patch.object(transport, "_send_response"):
            with patch("chuk_mcp_server.stdio_transport.sys.stdin") as mock_stdin:
                mock_stdin.readline = MagicMock(return_value=response_json)
                with pytest.raises(RuntimeError, match="Response ID mismatch"):
                    await transport._send_and_receive(request)


class TestStdioSyncTransportRun:
    """Cover lines 388 and 401-402: run() exception branches."""

    @pytest.fixture
    def transport(self):
        handler = _make_mock_handler()
        return StdioSyncTransport(handler)

    def test_run_processes_valid_message(self, transport):
        """run() reads a valid JSON-RPC message and calls _handle_message (line 388)."""
        msg = json.dumps({"jsonrpc": "2.0", "method": "ping", "id": 1})
        call_count = [0]

        def readline_side_effect():
            call_count[0] += 1
            if call_count[0] == 1:
                return msg + "\n"
            return ""  # EOF

        with patch("sys.stdin.readline", side_effect=readline_side_effect):
            with patch("builtins.print"):
                transport.run()

        transport.protocol.handle_request.assert_called_once()

    def test_run_exception_in_handle_message(self, transport):
        """run() catches exceptions from loop.run_until_complete and sends error (line 388 + 392-399)."""
        msg = json.dumps({"jsonrpc": "2.0", "method": "ping", "id": 1})
        call_count = [0]

        def readline_side_effect():
            call_count[0] += 1
            if call_count[0] == 1:
                return msg + "\n"
            return ""  # EOF

        # Make handle_request raise
        transport.protocol.handle_request = AsyncMock(side_effect=RuntimeError("handle boom"))

        captured_output = []

        def mock_print(line, flush=True):
            captured_output.append(line)

        with patch("sys.stdin.readline", side_effect=readline_side_effect):
            with patch("builtins.print", side_effect=mock_print):
                transport.run()

        # Should have printed an error response for the exception from _handle_message,
        # and also from the except block in the run() while loop
        all_output = " ".join(captured_output)
        # The exception is caught at _handle_message level or at run() level
        assert "error" in all_output.lower() or "Internal error" in all_output

    def test_run_outer_exception(self, transport):
        """run() outer except catches unexpected exceptions (lines 401-402).

        The outer except is reached when _send_response (called inside the
        inner except) itself raises, propagating out of the inner handler.
        """
        msg = json.dumps({"jsonrpc": "2.0", "method": "ping", "id": 1})
        call_count = [0]

        def readline_side_effect():
            call_count[0] += 1
            if call_count[0] == 1:
                return msg + "\n"
            return ""  # EOF

        # Make handle_request raise (enters inner except on line 392)
        transport.protocol.handle_request = AsyncMock(side_effect=RuntimeError("inner boom"))

        # Make _send_response also raise so the inner except block
        # propagates to the outer except (line 401)
        with patch("sys.stdin.readline", side_effect=readline_side_effect):
            with patch.object(transport, "_send_response", side_effect=RuntimeError("send also boom")):
                transport.run()

        # Should not crash -- outer except catches it and loop.close() runs

    def test_run_eof_immediate(self, transport):
        """run() exits cleanly on immediate EOF."""
        with patch("sys.stdin.readline", return_value=""):
            transport.run()

        transport.protocol.handle_request.assert_not_called()


class TestStdioSyncTransportHandleMessage:
    """Cover lines 421-426: _handle_message JSON decode error and general exception."""

    @pytest.fixture
    def transport(self):
        handler = _make_mock_handler()
        return StdioSyncTransport(handler)

    @pytest.mark.asyncio
    async def test_json_decode_error(self, transport):
        """Invalid JSON triggers orjson.JSONDecodeError path (lines 428-430)."""
        with patch.object(transport, "_send_error") as mock_send_error:
            await transport._handle_message("not json at all {{{")

        mock_send_error.assert_called_once()
        args = mock_send_error.call_args[0]
        assert args[0] == -32700  # PARSE_ERROR
        assert "Parse error" in args[1]

    @pytest.mark.asyncio
    async def test_general_exception_in_protocol(self, transport):
        """Exception in protocol handler triggers general except (lines 431-433)."""
        transport.protocol.handle_request = AsyncMock(side_effect=ValueError("unexpected failure"))

        valid_msg = json.dumps({"jsonrpc": "2.0", "method": "ping", "id": 1})

        with patch.object(transport, "_send_error") as mock_send_error:
            await transport._handle_message(valid_msg)

        mock_send_error.assert_called_once()
        args = mock_send_error.call_args[0]
        assert args[0] == -32603  # INTERNAL_ERROR
        assert "Internal error" in args[1]
        assert "unexpected failure" in args[1]

    @pytest.mark.asyncio
    async def test_handle_message_with_new_session_id(self, transport):
        """_handle_message stores session_id when returned (lines 421-422)."""
        transport.protocol.handle_request = AsyncMock(
            return_value=({"jsonrpc": "2.0", "id": 1, "result": {}}, "new-session-abc")
        )

        msg = json.dumps({"jsonrpc": "2.0", "method": "initialize", "id": 1, "params": {}})

        with patch.object(transport, "_send_response"):
            await transport._handle_message(msg)

        assert transport.session_id == "new-session-abc"

    @pytest.mark.asyncio
    async def test_handle_message_sends_response(self, transport):
        """_handle_message sends response when one is generated (lines 424-426)."""
        response = {"jsonrpc": "2.0", "id": 1, "result": {"ok": True}}
        transport.protocol.handle_request = AsyncMock(return_value=(response, None))

        msg = json.dumps({"jsonrpc": "2.0", "method": "ping", "id": 1})

        with patch.object(transport, "_send_response") as mock_send:
            await transport._handle_message(msg)

        mock_send.assert_called_once_with(response)

    @pytest.mark.asyncio
    async def test_handle_message_no_response(self, transport):
        """_handle_message does not send when response is empty (notification ack)."""
        transport.protocol.handle_request = AsyncMock(return_value=(None, None))

        msg = json.dumps({"jsonrpc": "2.0", "method": "notifications/initialized"})

        with patch.object(transport, "_send_response") as mock_send:
            await transport._handle_message(msg)

        mock_send.assert_not_called()


# ============================================================================
# Integration-style tests with real MCPProtocolHandler
# ============================================================================


class TestStdioTransportWithRealHandler:
    """Verify the transport works with a real MCPProtocolHandler."""

    @pytest.mark.asyncio
    async def test_send_and_receive_notification_real_handler(self):
        """_send_and_receive with real handler, notification path."""
        handler = _make_handler()
        transport = StdioTransport(handler)
        transport.writer = StringIO()

        result = await transport._send_and_receive(
            {
                "jsonrpc": "2.0",
                "method": "notifications/progress",
                "params": {"progressToken": "t", "progress": 1},
            }
        )

        assert result == {}

    @pytest.mark.asyncio
    async def test_send_and_receive_request_real_handler(self):
        """_send_and_receive with real handler, request path resolves correctly."""
        handler = _make_handler()
        transport = StdioTransport(handler)
        transport.writer = StringIO()

        request = {
            "jsonrpc": "2.0",
            "id": "real-1",
            "method": "sampling/createMessage",
            "params": {"messages": []},
        }

        expected = {"jsonrpc": "2.0", "id": "real-1", "result": {}}

        async def _resolve():
            await asyncio.sleep(0.05)
            f = transport._pending_requests.get("real-1")
            if f and not f.done():
                f.set_result(expected)

        result, _ = await asyncio.gather(
            transport._send_and_receive(request),
            _resolve(),
        )

        assert result == expected


class TestStdioSyncTransportWithRealHandler:
    """Verify StdioSyncTransport with a real MCPProtocolHandler."""

    @pytest.mark.asyncio
    async def test_send_and_receive_notification_real_handler(self):
        """Notification path with real handler."""
        handler = _make_handler()
        transport = StdioSyncTransport(handler)

        with patch.object(transport, "_send_response"):
            result = await transport._send_and_receive(
                {
                    "jsonrpc": "2.0",
                    "method": "notifications/message",
                    "params": {"level": "info", "data": "test"},
                }
            )

        assert result == {}

    @pytest.mark.asyncio
    async def test_send_and_receive_request_real_handler(self):
        """Request path with real handler."""
        handler = _make_handler()
        transport = StdioSyncTransport(handler)

        response_line = (
            orjson.dumps(
                {
                    "jsonrpc": "2.0",
                    "id": "sync-1",
                    "result": {"ok": True},
                }
            ).decode("utf-8")
            + "\n"
        )

        request = {
            "jsonrpc": "2.0",
            "id": "sync-1",
            "method": "sampling/createMessage",
            "params": {},
        }

        with patch.object(transport, "_send_response"):
            with patch("chuk_mcp_server.stdio_transport.sys.stdin") as mock_stdin:
                mock_stdin.readline = MagicMock(return_value=response_line)
                result = await transport._send_and_receive(request)

        assert result["id"] == "sync-1"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
