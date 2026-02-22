#!/usr/bin/env python3
"""
Tests for stdio transport functionality.
"""

import asyncio
import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from chuk_mcp_server.protocol import MCPProtocolHandler
from chuk_mcp_server.stdio_transport import StdioTransport, run_stdio_server


@pytest.fixture
def mock_protocol():
    """Create a mock protocol handler."""
    protocol = MagicMock(spec=MCPProtocolHandler)
    protocol.session_manager = MagicMock()
    protocol.session_manager.create_session = MagicMock(return_value="test-session-123")
    protocol.handle_request = AsyncMock(
        return_value=({"jsonrpc": "2.0", "id": 1, "result": {"test": "response"}}, None)
    )
    return protocol


@pytest.fixture
def stdio_transport(mock_protocol):
    """Create a stdio transport instance."""
    return StdioTransport(mock_protocol)


class TestStdioTransport:
    """Test stdio transport functionality."""

    def test_initialization(self, stdio_transport, mock_protocol):
        """Test transport initialization."""
        assert stdio_transport.protocol == mock_protocol
        assert stdio_transport.reader is None
        assert stdio_transport.writer is None
        assert stdio_transport.running is False
        assert stdio_transport.session_id is None

    @pytest.mark.asyncio
    async def test_handle_initialize_message(self, stdio_transport, mock_protocol):
        """Test handling of initialize message."""
        message = json.dumps(
            {
                "jsonrpc": "2.0",
                "method": "initialize",
                "params": {"clientInfo": {"name": "test-client"}, "protocolVersion": "2025-03-26"},
                "id": 1,
            }
        )

        # Mock stdout
        with patch.object(stdio_transport, "writer") as mock_writer:
            mock_writer.write = MagicMock()
            mock_writer.flush = MagicMock()

            await stdio_transport._handle_message(message)

            # Verify session creation
            mock_protocol.session_manager.create_session.assert_called_once_with({"name": "test-client"}, "2025-03-26")

            # Verify request handling
            mock_protocol.handle_request.assert_called_once()

            # Verify response sent
            mock_writer.write.assert_called_once()
            response_data = mock_writer.write.call_args[0][0]
            assert '"result"' in response_data
            assert response_data.endswith("\n")

    @pytest.mark.asyncio
    async def test_handle_tool_call(self, stdio_transport, mock_protocol):
        """Test handling of tool call message."""
        message = json.dumps(
            {
                "jsonrpc": "2.0",
                "method": "tools/call",
                "params": {"name": "echo", "arguments": {"message": "hello"}},
                "id": 2,
            }
        )

        with patch.object(stdio_transport, "writer") as mock_writer:
            mock_writer.write = MagicMock()
            mock_writer.flush = MagicMock()

            await stdio_transport._handle_message(message)

            # Verify request handling
            mock_protocol.handle_request.assert_called_once()
            call_args = mock_protocol.handle_request.call_args[0][0]
            assert call_args["method"] == "tools/call"
            assert call_args["params"]["name"] == "echo"

    @pytest.mark.asyncio
    async def test_handle_notification(self, stdio_transport, mock_protocol):
        """Test handling of notification (no id, no response expected)."""
        message = json.dumps(
            {"jsonrpc": "2.0", "method": "notifications/log", "params": {"level": "info", "message": "test log"}}
        )

        # For notifications, handle_request returns (None, None)
        mock_protocol.handle_request.return_value = (None, None)

        with patch.object(stdio_transport, "writer") as mock_writer:
            mock_writer.write = MagicMock()

            await stdio_transport._handle_message(message)

            # Verify request was handled
            mock_protocol.handle_request.assert_called_once()

            # Verify no response sent for notification
            mock_writer.write.assert_not_called()

    @pytest.mark.asyncio
    async def test_handle_invalid_json(self, stdio_transport):
        """Test handling of invalid JSON."""
        message = "not valid json {"

        with patch.object(stdio_transport, "_send_error") as mock_send_error:
            await stdio_transport._handle_message(message)

            # Verify error response sent
            mock_send_error.assert_called_once()
            error_args = mock_send_error.call_args[0]
            assert error_args[0] is None  # no request id
            assert error_args[1] == -32700  # Parse error code
            assert "Parse error" in error_args[2]

    @pytest.mark.asyncio
    async def test_send_error(self, stdio_transport):
        """Test sending error response."""
        with patch.object(stdio_transport, "writer") as mock_writer:
            mock_writer.write = MagicMock()
            mock_writer.flush = MagicMock()

            await stdio_transport._send_error(123, -32603, "Internal error")

            # Verify error response format
            mock_writer.write.assert_called_once()
            response = mock_writer.write.call_args[0][0]
            data = json.loads(response.rstrip("\n"))

            assert data["jsonrpc"] == "2.0"
            assert data["id"] == 123
            assert data["error"]["code"] == -32603
            assert data["error"]["message"] == "Internal error"

    def test_context_manager(self, stdio_transport):
        """Test context manager interface."""
        with stdio_transport as transport:
            assert transport == stdio_transport

    @pytest.mark.asyncio
    async def test_start_transport(self, stdio_transport):
        """Test starting the transport."""
        with patch("asyncio.get_event_loop") as mock_get_loop:
            mock_loop = MagicMock()
            # Make connect_read_pipe return an async coroutine
            mock_loop.connect_read_pipe = AsyncMock(return_value=(None, None))
            mock_get_loop.return_value = mock_loop

            with patch("asyncio.StreamReader") as mock_reader_class:
                mock_reader = MagicMock()
                mock_reader_class.return_value = mock_reader

                with patch("asyncio.StreamReaderProtocol"):
                    # Use AsyncMock for the async _listen method
                    with patch.object(stdio_transport, "_listen", new_callable=AsyncMock) as mock_listen:
                        await stdio_transport.start()

                        assert stdio_transport.running is True
                        assert stdio_transport.reader is not None
                        mock_listen.assert_called_once()

    @pytest.mark.asyncio
    async def test_stop_transport(self, stdio_transport):
        """Test stopping the transport."""
        # Set up transport as if it's running
        stdio_transport.running = True
        mock_reader = MagicMock()
        stdio_transport.reader = mock_reader

        await stdio_transport.stop()

        assert stdio_transport.running is False
        mock_reader.feed_eof.assert_called_once()

    @pytest.mark.asyncio
    async def test_listen_with_data(self, stdio_transport):
        """Test listening for messages."""
        # Create mock reader that returns data then EOF
        mock_reader = AsyncMock()
        mock_reader.read = AsyncMock(
            side_effect=[
                b'{"jsonrpc":"2.0","method":"test","id":1}\n',
                b"",  # EOF
            ]
        )
        stdio_transport.reader = mock_reader
        stdio_transport.running = True

        with patch.object(stdio_transport, "_handle_message", new_callable=AsyncMock) as mock_handle:
            # Patch running flag to stop after first message
            async def handle_side_effect(msg):
                stdio_transport.running = False

            mock_handle.side_effect = handle_side_effect

            await stdio_transport._listen()

            mock_handle.assert_called_once_with('{"jsonrpc":"2.0","method":"test","id":1}')

    @pytest.mark.asyncio
    async def test_listen_with_partial_messages(self, stdio_transport):
        """Test listening with partial message chunks."""
        mock_reader = AsyncMock()
        mock_reader.read = AsyncMock(
            side_effect=[
                b'{"jsonrpc":"2.0",',
                b'"method":"test","id":1}\n',
                b"",  # EOF
            ]
        )
        stdio_transport.reader = mock_reader
        stdio_transport.running = True

        with patch.object(stdio_transport, "_handle_message", new_callable=AsyncMock) as mock_handle:
            # Stop after handling message
            async def handle_side_effect(msg):
                stdio_transport.running = False

            mock_handle.side_effect = handle_side_effect

            await stdio_transport._listen()

            mock_handle.assert_called_once_with('{"jsonrpc":"2.0","method":"test","id":1}')

    @pytest.mark.asyncio
    async def test_listen_with_multiple_messages(self, stdio_transport):
        """Test listening with multiple messages in one chunk."""
        mock_reader = AsyncMock()
        mock_reader.read = AsyncMock(
            side_effect=[
                b'{"jsonrpc":"2.0","method":"test1","id":1}\n{"jsonrpc":"2.0","method":"test2","id":2}\n',
                b"",  # EOF
            ]
        )
        stdio_transport.reader = mock_reader
        stdio_transport.running = True

        messages_handled = []

        async def handle_side_effect(msg):
            messages_handled.append(msg)
            if len(messages_handled) >= 2:
                stdio_transport.running = False

        with patch.object(stdio_transport, "_handle_message", new_callable=AsyncMock) as mock_handle:
            mock_handle.side_effect = handle_side_effect

            await stdio_transport._listen()

            assert mock_handle.call_count == 2
            assert messages_handled[0] == '{"jsonrpc":"2.0","method":"test1","id":1}'
            assert messages_handled[1] == '{"jsonrpc":"2.0","method":"test2","id":2}'

    @pytest.mark.asyncio
    async def test_listen_cancelled(self, stdio_transport):
        """Test listening when cancelled."""

        mock_reader = AsyncMock()
        mock_reader.read = AsyncMock(side_effect=asyncio.CancelledError())
        stdio_transport.reader = mock_reader
        stdio_transport.running = True

        await stdio_transport._listen()

        # Should exit cleanly without error
        assert True

    @pytest.mark.asyncio
    async def test_listen_with_exception(self, stdio_transport):
        """Test listening with exception during read."""
        mock_reader = AsyncMock()
        mock_reader.read = AsyncMock(side_effect=Exception("Read error"))
        stdio_transport.reader = mock_reader
        stdio_transport.running = True

        with patch.object(stdio_transport, "_send_error", new_callable=AsyncMock) as mock_send_error:
            # Stop after error
            async def send_error_side_effect(*args):
                stdio_transport.running = False

            mock_send_error.side_effect = send_error_side_effect

            await stdio_transport._listen()

            mock_send_error.assert_called_once()
            error_args = mock_send_error.call_args[0]
            assert error_args[1] == -32603  # Internal error code (generic Exception)

    @pytest.mark.asyncio
    async def test_handle_message_with_exception(self, stdio_transport, mock_protocol):
        """Test handling message that causes exception."""
        mock_protocol.handle_request.side_effect = Exception("Handler error")

        message = json.dumps({"jsonrpc": "2.0", "method": "test", "id": 1})

        with patch.object(stdio_transport, "_send_error", new_callable=AsyncMock) as mock_send_error:
            await stdio_transport._handle_message(message)

            mock_send_error.assert_called_once()
            error_args = mock_send_error.call_args[0]
            assert error_args[0] == 1  # request id
            assert error_args[1] == -32603  # Internal error code
            assert "Handler error" in error_args[2]

    @pytest.mark.asyncio
    async def test_send_response_with_no_writer(self, stdio_transport):
        """Test sending response when writer is None."""
        stdio_transport.writer = None

        # Should not raise error
        await stdio_transport._send_response({"test": "data"})

    @pytest.mark.asyncio
    async def test_send_response_with_exception(self, stdio_transport):
        """Test sending response with write exception."""
        mock_writer = MagicMock()
        mock_writer.write.side_effect = Exception("Write error")
        stdio_transport.writer = mock_writer

        # Should log error but not raise
        await stdio_transport._send_response({"test": "data"})

    def test_context_manager_exit_with_running_loop(self, stdio_transport):
        """Test context manager exit with running event loop."""
        with patch("asyncio.get_running_loop") as mock_get_loop:
            mock_loop = MagicMock()
            mock_get_loop.return_value = mock_loop

            stdio_transport.__exit__(None, None, None)

            mock_loop.create_task.assert_called_once()

    def test_context_manager_exit_without_loop(self, stdio_transport):
        """Test context manager exit without event loop."""
        with patch("asyncio.get_running_loop", side_effect=RuntimeError("No loop")):
            stdio_transport.__exit__(None, None, None)

            assert stdio_transport.running is False


class TestRunStdioServer:
    """Test run_stdio_server function."""

    @patch("chuk_mcp_server.stdio_transport.asyncio.run")
    @patch("chuk_mcp_server.stdio_transport.logging.basicConfig")
    def test_run_stdio_server(self, mock_logging, mock_asyncio_run):
        """Test running the stdio server."""
        mock_protocol = MagicMock(spec=MCPProtocolHandler)

        run_stdio_server(mock_protocol)

        # Verify logging was configured
        mock_logging.assert_called_once()
        logging_kwargs = mock_logging.call_args[1]
        assert logging_kwargs["stream"] is not None

        # Verify asyncio.run was called
        mock_asyncio_run.assert_called_once()

    @patch("chuk_mcp_server.stdio_transport.asyncio.run")
    def test_run_stdio_server_keyboard_interrupt(self, mock_asyncio_run):
        """Test keyboard interrupt handling - note that KeyboardInterrupt is not caught."""
        mock_protocol = MagicMock(spec=MCPProtocolHandler)

        # Simulate KeyboardInterrupt
        mock_asyncio_run.side_effect = KeyboardInterrupt()

        # KeyboardInterrupt will propagate (not caught in run_stdio_server)
        with pytest.raises(KeyboardInterrupt):
            run_stdio_server(mock_protocol)
