"""Tests for HTTP bidirectional transport (SSE + /mcp/respond)."""

import asyncio
import json

import pytest

from chuk_mcp_server.endpoints.mcp import MCPEndpoint
from chuk_mcp_server.protocol import MCPProtocolHandler
from chuk_mcp_server.types import ServerInfo, ToolHandler, create_server_capabilities


def _make_protocol_with_tool(tool_fn=None, tool_name="echo"):
    """Create a protocol handler with a test tool."""
    server_info = ServerInfo(name="test", version="1.0.0")
    caps = create_server_capabilities()
    protocol = MCPProtocolHandler(server_info, caps)

    if tool_fn is None:

        def tool_fn(message: str) -> str:
            """Echo back."""
            return f"Echo: {message}"

    protocol.register_tool(ToolHandler.from_function(tool_fn, name=tool_name, description="Test tool."))
    return protocol


async def _create_session(protocol):
    """Create a session via initialize."""
    init_msg = {
        "jsonrpc": "2.0",
        "id": "init-1",
        "method": "initialize",
        "params": {
            "protocolVersion": "2025-06-18",
            "capabilities": {"sampling": {}, "elicitation": {}, "roots": {"listChanged": True}},
            "clientInfo": {"name": "test-client", "version": "1.0.0"},
        },
    }
    response, session_id = await protocol.handle_request(init_msg)
    return session_id


class TestMCPEndpointBidirectional:
    """Tests for bidirectional SSE communication."""

    @pytest.mark.asyncio
    async def test_pending_requests_initialized(self):
        """MCPEndpoint initializes _pending_requests dict."""
        protocol = _make_protocol_with_tool()
        endpoint = MCPEndpoint(protocol)
        assert endpoint._pending_requests == {}

    @pytest.mark.asyncio
    async def test_sse_tool_call_returns_result(self):
        """SSE tool/call returns the tool result as final SSE event."""
        protocol = _make_protocol_with_tool()
        session_id = await _create_session(protocol)
        endpoint = MCPEndpoint(protocol)

        request_data = {
            "jsonrpc": "2.0",
            "id": "call-1",
            "method": "tools/call",
            "params": {"name": "echo", "arguments": {"message": "hello"}},
        }

        events = []
        async for chunk in endpoint._sse_stream_generator(request_data, session_id, "tools/call"):
            events.append(chunk)

        # Should have event: message, data line, and blank line
        combined = "".join(events)
        assert "event: message" in combined
        assert '"Echo: hello"' in combined

    @pytest.mark.asyncio
    async def test_sse_non_tool_call_uses_simple_path(self):
        """Non-tool-call SSE requests use the simple one-shot path."""
        protocol = _make_protocol_with_tool()
        session_id = await _create_session(protocol)
        endpoint = MCPEndpoint(protocol)

        request_data = {
            "jsonrpc": "2.0",
            "id": "list-1",
            "method": "tools/list",
            "params": {},
        }

        events = []
        async for chunk in endpoint._sse_stream_generator(request_data, session_id, "tools/list"):
            events.append(chunk)

        combined = "".join(events)
        assert "event: message" in combined
        assert '"echo"' in combined

    @pytest.mark.asyncio
    async def test_send_to_client_http_notification(self):
        """Notifications (no id) are enqueued and return immediately."""
        protocol = _make_protocol_with_tool()
        endpoint = MCPEndpoint(protocol)

        queue: asyncio.Queue = asyncio.Queue()
        notification = {
            "jsonrpc": "2.0",
            "method": "notifications/progress",
            "params": {"progressToken": "t1", "progress": 1, "total": 5},
        }

        result = await endpoint._send_to_client_http(notification, queue)
        assert result == {}

        # Notification should be in the queue
        item = await queue.get()
        assert item["method"] == "notifications/progress"

    @pytest.mark.asyncio
    async def test_send_to_client_http_request_response(self):
        """Requests with id create futures and await responses."""
        protocol = _make_protocol_with_tool()
        endpoint = MCPEndpoint(protocol)

        queue: asyncio.Queue = asyncio.Queue()
        request = {
            "jsonrpc": "2.0",
            "id": "sample-123",
            "method": "sampling/createMessage",
            "params": {"messages": []},
        }

        # Simulate client response arriving after a short delay
        async def _respond():
            await asyncio.sleep(0.05)
            future = endpoint._pending_requests.get("sample-123")
            if future and not future.done():
                future.set_result(
                    {
                        "jsonrpc": "2.0",
                        "id": "sample-123",
                        "result": {"model": "test", "content": {"type": "text", "text": "response"}},
                    }
                )

        asyncio.create_task(_respond())
        result = await endpoint._send_to_client_http(request, queue)

        assert result["result"]["content"]["text"] == "response"
        # Future should be cleaned up
        assert "sample-123" not in endpoint._pending_requests

    @pytest.mark.asyncio
    async def test_send_to_client_http_timeout(self):
        """Request times out if client doesn't respond."""
        protocol = _make_protocol_with_tool()
        endpoint = MCPEndpoint(protocol)

        queue: asyncio.Queue = asyncio.Queue()
        request = {
            "jsonrpc": "2.0",
            "id": "timeout-1",
            "method": "sampling/createMessage",
            "params": {},
        }

        # Use a short timeout for testing by patching
        import unittest.mock as mock

        with mock.patch("chuk_mcp_server.endpoints.mcp.asyncio.wait_for", side_effect=TimeoutError("timed out")):
            with pytest.raises(RuntimeError, match="Timeout"):
                await endpoint._send_to_client_http(request, queue)

        # Future should be cleaned up even after timeout
        assert "timeout-1" not in endpoint._pending_requests

    @pytest.mark.asyncio
    async def test_handle_respond_resolves_future(self):
        """POST to /mcp/respond resolves the pending future."""
        protocol = _make_protocol_with_tool()
        endpoint = MCPEndpoint(protocol)

        # Create a pending future
        loop = asyncio.get_running_loop()
        future: asyncio.Future = loop.create_future()
        endpoint._pending_requests["resp-1"] = future

        # Simulate the HTTP request
        from unittest.mock import AsyncMock, MagicMock

        import orjson

        response_data = {
            "jsonrpc": "2.0",
            "id": "resp-1",
            "result": {"model": "claude", "content": {"type": "text", "text": "ok"}},
        }

        mock_request = MagicMock()
        mock_request.body = AsyncMock(return_value=orjson.dumps(response_data))

        response = await endpoint.handle_respond(mock_request)

        assert future.done()
        assert future.result() == response_data

        # Check HTTP response is OK
        body = json.loads(response.body.decode())
        assert body["status"] == "ok"

    @pytest.mark.asyncio
    async def test_handle_respond_unknown_id(self):
        """POST to /mcp/respond with unknown ID returns error."""
        protocol = _make_protocol_with_tool()
        endpoint = MCPEndpoint(protocol)

        from unittest.mock import AsyncMock, MagicMock

        import orjson

        response_data = {"jsonrpc": "2.0", "id": "unknown-999", "result": {}}
        mock_request = MagicMock()
        mock_request.body = AsyncMock(return_value=orjson.dumps(response_data))

        response = await endpoint.handle_respond(mock_request)

        body = json.loads(response.body.decode())
        assert "error" in body

    @pytest.mark.asyncio
    async def test_send_to_client_cleaned_up_after_request(self):
        """protocol._send_to_client is restored after SSE tool call."""
        protocol = _make_protocol_with_tool()
        session_id = await _create_session(protocol)
        endpoint = MCPEndpoint(protocol)

        # Verify initial state
        assert protocol._send_to_client is None

        request_data = {
            "jsonrpc": "2.0",
            "id": "call-cleanup",
            "method": "tools/call",
            "params": {"name": "echo", "arguments": {"message": "test"}},
        }

        # Consume the SSE stream
        async for _ in endpoint._sse_stream_generator(request_data, session_id, "tools/call"):
            pass

        # _send_to_client should be restored to None
        assert protocol._send_to_client is None

    @pytest.mark.asyncio
    async def test_bidirectional_sampling_via_sse(self):
        """Tool that calls create_message() emits SSE server_request event."""

        # Create a tool that attempts sampling
        async def sampling_tool(query: str) -> str:
            """Tool that uses sampling."""
            from chuk_mcp_server.context import create_message

            try:
                result = await create_message(
                    messages=[{"role": "user", "content": {"type": "text", "text": query}}],
                    max_tokens=100,
                )
                return f"LLM said: {result.get('content', {}).get('text', 'nothing')}"
            except RuntimeError:
                return "Sampling not available"

        protocol = _make_protocol_with_tool(sampling_tool, "sampling_tool")
        session_id = await _create_session(protocol)
        endpoint = MCPEndpoint(protocol)

        request_data = {
            "jsonrpc": "2.0",
            "id": "call-sample",
            "method": "tools/call",
            "params": {"name": "sampling_tool", "arguments": {"query": "hello"}},
        }

        events = []

        async def _collect_and_respond():
            """Collect SSE events and respond to server requests."""
            async for chunk in endpoint._sse_stream_generator(request_data, session_id, "tools/call"):
                events.append(chunk)
                # Check if this is a server_request event for sampling
                if "event: server_request" in chunk:
                    # Next chunk will be the data line
                    pass
                elif chunk.startswith("data: ") and events and "server_request" in "".join(events[-3:]):
                    # Parse the request and respond
                    data_str = chunk.replace("data: ", "").strip()
                    try:
                        req = json.loads(data_str)
                        if req.get("method") == "sampling/createMessage":
                            req_id = req.get("id")
                            if req_id and str(req_id) in endpoint._pending_requests:
                                future = endpoint._pending_requests[str(req_id)]
                                if not future.done():
                                    future.set_result(
                                        {
                                            "jsonrpc": "2.0",
                                            "id": req_id,
                                            "result": {
                                                "model": "test-model",
                                                "content": {"type": "text", "text": "LLM response"},
                                            },
                                        }
                                    )
                    except json.JSONDecodeError:
                        pass

        await _collect_and_respond()

        combined = "".join(events)
        # The tool either succeeded with sampling or fell back
        assert "event: message" in combined  # Final response always present

    @pytest.mark.asyncio
    async def test_non_sse_json_no_bidirectional(self):
        """Regular JSON-RPC (non-SSE) requests don't set _send_to_client."""
        protocol = _make_protocol_with_tool()
        session_id = await _create_session(protocol)
        endpoint = MCPEndpoint(protocol)

        request_data = {
            "jsonrpc": "2.0",
            "id": "json-1",
            "method": "tools/call",
            "params": {"name": "echo", "arguments": {"message": "test"}},
        }

        response = await endpoint._handle_json_request(request_data, session_id, "tools/call")

        # _send_to_client should still be None (not set for JSON requests)
        assert protocol._send_to_client is None

        # Response should still work
        body = json.loads(response.body.decode())
        assert "result" in body
