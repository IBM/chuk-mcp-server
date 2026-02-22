#!/usr/bin/env python3
"""
Tests targeting uncovered lines in src/chuk_mcp_server/endpoints/mcp.py.

Covers:
  - Lines 122-131: OAuth double-Bearer prefix handling and else branch
  - Line 135: Warning for non-Bearer auth header
  - Lines 200->202: handle_respond unknown request ID path
  - Lines 214-218: handle_respond exception paths (JSON decode, general)
  - Lines 292-301: SSE stream generator tool call exception path
  - Lines 313->317, 325-327: SSE stream generator task cleanup / cancel
"""

import asyncio
import contextlib
import json
from unittest.mock import AsyncMock, MagicMock, patch

import orjson
import pytest

from chuk_mcp_server.endpoints.mcp import MCPEndpoint
from chuk_mcp_server.protocol import MCPProtocolHandler
from chuk_mcp_server.types import ServerInfo, ToolHandler, create_server_capabilities

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_protocol(tool_fn=None, tool_name="echo"):
    """Create a protocol handler with a simple test tool."""
    server_info = ServerInfo(name="test-server", version="0.1.0")
    caps = create_server_capabilities()
    protocol = MCPProtocolHandler(server_info, caps)

    if tool_fn is None:

        def tool_fn(message: str) -> str:
            """Echo back."""
            return f"Echo: {message}"

    protocol.register_tool(ToolHandler.from_function(tool_fn, name=tool_name, description="test tool"))
    return protocol


async def _create_session(protocol: MCPProtocolHandler) -> str:
    """Create a session via the initialize handshake."""
    init_msg = {
        "jsonrpc": "2.0",
        "id": "init-1",
        "method": "initialize",
        "params": {
            "protocolVersion": "2025-06-18",
            "capabilities": {},
            "clientInfo": {"name": "test-client", "version": "1.0.0"},
        },
    }
    _, session_id = await protocol.handle_request(init_msg)
    return session_id


def _mock_request(*, body: bytes = b"", headers: dict | None = None, method: str = "POST"):
    """Build a lightweight mock Starlette Request."""
    req = MagicMock()
    req.method = method
    req.body = AsyncMock(return_value=body)
    req.headers = headers or {}
    return req


# ---------------------------------------------------------------------------
# Tests: OAuth double-Bearer prefix handling  (lines 122-131, 135)
# ---------------------------------------------------------------------------


class TestOAuthBearerPrefixHandling:
    """Cover the Authorization header parsing paths in _handle_post."""

    @pytest.mark.asyncio
    async def test_double_bearer_prefix_is_stripped(self):
        """Authorization: Bearer Bearer <token> -> double-Bearer path (lines 122-131)."""
        protocol = _make_protocol()
        session_id = await _create_session(protocol)
        endpoint = MCPEndpoint(protocol)

        body = orjson.dumps(
            {
                "jsonrpc": "2.0",
                "id": "db-1",
                "method": "tools/list",
                "params": {},
            }
        )

        req = _mock_request(
            body=body,
            headers={
                "accept": "application/json",
                "authorization": "Bearer Bearer actual-token-value",
                "mcp-session-id": session_id,
            },
        )

        with patch("chuk_mcp_server.endpoints.mcp.logger") as mock_logger:
            response = await endpoint.handle_request(req)

        # The request should succeed (tools/list doesn't require auth)
        resp_body = json.loads(response.body.decode())
        assert "result" in resp_body

        # Verify the double-Bearer warning was logged
        warning_calls = [call for call in mock_logger.warning.call_args_list if "Double-Bearer" in str(call)]
        assert len(warning_calls) >= 1

    @pytest.mark.asyncio
    async def test_single_bearer_prefix_extracted(self):
        """Authorization: Bearer <token> -> normal Bearer path (lines 122-124, 131-133)."""
        protocol = _make_protocol()
        session_id = await _create_session(protocol)
        endpoint = MCPEndpoint(protocol)

        body = orjson.dumps(
            {
                "jsonrpc": "2.0",
                "id": "sb-1",
                "method": "tools/list",
                "params": {},
            }
        )

        req = _mock_request(
            body=body,
            headers={
                "accept": "application/json",
                "authorization": "Bearer my-single-token",
                "mcp-session-id": session_id,
            },
        )

        with patch("chuk_mcp_server.endpoints.mcp.logger") as mock_logger:
            response = await endpoint.handle_request(req)

        resp_body = json.loads(response.body.decode())
        assert "result" in resp_body

        # No double-Bearer warning should have been logged
        warning_calls = [call for call in mock_logger.warning.call_args_list if "Double-Bearer" in str(call)]
        assert len(warning_calls) == 0

    @pytest.mark.asyncio
    async def test_non_bearer_auth_header_warning(self):
        """Authorization: Basic xyz -> non-Bearer warning (line 135)."""
        protocol = _make_protocol()
        session_id = await _create_session(protocol)
        endpoint = MCPEndpoint(protocol)

        body = orjson.dumps(
            {
                "jsonrpc": "2.0",
                "id": "nb-1",
                "method": "tools/list",
                "params": {},
            }
        )

        req = _mock_request(
            body=body,
            headers={
                "accept": "application/json",
                "authorization": "Basic dXNlcjpwYXNz",
                "mcp-session-id": session_id,
            },
        )

        with patch("chuk_mcp_server.endpoints.mcp.logger") as mock_logger:
            response = await endpoint.handle_request(req)

        resp_body = json.loads(response.body.decode())
        assert "result" in resp_body

        # Should have logged the non-Bearer warning
        warning_calls = [call for call in mock_logger.warning.call_args_list if "doesn't start with" in str(call)]
        assert len(warning_calls) >= 1


# ---------------------------------------------------------------------------
# Tests: handle_respond edge cases  (lines 200->202, 208-212, 214-218)
# ---------------------------------------------------------------------------


class TestHandleRespondEdgeCases:
    """Cover the error branches in handle_respond."""

    @pytest.mark.asyncio
    async def test_handle_respond_future_already_done(self):
        """If the future is already done, set_result is skipped (line 200->202)."""
        protocol = _make_protocol()
        endpoint = MCPEndpoint(protocol)

        loop = asyncio.get_running_loop()
        future: asyncio.Future = loop.create_future()
        future.set_result({"already": "resolved"})  # mark done before respond
        endpoint._pending_requests["done-1"] = future

        body = orjson.dumps({"jsonrpc": "2.0", "id": "done-1", "result": {"late": True}})
        req = _mock_request(body=body)

        response = await endpoint.handle_respond(req)
        resp_body = json.loads(response.body.decode())

        # Should still return ok (the id was found in pending requests)
        assert resp_body["status"] == "ok"
        # The original value should remain unchanged
        assert future.result() == {"already": "resolved"}

    @pytest.mark.asyncio
    async def test_handle_respond_unknown_request_id(self):
        """Unknown request ID in handle_respond returns error (lines 208-212)."""
        protocol = _make_protocol()
        endpoint = MCPEndpoint(protocol)

        body = orjson.dumps({"jsonrpc": "2.0", "id": "ghost-42", "result": {}})
        req = _mock_request(body=body)

        response = await endpoint.handle_respond(req)
        resp_body = json.loads(response.body.decode())

        assert "error" in resp_body
        assert "No pending request" in resp_body["error"]["message"]

    @pytest.mark.asyncio
    async def test_handle_respond_invalid_json(self):
        """Invalid JSON body triggers JSONDecodeError path (lines 214-215)."""
        protocol = _make_protocol()
        endpoint = MCPEndpoint(protocol)

        req = _mock_request(body=b"<<<not json>>>")

        response = await endpoint.handle_respond(req)
        resp_body = json.loads(response.body.decode())

        assert "error" in resp_body
        assert resp_body["error"]["code"] == -32700  # PARSE_ERROR

    @pytest.mark.asyncio
    async def test_handle_respond_general_exception(self):
        """General exception in handle_respond triggers catch-all (lines 216-218)."""
        protocol = _make_protocol()
        endpoint = MCPEndpoint(protocol)

        req = MagicMock()
        req.body = AsyncMock(side_effect=RuntimeError("boom"))

        response = await endpoint.handle_respond(req)
        resp_body = json.loads(response.body.decode())

        assert "error" in resp_body
        assert resp_body["error"]["code"] == -32603  # INTERNAL_ERROR
        assert "boom" in resp_body["error"]["message"]


# ---------------------------------------------------------------------------
# Tests: SSE stream generator exception path  (lines 292-301)
# ---------------------------------------------------------------------------


class TestSSEStreamToolCallException:
    """Cover the exception branch inside _execute() for tool calls."""

    @pytest.mark.asyncio
    async def test_sse_tool_call_protocol_exception(self):
        """When protocol.handle_request raises during tool call, an error SSE event is emitted (lines 292-301)."""
        protocol = _make_protocol()
        session_id = await _create_session(protocol)
        endpoint = MCPEndpoint(protocol)

        request_data = {
            "jsonrpc": "2.0",
            "id": "exc-1",
            "method": "tools/call",
            "params": {"name": "echo", "arguments": {"message": "hello"}},
        }

        # Make handle_request raise an exception during tool execution
        async def _exploding_handle(msg, sid=None, token=None):
            raise RuntimeError("tool execution blew up")

        protocol.handle_request = _exploding_handle

        events = []
        async for chunk in endpoint._sse_stream_generator(request_data, session_id, "tools/call"):
            events.append(chunk)

        combined = "".join(events)

        # Should contain an error response in the SSE stream
        assert "event: message" in combined
        assert "tool execution blew up" in combined

        # Verify JSON-RPC error structure
        for evt in events:
            if evt.startswith("data: "):
                data = json.loads(evt[len("data: ") :].strip())
                assert data["error"]["code"] == -32603
                assert data["id"] == "exc-1"
                break

        # _send_to_client should be restored
        assert protocol._send_to_client is None


# ---------------------------------------------------------------------------
# Tests: SSE stream generator task cleanup / cancel  (lines 313->317, 325-327)
# ---------------------------------------------------------------------------


class TestSSEStreamTaskCleanup:
    """Cover task cleanup paths in the SSE stream generator."""

    @pytest.mark.asyncio
    async def test_sse_stream_final_response_none_skips_yield(self):
        """When the final response is None, the yield is skipped (line 313->317)."""
        protocol = _make_protocol()
        session_id = await _create_session(protocol)
        endpoint = MCPEndpoint(protocol)

        request_data = {
            "jsonrpc": "2.0",
            "id": "none-resp",
            "method": "tools/call",
            "params": {"name": "echo", "arguments": {"message": "x"}},
        }

        # Make handle_request return (None, None) so the final response is None
        async def _none_handle(msg, sid=None, token=None):
            return None, None

        protocol.handle_request = _none_handle

        events = []
        async for chunk in endpoint._sse_stream_generator(request_data, session_id, "tools/call"):
            events.append(chunk)

        # No SSE events should be emitted (response was None)
        assert len(events) == 0

    @pytest.mark.asyncio
    async def test_sse_stream_task_cancelled_on_generator_close(self):
        """When the generator is closed early, the background task is cancelled (lines 325-327)."""
        # Create a tool that blocks so the task is still running when we break
        hang_event = asyncio.Event()

        async def slow_tool(message: str) -> str:
            """A slow tool that hangs."""
            await hang_event.wait()
            return "done"

        protocol = _make_protocol(tool_fn=slow_tool, tool_name="slow")
        session_id = await _create_session(protocol)
        endpoint = MCPEndpoint(protocol)

        request_data = {
            "jsonrpc": "2.0",
            "id": "cancel-1",
            "method": "tools/call",
            "params": {"name": "slow", "arguments": {"message": "x"}},
        }

        gen = endpoint._sse_stream_generator(request_data, session_id, "tools/call")

        # Start iterating -- since the tool hangs, no items appear on the queue.
        # We'll race: wait a tiny bit for the task to start, then close the generator.
        task_started = False

        async def _drain():
            nonlocal task_started
            # The generator will block on sse_queue.get() because the tool is hanging.
            # We use wait_for with a short timeout to let the task spin up.
            try:
                async for _ in gen:
                    task_started = True
                    break
            except asyncio.CancelledError:
                pass

        drain_task = asyncio.create_task(_drain())

        # Give the internal task time to start
        await asyncio.sleep(0.1)

        # Cancel the drain task to simulate the client disconnecting
        drain_task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await drain_task

        # Close the generator to trigger the finally block
        await gen.aclose()

        # Unblock the slow tool so cleanup can proceed
        hang_event.set()
        await asyncio.sleep(0.05)

        # _send_to_client should be restored
        assert protocol._send_to_client is None


# ---------------------------------------------------------------------------
# Tests: handle_request routing (method not allowed)
# ---------------------------------------------------------------------------


class TestHandleRequestRouting:
    """Cover basic routing edge cases in handle_request."""

    @pytest.mark.asyncio
    async def test_method_not_allowed(self):
        """Non-GET/POST/DELETE/OPTIONS methods return 405."""
        protocol = _make_protocol()
        endpoint = MCPEndpoint(protocol)

        req = _mock_request(method="PATCH")
        response = await endpoint.handle_request(req)
        assert response.status_code == 405

    @pytest.mark.asyncio
    async def test_options_returns_cors(self):
        """OPTIONS returns CORS preflight response."""
        protocol = _make_protocol()
        endpoint = MCPEndpoint(protocol)

        req = _mock_request(method="OPTIONS")
        response = await endpoint.handle_request(req)
        assert response.status_code == 200
        assert "Access-Control-Allow-Origin" in response.headers

    @pytest.mark.asyncio
    async def test_get_returns_server_info(self):
        """GET returns server information."""
        protocol = _make_protocol()
        endpoint = MCPEndpoint(protocol)

        req = _mock_request(method="GET")
        response = await endpoint.handle_request(req)

        body = json.loads(response.body.decode())
        assert body["name"] == "test-server"
        assert body["status"] == "ready"
