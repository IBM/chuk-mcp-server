#!/usr/bin/env python3
"""Tests for MCP endpoint."""

from unittest.mock import AsyncMock, MagicMock, patch

import orjson
import pytest
from starlette.responses import StreamingResponse

from chuk_mcp_server.endpoints.mcp import MCPEndpoint


class MockRequest:
    """Mock request for testing."""

    def __init__(self, method="POST", body_data=None, headers=None):
        self.method = method
        self.headers = headers or {}
        self._body_data = body_data or {}

    async def body(self):
        """Return mock body data."""
        if isinstance(self._body_data, bytes):
            return self._body_data
        return orjson.dumps(self._body_data)


class TestMCPEndpoint:
    """Tests for MCPEndpoint class."""

    def setup_method(self):
        """Set up test fixtures."""
        # Create mock protocol handler
        self.mock_protocol = MagicMock()

        # Mock server info
        self.mock_protocol.server_info.name = "TestServer"
        self.mock_protocol.server_info.version = "1.0.0"

        # Mock tools and resources
        self.mock_protocol.tools = {"tool1": MagicMock(), "tool2": MagicMock()}
        self.mock_protocol.resources = {"resource1": MagicMock()}

        # Mock session manager
        self.mock_protocol.session_manager = MagicMock()
        self.mock_protocol.session_manager.create_session.return_value = "test-session-123"
        self.mock_protocol.session_manager.get_session.return_value = None

        # Mock handle_request
        self.mock_protocol.handle_request = AsyncMock()

        # Mock SSE event ID methods (MCP 2025-11-25)
        self._sse_counter = 0

        def _next_sse_event_id(session_id):
            self._sse_counter += 1
            return self._sse_counter

        self.mock_protocol.next_sse_event_id = MagicMock(side_effect=_next_sse_event_id)
        self.mock_protocol.buffer_sse_event = MagicMock()

        # Mock terminate_session
        self.mock_protocol.terminate_session = MagicMock(return_value=True)

        self.endpoint = MCPEndpoint(self.mock_protocol)

    def test_init(self):
        """Test MCPEndpoint initialization."""
        assert self.endpoint.protocol == self.mock_protocol

    @pytest.mark.asyncio
    async def test_handle_request_options_cors(self):
        """Test handling OPTIONS request for CORS."""
        request = MockRequest(method="OPTIONS")

        response = await self.endpoint.handle_request(request)

        assert response.status_code == 200
        assert response.headers["Access-Control-Allow-Origin"] == "*"
        assert response.headers["Access-Control-Allow-Methods"] == "GET, POST, DELETE, OPTIONS"
        assert response.headers["Access-Control-Allow-Headers"] == "*"
        assert "Access-Control-Allow-Credentials" not in response.headers

    @pytest.mark.asyncio
    async def test_handle_request_get_server_info(self):
        """Test handling GET request for server info."""
        request = MockRequest(method="GET")

        response = await self.endpoint.handle_request(request)

        assert response.status_code == 200
        assert response.media_type == "application/json"
        assert "Access-Control-Allow-Origin" in response.headers

        body = orjson.loads(response.body)
        assert body["name"] == "TestServer"
        assert body["version"] == "1.0.0"
        assert body["protocol"] == "MCP 2025-03-26"
        assert body["status"] == "ready"
        assert body["tools"] == 2
        assert body["resources"] == 1
        assert body["powered_by"] == "ChukMCPServer with chuk_mcp"

    @pytest.mark.asyncio
    async def test_handle_request_method_not_allowed(self):
        """Test handling unsupported HTTP methods."""
        for method in ["PUT", "PATCH"]:
            request = MockRequest(method=method)

            response = await self.endpoint.handle_request(request)

            assert response.status_code == 405
            assert response.body == b"Method not allowed"

    @pytest.mark.asyncio
    async def test_handle_post_json_request_success(self):
        """Test handling successful POST JSON request."""
        request_data = {"jsonrpc": "2.0", "id": "test-id", "method": "tools/list", "params": {}}
        request = MockRequest(
            method="POST",
            body_data=request_data,
            headers={"mcp-session-id": "test-session", "accept": "application/json"},
        )

        # Mock successful response
        mock_response = {"jsonrpc": "2.0", "id": "test-id", "result": {"tools": []}}
        self.mock_protocol.handle_request.return_value = (mock_response, None)

        response = await self.endpoint.handle_request(request)

        assert response.status_code == 200
        assert response.media_type == "application/json"
        assert "Access-Control-Allow-Origin" in response.headers

        body = orjson.loads(response.body)
        assert body == mock_response

        # Verify protocol handler was called correctly (with oauth_token=None)
        self.mock_protocol.handle_request.assert_called_once_with(request_data, "test-session", None)

    @pytest.mark.asyncio
    async def test_handle_post_json_request_notification(self):
        """Test handling POST request that's a notification (no response)."""
        request_data = {"jsonrpc": "2.0", "method": "notifications/updated", "params": {"message": "test"}}
        request = MockRequest(method="POST", body_data=request_data, headers={"mcp-session-id": "test-session"})

        # Mock notification response (None)
        self.mock_protocol.handle_request.return_value = (None, None)

        response = await self.endpoint.handle_request(request)

        assert response.status_code == 202
        assert response.body == b""
        assert "Access-Control-Allow-Origin" in response.headers

    @pytest.mark.asyncio
    async def test_handle_post_json_request_with_new_session(self):
        """Test handling POST request that returns a new session ID."""
        request_data = {"jsonrpc": "2.0", "id": "test-id", "method": "initialize", "params": {}}
        request = MockRequest(method="POST", body_data=request_data)

        # Mock response with new session ID
        mock_response = {"jsonrpc": "2.0", "id": "test-id", "result": {}}
        new_session_id = "new-session-123"
        self.mock_protocol.handle_request.return_value = (mock_response, new_session_id)

        response = await self.endpoint.handle_request(request)

        assert response.status_code == 200
        assert response.headers["Mcp-Session-Id"] == new_session_id

        body = orjson.loads(response.body)
        assert body == mock_response

    @pytest.mark.asyncio
    async def test_handle_post_missing_session_id_non_initialize(self):
        """Test handling POST request without session ID for non-initialize method."""
        request_data = {"jsonrpc": "2.0", "id": "test-id", "method": "tools/list", "params": {}}
        request = MockRequest(method="POST", body_data=request_data)

        response = await self.endpoint.handle_request(request)

        assert response.status_code == 400
        body = orjson.loads(response.body)
        assert body["jsonrpc"] == "2.0"
        assert body["id"] == "test-id"
        assert body["error"]["code"] == -32600
        assert "Missing session ID" in body["error"]["message"]

    @pytest.mark.asyncio
    async def test_handle_post_json_decode_error(self):
        """Test handling POST request with invalid JSON."""
        request = MockRequest(method="POST", body_data=b"{invalid json}")

        response = await self.endpoint.handle_request(request)

        assert response.status_code == 400
        body = orjson.loads(response.body)
        assert body["jsonrpc"] == "2.0"
        assert body["error"]["code"] == -32700
        assert "Parse error" in body["error"]["message"]

    @pytest.mark.asyncio
    async def test_handle_post_empty_body(self):
        """Test handling POST request with empty body."""
        request = MockRequest(method="POST", body_data=b"", headers={"mcp-session-id": "test-session"})

        # Mock response for empty request
        mock_response = {"jsonrpc": "2.0", "error": {"code": -32600, "message": "Invalid Request"}}
        self.mock_protocol.handle_request.return_value = (mock_response, None)

        await self.endpoint.handle_request(request)

        # Should process empty body as {} (with oauth_token=None)
        self.mock_protocol.handle_request.assert_called_once_with({}, "test-session", None)

    @pytest.mark.asyncio
    async def test_handle_post_protocol_exception(self):
        """Test handling POST request when protocol handler raises exception."""
        request_data = {"jsonrpc": "2.0", "id": "test-id", "method": "tools/list", "params": {}}
        request = MockRequest(method="POST", body_data=request_data, headers={"mcp-session-id": "test-session"})

        # Mock protocol handler exception
        self.mock_protocol.handle_request.side_effect = Exception("Test error")

        response = await self.endpoint.handle_request(request)

        assert response.status_code == 500
        body = orjson.loads(response.body)
        assert body["jsonrpc"] == "2.0"
        assert body["error"]["code"] == -32603
        assert "Internal error: Test error" in body["error"]["message"]

    @pytest.mark.asyncio
    async def test_handle_post_sse_request_initialize(self):
        """Test handling POST request with SSE for initialize method."""
        request_data = {
            "jsonrpc": "2.0",
            "id": "test-id",
            "method": "initialize",
            "params": {"clientInfo": {"name": "Test Client", "version": "1.0"}, "protocolVersion": "2025-03-26"},
        }
        request = MockRequest(method="POST", body_data=request_data, headers={"accept": "text/event-stream"})

        # Mock successful response
        mock_response = {"jsonrpc": "2.0", "id": "test-id", "result": {}}
        self.mock_protocol.handle_request.return_value = (mock_response, None)

        response = await self.endpoint.handle_request(request)

        assert isinstance(response, StreamingResponse)
        assert response.media_type == "text/event-stream"
        assert "Access-Control-Allow-Origin" in response.headers
        assert "Cache-Control" in response.headers
        assert response.headers["Mcp-Session-Id"] == "test-session-123"

        # Verify session was created
        self.mock_protocol.session_manager.create_session.assert_called_once_with(
            {"name": "Test Client", "version": "1.0"}, "2025-03-26"
        )

    @pytest.mark.asyncio
    async def test_handle_post_sse_request_with_session_id(self):
        """Test handling POST request with SSE using existing session ID."""
        request_data = {"jsonrpc": "2.0", "id": "test-id", "method": "tools/list", "params": {}}
        request = MockRequest(
            method="POST",
            body_data=request_data,
            headers={"accept": "text/event-stream", "mcp-session-id": "existing-session"},
        )

        # Mock successful response
        mock_response = {"jsonrpc": "2.0", "id": "test-id", "result": {"tools": []}}
        self.mock_protocol.handle_request.return_value = (mock_response, None)

        response = await self.endpoint.handle_request(request)

        assert isinstance(response, StreamingResponse)
        assert response.media_type == "text/event-stream"

        # Should use existing session, not create new one
        self.mock_protocol.session_manager.create_session.assert_not_called()

    @pytest.mark.asyncio
    async def test_sse_stream_generator_success(self):
        """Test SSE stream generator with successful response."""
        request_data = {"jsonrpc": "2.0", "id": "test-id", "method": "tools/list"}
        mock_response = {"jsonrpc": "2.0", "id": "test-id", "result": {"tools": []}}
        self.mock_protocol.handle_request.return_value = (mock_response, None)

        # Collect streamed data
        stream_data = []
        async for chunk in self.endpoint._sse_stream_generator(request_data, "test-session", "tools/list"):
            stream_data.append(chunk)

        # Verify SSE format: event, id, data, blank line
        assert len(stream_data) == 4
        assert stream_data[0] == "event: message\r\n"
        assert stream_data[1] == "id: 1\r\n"
        assert stream_data[2] == f"data: {orjson.dumps(mock_response).decode()}\r\n"
        assert stream_data[3] == "\r\n"

    @pytest.mark.asyncio
    async def test_sse_stream_generator_notification(self):
        """Test SSE stream generator with notification (no response)."""
        request_data = {"jsonrpc": "2.0", "method": "notifications/updated"}
        self.mock_protocol.handle_request.return_value = (None, None)

        # Collect streamed data
        stream_data = []
        async for chunk in self.endpoint._sse_stream_generator(request_data, "test-session", "notifications/updated"):
            stream_data.append(chunk)

        # Should not send anything for notifications
        assert len(stream_data) == 0

    @pytest.mark.asyncio
    async def test_sse_stream_generator_exception(self):
        """Test SSE stream generator when protocol handler raises exception."""
        request_data = {"jsonrpc": "2.0", "id": "test-id", "method": "tools/list"}
        self.mock_protocol.handle_request.side_effect = Exception("Test SSE error")

        # Collect streamed data
        stream_data = []
        async for chunk in self.endpoint._sse_stream_generator(request_data, "test-session", "tools/list"):
            stream_data.append(chunk)

        # Verify error event format: event, id, data, blank line
        assert len(stream_data) == 4
        assert stream_data[0] == "event: error\r\n"
        assert stream_data[1] == "id: 1\r\n"

        # Parse error data
        error_data = stream_data[2].replace("data: ", "").replace("\r\n", "")
        error_response = orjson.loads(error_data)
        assert error_response["jsonrpc"] == "2.0"
        assert error_response["id"] == "test-id"
        assert error_response["error"]["code"] == -32603
        assert "Test SSE error" in error_response["error"]["message"]

        assert stream_data[3] == "\r\n"

    def test_cors_response(self):
        """Test CORS response generation."""
        response = self.endpoint._cors_response()

        assert response.status_code == 200
        assert response.body == b""
        assert response.headers["Access-Control-Allow-Origin"] == "*"
        assert response.headers["Access-Control-Allow-Methods"] == "GET, POST, DELETE, OPTIONS"
        assert response.headers["Access-Control-Allow-Headers"] == "*"
        assert "Access-Control-Allow-Credentials" not in response.headers

    def test_sse_headers_without_session(self):
        """Test SSE headers generation without session ID."""
        headers = self.endpoint._sse_headers(None)

        assert headers["Access-Control-Allow-Origin"] == "*"
        assert headers["Cache-Control"] == "no-cache"
        assert headers["Connection"] == "keep-alive"
        assert "Mcp-Session-Id" not in headers

    def test_sse_headers_with_session(self):
        """Test SSE headers generation with session ID."""
        headers = self.endpoint._sse_headers("test-session-123")

        assert headers["Access-Control-Allow-Origin"] == "*"
        assert headers["Cache-Control"] == "no-cache"
        assert headers["Connection"] == "keep-alive"
        assert headers["Mcp-Session-Id"] == "test-session-123"

    def test_error_response_parse_error(self):
        """Test error response generation for parse errors."""
        response = self.endpoint._error_response("test-id", -32700, "Parse error")

        assert response.status_code == 400
        assert response.media_type == "application/json"
        assert "Access-Control-Allow-Origin" in response.headers

        body = orjson.loads(response.body)
        assert body["jsonrpc"] == "2.0"
        assert body["id"] == "test-id"
        assert body["error"]["code"] == -32700
        assert body["error"]["message"] == "Parse error"

    def test_error_response_invalid_request(self):
        """Test error response generation for invalid requests."""
        response = self.endpoint._error_response("test-id", -32600, "Invalid request")

        assert response.status_code == 400

    def test_error_response_internal_error(self):
        """Test error response generation for internal errors."""
        response = self.endpoint._error_response("test-id", -32603, "Internal error")

        assert response.status_code == 500

        body = orjson.loads(response.body)
        assert body["jsonrpc"] == "2.0"
        assert body["id"] == "test-id"
        assert body["error"]["code"] == -32603
        assert body["error"]["message"] == "Internal error"

    def test_error_response_custom_error_code(self):
        """Test error response generation for custom error codes."""
        response = self.endpoint._error_response("test-id", -32001, "Custom error")

        assert response.status_code == 500  # Non-standard codes default to 500

    @pytest.mark.asyncio
    async def test_logging_integration(self):
        """Test that logging is properly integrated."""
        request_data = {"jsonrpc": "2.0", "id": "test-id", "method": "tools/list", "params": {}}
        request = MockRequest(method="POST", body_data=request_data, headers={"mcp-session-id": "test-session"})

        mock_response = {"jsonrpc": "2.0", "id": "test-id", "result": {"tools": []}}
        self.mock_protocol.handle_request.return_value = (mock_response, None)

        with patch("chuk_mcp_server.endpoints.mcp.logger") as mock_logger:
            await self.endpoint.handle_request(request)

            # Verify debug logging was called
            mock_logger.debug.assert_called_with("Processing tools/list request")

    @pytest.mark.asyncio
    async def test_complex_sse_initialize_flow(self):
        """Test complete SSE initialize flow with all components."""
        request_data = {
            "jsonrpc": "2.0",
            "id": "test-id",
            "method": "initialize",
            "params": {
                "clientInfo": {"name": "MCP Inspector", "version": "1.0.0"},
                "protocolVersion": "2025-03-26",
                "capabilities": {},
            },
        }
        request = MockRequest(method="POST", body_data=request_data, headers={"accept": "text/event-stream"})

        # Mock initialize response
        mock_response = {
            "jsonrpc": "2.0",
            "id": "test-id",
            "result": {
                "protocolVersion": "2025-03-26",
                "capabilities": {"tools": {}, "resources": {}},
                "serverInfo": {"name": "TestServer", "version": "1.0.0"},
            },
        }
        self.mock_protocol.handle_request.return_value = (mock_response, None)

        with patch("chuk_mcp_server.endpoints.mcp.logger") as mock_logger:
            response = await self.endpoint.handle_request(request)

            assert isinstance(response, StreamingResponse)
            assert response.headers["Mcp-Session-Id"] == "test-session-123"

            # Verify session creation logging
            mock_logger.info.assert_called_with("Created SSE session: test-ses...")

    # ================================================================
    # DELETE endpoint tests (MCP 2025-11-25)
    # ================================================================

    @pytest.mark.asyncio
    async def test_handle_delete_success(self):
        """Test successful session termination via DELETE."""
        request = MockRequest(method="DELETE", headers={"mcp-session-id": "test-session-123"})

        response = await self.endpoint.handle_request(request)

        assert response.status_code == 200
        assert "Access-Control-Allow-Origin" in response.headers
        assert "MCP-Protocol-Version" in response.headers
        self.mock_protocol.terminate_session.assert_called_once_with("test-session-123")

    @pytest.mark.asyncio
    async def test_handle_delete_missing_session_id(self):
        """Test DELETE without session ID returns error."""
        request = MockRequest(method="DELETE", headers={})

        response = await self.endpoint.handle_request(request)

        assert response.status_code == 400
        body = orjson.loads(response.body)
        assert body["error"]["code"] == -32600
        assert "Missing session ID" in body["error"]["message"]

    @pytest.mark.asyncio
    async def test_handle_delete_unknown_session(self):
        """Test DELETE with unknown session ID returns error."""
        self.mock_protocol.terminate_session.return_value = False
        request = MockRequest(method="DELETE", headers={"mcp-session-id": "unknown-session"})

        response = await self.endpoint.handle_request(request)

        assert response.status_code == 400
        body = orjson.loads(response.body)
        assert "Unknown session" in body["error"]["message"]

    # ================================================================
    # MCP-Protocol-Version header tests (MCP 2025-11-25)
    # ================================================================

    @pytest.mark.asyncio
    async def test_json_response_includes_protocol_version_header(self):
        """Test that JSON responses include MCP-Protocol-Version header."""
        request_data = {"jsonrpc": "2.0", "id": "test-id", "method": "tools/list", "params": {}}
        request = MockRequest(
            method="POST",
            body_data=request_data,
            headers={"mcp-session-id": "test-session", "accept": "application/json"},
        )
        mock_response = {"jsonrpc": "2.0", "id": "test-id", "result": {"tools": []}}
        self.mock_protocol.handle_request.return_value = (mock_response, None)

        response = await self.endpoint.handle_request(request)

        assert "MCP-Protocol-Version" in response.headers

    def test_sse_headers_include_protocol_version(self):
        """Test that SSE headers include MCP-Protocol-Version."""
        headers = self.endpoint._sse_headers("test-session")
        assert "MCP-Protocol-Version" in headers

    def test_error_response_includes_protocol_version(self):
        """Test that error responses include MCP-Protocol-Version header."""
        response = self.endpoint._error_response("test-id", -32600, "Bad request")
        assert "MCP-Protocol-Version" in response.headers
