#!/usr/bin/env python3
# src/chuk_mcp_server/endpoints/mcp.py
"""
MCP Endpoint - Handles core MCP protocol requests with SSE support.

Supports bidirectional communication over SSE: during tool execution,
the server can send requests (sampling, elicitation, roots) to the client
as SSE events. The client responds via POST to /mcp/respond.
"""

import asyncio
import contextlib
import logging
from typing import Any

import orjson

# starlette
from starlette.requests import Request
from starlette.responses import Response, StreamingResponse

# chuk_mcp_server - Fix import path
from ..protocol import MCPProtocolHandler
from .constants import (
    BEARER_PREFIX,
    CACHE_NO_CACHE,
    CONNECTION_KEEP_ALIVE,
    CONTENT_TYPE_JSON,
    CONTENT_TYPE_SSE,
    CORS_ALLOW_ALL,
    ERROR_METHOD_NOT_ALLOWED,
    FRAMEWORK_DESCRIPTION,
    HEADER_ACCEPT,
    HEADER_AUTHORIZATION,
    HEADER_CACHE_CONTROL,
    HEADER_CONNECTION,
    HEADER_CORS_HEADERS,
    HEADER_CORS_METHODS,
    HEADER_CORS_ORIGIN,
    HEADER_MCP_PROTOCOL_VERSION,
    HEADER_MCP_SESSION_ID,
    HEADERS_CORS_ONLY,
    JSONRPC_VERSION,
    MCP_PROTOCOL_FULL,
    MCP_PROTOCOL_VERSION,
    SSE_EVENT_ERROR,
    SSE_EVENT_MESSAGE,
    SSE_LINE_END,
    STATUS_READY,
    HttpStatus,
    JsonRpcErrorCode,
)

# logger
logger = logging.getLogger(__name__)

# JSON-RPC response key name
JSONRPC_VERSION_KEY = "jsonrpc"


class MCPEndpoint:
    """Core MCP endpoint handler with SSE and bidirectional support."""

    def __init__(self, protocol_handler: MCPProtocolHandler):
        self.protocol = protocol_handler
        # Pending server-to-client requests awaiting responses via /mcp/respond
        self._pending_requests: dict[str, asyncio.Future[dict[str, Any]]] = {}

    def _get_protocol_version(self, session_id: str | None) -> str:
        """Get the negotiated protocol version for a session."""
        if session_id:
            session = self.protocol.session_manager.get_session(session_id)
            if session:
                return str(session.get("protocol_version", MCP_PROTOCOL_VERSION))
        return MCP_PROTOCOL_VERSION

    async def handle_request(self, request: Request) -> Response:
        """Main MCP endpoint handler."""

        # Handle CORS preflight
        if request.method == "OPTIONS":
            return self._cors_response()

        # Handle GET - return server info / SSE stream
        if request.method == "GET":
            return await self._handle_get(request)

        # Handle POST - process MCP requests
        if request.method == "POST":
            return await self._handle_post(request)

        # Handle DELETE - session termination (MCP 2025-11-25)
        if request.method == "DELETE":
            return await self._handle_delete(request)

        return Response(ERROR_METHOD_NOT_ALLOWED, status_code=HttpStatus.METHOD_NOT_ALLOWED)

    async def _handle_delete(self, request: Request) -> Response:
        """Handle DELETE - terminate session (MCP 2025-11-25)."""
        session_id = request.headers.get(HEADER_MCP_SESSION_ID.lower())
        if not session_id:
            return self._error_response(None, JsonRpcErrorCode.INVALID_REQUEST, "Missing session ID")

        protocol_version = self._get_protocol_version(session_id)
        terminated = self.protocol.terminate_session(session_id)
        if not terminated:
            return self._error_response(
                None, JsonRpcErrorCode.INVALID_REQUEST, f"Unknown session: {session_id}", session_id
            )

        headers: dict[str, str] = {
            HEADER_CORS_ORIGIN: CORS_ALLOW_ALL,
            HEADER_MCP_PROTOCOL_VERSION: protocol_version,
        }
        return Response("", status_code=HttpStatus.OK, headers=headers)

    def _cors_response(self) -> Response:
        """Return CORS preflight response."""
        return Response(
            "",
            headers={
                HEADER_CORS_ORIGIN: CORS_ALLOW_ALL,
                HEADER_CORS_METHODS: "GET, POST, DELETE, OPTIONS",
                HEADER_CORS_HEADERS: CORS_ALLOW_ALL,
            },
        )

    async def _handle_get(self, request: Request) -> Response:  # noqa: ARG002
        """Handle GET request - return server information."""
        server_info = {
            "name": self.protocol.server_info.name,
            "version": self.protocol.server_info.version,
            "protocol": MCP_PROTOCOL_FULL,
            "status": STATUS_READY,
            "tools": len(self.protocol.tools),
            "resources": len(self.protocol.resources),
            "powered_by": FRAMEWORK_DESCRIPTION,
        }

        return Response(orjson.dumps(server_info), media_type=CONTENT_TYPE_JSON, headers=HEADERS_CORS_ONLY)

    async def _handle_post(self, request: Request) -> Response:
        """Handle POST request - process MCP protocol messages."""
        accept_header = request.headers.get(HEADER_ACCEPT, "")
        session_id = request.headers.get(HEADER_MCP_SESSION_ID.lower())

        # Extract OAuth token from Authorization header (case-insensitive)
        auth_header = request.headers.get(HEADER_AUTHORIZATION, "")
        oauth_token = None

        # Check for Bearer token (case-insensitive)
        if auth_header.lower().startswith(BEARER_PREFIX):
            # Find where "bearer " ends (handle any casing)
            bearer_prefix_len = len(BEARER_PREFIX)
            oauth_token = auth_header[bearer_prefix_len:]  # Remove first "Bearer " prefix

            # Handle double-Bearer bug in some MCP clients (e.g., "Bearer Bearer token")
            # This happens when clients incorrectly store "Bearer token" as the access_token value
            if oauth_token.lower().startswith(BEARER_PREFIX):
                logger.warning("Double-Bearer prefix detected in Authorization header, stripping again")
                oauth_token = oauth_token[len(BEARER_PREFIX) :]  # Strip second "Bearer " prefix

            logger.debug(
                f"Extracted OAuth token: {oauth_token[:16] if oauth_token else 'None'}... (original header: {auth_header[:30]}...)"
            )
        elif auth_header:
            logger.warning(f"Authorization header present but doesn't start with 'Bearer ': {auth_header[:30]}...")

        try:
            # Parse request body
            body = await request.body()

            # Reject oversized request bodies
            from ..constants import MAX_REQUEST_BODY_BYTES

            if len(body) > MAX_REQUEST_BODY_BYTES:
                return self._error_response(
                    None,
                    JsonRpcErrorCode.INVALID_REQUEST,
                    f"Request body too large ({len(body)} bytes, max {MAX_REQUEST_BODY_BYTES})",
                )

            request_data = orjson.loads(body) if body else {}
            method = request_data.get("method")

            logger.debug(f"Processing {method} request")

            # Route based on Accept header
            if CONTENT_TYPE_SSE in accept_header:
                # SSE streaming
                return await self._handle_sse_request(request_data, session_id, oauth_token)
            else:
                # Regular JSON-RPC request
                return await self._handle_json_request(request_data, session_id, method, oauth_token)

        except orjson.JSONDecodeError as e:
            logger.error(f"JSON decode error: {e}")
            return self._error_response(None, JsonRpcErrorCode.PARSE_ERROR, f"Parse error: {str(e)}")
        except Exception as e:
            logger.error(f"Request processing error: {e}")
            return self._error_response(None, JsonRpcErrorCode.INTERNAL_ERROR, f"Internal error: {str(e)}")

    async def _handle_json_request(
        self, request_data: dict[str, Any], session_id: str | None, method: str, oauth_token: str | None = None
    ) -> Response:
        """Handle regular JSON-RPC request."""

        # Validate session ID for non-initialize requests
        if method != "initialize" and not session_id:
            return self._error_response(
                request_data.get("id", "server-error"),
                JsonRpcErrorCode.INVALID_REQUEST,
                "Bad Request: Missing session ID",
            )

        # Process the request through protocol handler
        response, new_session_id = await self.protocol.handle_request(request_data, session_id, oauth_token)

        effective_session = new_session_id or session_id
        protocol_version = self._get_protocol_version(effective_session)

        # Handle notifications (no response)
        if response is None:
            headers = {
                HEADER_CORS_ORIGIN: CORS_ALLOW_ALL,
                HEADER_MCP_PROTOCOL_VERSION: protocol_version,
            }
            return Response("", status_code=HttpStatus.ACCEPTED, headers=headers)

        # Build response headers
        headers = {
            HEADER_CORS_ORIGIN: CORS_ALLOW_ALL,
            HEADER_MCP_PROTOCOL_VERSION: protocol_version,
        }
        if new_session_id:
            headers[HEADER_MCP_SESSION_ID] = new_session_id

        return Response(orjson.dumps(response), media_type=CONTENT_TYPE_JSON, headers=headers)

    async def handle_respond(self, request: Request) -> Response:
        """Handle client responses to server-initiated requests.

        When the server sends a request (sampling, elicitation, roots) via SSE,
        the client responds by POSTing to /mcp/respond with the JSON-RPC response.
        """
        try:
            body = await request.body()
            data = orjson.loads(body) if body else {}
            request_id = str(data.get("id", ""))

            if request_id in self._pending_requests:
                future = self._pending_requests[request_id]
                if not future.done():
                    future.set_result(data)
                return Response(
                    orjson.dumps({"status": "ok"}),
                    media_type=CONTENT_TYPE_JSON,
                    headers=HEADERS_CORS_ONLY,
                )

            return self._error_response(
                data.get("id"),
                JsonRpcErrorCode.INVALID_REQUEST,
                f"No pending request with ID: {request_id}",
            )

        except orjson.JSONDecodeError as e:
            return self._error_response(None, JsonRpcErrorCode.PARSE_ERROR, f"Parse error: {str(e)}")
        except Exception as e:
            logger.error(f"Error handling /mcp/respond: {e}")
            return self._error_response(None, JsonRpcErrorCode.INTERNAL_ERROR, f"Internal error: {str(e)}")

    async def _send_to_client_http(self, request: dict[str, Any], sse_queue: asyncio.Queue[Any]) -> dict[str, Any]:
        """Send a server-to-client request via SSE and await response.

        For notifications (no id): enqueue and return immediately.
        For requests (with id): create a future, enqueue, await response via /mcp/respond.
        """
        request_id = request.get("id")
        if request_id is None:
            # Notification (e.g., progress) — fire and forget
            await sse_queue.put(request)
            return {}

        # Request-response — create future, enqueue, await
        loop = asyncio.get_running_loop()
        future: asyncio.Future[dict[str, Any]] = loop.create_future()
        request_id_str = str(request_id)
        self._pending_requests[request_id_str] = future
        await sse_queue.put(request)

        try:
            return await asyncio.wait_for(future, timeout=120.0)
        except TimeoutError:
            raise RuntimeError(f"Timeout waiting for client response to request {request_id}")
        finally:
            self._pending_requests.pop(request_id_str, None)

    async def _handle_sse_request(
        self, request_data: dict[str, Any], session_id: str | None, oauth_token: str | None = None
    ) -> StreamingResponse:
        """Handle SSE request for Inspector compatibility."""

        created_session_id = None
        method = request_data.get("method")

        # Create session ID for initialize requests
        if method == "initialize":
            client_info = request_data.get("params", {}).get("clientInfo", {})
            protocol_version = request_data.get("params", {}).get("protocolVersion", MCP_PROTOCOL_VERSION)
            created_session_id = self.protocol.session_manager.create_session(client_info, protocol_version)
            logger.info(f"Created SSE session: {created_session_id[:8]}...")

        return StreamingResponse(
            self._sse_stream_generator(request_data, created_session_id or session_id, method, oauth_token),
            media_type=CONTENT_TYPE_SSE,
            headers=self._sse_headers(created_session_id),
        )

    def _emit_sse_event(self, event_type: str, data: dict[str, Any], session_id: str | None) -> tuple[str, ...]:
        """Build SSE event lines with optional event ID for resumability."""
        lines: list[str] = [event_type]
        if session_id:
            event_id = self.protocol.next_sse_event_id(session_id)
            self.protocol.buffer_sse_event(session_id, event_id, data)
            lines.append(f"id: {event_id}\r\n")
        lines.append(f"data: {orjson.dumps(data).decode()}\r\n")
        lines.append(SSE_LINE_END)
        return tuple(lines)

    async def _sse_stream_generator(
        self, request_data: dict[str, Any], session_id: str | None, method: str, oauth_token: str | None = None
    ):
        """Generate SSE stream response with bidirectional support.

        For tools/call requests, sets up a bidirectional channel so the server
        can send requests (sampling, elicitation, roots, progress) to the client
        as SSE events during tool execution.
        """
        is_tool_call = method == "tools/call"

        if is_tool_call:
            # Bidirectional mode: use queue for server-to-client messages
            sse_queue: asyncio.Queue[Any] = asyncio.Queue()

            async def _send_fn(request: dict[str, Any]) -> dict[str, Any]:
                return await self._send_to_client_http(request, sse_queue)

            async def _execute() -> None:
                prev_send = self.protocol._send_to_client
                self.protocol._send_to_client = _send_fn
                try:
                    response, _ = await self.protocol.handle_request(request_data, session_id, oauth_token)
                    await sse_queue.put(("_final_", response))
                except Exception as exc:
                    error_response = {
                        JSONRPC_VERSION_KEY: JSONRPC_VERSION,
                        "id": request_data.get("id"),
                        "error": {
                            "code": JsonRpcErrorCode.INTERNAL_ERROR,
                            "message": str(exc),
                        },
                    }
                    await sse_queue.put(("_final_", error_response))
                finally:
                    self.protocol._send_to_client = prev_send

            task = asyncio.create_task(_execute())

            try:
                while True:
                    item = await sse_queue.get()
                    if isinstance(item, tuple) and len(item) == 2 and item[0] == "_final_":
                        # Final response from tool execution
                        response = item[1]
                        if response:
                            for line in self._emit_sse_event(SSE_EVENT_MESSAGE, response, session_id):
                                yield line
                        break
                    else:
                        # Server-to-client request or notification
                        for line in self._emit_sse_event("event: server_request\r\n", item, session_id):
                            yield line
            finally:
                if not task.done():
                    task.cancel()
                    with contextlib.suppress(asyncio.CancelledError):
                        await task
        else:
            # Non-tool-call: simple one-shot SSE response (no bidirectional needed)
            try:
                response, _ = await self.protocol.handle_request(request_data, session_id, oauth_token)

                if response:
                    logger.debug(f"Streaming SSE response for {method}")
                    for line in self._emit_sse_event(SSE_EVENT_MESSAGE, response, session_id):
                        yield line

            except Exception as e:
                logger.error(f"SSE stream error: {e}")
                error_response = {
                    JSONRPC_VERSION_KEY: JSONRPC_VERSION,
                    "id": request_data.get("id"),
                    "error": {
                        "code": JsonRpcErrorCode.INTERNAL_ERROR,
                        "message": str(e),
                    },
                }
                for line in self._emit_sse_event(SSE_EVENT_ERROR, error_response, session_id):
                    yield line

    def _sse_headers(self, session_id: str | None) -> dict[str, str]:
        """Build SSE response headers."""
        headers = {
            HEADER_CORS_ORIGIN: CORS_ALLOW_ALL,
            HEADER_CACHE_CONTROL: CACHE_NO_CACHE,
            HEADER_CONNECTION: CONNECTION_KEEP_ALIVE,
            HEADER_MCP_PROTOCOL_VERSION: self._get_protocol_version(session_id),
        }

        if session_id:
            headers[HEADER_MCP_SESSION_ID] = session_id

        return headers

    def _error_response(self, msg_id: Any, code: int, message: str, session_id: str | None = None) -> Response:
        """Create error response."""
        error_response = {
            JSONRPC_VERSION_KEY: JSONRPC_VERSION,
            "id": msg_id,
            "error": {"code": code, "message": message},
        }

        status_code = (
            HttpStatus.BAD_REQUEST
            if code in [JsonRpcErrorCode.PARSE_ERROR, JsonRpcErrorCode.INVALID_REQUEST]
            else HttpStatus.INTERNAL_SERVER_ERROR
        )

        headers = {
            HEADER_CORS_ORIGIN: CORS_ALLOW_ALL,
            HEADER_MCP_PROTOCOL_VERSION: self._get_protocol_version(session_id),
        }

        return Response(
            orjson.dumps(error_response),
            status_code=status_code,
            media_type=CONTENT_TYPE_JSON,
            headers=headers,
        )
