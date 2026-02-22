#!/usr/bin/env python3
# src/chuk_mcp_server/stdio_transport.py
"""
STDIO Transport - Support for MCP protocol over standard input/output.

Provides both asynchronous and synchronous transport implementations for
communicating with MCP clients over stdin/stdout.
"""

import asyncio
import logging
import sys
from typing import Any, TextIO

import orjson

from .constants import (
    DEFAULT_ENCODING,
    JSONRPC_KEY,
    JSONRPC_VERSION,
    KEY_CLIENT_INFO,
    KEY_ERROR,
    KEY_ID,
    KEY_METHOD,
    KEY_PARAMS,
    KEY_PROTOCOL_VERSION,
    MCP_PROTOCOL_VERSION_2025_03,
    JsonRpcError,
    McpMethod,
)
from .protocol import MCPProtocolHandler

logger = logging.getLogger(__name__)


class StdioTransport:
    """
    Handle MCP protocol communication over stdio (stdin/stdout).

    This transport enables the server to communicate with clients via standard
    input/output streams, supporting the full MCP protocol specification.
    """

    def __init__(self, protocol_handler: MCPProtocolHandler) -> None:
        """
        Initialize stdio transport.

        Args:
            protocol_handler: The MCP protocol handler instance
        """
        self.protocol = protocol_handler
        self.reader: asyncio.StreamReader | None = None
        self.writer: TextIO | None = None
        self.running = False
        self.session_id: str | None = None

        # Pending server-to-client requests awaiting responses
        self._pending_requests: dict[str, asyncio.Future[dict[str, Any]]] = {}

        # Set the transport callback on the protocol handler
        self.protocol._send_to_client = self._send_and_receive

    async def _send_and_receive(self, request: dict[str, Any]) -> dict[str, Any]:
        """
        Send a JSON-RPC request to the client and await the response.

        Used for server-initiated requests like sampling/createMessage.
        For notifications (no id), sends fire-and-forget and returns immediately.

        Args:
            request: JSON-RPC request dict

        Returns:
            JSON-RPC response dict from the client
        """
        request_id = request.get(KEY_ID)
        if request_id is None:
            # Notification — fire and forget, no response expected
            await self._send_response(request)
            return {}

        # Enforce pending request limit
        from .constants import MAX_PENDING_REQUESTS

        if len(self._pending_requests) >= MAX_PENDING_REQUESTS:
            raise RuntimeError(f"Too many pending requests ({MAX_PENDING_REQUESTS})")

        # Create a future for this request
        loop = asyncio.get_event_loop()
        future: asyncio.Future[dict[str, Any]] = loop.create_future()
        self._pending_requests[str(request_id)] = future

        try:
            # Send the request to the client via stdout
            await self._send_response(request)

            # Wait for the client's response (with timeout)
            return await asyncio.wait_for(future, timeout=120.0)
        except TimeoutError:
            raise RuntimeError(f"Timeout waiting for client response to request {request_id}")
        finally:
            self._pending_requests.pop(str(request_id), None)

    async def start(self) -> None:
        """Start the stdio transport server."""
        # Don't log in stdio mode to keep output clean
        self.running = True

        # Set up async stdio
        loop = asyncio.get_event_loop()
        self.reader = asyncio.StreamReader()
        protocol = asyncio.StreamReaderProtocol(self.reader)
        await loop.connect_read_pipe(lambda: protocol, sys.stdin)

        # Use stdout directly for writing
        self.writer = sys.stdout

        # Start listening for messages
        await self._listen()

    async def _listen(self) -> None:
        """Listen for incoming JSON-RPC messages on stdin."""
        buffer = ""

        while self.running:
            try:
                # Read from stdin
                if not self.reader:
                    break
                chunk = await self.reader.read(4096)
                if not chunk:
                    # Stdin closed, shutting down
                    break

                # Decode and add to buffer
                buffer += chunk.decode(DEFAULT_ENCODING)

                # Process complete messages
                while "\n" in buffer:
                    line, buffer = buffer.split("\n", 1)
                    line = line.strip()

                    if not line:
                        continue

                    # Parse and handle the message
                    await self._handle_message(line)

            except asyncio.CancelledError:
                break
            except (orjson.JSONDecodeError, ValueError) as e:
                logger.debug(f"Parse error in stdio listener: {e}")
                await self._send_error(None, JsonRpcError.PARSE_ERROR, f"Parse error: {str(e)}")
            except Exception as e:
                logger.debug(f"Error in stdio listener: {e}")
                await self._send_error(None, JsonRpcError.INTERNAL_ERROR, f"Internal error: {str(e)}")

    async def _handle_message(self, message: str) -> None:
        """
        Handle a single JSON-RPC message.

        Routes client responses (no method key) to pending server-initiated requests,
        and client requests (with method key) to the protocol handler.

        Args:
            message: Raw JSON-RPC message string
        """
        # Reject oversized messages
        from .constants import MAX_REQUEST_BODY_BYTES

        if len(message.encode("utf-8")) > MAX_REQUEST_BODY_BYTES:
            await self._send_error(
                None, JsonRpcError.INVALID_REQUEST, f"Message too large (max {MAX_REQUEST_BODY_BYTES} bytes)"
            )
            return

        try:
            # Parse the JSON-RPC message
            request_data = orjson.loads(message)

            # Check if this is a response to a server-initiated request
            method = request_data.get(KEY_METHOD)
            request_id = request_data.get(KEY_ID)

            if method is None and request_id is not None:
                # This is a response (no method key) - route to pending request
                request_id_str = str(request_id)
                if request_id_str in self._pending_requests:
                    future = self._pending_requests[request_id_str]
                    if not future.done():
                        future.set_result(request_data)
                    return
                # Response for unknown request ID - ignore
                logger.debug(f"Received response for unknown request ID: {request_id}")
                return

            # Extract params
            params = request_data.get(KEY_PARAMS, {})

            # Handle initialize specially to create session
            if method == McpMethod.INITIALIZE:
                client_info = params.get(KEY_CLIENT_INFO, {})
                protocol_version = params.get(KEY_PROTOCOL_VERSION, MCP_PROTOCOL_VERSION_2025_03)
                session_id = self.protocol.session_manager.create_session(client_info, protocol_version)
                self.session_id = session_id

            # Process through protocol handler
            response, error = await self.protocol.handle_request(request_data, self.session_id)

            # Send response if this was a request (not a notification)
            if request_id is not None and response:
                await self._send_response(response)

        except (orjson.JSONDecodeError, ValueError) as e:
            logger.debug(f"Invalid JSON in stdio message: {e}")
            await self._send_error(None, JsonRpcError.PARSE_ERROR, f"Parse error: {str(e)}")
        except Exception as e:
            logger.debug(f"Error handling stdio message: {e}")
            request_id = request_data.get(KEY_ID) if "request_data" in locals() else None
            await self._send_error(request_id, JsonRpcError.INTERNAL_ERROR, f"Internal error: {str(e)}")

    async def _send_response(self, response: dict[str, Any]) -> None:
        """
        Send a response over stdout.

        Args:
            response: Response dictionary to send
        """
        try:
            # Serialize with orjson for performance
            json_str = orjson.dumps(response).decode(DEFAULT_ENCODING)

            # Write to stdout with newline
            if self.writer:
                self.writer.write(json_str + "\n")
                self.writer.flush()

            # Sent response

        except Exception as e:
            logger.error(f"Critical error sending stdio response: {e}")

    async def _send_error(self, request_id: Any, code: int, message: str) -> None:
        """
        Send an error response.

        Args:
            request_id: The request ID (if available)
            code: JSON-RPC error code
            message: Error message
        """
        error_response = {
            JSONRPC_KEY: JSONRPC_VERSION,
            KEY_ID: request_id,
            KEY_ERROR: {"code": code, "message": message},
        }
        await self._send_response(error_response)

    async def stop(self) -> None:
        """Stop the stdio transport."""
        self.running = False

        # Cancel all pending server-to-client requests
        for req_id, future in self._pending_requests.items():
            if not future.done():
                future.cancel()
        self._pending_requests.clear()

        # Close reader if available
        if self.reader:
            self.reader.feed_eof()

    def __enter__(self) -> "StdioTransport":
        """Context manager entry."""
        return self

    def __exit__(self, exc_type: type[BaseException] | None, exc_val: BaseException | None, exc_tb: Any) -> None:
        """Context manager exit."""
        # Only create task if there's a running event loop
        try:
            loop = asyncio.get_running_loop()
            loop.create_task(self.stop())
        except RuntimeError:
            # No event loop running, just set the flag
            self.running = False


def run_stdio_server(protocol_handler: MCPProtocolHandler) -> None:
    """
    Run the MCP server in stdio mode.

    Args:
        protocol_handler: The MCP protocol handler instance
    """

    async def _run() -> None:
        transport = StdioTransport(protocol_handler)

        try:
            await transport.start()
        except KeyboardInterrupt:
            # Stdio server interrupted
            pass
        finally:
            await transport.stop()

    # Configure logging to stderr to keep stdout clean
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", stream=sys.stderr
    )

    # Run the async server
    asyncio.run(_run())


# ============================================================================
# Synchronous STDIO Transport
# ============================================================================


class StdioSyncTransport:
    """
    Synchronous MCP transport over stdin/stdout.

    This is a simpler, synchronous alternative to StdioTransport that uses
    asyncio.run() for each message, making it easier to integrate into
    non-async codebases.
    """

    def __init__(self, protocol_handler: Any) -> None:
        """
        Initialize synchronous stdio transport.

        Args:
            protocol_handler: The MCP protocol handler instance
        """
        self.protocol = protocol_handler
        self.session_id: str | None = None

        # Set the transport callback on the protocol handler for sampling support
        self.protocol._send_to_client = self._send_and_receive

    async def _send_and_receive(self, request: dict[str, Any]) -> dict[str, Any]:
        """
        Send a JSON-RPC request to the client and await the response.

        Used for server-initiated requests like sampling/createMessage.
        For notifications (no id), sends fire-and-forget and returns immediately.
        Reads from stdin in a thread to avoid blocking the event loop.

        Args:
            request: JSON-RPC request dict

        Returns:
            JSON-RPC response dict from the client
        """
        request_id = request.get(KEY_ID)
        if request_id is None:
            # Notification — fire and forget, no response expected
            self._send_response(request)
            return {}

        # Send the request to the client via stdout
        self._send_response(request)

        # Read the response from stdin (in a thread to avoid blocking)
        try:
            line = await asyncio.wait_for(
                asyncio.to_thread(sys.stdin.readline),
                timeout=120.0,
            )
        except TimeoutError:
            raise RuntimeError(f"Timeout waiting for client response to request {request_id}")

        if not line:
            raise RuntimeError("Client closed stdin while waiting for sampling response")

        line = line.strip()
        if not line:
            raise RuntimeError("Empty response from client")

        response = orjson.loads(line)

        # Verify response ID matches
        response_id = response.get(KEY_ID)
        if str(response_id) != str(request_id):
            raise RuntimeError(f"Response ID mismatch: expected {request_id}, got {response_id}")

        return dict(response)

    def run(self) -> None:
        """Run the STDIO transport synchronously."""
        logger.info("🔌 Starting MCP STDIO transport (sync)")

        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            while True:
                try:
                    # Read line from stdin
                    line = sys.stdin.readline()
                    if not line:  # EOF
                        break

                    line = line.strip()
                    if not line:
                        continue

                    # Process message
                    loop.run_until_complete(self._handle_message(line))

                except KeyboardInterrupt:
                    break
                except Exception as e:
                    logger.error(f"Error in main loop: {e}")
                    error_response = {
                        JSONRPC_KEY: JSONRPC_VERSION,
                        KEY_ID: None,
                        KEY_ERROR: {"code": JsonRpcError.INTERNAL_ERROR, "message": f"Transport error: {str(e)}"},
                    }
                    self._send_response(error_response)

        except Exception as e:
            logger.error(f"STDIO transport error: {e}")
        finally:
            loop.close()
            logger.info("🔌 STDIO transport stopped")

    async def _handle_message(self, line: str) -> None:
        """
        Handle incoming JSON-RPC message.

        Args:
            line: Raw JSON-RPC message string
        """
        # Reject oversized messages
        from .constants import MAX_REQUEST_BODY_BYTES

        if len(line.encode("utf-8")) > MAX_REQUEST_BODY_BYTES:
            self._send_error(JsonRpcError.INVALID_REQUEST, f"Message too large (max {MAX_REQUEST_BODY_BYTES} bytes)")
            return

        try:
            message = orjson.loads(line)

            # Process with protocol handler
            response, new_session_id = await self.protocol.handle_request(message, self.session_id)

            # Update session ID if this was initialization
            if new_session_id:
                self.session_id = new_session_id

            # Send response if one was generated
            if response:
                self._send_response(response)

        except orjson.JSONDecodeError as e:
            logger.error(f"Invalid JSON: {e}")
            self._send_error(JsonRpcError.PARSE_ERROR, "Parse error")
        except Exception as e:
            logger.error(f"Message handling error: {e}")
            self._send_error(JsonRpcError.INTERNAL_ERROR, f"Internal error: {str(e)}")

    def _send_response(self, response: dict[str, Any]) -> None:
        """
        Send response to stdout.

        Args:
            response: Response dictionary to send
        """
        try:
            response_line = orjson.dumps(response).decode(DEFAULT_ENCODING)
            print(response_line, flush=True)

        except Exception as e:
            logger.error(f"Error sending response: {e}")

    def _send_error(self, code: int, message: str, request_id: Any = None) -> None:
        """
        Send error response.

        Args:
            code: JSON-RPC error code
            message: Error message
            request_id: The request ID (if available)
        """
        error_response = {
            JSONRPC_KEY: JSONRPC_VERSION,
            KEY_ID: request_id,
            KEY_ERROR: {"code": code, "message": message},
        }
        self._send_response(error_response)
