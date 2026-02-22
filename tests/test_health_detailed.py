#!/usr/bin/env python3
"""Tests for health check endpoints: /health, /health/ready, /health/detailed."""

from unittest.mock import Mock

import orjson
import pytest
from starlette.requests import Request

from chuk_mcp_server.endpoints.health import (
    _protocol_handler as _original_handler,
)
from chuk_mcp_server.endpoints.health import (
    handle_health_detailed,
    handle_health_ready,
    handle_health_ultra_fast,
)
from chuk_mcp_server.protocol import MCPProtocolHandler
from chuk_mcp_server.types import ServerInfo, ToolHandler, create_server_capabilities


def _make_mock_request() -> Mock:
    """Create a minimal mock Starlette Request."""
    return Mock(spec=Request)


def _make_protocol(tools: int = 0, resources: int = 0, prompts: int = 0) -> MCPProtocolHandler:
    """Create a protocol handler with a configurable number of tools/resources/prompts."""
    server_info = ServerInfo(name="test-server", version="1.0.0")
    caps = create_server_capabilities()
    protocol = MCPProtocolHandler(server_info, caps)

    for i in range(tools):

        def _tool_fn(x: str = "default") -> str:
            return x

        protocol.register_tool(ToolHandler.from_function(_tool_fn, name=f"tool_{i}", description=f"Tool {i}"))

    for i in range(resources):
        from chuk_mcp_server.types import ResourceHandler

        async def _read() -> str:
            return "data"

        protocol.register_resource(
            ResourceHandler.from_function(
                uri=f"test://resource_{i}",
                func=_read,
                name=f"resource_{i}",
                description=f"Resource {i}",
            )
        )

    for i in range(prompts):
        from chuk_mcp_server.types import PromptHandler

        async def _get_prompt() -> str:
            return "prompt result"

        protocol.register_prompt(
            PromptHandler.from_function(
                _get_prompt,
                name=f"prompt_{i}",
                description=f"Prompt {i}",
            )
        )

    return protocol


def _set_protocol(protocol: MCPProtocolHandler | None) -> None:
    """Set the module-level _protocol_handler in the health module."""
    import chuk_mcp_server.endpoints.health as health_mod

    health_mod._protocol_handler = protocol


@pytest.fixture(autouse=True)
def _restore_protocol_handler():
    """Restore the original protocol handler after each test."""
    yield
    import chuk_mcp_server.endpoints.health as health_mod

    health_mod._protocol_handler = _original_handler


# =========================================================================
# /health endpoint tests
# =========================================================================


class TestHealthEndpoint:
    """Test the basic /health endpoint."""

    @pytest.mark.asyncio
    async def test_health_returns_200(self):
        """/health returns 200 with status and uptime."""
        request = _make_mock_request()
        response = await handle_health_ultra_fast(request)

        assert response.status_code == 200
        data = orjson.loads(response.body)
        assert data["status"] == "healthy"
        assert "uptime" in data
        assert "timestamp" in data
        assert data["server"] == "ChukMCPServer"

    @pytest.mark.asyncio
    async def test_health_json_content_type(self):
        """/health returns proper JSON content type."""
        request = _make_mock_request()
        response = await handle_health_ultra_fast(request)

        assert response.media_type == "application/json"

    @pytest.mark.asyncio
    async def test_health_cors_headers(self):
        """/health response includes CORS headers."""
        request = _make_mock_request()
        response = await handle_health_ultra_fast(request)

        assert response.headers.get("access-control-allow-origin") == "*"


# =========================================================================
# /health/ready endpoint tests
# =========================================================================


class TestHealthReadyEndpoint:
    """Test the /health/ready readiness probe."""

    @pytest.mark.asyncio
    async def test_ready_returns_200_when_tools_registered(self):
        """/health/ready returns 200 when tools are registered."""
        protocol = _make_protocol(tools=2)
        _set_protocol(protocol)

        request = _make_mock_request()
        response = await handle_health_ready(request)

        assert response.status_code == 200
        data = orjson.loads(response.body)
        assert data["status"] == "ready"

    @pytest.mark.asyncio
    async def test_ready_returns_503_when_no_tools(self):
        """/health/ready returns 503 when no tools are registered."""
        protocol = _make_protocol(tools=0)
        _set_protocol(protocol)

        request = _make_mock_request()
        response = await handle_health_ready(request)

        assert response.status_code == 503
        data = orjson.loads(response.body)
        assert data["status"] == "not_ready"

    @pytest.mark.asyncio
    async def test_ready_returns_503_when_no_protocol(self):
        """/health/ready returns 503 when protocol handler is not set."""
        _set_protocol(None)

        request = _make_mock_request()
        response = await handle_health_ready(request)

        assert response.status_code == 503
        data = orjson.loads(response.body)
        assert data["status"] == "not_ready"

    @pytest.mark.asyncio
    async def test_ready_json_content_type(self):
        """/health/ready returns proper JSON content type."""
        protocol = _make_protocol(tools=1)
        _set_protocol(protocol)

        request = _make_mock_request()
        response = await handle_health_ready(request)

        assert response.media_type == "application/json"

    @pytest.mark.asyncio
    async def test_ready_cors_headers(self):
        """/health/ready response includes CORS headers."""
        protocol = _make_protocol(tools=1)
        _set_protocol(protocol)

        request = _make_mock_request()
        response = await handle_health_ready(request)

        assert response.headers.get("access-control-allow-origin") == "*"


# =========================================================================
# /health/detailed endpoint tests
# =========================================================================


class TestHealthDetailedEndpoint:
    """Test the /health/detailed endpoint."""

    @pytest.mark.asyncio
    async def test_detailed_returns_session_and_tool_counts(self):
        """/health/detailed returns session count, tool count, resource count."""
        protocol = _make_protocol(tools=3, resources=2, prompts=1)
        # Create a session to verify session count
        protocol.session_manager.create_session({"name": "test-client"}, "2025-03-26")
        _set_protocol(protocol)

        request = _make_mock_request()
        response = await handle_health_detailed(request)

        assert response.status_code == 200
        data = orjson.loads(response.body)

        assert data["status"] == "healthy"
        assert data["tools"] == 3
        assert data["resources"] == 2
        assert data["prompts"] == 1
        assert data["sessions"] == 1
        assert data["in_flight_requests"] == 0
        assert "uptime" in data
        assert "timestamp" in data
        assert data["server"] == "ChukMCPServer"

    @pytest.mark.asyncio
    async def test_detailed_returns_zeros_when_no_protocol(self):
        """/health/detailed returns zero counts when protocol handler is not set."""
        _set_protocol(None)

        request = _make_mock_request()
        response = await handle_health_detailed(request)

        assert response.status_code == 200
        data = orjson.loads(response.body)

        assert data["tools"] == 0
        assert data["resources"] == 0
        assert data["prompts"] == 0
        assert data["sessions"] == 0
        assert data["in_flight_requests"] == 0

    @pytest.mark.asyncio
    async def test_detailed_json_content_type(self):
        """/health/detailed returns proper JSON content type."""
        protocol = _make_protocol(tools=1)
        _set_protocol(protocol)

        request = _make_mock_request()
        response = await handle_health_detailed(request)

        assert response.media_type == "application/json"

    @pytest.mark.asyncio
    async def test_detailed_cors_headers(self):
        """/health/detailed response includes CORS headers."""
        protocol = _make_protocol(tools=1)
        _set_protocol(protocol)

        request = _make_mock_request()
        response = await handle_health_detailed(request)

        assert response.headers.get("access-control-allow-origin") == "*"
