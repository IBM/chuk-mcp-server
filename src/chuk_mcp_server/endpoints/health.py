#!/usr/bin/env python3
"""
Health check endpoints with dynamic fixed-length timestamps.

Provides three health endpoints:
- /health       - Basic liveness probe (ultra-fast)
- /health/ready - Readiness probe (checks that tools are registered)
- /health/detailed - Detailed health with session count, tool count, etc.
"""

import time

import orjson
from starlette.requests import Request
from starlette.responses import Response

from ..protocol import MCPProtocolHandler
from .constants import CONTENT_TYPE_JSON, HEADERS_CORS_NOCACHE, SERVER_NAME, STATUS_HEALTHY, STATUS_READY

_SERVER_START_TIME = time.time()

# Module-level reference set by HealthEndpoint.__init__
_protocol_handler: MCPProtocolHandler | None = None


class HealthEndpoint:
    def __init__(self, protocol_handler: MCPProtocolHandler):
        global _protocol_handler
        self.protocol = protocol_handler
        self.start_time = time.time()
        _protocol_handler = protocol_handler

    async def handle_request(self, request: Request) -> Response:
        return await handle_health_ultra_fast(request)


async def handle_health_ultra_fast(_request: Request) -> Response:
    """Dynamic health check with Unix millisecond timestamp"""
    current_time = time.time()
    timestamp_ms = int(current_time * 1000)
    uptime_seconds = int(current_time - _SERVER_START_TIME)

    response_data = {
        "status": STATUS_HEALTHY,
        "server": SERVER_NAME,
        "timestamp": timestamp_ms,
        "uptime": uptime_seconds,
    }

    return Response(orjson.dumps(response_data), media_type=CONTENT_TYPE_JSON, headers=HEADERS_CORS_NOCACHE)


async def handle_health_ready(_request: Request) -> Response:
    """Readiness probe - checks that tools are registered."""
    protocol = _protocol_handler
    ready = len(protocol.tools) > 0 if protocol is not None else False

    status_code = 200 if ready else 503
    return Response(
        orjson.dumps({"status": STATUS_READY if ready else "not_ready"}),
        status_code=status_code,
        media_type=CONTENT_TYPE_JSON,
        headers=HEADERS_CORS_NOCACHE,
    )


async def handle_health_detailed(_request: Request) -> Response:
    """Detailed health check with session count, tool count, resource count, etc."""
    current_time = time.time()
    timestamp_ms = int(current_time * 1000)
    uptime_seconds = int(current_time - _SERVER_START_TIME)

    protocol = _protocol_handler
    if protocol is not None:
        tool_count = len(protocol.tools)
        resource_count = len(protocol.resources)
        prompt_count = len(protocol.prompts)
        session_count = len(protocol.session_manager.sessions)
        in_flight_requests = len(protocol._in_flight_requests)
    else:
        tool_count = 0
        resource_count = 0
        prompt_count = 0
        session_count = 0
        in_flight_requests = 0

    response_data = {
        "status": STATUS_HEALTHY,
        "server": SERVER_NAME,
        "timestamp": timestamp_ms,
        "uptime": uptime_seconds,
        "tools": tool_count,
        "resources": resource_count,
        "prompts": prompt_count,
        "sessions": session_count,
        "in_flight_requests": in_flight_requests,
    }

    return Response(orjson.dumps(response_data), media_type=CONTENT_TYPE_JSON, headers=HEADERS_CORS_NOCACHE)
