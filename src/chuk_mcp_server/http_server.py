#!/usr/bin/env python3
# src/chuk_mcp_server/http_server.py
"""
HTTP Server with systematic performance bottleneck fixes
Target: Break through the 3,600 RPS ceiling
"""

import logging

import uvicorn
from starlette.applications import Starlette
from starlette.middleware import Middleware
from starlette.middleware.cors import CORSMiddleware
from starlette.requests import Request
from starlette.responses import Response

from .endpoint_registry import http_endpoint_registry

# Import optimized endpoints
from .endpoints import (
    HealthEndpoint,
    InfoEndpoint,
    MCPEndpoint,
    handle_health_detailed,
    handle_health_ready,
    handle_health_ultra_fast,
    handle_ping,
    handle_version,
)
from .middlewares import ContextMiddleware
from .openapi import generate_openapi_spec
from .protocol import MCPProtocolHandler

logger = logging.getLogger(__name__)


def internal_error_response() -> Response:
    """Simple internal error response"""
    return Response(
        '{"error": "Internal server error", "code": 500}',
        status_code=500,
        media_type="application/json",
        headers={"Access-Control-Allow-Origin": "*"},
    )


class HTTPServer:
    """HTTP server optimized to break through performance bottlenecks."""

    def __init__(self, protocol_handler: MCPProtocolHandler, post_register_hook=None):
        """
        Initialize HTTP server.

        Args:
            protocol_handler: MCP protocol handler
            post_register_hook: Optional callback to register additional endpoints after default endpoints are registered
        """
        self.protocol = protocol_handler
        self._register_endpoints()

        # Call post-register hook if provided (for custom endpoints like OAuth)
        if post_register_hook:
            post_register_hook()

        self.app = self._create_app()
        logger.info("🚀 HTTP server initialized with bottleneck fixes")

    def _register_endpoints(self):
        """Register all endpoints."""

        # Clear any existing endpoints first
        http_endpoint_registry.clear_endpoints()

        # Create endpoint instances
        mcp_endpoint = MCPEndpoint(self.protocol)
        HealthEndpoint(self.protocol)
        info_endpoint = InfoEndpoint(self.protocol)

        # Create OpenAPI handler
        openapi_handler = self._create_openapi_handler()

        # Create docs handler
        docs_handler = self._create_docs_handler(info_endpoint)

        # Define all endpoints
        endpoints = [
            ("/ping", handle_ping, ["GET"], "ping"),
            ("/version", handle_version, ["GET"], "version"),
            ("/health", handle_health_ultra_fast, ["GET"], "health_fast"),
            ("/health/ready", handle_health_ready, ["GET"], "health_ready"),
            ("/health/detailed", handle_health_detailed, ["GET"], "health_detailed"),
            ("/mcp", mcp_endpoint.handle_request, ["GET", "POST", "DELETE", "OPTIONS"], "mcp_protocol"),
            ("/mcp/respond", mcp_endpoint.handle_respond, ["POST", "OPTIONS"], "mcp_respond"),
            ("/", info_endpoint.handle_request, ["GET"], "server_info"),
            ("/info", info_endpoint.handle_request, ["GET"], "server_info_explicit"),
            ("/docs", docs_handler, ["GET"], "documentation"),
            ("/openapi.json", openapi_handler, ["GET"], "openapi_spec"),
        ]

        # Register endpoints
        for path, handler, methods, name in endpoints:
            http_endpoint_registry.register_endpoint(
                path=path, handler=handler, methods=methods, name=name, description=f"Endpoint: {name}"
            )

        logger.info(f"📊 Registered {len(endpoints)} endpoints")

    def _create_openapi_handler(self):
        """Create OpenAPI spec handler."""
        import orjson

        protocol = self.protocol

        async def openapi_handler(request: Request) -> Response:  # noqa: ARG001
            spec = generate_openapi_spec(protocol)
            return Response(
                orjson.dumps(spec),
                media_type="application/json",
                headers={"Access-Control-Allow-Origin": "*"},
            )

        return openapi_handler

    def _create_docs_handler(self, info_endpoint):
        """Create docs handler."""

        async def docs_handler(request: Request) -> Response:
            request.state.format_override = "docs"
            return await info_endpoint.handle_request(request)

        return docs_handler

    def _create_app(self) -> Starlette:
        """Create Starlette application with minimal overhead."""

        # MINIMAL middleware stack to reduce overhead
        middleware = [
            # Simplified CORS middleware
            Middleware(
                CORSMiddleware,
                allow_origins=["*"],
                allow_methods=["GET", "POST", "DELETE", "OPTIONS"],
                allow_headers=["*"],
                expose_headers=["Mcp-Session-Id"],
                max_age=86400,  # Long cache for preflight
            ),
            # Remove GZip middleware for benchmarking (it adds overhead)
            # Middleware(GZipMiddleware, minimum_size=2048)
        ]

        for custom_middleware in http_endpoint_registry.get_middleware():
            middleware.append(Middleware(custom_middleware.middleware_class))

        middleware.append(Middleware(ContextMiddleware))

        routes = http_endpoint_registry.get_routes()
        logger.info(f"🔗 Creating Starlette app with {len(routes)} routes")

        return Starlette(
            debug=False,
            routes=routes,
            middleware=middleware,
            exception_handlers={Exception: self._global_exception_handler},
        )

    async def _global_exception_handler(self, request: Request, exc: Exception) -> Response:
        """Minimal exception handler."""
        logger.error(f"Exception in {request.method} {request.url.path}: {exc}")
        return internal_error_response()

    def run(
        self,
        host: str = "localhost",
        port: int = 8000,
        debug: bool = False,
        log_level: str = "warning",
        reload: bool = False,
    ):
        """Run with maximum performance configuration to break bottlenecks.

        Args:
            host: Host to bind to
            port: Port to bind to
            debug: Enable debug mode (more verbose logging)
            log_level: Logging level for application logs (debug, info, warning, error, critical)
            reload: Enable hot reload (auto-restart on file changes)
        """

        # Logging is already configured in core.py before this is called
        # Just determine the uvicorn log level based on the passed log_level
        import os
        import sys

        # If debug is explicitly set to True, override log_level
        if debug is True:
            log_level = "debug"

        # Handle log_level - ensure it's a string
        default_level = log_level if isinstance(log_level, str) else "warning"
        app_log_level = os.getenv("MCP_LOG_LEVEL", default_level).upper()

        # Set uvicorn log level based on app log level
        if app_log_level == "DEBUG":
            uvicorn_log_level = "debug"
        elif app_log_level == "INFO":
            uvicorn_log_level = "info"
        else:
            uvicorn_log_level = "warning"

        # PERFORMANCE-FOCUSED uvicorn configuration
        uvicorn_config = {
            "app": self.app,
            "host": host,
            "port": port,
            # Core performance settings
            "workers": 1,
            "http": "httptools",  # Force httptools
            # Disable overhead features
            "access_log": debug,  # Only show access logs in debug mode
            "server_header": False,
            "date_header": False,
            "log_level": uvicorn_log_level,  # Configurable logging
            # Connection optimizations
            "backlog": 4096,  # Increase from 2048
            "limit_concurrency": 2000,  # Increase from 1000
            "timeout_keep_alive": 60,  # Longer keep-alive
            # Buffer optimizations
            "h11_max_incomplete_event_size": 16384,  # Increase buffer
        }

        # Add hot reload if requested
        if reload:
            uvicorn_config["reload"] = True

        # Add uvloop only on non-Windows platforms (uvloop doesn't support Windows)
        if sys.platform != "win32":
            try:
                import uvloop  # noqa: F401

                uvicorn_config["loop"] = "uvloop"
            except ImportError:
                pass  # Fall back to default asyncio loop

        try:
            import httptools  # noqa: F401
        except ImportError:
            uvicorn_config.pop("http", None)

        try:
            uvicorn.run(**uvicorn_config)
        except KeyboardInterrupt:
            logger.info("\n👋 Server shutting down gracefully...")
        except Exception as e:
            logger.error(f"❌ Server startup error: {e}")
            raise


# Factory function
def create_server(protocol_handler: MCPProtocolHandler, post_register_hook=None) -> HTTPServer:
    """
    Create HTTP server with bottleneck fixes.

    Args:
        protocol_handler: MCP protocol handler
        post_register_hook: Optional callback to register additional endpoints after default endpoints
    """
    return HTTPServer(protocol_handler, post_register_hook=post_register_hook)
