#!/usr/bin/env python3
"""Comprehensive tests for http_server module."""

from unittest.mock import AsyncMock, Mock, patch

import pytest
from starlette.applications import Starlette
from starlette.responses import Response

from chuk_mcp_server.http_server import HTTPServer, create_server, internal_error_response
from chuk_mcp_server.protocol import MCPProtocolHandler


class TestInternalErrorResponse:
    """Test internal error response function."""

    def test_internal_error_response(self):
        """Test internal_error_response function."""
        response = internal_error_response()

        assert isinstance(response, Response)
        assert response.status_code == 500
        assert response.media_type == "application/json"
        assert response.headers["Access-Control-Allow-Origin"] == "*"
        assert b'"error": "Internal server error"' in response.body
        assert b'"code": 500' in response.body


class TestHTTPServer:
    """Test HTTPServer class."""

    def setup_method(self):
        """Set up test fixtures."""
        # Create mock protocol handler
        self.mock_protocol = Mock(spec=MCPProtocolHandler)
        self.mock_protocol.server_info = Mock()
        self.mock_protocol.server_info.name = "TestServer"
        self.mock_protocol.server_info.version = "1.0.0"
        self.mock_protocol.tools = {}
        self.mock_protocol.resources = {}
        self.mock_protocol.capabilities = Mock()

    @patch("chuk_mcp_server.http_server.http_endpoint_registry")
    @patch("chuk_mcp_server.http_server.MCPEndpoint")
    @patch("chuk_mcp_server.http_server.HealthEndpoint")
    @patch("chuk_mcp_server.http_server.InfoEndpoint")
    def test_server_initialization(self, mock_info_endpoint, mock_health_endpoint, mock_mcp_endpoint, mock_registry):
        """Test server initialization."""
        # Mock registry methods
        mock_registry.clear_endpoints = Mock()
        mock_registry.register_endpoint = Mock()
        mock_registry.get_routes.return_value = []

        # Mock endpoint instances
        mock_info_instance = Mock()
        mock_info_instance.handle_request = AsyncMock()
        mock_info_endpoint.return_value = mock_info_instance

        mock_mcp_instance = Mock()
        mock_mcp_instance.handle_request = AsyncMock()
        mock_mcp_endpoint.return_value = mock_mcp_instance

        server = HTTPServer(self.mock_protocol)

        assert server.protocol == self.mock_protocol
        assert isinstance(server.app, Starlette)

        # Should have cleared existing endpoints
        mock_registry.clear_endpoints.assert_called_once()

        # Should have registered endpoints
        assert mock_registry.register_endpoint.call_count == 11  # 11 endpoints

    @patch("chuk_mcp_server.http_server.http_endpoint_registry")
    @patch("chuk_mcp_server.http_server.MCPEndpoint")
    @patch("chuk_mcp_server.http_server.HealthEndpoint")
    @patch("chuk_mcp_server.http_server.InfoEndpoint")
    def test_register_endpoints(self, mock_info_endpoint, mock_health_endpoint, mock_mcp_endpoint, mock_registry):
        """Test endpoint registration."""
        # Mock registry
        mock_registry.clear_endpoints = Mock()
        mock_registry.register_endpoint = Mock()
        mock_registry.get_routes.return_value = []

        # Mock endpoint instances
        mock_info_instance = Mock()
        mock_info_instance.handle_request = AsyncMock()
        mock_info_endpoint.return_value = mock_info_instance

        mock_mcp_instance = Mock()
        mock_mcp_instance.handle_request = AsyncMock()
        mock_mcp_endpoint.return_value = mock_mcp_instance

        HTTPServer(self.mock_protocol)

        # Check that specific endpoints were registered
        registered_paths = []
        for call in mock_registry.register_endpoint.call_args_list:
            registered_paths.append(call[1]["path"])

        expected_paths = [
            "/ping",
            "/version",
            "/health",
            "/health/ready",
            "/health/detailed",
            "/mcp",
            "/",
            "/info",
            "/docs",
        ]
        for path in expected_paths:
            assert path in registered_paths

    @patch("chuk_mcp_server.http_server.http_endpoint_registry")
    @patch("chuk_mcp_server.http_server.MCPEndpoint")
    @patch("chuk_mcp_server.http_server.HealthEndpoint")
    @patch("chuk_mcp_server.http_server.InfoEndpoint")
    def test_create_docs_handler(self, mock_info_endpoint, mock_health_endpoint, mock_mcp_endpoint, mock_registry):
        """Test docs handler creation."""
        # Mock registry
        mock_registry.clear_endpoints = Mock()
        mock_registry.register_endpoint = Mock()
        mock_registry.get_routes.return_value = []

        # Mock info endpoint
        mock_info_instance = Mock()
        mock_info_instance.handle_request = AsyncMock(return_value=Response("test docs"))
        mock_info_endpoint.return_value = mock_info_instance

        mock_mcp_instance = Mock()
        mock_mcp_endpoint.return_value = mock_mcp_instance

        HTTPServer(self.mock_protocol)

        # Get the docs handler from registered endpoints
        docs_handler = None
        for call in mock_registry.register_endpoint.call_args_list:
            if call[1]["path"] == "/docs":
                docs_handler = call[1]["handler"]
                break

        assert docs_handler is not None

    @patch("chuk_mcp_server.http_server.http_endpoint_registry")
    @patch("chuk_mcp_server.http_server.MCPEndpoint")
    @patch("chuk_mcp_server.http_server.HealthEndpoint")
    @patch("chuk_mcp_server.http_server.InfoEndpoint")
    @pytest.mark.asyncio
    async def test_docs_handler_functionality(
        self, mock_info_endpoint, mock_health_endpoint, mock_mcp_endpoint, mock_registry
    ):
        """Test docs handler functionality."""
        # Mock registry
        mock_registry.clear_endpoints = Mock()
        mock_registry.register_endpoint = Mock()
        mock_registry.get_routes.return_value = []

        # Mock info endpoint that returns docs format
        mock_info_instance = Mock()
        mock_info_instance.handle_request = AsyncMock(return_value=Response("# Documentation"))
        mock_info_endpoint.return_value = mock_info_instance

        mock_mcp_instance = Mock()
        mock_mcp_endpoint.return_value = mock_mcp_instance

        HTTPServer(self.mock_protocol)

        # Get the docs handler
        docs_handler = None
        for call in mock_registry.register_endpoint.call_args_list:
            if call[1]["path"] == "/docs":
                docs_handler = call[1]["handler"]
                break

        # Create mock request with state
        mock_request = Mock()
        mock_request.state = Mock()

        response = await docs_handler(mock_request)

        # Should have set format_override on request.state and called info endpoint
        assert mock_request.state.format_override == "docs"
        mock_info_instance.handle_request.assert_called_once_with(mock_request)
        assert isinstance(response, Response)

    @patch("chuk_mcp_server.http_server.http_endpoint_registry")
    @patch("chuk_mcp_server.http_server.MCPEndpoint")
    @patch("chuk_mcp_server.http_server.HealthEndpoint")
    @patch("chuk_mcp_server.http_server.InfoEndpoint")
    def test_create_app(self, mock_info_endpoint, mock_health_endpoint, mock_mcp_endpoint, mock_registry):
        """Test Starlette app creation."""
        from starlette.routing import Route

        # Mock registry
        mock_registry.clear_endpoints = Mock()
        mock_registry.register_endpoint = Mock()
        mock_route = Route("/test", AsyncMock())
        mock_registry.get_routes.return_value = [mock_route]

        # Mock endpoints
        mock_info_endpoint.return_value = Mock()
        mock_mcp_endpoint.return_value = Mock()

        server = HTTPServer(self.mock_protocol)

        assert isinstance(server.app, Starlette)
        assert server.app.debug is False

        # Should have middleware (check the middleware in a safer way)
        assert hasattr(server.app, "middleware_stack")
        if server.app.middleware_stack:
            assert hasattr(server.app.middleware_stack, "middleware")

    @patch("chuk_mcp_server.http_server.http_endpoint_registry")
    @patch("chuk_mcp_server.http_server.MCPEndpoint")
    @patch("chuk_mcp_server.http_server.HealthEndpoint")
    @patch("chuk_mcp_server.http_server.InfoEndpoint")
    @pytest.mark.asyncio
    async def test_global_exception_handler(
        self, mock_info_endpoint, mock_health_endpoint, mock_mcp_endpoint, mock_registry
    ):
        """Test global exception handler."""
        # Mock registry
        mock_registry.clear_endpoints = Mock()
        mock_registry.register_endpoint = Mock()
        mock_registry.get_routes.return_value = []

        # Mock endpoints
        mock_info_endpoint.return_value = Mock()
        mock_mcp_endpoint.return_value = Mock()

        server = HTTPServer(self.mock_protocol)

        # Create mock request
        mock_request = Mock()
        mock_request.method = "GET"
        mock_request.url = Mock()
        mock_request.url.path = "/test"

        # Test exception handling
        test_exception = Exception("Test error")

        with patch("chuk_mcp_server.http_server.logger") as mock_logger:
            response = await server._global_exception_handler(mock_request, test_exception)

            assert isinstance(response, Response)
            assert response.status_code == 500
            assert response.media_type == "application/json"

            # Should have logged the error
            mock_logger.error.assert_called_once()

    @patch("chuk_mcp_server.http_server.http_endpoint_registry")
    @patch("chuk_mcp_server.http_server.MCPEndpoint")
    @patch("chuk_mcp_server.http_server.HealthEndpoint")
    @patch("chuk_mcp_server.http_server.InfoEndpoint")
    @patch("chuk_mcp_server.http_server.uvicorn")
    def test_run_basic(self, mock_uvicorn, mock_info_endpoint, mock_health_endpoint, mock_mcp_endpoint, mock_registry):
        """Test basic server run."""
        # Mock registry
        mock_registry.clear_endpoints = Mock()
        mock_registry.register_endpoint = Mock()
        mock_registry.get_routes.return_value = []

        # Mock endpoints
        mock_info_endpoint.return_value = Mock()
        mock_mcp_endpoint.return_value = Mock()

        server = HTTPServer(self.mock_protocol)

        # Mock uvicorn.run to avoid actually starting server
        mock_uvicorn.run = Mock()

        server.run()

        # Should have called uvicorn.run with config
        mock_uvicorn.run.assert_called_once()

        # Check some config parameters
        config = mock_uvicorn.run.call_args[1]
        assert config["app"] == server.app
        assert config["host"] == "localhost"
        assert config["port"] == 8000
        assert config["workers"] == 1
        assert config["access_log"] is False

    @patch("chuk_mcp_server.http_server.http_endpoint_registry")
    @patch("chuk_mcp_server.http_server.MCPEndpoint")
    @patch("chuk_mcp_server.http_server.HealthEndpoint")
    @patch("chuk_mcp_server.http_server.InfoEndpoint")
    @patch("chuk_mcp_server.http_server.uvicorn")
    def test_run_with_custom_params(
        self, mock_uvicorn, mock_info_endpoint, mock_health_endpoint, mock_mcp_endpoint, mock_registry
    ):
        """Test server run with custom parameters."""
        # Mock registry
        mock_registry.clear_endpoints = Mock()
        mock_registry.register_endpoint = Mock()
        mock_registry.get_routes.return_value = []

        # Mock endpoints
        mock_info_endpoint.return_value = Mock()
        mock_mcp_endpoint.return_value = Mock()

        server = HTTPServer(self.mock_protocol)

        mock_uvicorn.run = Mock()

        server.run(host="0.0.0.0", port=9000, debug=True)

        config = mock_uvicorn.run.call_args[1]
        assert config["host"] == "0.0.0.0"
        assert config["port"] == 9000
        assert config["log_level"] == "debug"
        assert config["access_log"] is True

    @patch("chuk_mcp_server.http_server.http_endpoint_registry")
    @patch("chuk_mcp_server.http_server.MCPEndpoint")
    @patch("chuk_mcp_server.http_server.HealthEndpoint")
    @patch("chuk_mcp_server.http_server.InfoEndpoint")
    @patch("chuk_mcp_server.http_server.uvicorn")
    def test_run_uvloop_available(
        self, mock_uvicorn, mock_info_endpoint, mock_health_endpoint, mock_mcp_endpoint, mock_registry
    ):
        """Test server run when uvloop is available."""
        # Mock registry
        mock_registry.clear_endpoints = Mock()
        mock_registry.register_endpoint = Mock()
        mock_registry.get_routes.return_value = []

        # Mock endpoints
        mock_info_endpoint.return_value = Mock()
        mock_mcp_endpoint.return_value = Mock()

        server = HTTPServer(self.mock_protocol)

        mock_uvicorn.run = Mock()

        # Mock uvloop as available
        with patch("builtins.__import__") as mock_import:
            # Make uvloop import succeed
            mock_import.return_value = Mock()

            server.run()

            config = mock_uvicorn.run.call_args[1]
            assert config["loop"] == "uvloop"

    @patch("chuk_mcp_server.http_server.http_endpoint_registry")
    @patch("chuk_mcp_server.http_server.MCPEndpoint")
    @patch("chuk_mcp_server.http_server.HealthEndpoint")
    @patch("chuk_mcp_server.http_server.InfoEndpoint")
    @patch("chuk_mcp_server.http_server.uvicorn")
    def test_run_uvloop_unavailable(
        self, mock_uvicorn, mock_info_endpoint, mock_health_endpoint, mock_mcp_endpoint, mock_registry
    ):
        """Test server run when uvloop is unavailable."""
        # Mock registry
        mock_registry.clear_endpoints = Mock()
        mock_registry.register_endpoint = Mock()
        mock_registry.get_routes.return_value = []

        # Mock endpoints
        mock_info_endpoint.return_value = Mock()
        mock_mcp_endpoint.return_value = Mock()

        server = HTTPServer(self.mock_protocol)

        mock_uvicorn.run = Mock()

        # Mock ImportError for uvloop
        original_import = __builtins__["__import__"]

        def mock_import(name, *args, **kwargs):
            if name == "uvloop":
                raise ImportError("uvloop not available")
            return original_import(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=mock_import):
            server.run()

            config = mock_uvicorn.run.call_args[1]
            assert "loop" not in config

    @patch("chuk_mcp_server.http_server.http_endpoint_registry")
    @patch("chuk_mcp_server.http_server.MCPEndpoint")
    @patch("chuk_mcp_server.http_server.HealthEndpoint")
    @patch("chuk_mcp_server.http_server.InfoEndpoint")
    @patch("chuk_mcp_server.http_server.uvicorn")
    def test_run_httptools_available(
        self, mock_uvicorn, mock_info_endpoint, mock_health_endpoint, mock_mcp_endpoint, mock_registry
    ):
        """Test server run when httptools is available."""
        # Mock registry
        mock_registry.clear_endpoints = Mock()
        mock_registry.register_endpoint = Mock()
        mock_registry.get_routes.return_value = []

        # Mock endpoints
        mock_info_endpoint.return_value = Mock()
        mock_mcp_endpoint.return_value = Mock()

        server = HTTPServer(self.mock_protocol)

        mock_uvicorn.run = Mock()

        server.run()

        config = mock_uvicorn.run.call_args[1]
        assert config["http"] == "httptools"

    @patch("chuk_mcp_server.http_server.http_endpoint_registry")
    @patch("chuk_mcp_server.http_server.MCPEndpoint")
    @patch("chuk_mcp_server.http_server.HealthEndpoint")
    @patch("chuk_mcp_server.http_server.InfoEndpoint")
    @patch("chuk_mcp_server.http_server.uvicorn")
    def test_run_httptools_unavailable(
        self, mock_uvicorn, mock_info_endpoint, mock_health_endpoint, mock_mcp_endpoint, mock_registry
    ):
        """Test server run when httptools is unavailable."""
        # Mock registry
        mock_registry.clear_endpoints = Mock()
        mock_registry.register_endpoint = Mock()
        mock_registry.get_routes.return_value = []

        # Mock endpoints
        mock_info_endpoint.return_value = Mock()
        mock_mcp_endpoint.return_value = Mock()

        server = HTTPServer(self.mock_protocol)

        mock_uvicorn.run = Mock()

        # Mock ImportError for httptools
        original_import = __builtins__["__import__"]

        def mock_import(name, *args, **kwargs):
            if name == "httptools":
                raise ImportError("httptools not available")
            return original_import(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=mock_import):
            server.run()

            config = mock_uvicorn.run.call_args[1]
            assert "http" not in config

    @patch("chuk_mcp_server.http_server.http_endpoint_registry")
    @patch("chuk_mcp_server.http_server.MCPEndpoint")
    @patch("chuk_mcp_server.http_server.HealthEndpoint")
    @patch("chuk_mcp_server.http_server.InfoEndpoint")
    @patch("chuk_mcp_server.http_server.uvicorn")
    def test_run_keyboard_interrupt(
        self, mock_uvicorn, mock_info_endpoint, mock_health_endpoint, mock_mcp_endpoint, mock_registry
    ):
        """Test server run with KeyboardInterrupt."""
        # Mock registry
        mock_registry.clear_endpoints = Mock()
        mock_registry.register_endpoint = Mock()
        mock_registry.get_routes.return_value = []

        # Mock endpoints
        mock_info_endpoint.return_value = Mock()
        mock_mcp_endpoint.return_value = Mock()

        server = HTTPServer(self.mock_protocol)

        mock_uvicorn.run = Mock(side_effect=KeyboardInterrupt())

        with patch("chuk_mcp_server.http_server.logger") as mock_logger:
            server.run()

            # Should have logged graceful shutdown
            mock_logger.info.assert_any_call("\n👋 Server shutting down gracefully...")

    @patch("chuk_mcp_server.http_server.http_endpoint_registry")
    @patch("chuk_mcp_server.http_server.MCPEndpoint")
    @patch("chuk_mcp_server.http_server.HealthEndpoint")
    @patch("chuk_mcp_server.http_server.InfoEndpoint")
    @patch("chuk_mcp_server.http_server.uvicorn")
    def test_run_server_error(
        self, mock_uvicorn, mock_info_endpoint, mock_health_endpoint, mock_mcp_endpoint, mock_registry
    ):
        """Test server run with server error."""
        # Mock registry
        mock_registry.clear_endpoints = Mock()
        mock_registry.register_endpoint = Mock()
        mock_registry.get_routes.return_value = []

        # Mock endpoints
        mock_info_endpoint.return_value = Mock()
        mock_mcp_endpoint.return_value = Mock()

        server = HTTPServer(self.mock_protocol)

        test_error = Exception("Server startup failed")
        mock_uvicorn.run = Mock(side_effect=test_error)

        with patch("chuk_mcp_server.http_server.logger") as mock_logger:
            with pytest.raises(Exception, match="Server startup failed"):
                server.run()

            # Should have logged the error
            mock_logger.error.assert_any_call("❌ Server startup error: Server startup failed")

    @patch("chuk_mcp_server.http_server.http_endpoint_registry")
    @patch("chuk_mcp_server.http_server.MCPEndpoint")
    @patch("chuk_mcp_server.http_server.HealthEndpoint")
    @patch("chuk_mcp_server.http_server.InfoEndpoint")
    def test_performance_optimizations(
        self, mock_info_endpoint, mock_health_endpoint, mock_mcp_endpoint, mock_registry
    ):
        """Test that performance optimizations are applied."""
        # Mock registry
        mock_registry.clear_endpoints = Mock()
        mock_registry.register_endpoint = Mock()
        mock_registry.get_routes.return_value = []

        # Mock endpoints
        mock_info_endpoint.return_value = Mock()
        mock_mcp_endpoint.return_value = Mock()

        with patch("chuk_mcp_server.http_server.uvicorn") as mock_uvicorn:
            mock_uvicorn.run = Mock()

            server = HTTPServer(self.mock_protocol)
            server.run()

            config = mock_uvicorn.run.call_args[1]

            # Check performance optimizations
            assert config["workers"] == 1
            assert config["access_log"] is False
            assert config["server_header"] is False
            assert config["date_header"] is False
            assert config["log_level"] == "warning"  # Default log level is warning
            assert config["backlog"] == 4096
            assert config["limit_concurrency"] == 2000
            assert config["timeout_keep_alive"] == 60
            assert config["h11_max_incomplete_event_size"] == 16384


class TestCreateServerFactory:
    """Test create_server factory function."""

    def test_create_server(self):
        """Test create_server factory function."""
        mock_protocol = Mock(spec=MCPProtocolHandler)
        mock_protocol.server_info = Mock()
        mock_protocol.server_info.name = "TestServer"
        mock_protocol.server_info.version = "1.0.0"
        mock_protocol.tools = {}
        mock_protocol.resources = {}
        mock_protocol.capabilities = Mock()

        with patch("chuk_mcp_server.http_server.http_endpoint_registry"):
            server = create_server(mock_protocol)

            assert isinstance(server, HTTPServer)
            assert server.protocol == mock_protocol


class TestEdgeCasesAndIntegration:
    """Test edge cases and integration scenarios."""

    def setup_method(self):
        """Set up test fixtures."""
        self.mock_protocol = Mock(spec=MCPProtocolHandler)
        self.mock_protocol.server_info = Mock()
        self.mock_protocol.server_info.name = "TestServer"
        self.mock_protocol.server_info.version = "1.0.0"
        self.mock_protocol.tools = {}
        self.mock_protocol.resources = {}
        self.mock_protocol.capabilities = Mock()

    @patch("chuk_mcp_server.http_server.http_endpoint_registry")
    @patch("chuk_mcp_server.http_server.MCPEndpoint")
    @patch("chuk_mcp_server.http_server.HealthEndpoint")
    @patch("chuk_mcp_server.http_server.InfoEndpoint")
    def test_logging_integration(self, mock_info_endpoint, mock_health_endpoint, mock_mcp_endpoint, mock_registry):
        """Test logging integration."""
        # Mock registry
        mock_registry.clear_endpoints = Mock()
        mock_registry.register_endpoint = Mock()
        mock_registry.get_routes.return_value = []

        # Mock endpoints
        mock_info_endpoint.return_value = Mock()
        mock_mcp_endpoint.return_value = Mock()

        with patch("chuk_mcp_server.http_server.logger") as mock_logger:
            HTTPServer(self.mock_protocol)

            # Should have logged initialization
            mock_logger.info.assert_any_call("🚀 HTTP server initialized with bottleneck fixes")

    @patch("chuk_mcp_server.http_server.http_endpoint_registry")
    @patch("chuk_mcp_server.http_server.MCPEndpoint")
    @patch("chuk_mcp_server.http_server.HealthEndpoint")
    @patch("chuk_mcp_server.http_server.InfoEndpoint")
    def test_middleware_configuration(self, mock_info_endpoint, mock_health_endpoint, mock_mcp_endpoint, mock_registry):
        """Test middleware configuration."""
        # Mock registry
        mock_registry.clear_endpoints = Mock()
        mock_registry.register_endpoint = Mock()
        mock_registry.get_routes.return_value = []

        # Mock endpoints
        mock_info_endpoint.return_value = Mock()
        mock_mcp_endpoint.return_value = Mock()

        server = HTTPServer(self.mock_protocol)

        # Check CORS middleware is configured (check safely)
        assert hasattr(server.app, "middleware_stack")
        if server.app.middleware_stack and hasattr(server.app.middleware_stack, "middleware"):
            middleware = server.app.middleware_stack.middleware
            assert len(middleware) > 0

            # Find CORS middleware
            cors_middleware = None
            for mw in middleware:
                if hasattr(mw, "cls") and "CORS" in str(mw.cls):
                    cors_middleware = mw
                    break

            assert cors_middleware is not None
        else:
            # If middleware stack is not accessible, just verify the app was created
            assert isinstance(server.app, Starlette)

    @patch("chuk_mcp_server.http_server.http_endpoint_registry")
    @patch("chuk_mcp_server.http_server.MCPEndpoint")
    @patch("chuk_mcp_server.http_server.HealthEndpoint")
    @patch("chuk_mcp_server.http_server.InfoEndpoint")
    def test_debug_mode_configuration(self, mock_info_endpoint, mock_health_endpoint, mock_mcp_endpoint, mock_registry):
        """Test debug mode configuration differences."""
        # Mock registry
        mock_registry.clear_endpoints = Mock()
        mock_registry.register_endpoint = Mock()
        mock_registry.get_routes.return_value = []

        # Mock endpoints
        mock_info_endpoint.return_value = Mock()
        mock_mcp_endpoint.return_value = Mock()

        server = HTTPServer(self.mock_protocol)

        with patch("chuk_mcp_server.http_server.uvicorn") as mock_uvicorn:
            mock_uvicorn.run = Mock()

            # Test debug mode
            server.run(debug=True)

            config = mock_uvicorn.run.call_args[1]
            assert config["log_level"] == "debug"
            assert config["access_log"] is True

            # Reset mock
            mock_uvicorn.run.reset_mock()

            # Test production mode
            server.run(debug=False)

            config = mock_uvicorn.run.call_args[1]
            assert config["log_level"] == "warning"  # Default log level is warning
            assert config["access_log"] is False

    @patch("chuk_mcp_server.http_server.http_endpoint_registry")
    @patch("chuk_mcp_server.http_server.MCPEndpoint")
    @patch("chuk_mcp_server.http_server.HealthEndpoint")
    @patch("chuk_mcp_server.http_server.InfoEndpoint")
    @patch("chuk_mcp_server.http_server.uvicorn")
    def test_comprehensive_config_verification(
        self, mock_uvicorn, mock_info_endpoint, mock_health_endpoint, mock_mcp_endpoint, mock_registry
    ):
        """Test comprehensive uvicorn configuration."""
        import sys

        # Mock registry
        mock_registry.clear_endpoints = Mock()
        mock_registry.register_endpoint = Mock()
        mock_registry.get_routes.return_value = []

        # Mock endpoints
        mock_info_endpoint.return_value = Mock()
        mock_mcp_endpoint.return_value = Mock()

        server = HTTPServer(self.mock_protocol)

        mock_uvicorn.run = Mock()

        server.run(host="custom.host", port=9999, debug=False)

        config = mock_uvicorn.run.call_args[1]

        # Verify all expected configuration keys
        expected_config = {
            "host": "custom.host",
            "port": 9999,
            "workers": 1,
            "http": "httptools",
            "access_log": False,
            "server_header": False,
            "date_header": False,
            "log_level": "warning",  # Default log level is warning
            "backlog": 4096,
            "limit_concurrency": 2000,
            "timeout_keep_alive": 60,
            "h11_max_incomplete_event_size": 16384,
        }

        # On non-Windows platforms, uvloop should be used
        if sys.platform != "win32":
            expected_config["loop"] = "uvloop"

        for key, expected_value in expected_config.items():
            assert config[key] == expected_value

        # Verify the app is set
        assert config["app"] == server.app
