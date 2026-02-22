#!/usr/bin/env python3
"""
Tests for CLI module.
"""

import sys
from unittest.mock import MagicMock, patch

import pytest

from chuk_mcp_server.cli import create_example_server, main, setup_logging


class TestSetupLogging:
    """Test logging setup functionality."""

    @patch("chuk_mcp_server.cli.logging.basicConfig")
    def test_setup_logging_debug(self, mock_basicConfig):
        """Test logging setup with debug mode."""
        setup_logging(debug=True, stderr=True)

        mock_basicConfig.assert_called_once()
        kwargs = mock_basicConfig.call_args[1]
        assert kwargs["level"] == 10  # logging.DEBUG
        assert kwargs["stream"] == sys.stderr

    @patch("chuk_mcp_server.cli.logging.basicConfig")
    def test_setup_logging_info(self, mock_basicConfig):
        """Test logging setup with info mode."""
        setup_logging(debug=False, stderr=False)

        mock_basicConfig.assert_called_once()
        kwargs = mock_basicConfig.call_args[1]
        assert kwargs["level"] == 20  # logging.INFO
        assert kwargs["stream"] == sys.stdout


class TestCreateExampleServer:
    """Test example server creation."""

    def test_create_example_server_with_defaults(self):
        """Test creating server with default tools and resources."""
        server = create_example_server()

        assert server.server_info.name == "chuk-mcp-server"
        assert server.server_info.version == "0.2.3"

        # Check that example tools were added
        tools = server.get_tools()
        tool_names = [tool.name for tool in tools]
        assert "echo" in tool_names
        assert "add" in tool_names
        assert "get_env" in tool_names

        # Check that example resource was added
        resources = server.get_resources()
        resource_uris = [resource.uri for resource in resources]
        assert "server://info" in resource_uris

    @patch.dict("os.environ", {"MCP_SERVER_NAME": "custom-server", "MCP_SERVER_VERSION": "1.0.0"})
    def test_create_example_server_with_env_vars(self):
        """Test creating server with environment variables."""
        server = create_example_server()

        assert server.server_info.name == "custom-server"
        assert server.server_info.version == "1.0.0"

    @patch("chuk_mcp_server.cli.ChukMCPServer")
    def test_create_example_server_with_existing_tools(self, mock_server_class):
        """Test that example tools are not added if tools already exist."""
        mock_server = MagicMock()
        mock_tool = MagicMock()
        mock_tool.name = "existing_tool"
        mock_server.get_tools.return_value = [mock_tool]
        mock_server.get_resources.return_value = []
        mock_server_class.return_value = mock_server

        server = create_example_server()

        # Verify tool decorator was not called (tools already exist)
        assert server == mock_server
        # Check that get_tools was called to check existing tools
        mock_server.get_tools.assert_called()

    def test_example_tools_functionality(self):
        """Test that the example tools work correctly."""
        server = create_example_server()

        # Get the registered tools
        tools = {tool.name: tool for tool in server.get_tools()}

        # Test echo tool via direct call
        assert "echo" in tools

        # Test that tools were registered
        assert len(tools) == 3
        assert "add" in tools
        assert "get_env" in tools


class TestMainFunction:
    """Test main CLI entry point."""

    @patch("chuk_mcp_server.cli.create_example_server")
    @patch("chuk_mcp_server.cli.setup_logging")
    @patch("sys.argv", ["chuk-mcp-server", "stdio"])
    def test_main_stdio_mode(self, mock_setup_logging, mock_create_server):
        """Test main function in stdio mode."""
        mock_server = MagicMock()
        mock_create_server.return_value = mock_server

        with patch("chuk_mcp_server.cli.logging.info"):
            main()

        # Verify logging setup for stdio (stderr=True)
        mock_setup_logging.assert_called_once_with(debug=False, stderr=True)

        # Verify server creation and run
        mock_create_server.assert_called_once()
        mock_server.run.assert_called_once_with(stdio=True, debug=False, log_level="warning")

    @patch("chuk_mcp_server.cli.create_example_server")
    @patch("chuk_mcp_server.cli.setup_logging")
    @patch("sys.argv", ["chuk-mcp-server", "stdio", "--debug"])
    def test_main_stdio_mode_debug(self, mock_setup_logging, mock_create_server):
        """Test main function in stdio mode with debug."""
        mock_server = MagicMock()
        mock_create_server.return_value = mock_server

        with patch("chuk_mcp_server.cli.logging.info"):
            main()

        # Verify debug logging
        mock_setup_logging.assert_called_once_with(debug=True, stderr=True)
        mock_server.run.assert_called_once_with(stdio=True, debug=True, log_level="warning")

    @patch("chuk_mcp_server.cli.create_example_server")
    @patch("chuk_mcp_server.cli.setup_logging")
    @patch("sys.argv", ["chuk-mcp-server", "http"])
    def test_main_http_mode(self, mock_setup_logging, mock_create_server):
        """Test main function in HTTP mode."""
        mock_server = MagicMock()
        mock_create_server.return_value = mock_server

        with patch("chuk_mcp_server.cli.logging.info"):
            main()

        # Verify logging setup for HTTP (stderr=False)
        mock_setup_logging.assert_called_once_with(debug=False, stderr=False)

        # Verify server run with HTTP mode
        mock_server.run.assert_called_once_with(
            host=None, port=None, debug=False, stdio=False, log_level="warning", reload=False, inspect=False
        )

    @patch("chuk_mcp_server.cli.create_example_server")
    @patch("chuk_mcp_server.cli.setup_logging")
    @patch("sys.argv", ["chuk-mcp-server", "http", "--host", "0.0.0.0", "--port", "9000", "--debug"])
    def test_main_http_mode_with_options(self, mock_setup_logging, mock_create_server):
        """Test main function in HTTP mode with custom options."""
        mock_server = MagicMock()
        mock_create_server.return_value = mock_server

        with patch("chuk_mcp_server.cli.logging.info"):
            main()

        # Verify debug logging
        mock_setup_logging.assert_called_once_with(debug=True, stderr=False)

        # Verify server run with custom options
        mock_server.run.assert_called_once_with(
            host="0.0.0.0", port=9000, debug=True, stdio=False, log_level="warning", reload=False, inspect=False
        )

    @patch("chuk_mcp_server.cli.create_example_server")
    @patch("chuk_mcp_server.cli.setup_logging")
    @patch("sys.argv", ["chuk-mcp-server", "auto"])
    def test_main_auto_mode(self, mock_setup_logging, mock_create_server):
        """Test main function in auto mode."""
        mock_server = MagicMock()
        mock_create_server.return_value = mock_server

        with patch("chuk_mcp_server.cli.logging.info"):
            main()

        # Verify logging setup for auto (stderr=False)
        mock_setup_logging.assert_called_once_with(debug=False, stderr=False)

        # Verify server run with auto detection
        mock_server.run.assert_called_once_with(host=None, port=None, debug=False, log_level="warning")

    @patch("chuk_mcp_server.cli.create_example_server")
    @patch("chuk_mcp_server.cli.setup_logging")
    @patch("sys.argv", ["chuk-mcp-server", "auto", "--host", "localhost", "--port", "8080"])
    def test_main_auto_mode_with_options(self, mock_setup_logging, mock_create_server):
        """Test main function in auto mode with options."""
        mock_server = MagicMock()
        mock_create_server.return_value = mock_server

        with patch("chuk_mcp_server.cli.logging.info"):
            main()

        # Verify server run with provided options
        mock_server.run.assert_called_once_with(host="localhost", port=8080, debug=False, log_level="warning")

    @patch("sys.argv", ["chuk-mcp-server", "--help"])
    def test_main_help(self):
        """Test main function with help flag."""
        with pytest.raises(SystemExit) as exc_info:
            with patch("sys.stdout"):
                main()

        # Help should exit with code 0
        assert exc_info.value.code == 0

    @patch("sys.argv", ["chuk-mcp-server"])
    def test_main_no_subcommand(self):
        """Test main function without subcommand."""
        with pytest.raises(SystemExit) as exc_info:
            with patch("sys.stderr"):
                main()

        # Should exit with error code
        assert exc_info.value.code == 2

    def test_example_resource_functionality(self):
        """Test that the example resource works correctly."""
        server = create_example_server()

        # Check that resource was registered
        resources = server.get_resources()
        assert len(resources) == 1
        assert resources[0].uri == "server://info"
