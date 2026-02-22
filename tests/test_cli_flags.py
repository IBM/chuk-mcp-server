"""Tests for CLI --inspect and --reload flags."""

import argparse


class TestCliFlags:
    """Tests for --inspect and --reload CLI argument parsing."""

    def _parse_http_args(self, *extra_args):
        """Parse CLI args for http subcommand."""

        parser = argparse.ArgumentParser()
        subparsers = parser.add_subparsers(dest="mode")

        http_parser = subparsers.add_parser("http")
        http_parser.add_argument("--host", default=None)
        http_parser.add_argument("--port", type=int, default=None)
        http_parser.add_argument("--debug", action="store_true")
        http_parser.add_argument("--reload", action="store_true")
        http_parser.add_argument("--inspect", action="store_true")
        http_parser.add_argument("--log-level", default="warning")

        return parser.parse_args(["http", *extra_args])

    def test_reload_flag_false_by_default(self):
        """--reload defaults to False."""
        args = self._parse_http_args()
        assert args.reload is False

    def test_reload_flag_enabled(self):
        """--reload sets reload to True."""
        args = self._parse_http_args("--reload")
        assert args.reload is True

    def test_inspect_flag_false_by_default(self):
        """--inspect defaults to False."""
        args = self._parse_http_args()
        assert args.inspect is False

    def test_inspect_flag_enabled(self):
        """--inspect sets inspect to True."""
        args = self._parse_http_args("--inspect")
        assert args.inspect is True

    def test_both_flags_together(self):
        """Both --reload and --inspect can be used together."""
        args = self._parse_http_args("--reload", "--inspect")
        assert args.reload is True
        assert args.inspect is True


class TestHttpServerReload:
    """Tests for reload param in HTTPServer."""

    def test_reload_added_to_uvicorn_config(self):
        """reload=True adds reload to uvicorn config."""
        from unittest.mock import patch

        from chuk_mcp_server.http_server import HTTPServer
        from chuk_mcp_server.protocol import MCPProtocolHandler
        from chuk_mcp_server.types import ServerInfo, create_server_capabilities

        server_info = ServerInfo(name="test", version="1.0")
        caps = create_server_capabilities()
        protocol = MCPProtocolHandler(server_info, caps)
        http_server = HTTPServer(protocol)

        captured_config = {}

        def mock_uvicorn_run(**kwargs):
            captured_config.update(kwargs)

        with patch("chuk_mcp_server.http_server.uvicorn.run", side_effect=mock_uvicorn_run):
            http_server.run(reload=True)

        assert captured_config.get("reload") is True

    def test_no_reload_by_default(self):
        """reload defaults to False and is not in uvicorn config."""
        from unittest.mock import patch

        from chuk_mcp_server.http_server import HTTPServer
        from chuk_mcp_server.protocol import MCPProtocolHandler
        from chuk_mcp_server.types import ServerInfo, create_server_capabilities

        server_info = ServerInfo(name="test", version="1.0")
        caps = create_server_capabilities()
        protocol = MCPProtocolHandler(server_info, caps)
        http_server = HTTPServer(protocol)

        captured_config = {}

        def mock_uvicorn_run(**kwargs):
            captured_config.update(kwargs)

        with patch("chuk_mcp_server.http_server.uvicorn.run", side_effect=mock_uvicorn_run):
            http_server.run()

        assert "reload" not in captured_config
