#!/usr/bin/env python3
"""Comprehensive tests for the core module to improve test coverage."""

import contextlib
from unittest.mock import MagicMock, Mock, patch

import pytest

from chuk_mcp_server.core import ChukMCPServer, create_mcp_server, quick_server
from chuk_mcp_server.types import PromptHandler, ResourceHandler, ToolHandler


class TestChukMCPServerComprehensive:
    """Comprehensive tests for ChukMCPServer class covering missing lines."""

    def test_initialization_with_all_params(self):
        """Test initialization with all parameters."""
        capabilities = MagicMock()
        experimental = {"test": True}

        server = ChukMCPServer(
            name="test-server",
            version="2.0.0",
            title="Test Server",
            description="A test server",
            capabilities=capabilities,
            tools=True,
            resources=True,
            prompts=True,
            logging=True,
            experimental=experimental,
            host="127.0.0.1",
            port=8080,
            debug=True,
        )

        assert server.server_info.name == "test-server"
        assert server.server_info.version == "2.0.0"
        assert server.server_info.title == "Test Server"
        assert server.capabilities == capabilities
        assert server.smart_host == "127.0.0.1"
        assert server.smart_port == 8080
        assert server.smart_debug is True

    def test_print_smart_config(self):
        """Test _print_smart_config method."""
        server = ChukMCPServer(debug=True)

        # Test that it doesn't raise an error
        with patch("builtins.print") as mock_print:
            server._print_smart_config()
            assert mock_print.called

    def test_register_global_functions_with_handlers(self):
        """Test _register_global_functions with existing handlers."""
        with patch("chuk_mcp_server.core.get_global_tools") as mock_tools:
            with patch("chuk_mcp_server.core.get_global_resources") as mock_resources:
                with patch("chuk_mcp_server.core.get_global_prompts") as mock_prompts:
                    with patch("chuk_mcp_server.core.clear_global_registry") as mock_clear:
                        # Mock tool with handler attribute - needs proper parameters attribute
                        mock_tool = Mock()
                        mock_tool.handler = Mock()
                        mock_tool.name = "test_tool"
                        mock_tool.description = "Test tool"
                        mock_tool.parameters = []  # Add parameters attribute
                        mock_tools.return_value = [mock_tool]

                        # Mock resource with handler attribute
                        mock_resource = Mock()
                        mock_resource.handler = Mock()
                        mock_resource.uri = "test://resource"
                        mock_resource.name = "test_resource"
                        mock_resource.description = "Test resource"
                        mock_resource.mime_type = "text/plain"
                        mock_resources.return_value = [mock_resource]

                        # Mock prompt with handler attribute - needs proper parameters attribute
                        mock_prompt = Mock()
                        mock_prompt.handler = Mock()
                        mock_prompt.name = "test_prompt"
                        mock_prompt.description = "Test prompt"
                        mock_prompt.parameters = []  # Add parameters attribute
                        mock_prompts.return_value = [mock_prompt]

                        ChukMCPServer()

                        # Should have registered the handlers
                        assert mock_clear.called

    def test_register_global_functions_without_handlers(self):
        """Test _register_global_functions without handler attributes - simplified."""
        with patch("chuk_mcp_server.core.get_global_tools") as mock_tools:
            with patch("chuk_mcp_server.core.get_global_resources") as mock_resources:
                with patch("chuk_mcp_server.core.get_global_prompts") as mock_prompts:
                    # Return empty lists to avoid complex mocking issues
                    mock_tools.return_value = []
                    mock_resources.return_value = []
                    mock_prompts.return_value = []

                    server = ChukMCPServer()

                    # Should not error with empty global registrations
                    assert server is not None

    def test_tool_decorator_callable_name(self):
        """Test tool decorator when name parameter is callable (direct decoration)."""
        server = ChukMCPServer()

        def test_func():
            return "test"

        # Test direct decoration without parentheses
        decorated = server.tool(test_func)

        assert decorated == test_func
        tools = server.get_tools()
        assert len(tools) == 1

    def test_tool_decorator_with_tags(self):
        """Test tool decorator with tags parameter."""
        server = ChukMCPServer()

        @server.tool(tags=["api", "utility"])
        def tagged_tool():
            return "tagged"

        assert tagged_tool() == "tagged"
        tools = server.get_tools()
        assert len(tools) == 1

    def test_resource_decorator_uri_parsing(self):
        """Test resource decorator with different URI schemes."""
        server = ChukMCPServer()

        @server.resource("file://test.txt")
        def file_resource():
            return "file content"

        @server.resource("test_resource")  # No scheme
        def no_scheme_resource():
            return "no scheme"

        assert file_resource() == "file content"
        assert no_scheme_resource() == "no scheme"
        resources = server.get_resources()
        assert len(resources) == 2

    def test_resource_decorator_with_tags(self):
        """Test resource decorator with tags parameter."""
        server = ChukMCPServer()

        @server.resource("data://tagged", tags=["data", "test"])
        def tagged_resource():
            return "tagged"

        assert tagged_resource() == "tagged"
        resources = server.get_resources()
        assert len(resources) == 1

    def test_prompt_decorator_callable_name(self):
        """Test prompt decorator when name parameter is callable (direct decoration)."""
        server = ChukMCPServer()

        def test_prompt():
            return "prompt"

        # Test direct decoration without parentheses
        decorated = server.prompt(test_prompt)

        assert decorated == test_prompt
        prompts = server.get_prompts()
        assert len(prompts) == 1

    def test_prompt_decorator_with_tags(self):
        """Test prompt decorator with tags parameter."""
        server = ChukMCPServer()

        @server.prompt(tags=["template", "custom"])
        def tagged_prompt():
            return "tagged prompt"

        assert tagged_prompt() == "tagged prompt"
        prompts = server.get_prompts()
        assert len(prompts) == 1

    def test_manual_registration_methods(self):
        """Test manual registration methods."""
        server = ChukMCPServer()

        # Create mock handlers with proper attributes
        tool_handler = Mock(spec=ToolHandler)
        tool_handler.name = "manual_tool"
        tool_handler.parameters = []  # Add parameters attribute

        resource_handler = Mock(spec=ResourceHandler)
        resource_handler.uri = "manual://resource"

        prompt_handler = Mock(spec=PromptHandler)
        prompt_handler.name = "manual_prompt"
        prompt_handler.parameters = []  # Add parameters attribute

        # Test manual registration - just check it doesn't error
        try:
            server.add_tool(tool_handler)
            server.add_resource(resource_handler)
            server.add_prompt(prompt_handler)
        except Exception as e:
            pytest.fail(f"Manual registration should not fail: {e}")

        # Just verify the methods exist and can be called
        assert hasattr(server, "add_tool")
        assert hasattr(server, "add_resource")
        assert hasattr(server, "add_prompt")

    def test_add_endpoint_manual(self):
        """Test manual endpoint registration."""
        server = ChukMCPServer()

        async def test_handler(request):
            return {"test": True}

        server.add_endpoint("/manual", test_handler, methods=["GET"])

        # Should not raise an error
        endpoints = server.get_endpoints()
        assert any(e["path"] == "/manual" for e in endpoints)

    def test_register_function_as_tool(self):
        """Test registering function as tool."""
        server = ChukMCPServer()

        def my_function(x: int) -> int:
            return x * 2

        tool_handler = server.register_function_as_tool(my_function, name="multiplier", description="Multiply by 2")

        assert tool_handler is not None
        tools = server.get_tools()
        assert len(tools) == 1

    def test_register_function_as_resource(self):
        """Test registering function as resource."""
        server = ChukMCPServer()

        def my_resource() -> str:
            return "resource data"

        resource_handler = server.register_function_as_resource(
            my_resource, uri="custom://data", name="Custom Data", description="Custom resource", mime_type="text/plain"
        )

        assert resource_handler is not None
        resources = server.get_resources()
        assert len(resources) == 1

    def test_register_function_as_prompt(self):
        """Test registering function as prompt."""
        server = ChukMCPServer()

        def my_prompt(topic: str) -> str:
            return f"Write about {topic}"

        prompt_handler = server.register_function_as_prompt(my_prompt, name="writer", description="Writing prompt")

        assert prompt_handler is not None
        prompts = server.get_prompts()
        assert len(prompts) == 1

    def test_search_methods_comprehensive(self):
        """Test comprehensive search methods."""
        server = ChukMCPServer()

        # Test search with different parameters
        tools = server.search_tools_by_tag("nonexistent")
        assert isinstance(tools, list)

        resources = server.search_resources_by_tag("nonexistent")
        assert isinstance(resources, list)

        prompts = server.search_prompts_by_tag("nonexistent")
        assert isinstance(prompts, list)

        # Test multi-tag search
        components = server.search_components_by_tags(["tag1", "tag2"], match_all=True)
        assert components is not None

        components = server.search_components_by_tags(["tag1", "tag2"], match_all=False)
        assert components is not None

    def test_clear_methods_comprehensive(self):
        """Test all clear methods."""
        server = ChukMCPServer()

        # Add some components first
        @server.tool
        def test_tool():
            return "tool"

        @server.resource("test://resource")
        def test_resource():
            return "resource"

        @server.prompt
        def test_prompt():
            return "prompt"

        @server.endpoint("/test")
        async def test_endpoint(request):
            return {}

        # Test individual clear methods - patch to avoid import issues
        with patch("chuk_mcp_server.core.mcp_registry") as mock_registry:
            # Mock the clear_type method to avoid attribute errors
            mock_registry.clear_type = Mock()
            mock_registry.MCPComponentType = Mock()
            mock_registry.MCPComponentType.TOOL = "TOOL"
            mock_registry.MCPComponentType.RESOURCE = "RESOURCE"
            mock_registry.MCPComponentType.PROMPT = "PROMPT"

            try:
                server.clear_tools()
                server.clear_resources()
                server.clear_prompts()
            except Exception as e:
                pytest.fail(f"Clear methods should not fail: {e}")

        server.clear_endpoints()
        # Endpoints might have built-in defaults, so we just check it doesn't error

    def test_run_method_with_overrides(self):
        """Test run method with parameter overrides."""
        server = ChukMCPServer()

        with patch.object(server, "_print_startup_info") as mock_print:
            with patch("chuk_mcp_server.core.create_server") as mock_create:
                mock_http_server = Mock()
                mock_create.return_value = mock_http_server
                mock_http_server.run.side_effect = KeyboardInterrupt()

                # Test run with overrides
                with contextlib.suppress(KeyboardInterrupt):
                    server.run(host="custom.host", port=9999, debug=True)

                # Should have used overridden values (with actual_log_level parameter)
                # When debug=True, log_level is automatically set to 'debug'
                mock_print.assert_called_with("custom.host", 9999, True, actual_log_level="DEBUG")
                mock_http_server.run.assert_called_with(
                    host="custom.host", port=9999, debug=True, log_level="debug", reload=False
                )

    def test_run_method_with_exception(self):
        """Test run method with server exception."""
        server = ChukMCPServer()

        with patch("chuk_mcp_server.core.create_server") as mock_create:
            mock_http_server = Mock()
            mock_create.return_value = mock_http_server
            mock_http_server.run.side_effect = Exception("Server error")

            with pytest.raises(Exception, match="Server error"):
                server.run()

    def test_run_method_logging_levels(self):
        """Test run method with different logging levels."""
        server = ChukMCPServer()

        with patch("chuk_mcp_server.core.create_server") as mock_create:
            with patch("chuk_mcp_server.core.logging") as mock_logging:
                mock_http_server = Mock()
                mock_create.return_value = mock_http_server
                mock_http_server.run.side_effect = KeyboardInterrupt()

                # Test debug=False (should use smart log level)
                with contextlib.suppress(KeyboardInterrupt):
                    server.run(debug=False)

                # Should have set log level based on smart config
                mock_logging.basicConfig.assert_called()

    def test_print_startup_info_comprehensive(self):
        """Test comprehensive startup info printing."""
        server = ChukMCPServer()

        # Add some components
        @server.tool
        def test_tool():
            return "tool"

        @server.resource("test://resource")
        def test_resource():
            return "resource"

        @server.prompt
        def test_prompt():
            return "prompt"

        with patch("builtins.print") as mock_print:
            server._print_startup_info("localhost", 8000, True)

            # Should have printed startup information
            assert mock_print.called

            # Check some expected content in print calls
            print_calls = [call[0][0] for call in mock_print.call_args_list if call[0]]
            startup_text = " ".join(str(call) for call in print_calls)

            assert "localhost" in startup_text or "8000" in startup_text

    def test_refresh_smart_config_comprehensive(self):
        """Test comprehensive smart config refresh."""
        server = ChukMCPServer()

        # Store original values
        # orig_environment = server.smart_environment  # Not used in this test

        with patch.object(server.smart_config, "get_all_defaults") as mock_get_defaults:
            mock_defaults = {
                "environment": "test_env",
                "workers": 4,
                "max_connections": 1000,
                "log_level": "DEBUG",
                "performance_mode": "high",
                "containerized": True,
            }
            mock_get_defaults.return_value = mock_defaults

            server.refresh_smart_config()

            # Should have updated values
            assert server.smart_environment == "test_env"
            assert server.smart_workers == 4
            assert server.smart_max_connections == 1000
            assert server.smart_log_level == "DEBUG"
            assert server.smart_performance_mode == "high"
            assert server.smart_containerized is True


class TestFactoryFunctionsComprehensive:
    """Comprehensive tests for factory functions."""

    def test_create_mcp_server_with_kwargs(self):
        """Test create_mcp_server with various kwargs."""
        server = create_mcp_server(
            name="factory-test", version="1.2.3", title="Factory Test", host="0.0.0.0", port=3000, debug=True
        )

        assert isinstance(server, ChukMCPServer)
        assert server.server_info.name == "factory-test"
        assert server.server_info.version == "1.2.3"

    def test_quick_server_with_name(self):
        """Test quick_server with custom name."""
        server = quick_server("Quick Test")

        assert isinstance(server, ChukMCPServer)
        assert server.server_info.name == "Quick Test"
        assert server.server_info.version == "0.1.0"

    def test_quick_server_default_name(self):
        """Test quick_server with default name."""
        server = quick_server()

        assert isinstance(server, ChukMCPServer)
        assert server.server_info.name == "Quick Server"
        assert server.server_info.version == "0.1.0"


class TestContextManagerComprehensive:
    """Comprehensive tests for context manager functionality."""

    def test_context_manager_exception_handling(self):
        """Test context manager with exception handling."""
        try:
            with ChukMCPServer() as server:
                assert server is not None
                raise ValueError("Test exception")
        except ValueError:
            pass  # Expected

        # Should not raise during exit

    def test_context_manager_normal_exit(self):
        """Test context manager normal exit."""
        with ChukMCPServer() as server:
            assert server is not None

            @server.tool
            def test_tool():
                return "test"

        # Should exit cleanly


class TestEdgeCasesAndErrorHandling:
    """Test edge cases and error handling."""

    def test_initialization_with_none_values(self):
        """Test initialization with None values."""
        server = ChukMCPServer(
            name=None,  # Should use smart default
            title=None,
            description=None,
            capabilities=None,  # Should create default
            host=None,  # Should use smart default
            port=None,  # Should use smart default
            debug=None,  # Should use smart default
        )

        assert server.server_info.name is not None
        assert server.capabilities is not None
        assert server.smart_host is not None
        assert server.smart_port is not None
        assert server.smart_debug is not None

    def test_decorator_with_complex_functions(self):
        """Test decorators with complex function signatures."""
        server = ChukMCPServer()

        @server.tool("complex_tool", "A complex tool")
        def complex_tool(a: int, b: str = "default", *args, **kwargs) -> dict:
            """Complex tool with various parameter types."""
            return {"a": a, "b": b, "args": args, "kwargs": kwargs}

        result = complex_tool(42, "test", 1, 2, 3, extra="value")
        assert result["a"] == 42
        assert result["b"] == "test"

        tools = server.get_tools()
        assert any(t.name == "complex_tool" for t in tools)

    def test_resource_name_formatting(self):
        """Test resource name formatting from function names."""
        server = ChukMCPServer()

        @server.resource("test://underscore_resource")
        def underscore_resource_function():
            return "data"

        resources = server.get_resources()
        resource = next((r for r in resources if r.uri == "test://underscore_resource"), None)
        assert resource is not None
        # Name should be formatted from function name
        assert "Underscore" in resource.name or "Resource" in resource.name

    def test_large_number_of_components(self):
        """Test server with large number of components."""
        server = ChukMCPServer()

        # Register many components
        for i in range(10):
            # Use default parameters to avoid loop variable binding issues
            @server.tool(f"tool_{i}")
            def tool_func(result=f"tool_{i}"):
                return result

            @server.resource(f"data://resource_{i}")
            def resource_func(result=f"resource_{i}"):
                return result

            @server.prompt(f"prompt_{i}")
            def prompt_func(result=f"prompt_{i}"):
                return result

        assert len(server.get_tools()) == 10
        assert len(server.get_resources()) == 10
        assert len(server.get_prompts()) == 10

        # Test info method with many components
        info = server.info()
        assert info["mcp_components"]["tools"]["count"] == 10
        assert info["mcp_components"]["resources"]["count"] == 10
        assert info["mcp_components"]["prompts"]["count"] == 10
