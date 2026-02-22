#!/usr/bin/env python3
"""
Additional tests for protocol.py to achieve 90%+ coverage.
Focuses on error paths, OAuth flows, and edge cases.
"""

from unittest.mock import AsyncMock, Mock

import pytest

from chuk_mcp_server.protocol import MCPProtocolHandler
from chuk_mcp_server.types import PromptHandler, ResourceHandler, ServerCapabilities, ServerInfo, ToolHandler


@pytest.fixture
def handler():
    """Create a protocol handler for testing."""
    server_info = ServerInfo(name="test-server", version="1.0.0")
    capabilities = ServerCapabilities(tools={"listChanged": True})
    return MCPProtocolHandler(server_info, capabilities)


@pytest.fixture
def handler_with_oauth():
    """Create a protocol handler with OAuth support."""
    server_info = ServerInfo(name="test-server", version="1.0.0")
    capabilities = ServerCapabilities(tools={"listChanged": True})

    # Mock OAuth provider
    oauth_provider = Mock()
    oauth_provider.validate_access_token = AsyncMock(
        return_value={"user_id": "user123", "external_access_token": "ext_token"}
    )

    def get_oauth_provider():
        return oauth_provider

    return MCPProtocolHandler(server_info, capabilities, oauth_provider_getter=get_oauth_provider)


class TestHandleRequestErrors:
    """Test error handling in handle_request."""

    @pytest.mark.asyncio
    async def test_handle_request_generic_exception(self, handler):
        """Test handle_request with generic exception (lines 203-205)."""
        # Create a request that will cause an exception during processing
        # We'll force an exception by making the handler's method fail
        request = {"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {}}

        # Mock _handle_initialize to raise an exception
        async def raise_exception(*args, **kwargs):
            raise RuntimeError("Test exception")

        handler._handle_initialize = raise_exception

        response, session_id = await handler.handle_request(request)

        assert response["jsonrpc"] == "2.0"
        assert response["id"] == 1
        assert "error" in response
        assert response["error"]["code"] == -32603
        assert response["error"]["message"] == "Internal server error"
        assert session_id is None


class TestOAuthToolHandling:
    """Test OAuth-required tool handling (lines 256-299)."""

    @pytest.mark.asyncio
    async def test_tool_requires_auth_no_token(self, handler_with_oauth):
        """Test tool requiring OAuth without token (lines 258-261)."""

        # Register a tool that requires auth
        def oauth_tool(name: str) -> str:
            return f"Hello, {name}!"

        oauth_tool._requires_auth = True
        oauth_tool._auth_scopes = ["read"]

        tool = ToolHandler.from_function(oauth_tool)
        handler_with_oauth.register_tool(tool)

        # Call without OAuth token (oauth_token=None)
        request = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "tools/call",
            "params": {"name": "oauth_tool", "arguments": {"name": "World"}},
        }

        response, _ = await handler_with_oauth.handle_request(request, oauth_token=None)

        assert "error" in response
        assert response["error"]["code"] == -32603
        assert "requires OAuth authorization" in response["error"]["message"]
        assert "authenticate first" in response["error"]["message"]

    @pytest.mark.asyncio
    async def test_tool_requires_auth_no_oauth_configured(self, handler):
        """Test tool requiring OAuth when OAuth not configured (lines 263-266)."""

        # Register a tool that requires auth
        def oauth_tool(name: str) -> str:
            return f"Hello, {name}!"

        oauth_tool._requires_auth = True

        tool = ToolHandler.from_function(oauth_tool)
        handler.register_tool(tool)

        # Call with fake token but no OAuth provider (passed as parameter)
        request = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "tools/call",
            "params": {"name": "oauth_tool", "arguments": {"name": "World"}},
        }

        response, _ = await handler.handle_request(request, oauth_token="fake_token")

        assert "error" in response
        assert response["error"]["code"] == -32603
        assert "OAuth is not configured" in response["error"]["message"]

    @pytest.mark.asyncio
    async def test_tool_requires_auth_provider_not_available(self, handler_with_oauth):
        """Test tool requiring OAuth when provider returns None (lines 271-272)."""

        # Register a tool that requires auth
        def oauth_tool(name: str) -> str:
            return f"Hello, {name}!"

        oauth_tool._requires_auth = True

        tool = ToolHandler.from_function(oauth_tool)
        handler_with_oauth.register_tool(tool)

        # Make oauth_provider_getter return None
        handler_with_oauth.oauth_provider_getter = lambda: None

        request = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "tools/call",
            "params": {"name": "oauth_tool", "arguments": {"name": "World"}},
        }

        response, _ = await handler_with_oauth.handle_request(request, oauth_token="valid_token")

        assert "error" in response
        assert response["error"]["code"] == -32603
        assert "OAuth provider not available" in response["error"]["message"]

    @pytest.mark.asyncio
    async def test_tool_requires_auth_no_external_token(self, handler_with_oauth):
        """Test tool requiring OAuth when external token missing (lines 278-281)."""

        # Register a tool that requires auth
        def oauth_tool(name: str) -> str:
            return f"Hello, {name}!"

        oauth_tool._requires_auth = True

        tool = ToolHandler.from_function(oauth_tool)
        handler_with_oauth.register_tool(tool)

        # Mock provider to return no external token
        provider = handler_with_oauth.oauth_provider_getter()
        provider.validate_access_token = AsyncMock(return_value={"user_id": "user123"})  # Missing external_access_token

        request = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "tools/call",
            "params": {"name": "oauth_tool", "arguments": {"name": "World"}},
        }

        response, _ = await handler_with_oauth.handle_request(request, oauth_token="valid_token")

        assert "error" in response
        assert response["error"]["code"] == -32603
        assert "external provider token is missing" in response["error"]["message"]

    @pytest.mark.asyncio
    async def test_tool_requires_auth_validation_fails(self, handler_with_oauth):
        """Test tool requiring OAuth when validation fails (lines 296-299)."""

        # Register a tool that requires auth
        def oauth_tool(name: str) -> str:
            return f"Hello, {name}!"

        oauth_tool._requires_auth = True

        tool = ToolHandler.from_function(oauth_tool)
        handler_with_oauth.register_tool(tool)

        # Mock provider to raise exception
        provider = handler_with_oauth.oauth_provider_getter()
        provider.validate_access_token = AsyncMock(side_effect=Exception("Invalid token"))

        request = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "tools/call",
            "params": {"name": "oauth_tool", "arguments": {"name": "World"}},
        }

        response, _ = await handler_with_oauth.handle_request(request, oauth_token="invalid_token")

        assert "error" in response
        assert response["error"]["code"] == -32603
        assert "OAuth validation failed" in response["error"]["message"]

    @pytest.mark.asyncio
    async def test_tool_requires_auth_success_with_user_id(self, handler_with_oauth):
        """Test successful OAuth tool call with user_id context (lines 285-292)."""

        # Register a tool that requires auth
        def oauth_tool(name: str, _external_access_token: str = None, _user_id: str = None) -> str:
            return f"Hello, {name}! User: {_user_id}"

        oauth_tool._requires_auth = True

        tool = ToolHandler.from_function(oauth_tool)
        handler_with_oauth.register_tool(tool)

        request = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "tools/call",
            "params": {"name": "oauth_tool", "arguments": {"name": "World"}},
        }

        response, _ = await handler_with_oauth.handle_request(request, oauth_token="valid_token")

        assert "result" in response
        assert "content" in response["result"]


class TestResourcesReadError:
    """Test resources/read error handling."""

    @pytest.mark.asyncio
    async def test_resources_read_unknown_uri(self, handler):
        """Test resources/read with unknown URI (line 331)."""
        request = {"jsonrpc": "2.0", "id": 1, "method": "resources/read", "params": {"uri": "unknown://resource"}}

        response, _ = await handler.handle_request(request)

        assert "error" in response
        assert response["error"]["code"] == -32602
        assert "Unknown resource" in response["error"]["message"]

    @pytest.mark.asyncio
    async def test_resources_read_exception(self, handler):
        """Test resources/read with exception during read (lines 345-347)."""

        # Register a resource that raises an exception
        async def failing_resource() -> str:
            raise RuntimeError("Resource read failed")

        resource = ResourceHandler.from_function(uri="test://failing", func=failing_resource)
        handler.register_resource(resource)

        request = {"jsonrpc": "2.0", "id": 1, "method": "resources/read", "params": {"uri": "test://failing"}}

        response, _ = await handler.handle_request(request)

        assert "error" in response
        assert response["error"]["code"] == -32603
        assert "Resource read error" in response["error"]["message"]


class TestPromptsGetError:
    """Test prompts/get error handling."""

    @pytest.mark.asyncio
    async def test_prompts_get_unknown_prompt(self, handler):
        """Test prompts/get with unknown prompt (line 365)."""
        request = {"jsonrpc": "2.0", "id": 1, "method": "prompts/get", "params": {"name": "unknown_prompt"}}

        response, _ = await handler.handle_request(request)

        assert "error" in response
        assert response["error"]["code"] == -32602
        assert "Unknown prompt" in response["error"]["message"]

    @pytest.mark.asyncio
    async def test_prompts_get_dict_result(self, handler):
        """Test prompts/get with dict result (lines 375-377)."""

        def dict_prompt() -> dict:
            return {"messages": [{"role": "user", "content": {"type": "text", "text": "Test"}}]}

        prompt = PromptHandler.from_function(dict_prompt)
        handler.register_prompt(prompt)

        request = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "prompts/get",
            "params": {"name": "dict_prompt", "arguments": {}},
        }

        response, _ = await handler.handle_request(request)

        assert "result" in response
        assert "messages" in response["result"]

    @pytest.mark.asyncio
    async def test_prompts_get_other_type_result(self, handler):
        """Test prompts/get with non-string/dict result (lines 378-380)."""

        def int_prompt() -> int:
            return 42

        prompt = PromptHandler.from_function(int_prompt)
        handler.register_prompt(prompt)

        request = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "prompts/get",
            "params": {"name": "int_prompt", "arguments": {}},
        }

        response, _ = await handler.handle_request(request)

        assert "result" in response
        assert "messages" in response["result"]

    @pytest.mark.asyncio
    async def test_prompts_get_dict_content_with_role(self, handler):
        """Test prompts/get with dict containing messages with role (line 390)."""

        def formatted_prompt() -> dict:
            # Return dict with messages key - triggers line 377, then line 390
            return {"messages": [{"role": "assistant", "content": {"type": "text", "text": "Already formatted"}}]}

        prompt = PromptHandler.from_function(formatted_prompt)
        handler.register_prompt(prompt)

        request = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "prompts/get",
            "params": {"name": "formatted_prompt", "arguments": {}},
        }

        response, _ = await handler.handle_request(request)

        assert "result" in response
        assert response["result"]["messages"][0]["role"] == "assistant"

    @pytest.mark.asyncio
    async def test_prompts_get_non_formatted_content(self, handler):
        """Test prompts/get with dict containing non-formatted content (lines 392-393, 396-399)."""

        def dict_with_content_prompt() -> dict:
            # Return dict with messages as list of content items (not full messages)
            # This triggers line 377, then line 392-393 for dict without role/content
            return {"messages": [{"type": "text", "text": "Content item"}]}

        prompt = PromptHandler.from_function(dict_with_content_prompt)
        handler.register_prompt(prompt)

        request = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "prompts/get",
            "params": {"name": "dict_with_content_prompt", "arguments": {}},
        }

        response, _ = await handler.handle_request(request)

        assert "result" in response
        assert "messages" in response["result"]
        # Should wrap the content dict in a message with role "user"
        assert response["result"]["messages"][0]["role"] == "user"
        assert response["result"]["messages"][0]["content"]["type"] == "text"

    @pytest.mark.asyncio
    async def test_prompts_get_exception(self, handler):
        """Test prompts/get with exception during generation (lines 414-416)."""

        def failing_prompt() -> str:
            raise RuntimeError("Prompt generation failed")

        prompt = PromptHandler.from_function(failing_prompt)
        handler.register_prompt(prompt)

        request = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "prompts/get",
            "params": {"name": "failing_prompt", "arguments": {}},
        }

        response, _ = await handler.handle_request(request)

        assert "error" in response
        assert response["error"]["code"] == -32603
        assert "Prompt generation error" in response["error"]["message"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
