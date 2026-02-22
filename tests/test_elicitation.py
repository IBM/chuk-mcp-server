#!/usr/bin/env python3
"""
Tests for elicitation feature.

Tests cover elicitation context management, protocol support, and integration
with the tool execution lifecycle.
"""

import pytest

from chuk_mcp_server.context import (
    clear_all,
    create_elicitation,
    get_elicitation_fn,
    set_elicitation_fn,
    set_session_id,
)
from chuk_mcp_server.protocol import MCPProtocolHandler
from chuk_mcp_server.types import ServerInfo, ToolHandler, create_server_capabilities


@pytest.fixture(autouse=True)
def cleanup_context():
    """Ensure context is clean before and after each test."""
    clear_all()
    yield
    clear_all()


class TestElicitation:
    """Test elicitation context management."""

    def test_get_elicitation_fn_default(self):
        """get_elicitation_fn returns None by default."""
        assert get_elicitation_fn() is None

    def test_set_and_get_elicitation_fn(self):
        """set_elicitation_fn and get_elicitation_fn round-trip."""

        async def mock_fn(**kwargs):
            return {"result": "ok"}

        set_elicitation_fn(mock_fn)
        assert get_elicitation_fn() is mock_fn

    def test_set_elicitation_fn_none(self):
        """Setting elicitation fn to None clears it."""

        async def mock_fn(**kwargs):
            return {}

        set_elicitation_fn(mock_fn)
        set_elicitation_fn(None)
        assert get_elicitation_fn() is None

    @pytest.mark.asyncio
    async def test_create_elicitation_raises_when_no_fn(self):
        """create_elicitation raises RuntimeError when no elicitation fn set."""
        with pytest.raises(RuntimeError, match="MCP elicitation is not available"):
            await create_elicitation(
                message="Please provide input",
                schema={"type": "object", "properties": {"name": {"type": "string"}}},
            )

    @pytest.mark.asyncio
    async def test_create_elicitation_delegates_correctly(self):
        """create_elicitation delegates to the set elicitation fn."""
        captured_kwargs = {}

        async def mock_fn(**kwargs):
            captured_kwargs.update(kwargs)
            return {"name": "John Doe"}

        set_elicitation_fn(mock_fn)
        result = await create_elicitation(
            message="Enter your name",
            schema={"type": "object", "properties": {"name": {"type": "string"}}},
            title="User Input",
            description="Please provide your name",
        )

        assert result == {"name": "John Doe"}
        assert captured_kwargs["message"] == "Enter your name"
        assert captured_kwargs["schema"] == {"type": "object", "properties": {"name": {"type": "string"}}}
        assert captured_kwargs["title"] == "User Input"
        assert captured_kwargs["description"] == "Please provide your name"

    def test_clear_all_resets_elicitation_fn(self):
        """clear_all() resets the elicitation fn."""

        async def mock_fn(**kwargs):
            return {}

        set_elicitation_fn(mock_fn)
        clear_all()
        assert get_elicitation_fn() is None


@pytest.mark.asyncio
class TestProtocolElicitation:
    """Test elicitation support in protocol handler."""

    async def test_protocol_stores_elicitation_capability(self):
        """initialize with elicitation in client capabilities, verify _client_supports_elicitation returns True."""
        server_info = ServerInfo(name="test", version="1.0.0")
        capabilities = create_server_capabilities()
        handler = MCPProtocolHandler(server_info, capabilities)

        # Set up transport (required for _client_supports_elicitation to return True)
        async def mock_send(request):
            return {"jsonrpc": "2.0", "id": request["id"], "result": {}}

        handler._send_to_client = mock_send

        # Initialize with elicitation capability
        params = {
            "clientInfo": {"name": "test-client"},
            "protocolVersion": "2025-03-26",
            "capabilities": {"elicitation": {}},
        }
        _, session_id = await handler._handle_initialize(params, 1)
        set_session_id(session_id)

        # Verify client supports elicitation
        assert handler._client_supports_elicitation({}) is True

    async def test_send_elicitation_request_builds_json_rpc(self):
        """send_elicitation_request builds correct JSON-RPC request."""
        server_info = ServerInfo(name="test", version="1.0.0")
        capabilities = create_server_capabilities()
        handler = MCPProtocolHandler(server_info, capabilities)

        captured_request = {}

        async def mock_send(request):
            captured_request.update(request)
            return {
                "jsonrpc": "2.0",
                "id": request["id"],
                "result": {"name": "Alice"},
            }

        handler._send_to_client = mock_send

        result = await handler.send_elicitation_request(
            message="What is your name?",
            schema={"type": "object", "properties": {"name": {"type": "string"}}},
            title="Name Input",
            description="Please enter your name",
        )

        assert captured_request["method"] == "elicitation/create"
        assert captured_request["params"]["message"] == "What is your name?"
        assert captured_request["params"]["requestedSchema"] == {
            "type": "object",
            "properties": {"name": {"type": "string"}},
        }
        assert captured_request["params"]["title"] == "Name Input"
        assert captured_request["params"]["description"] == "Please enter your name"
        assert result == {"name": "Alice"}

    async def test_tool_context_elicitation_injection_and_cleanup(self):
        """verify fn injected before tool execution and cleared after."""
        from chuk_mcp_server.context import get_elicitation_fn

        server_info = ServerInfo(name="test", version="1.0.0")
        capabilities = create_server_capabilities()
        handler = MCPProtocolHandler(server_info, capabilities)

        # Track whether elicitation was available during tool execution
        elicitation_was_available = {}

        async def dummy_tool(x: int) -> str:
            elicitation_was_available["value"] = get_elicitation_fn() is not None
            return str(x)

        tool = ToolHandler.from_function(dummy_tool, name="dummy_tool", description="test")
        handler.register_tool(tool)

        # Initialize with elicitation capability
        params = {
            "clientInfo": {"name": "test-client"},
            "protocolVersion": "2025-03-26",
            "capabilities": {"elicitation": {}},
        }
        _, session_id = await handler._handle_initialize(params, 1)
        set_session_id(session_id)

        # Set up transport
        async def mock_send(request):
            return {"jsonrpc": "2.0", "id": request["id"], "result": {}}

        handler._send_to_client = mock_send

        # Call the tool
        await handler._handle_tools_call({"name": "dummy_tool", "arguments": {"x": 42}}, 2)

        # Elicitation fn should have been available during tool execution
        assert elicitation_was_available["value"] is True

        # Elicitation fn should be cleared after tool execution
        assert get_elicitation_fn() is None


@pytest.mark.asyncio
class TestElicitationDefaults:
    """Test elicitation with default values in schemas (MCP 2025-11-25)."""

    async def test_schema_with_default_values_passthrough(self):
        """Default values in JSON Schema are passed through to the client."""
        server_info = ServerInfo(name="test", version="1.0.0")
        capabilities = create_server_capabilities()
        handler = MCPProtocolHandler(server_info, capabilities)

        captured_request = {}

        async def mock_send(request):
            captured_request.update(request)
            return {
                "jsonrpc": "2.0",
                "id": request["id"],
                "result": {"name": "Alice", "age": 30},
            }

        handler._send_to_client = mock_send

        schema = {
            "type": "object",
            "properties": {
                "name": {"type": "string", "default": "Anonymous"},
                "age": {"type": "integer", "default": 25},
                "active": {"type": "boolean", "default": True},
            },
        }

        await handler.send_elicitation_request(
            message="Enter details",
            schema=schema,
        )

        # Verify defaults are preserved in the schema sent to client
        sent_schema = captured_request["params"]["requestedSchema"]
        assert sent_schema["properties"]["name"]["default"] == "Anonymous"
        assert sent_schema["properties"]["age"]["default"] == 25
        assert sent_schema["properties"]["active"]["default"] is True

    async def test_schema_with_enum_and_default(self):
        """Enum fields with defaults are properly handled."""
        server_info = ServerInfo(name="test", version="1.0.0")
        capabilities = create_server_capabilities()
        handler = MCPProtocolHandler(server_info, capabilities)

        captured_request = {}

        async def mock_send(request):
            captured_request.update(request)
            return {
                "jsonrpc": "2.0",
                "id": request["id"],
                "result": {"priority": "high"},
            }

        handler._send_to_client = mock_send

        schema = {
            "type": "object",
            "properties": {
                "priority": {
                    "type": "string",
                    "enum": ["low", "medium", "high"],
                    "default": "medium",
                },
            },
        }

        await handler.send_elicitation_request(
            message="Select priority",
            schema=schema,
        )

        sent_schema = captured_request["params"]["requestedSchema"]
        assert sent_schema["properties"]["priority"]["default"] == "medium"
        assert sent_schema["properties"]["priority"]["enum"] == ["low", "medium", "high"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
