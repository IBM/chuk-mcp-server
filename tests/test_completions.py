"""Tests for MCP completion feature.

Tests the server-side completion provider registration and handling
of client completion requests for resources and prompts.
"""

import pytest

from chuk_mcp_server.context import clear_all
from chuk_mcp_server.protocol import MCPProtocolHandler
from chuk_mcp_server.types import ServerInfo, create_server_capabilities


@pytest.fixture(autouse=True)
def cleanup():
    """Clear context before and after each test."""
    clear_all()
    yield
    clear_all()


@pytest.fixture
def server_info():
    """Create test server info."""
    return ServerInfo(name="test-server", version="1.0.0")


@pytest.fixture
def handler(server_info):
    """Create protocol handler with completions enabled."""
    capabilities = create_server_capabilities(completions=True)
    return MCPProtocolHandler(server_info, capabilities)


@pytest.mark.asyncio
async def test_register_completion_provider(handler):
    """Test registering a completion provider."""

    async def test_completer(ref, argument):
        return {"values": ["test1", "test2"], "hasMore": False}

    handler.register_completion_provider("ref/resource", test_completer)

    assert "ref/resource" in handler.completion_providers
    assert handler.completion_providers["ref/resource"] == test_completer


@pytest.mark.asyncio
async def test_handle_resource_completion(handler):
    """Test handling completion request for resources."""

    async def resource_completer(ref, argument):
        value = argument.get("value", "")
        all_values = ["config://app", "config://db", "config://cache"]
        matches = [v for v in all_values if v.startswith(value)]
        return {"values": matches, "hasMore": False}

    handler.register_completion_provider("ref/resource", resource_completer)

    msg = {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "completion/complete",
        "params": {
            "ref": {"type": "ref/resource", "uri": "config://app"},
            "argument": {"name": "uri", "value": "config://"},
        },
    }

    response, _ = await handler.handle_request(msg, session_id="test-session")

    assert response["jsonrpc"] == "2.0"
    assert response["id"] == 1
    assert "result" in response
    completion = response["result"]["completion"]
    assert len(completion["values"]) == 3
    assert "config://app" in completion["values"]
    assert "config://db" in completion["values"]
    assert "config://cache" in completion["values"]
    assert completion["hasMore"] is False


@pytest.mark.asyncio
async def test_handle_prompt_completion(handler):
    """Test handling completion request for prompts."""

    async def prompt_completer(ref, argument):
        value = argument.get("value", "")
        all_prompts = ["analyze-code", "analyze-data", "summarize-text"]
        matches = [p for p in all_prompts if p.startswith(value)]
        return {"values": matches, "hasMore": False}

    handler.register_completion_provider("ref/prompt", prompt_completer)

    msg = {
        "jsonrpc": "2.0",
        "id": 2,
        "method": "completion/complete",
        "params": {
            "ref": {"type": "ref/prompt", "name": "analyze-code"},
            "argument": {"name": "prompt", "value": "analyze-"},
        },
    }

    response, _ = await handler.handle_request(msg, session_id="test-session")

    assert response["jsonrpc"] == "2.0"
    assert response["id"] == 2
    completion = response["result"]["completion"]
    assert len(completion["values"]) == 2
    assert "analyze-code" in completion["values"]
    assert "analyze-data" in completion["values"]
    assert completion["hasMore"] is False


@pytest.mark.asyncio
async def test_no_provider_returns_empty(handler):
    """Test completion request without registered provider returns empty result."""
    msg = {
        "jsonrpc": "2.0",
        "id": 3,
        "method": "completion/complete",
        "params": {
            "ref": {"type": "ref/resource", "uri": "config://app"},
            "argument": {"name": "uri", "value": "config://"},
        },
    }

    response, _ = await handler.handle_request(msg, session_id="test-session")

    assert response["jsonrpc"] == "2.0"
    assert response["id"] == 3
    completion = response["result"]["completion"]
    assert completion["values"] == []
    assert completion["hasMore"] is False


@pytest.mark.asyncio
async def test_multiple_completion_values(handler):
    """Test provider returning multiple completion values."""

    async def multi_completer(ref, argument):
        value = argument.get("value", "")
        all_values = [f"item-{i:03d}" for i in range(100)]
        matches = [v for v in all_values if v.startswith(value)]
        return {"values": matches, "hasMore": len(matches) > 50}

    handler.register_completion_provider("ref/resource", multi_completer)

    msg = {
        "jsonrpc": "2.0",
        "id": 4,
        "method": "completion/complete",
        "params": {"ref": {"type": "ref/resource", "uri": "item-001"}, "argument": {"name": "uri", "value": "item-0"}},
    }

    response, _ = await handler.handle_request(msg, session_id="test-session")

    assert response["jsonrpc"] == "2.0"
    assert response["id"] == 4
    completion = response["result"]["completion"]
    assert len(completion["values"]) > 0
    assert all(v.startswith("item-0") for v in completion["values"])
    assert completion["hasMore"] is True


@pytest.mark.asyncio
async def test_completion_capability_in_server_info():
    """Test that completions capability is included in server info."""
    server_info = ServerInfo(name="test-server", version="1.0.0")
    capabilities = create_server_capabilities(completions=True)
    handler = MCPProtocolHandler(server_info, capabilities)

    msg = {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "initialize",
        "params": {
            "protocolVersion": "2025-06-18",
            "capabilities": {},
            "clientInfo": {"name": "test-client", "version": "1.0.0"},
        },
    }

    response, _ = await handler.handle_request(msg, session_id="test-session")

    assert response["jsonrpc"] == "2.0"
    assert response["id"] == 1
    assert "result" in response
    assert "capabilities" in response["result"]

    caps_dict = response["result"]["capabilities"]
    assert "completion" in caps_dict
    assert caps_dict["completion"] is not None


@pytest.mark.asyncio
async def test_unknown_ref_type_returns_empty(handler):
    """Test completion request with unknown ref type returns empty result."""

    async def resource_completer(ref, argument):
        return {"values": ["test1", "test2"], "hasMore": False}

    handler.register_completion_provider("ref/resource", resource_completer)

    msg = {
        "jsonrpc": "2.0",
        "id": 5,
        "method": "completion/complete",
        "params": {"ref": {"type": "ref/unknown", "uri": "something"}, "argument": {"name": "uri", "value": "some"}},
    }

    response, _ = await handler.handle_request(msg, session_id="test-session")

    assert response["jsonrpc"] == "2.0"
    assert response["id"] == 5
    completion = response["result"]["completion"]
    assert completion["values"] == []
    assert completion["hasMore"] is False
