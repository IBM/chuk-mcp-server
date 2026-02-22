"""Tests for roots support in chuk_mcp_server."""

from unittest.mock import AsyncMock

import pytest

from chuk_mcp_server.context import (
    clear_all,
    get_roots_fn,
    list_roots,
    set_roots_fn,
    set_session_id,
)
from chuk_mcp_server.protocol import MCPProtocolHandler
from chuk_mcp_server.types import (
    ServerInfo,
    ToolHandler,
    create_server_capabilities,
)


@pytest.fixture(autouse=True)
def cleanup():
    """Clear context before and after each test."""
    clear_all()
    yield
    clear_all()


@pytest.fixture
def handler():
    """Create a basic MCPProtocolHandler for testing."""
    server_info = ServerInfo(name="test", version="1.0.0")
    capabilities = create_server_capabilities()
    return MCPProtocolHandler(server_info, capabilities)


async def _create_session(handler, client_capabilities=None):
    """Helper to create a session via initialize."""
    init_msg = {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "initialize",
        "params": {
            "protocolVersion": "2025-06-18",
            "capabilities": client_capabilities or {},
            "clientInfo": {"name": "test-client", "version": "1.0.0"},
        },
    }
    response, session_id = await handler.handle_request(init_msg)
    set_session_id(session_id)
    return session_id


@pytest.mark.asyncio
async def test_list_roots_raises_when_no_fn():
    """Test that list_roots raises RuntimeError when no function is set."""
    assert get_roots_fn() is None

    with pytest.raises(RuntimeError, match="MCP roots is not available"):
        await list_roots()


@pytest.mark.asyncio
async def test_list_roots_delegates_correctly():
    """Test that list_roots correctly delegates to the registered function."""
    mock_roots = [
        {"uri": "file:///project", "name": "Project Root"},
        {"uri": "file:///home", "name": "Home Directory"},
    ]

    async def mock_roots_fn():
        return mock_roots

    set_roots_fn(mock_roots_fn)
    result = await list_roots()

    assert len(result) == 2
    assert result[0]["uri"] == "file:///project"
    assert result[1]["name"] == "Home Directory"


@pytest.mark.asyncio
async def test_send_roots_request_builds_json_rpc(handler):
    """Test that send_roots_request builds correct JSON-RPC request."""
    mock_send = AsyncMock(
        return_value={
            "jsonrpc": "2.0",
            "id": "roots-abc",
            "result": {"roots": [{"uri": "file:///test", "name": "Test"}]},
        }
    )
    handler._send_to_client = mock_send

    result = await handler.send_roots_request()

    assert mock_send.called
    call_args = mock_send.call_args[0][0]
    assert call_args["jsonrpc"] == "2.0"
    assert call_args["method"] == "roots/list"
    assert "id" in call_args
    assert call_args["id"].startswith("roots-")
    assert result == [{"uri": "file:///test", "name": "Test"}]


@pytest.mark.asyncio
async def test_roots_list_changed_notification(handler):
    """Test that roots/list_changed notification is handled correctly."""
    session_id = await _create_session(handler)

    notification = {
        "jsonrpc": "2.0",
        "method": "notifications/roots/list_changed",
    }

    result, error = await handler.handle_request(notification, session_id)
    assert result is None
    assert error is None


@pytest.mark.asyncio
async def test_tool_context_roots_injection_and_cleanup(handler):
    """Test that roots function is injected before tool execution and cleared after."""
    roots_fn_during_execution = None

    async def test_tool_fn():
        nonlocal roots_fn_during_execution
        roots_fn_during_execution = get_roots_fn()
        return "executed"

    tool_handler = ToolHandler.from_function(test_tool_fn, "test_tool", "Test tool")
    handler.register_tool(tool_handler)

    # Initialize session with roots support
    session_id = await _create_session(handler, {"roots": {"listChanged": True}})

    mock_send = AsyncMock()
    handler._send_to_client = mock_send

    assert get_roots_fn() is None

    call_msg = {
        "jsonrpc": "2.0",
        "id": 2,
        "method": "tools/call",
        "params": {"name": "test_tool", "arguments": {}},
    }
    await handler.handle_request(call_msg, session_id)

    assert roots_fn_during_execution is not None
    assert get_roots_fn() is None
