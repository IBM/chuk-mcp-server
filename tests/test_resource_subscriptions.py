"""Tests for resource subscriptions in chuk_mcp_server."""

from unittest.mock import AsyncMock, patch

import pytest

from chuk_mcp_server.context import clear_all, set_session_id
from chuk_mcp_server.protocol import MCPProtocolHandler
from chuk_mcp_server.types import ServerInfo, create_server_capabilities


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
async def test_subscribe_to_resource(handler):
    """Test subscribing to a resource."""
    session_id = await _create_session(handler)

    subscribe_msg = {
        "jsonrpc": "2.0",
        "id": 2,
        "method": "resources/subscribe",
        "params": {"uri": "config://settings"},
    }

    response, _ = await handler.handle_request(subscribe_msg, session_id)

    assert response["jsonrpc"] == "2.0"
    assert response["id"] == 2
    assert response["result"] == {}


@pytest.mark.asyncio
async def test_unsubscribe_from_resource(handler):
    """Test subscribing and then unsubscribing from a resource."""
    session_id = await _create_session(handler)

    # First subscribe
    subscribe_msg = {
        "jsonrpc": "2.0",
        "id": 2,
        "method": "resources/subscribe",
        "params": {"uri": "config://settings"},
    }
    response, _ = await handler.handle_request(subscribe_msg, session_id)
    assert response["result"] == {}

    # Then unsubscribe
    unsubscribe_msg = {
        "jsonrpc": "2.0",
        "id": 3,
        "method": "resources/unsubscribe",
        "params": {"uri": "config://settings"},
    }
    response, _ = await handler.handle_request(unsubscribe_msg, session_id)
    assert response["result"] == {}


@pytest.mark.asyncio
async def test_multiple_subscriptions_per_session(handler):
    """Test subscribing to multiple URIs in the same session."""
    session_id = await _create_session(handler)

    uris = [
        "config://settings",
        "file:///project/data.json",
        "custom://resource",
    ]

    for idx, uri in enumerate(uris, start=2):
        subscribe_msg = {
            "jsonrpc": "2.0",
            "id": idx,
            "method": "resources/subscribe",
            "params": {"uri": uri},
        }
        response, _ = await handler.handle_request(subscribe_msg, session_id)
        assert response["result"] == {}

    # Verify all subscriptions are tracked
    assert session_id in handler._resource_subscriptions
    subscriptions = handler._resource_subscriptions[session_id]
    assert len(subscriptions) == 3
    for uri in uris:
        assert uri in subscriptions


@pytest.mark.asyncio
async def test_subscribe_to_nonexistent_resource(handler):
    """Test that subscribing to a nonexistent resource still succeeds."""
    session_id = await _create_session(handler)

    subscribe_msg = {
        "jsonrpc": "2.0",
        "id": 2,
        "method": "resources/subscribe",
        "params": {"uri": "nonexistent://resource"},
    }

    response, _ = await handler.handle_request(subscribe_msg, session_id)

    # Per MCP spec, subscription to nonexistent resource is allowed
    assert response["result"] == {}


@pytest.mark.asyncio
async def test_unsubscribe_when_not_subscribed(handler):
    """Test that unsubscribing when not subscribed produces no error."""
    session_id = await _create_session(handler)

    unsubscribe_msg = {
        "jsonrpc": "2.0",
        "id": 2,
        "method": "resources/unsubscribe",
        "params": {"uri": "never://subscribed"},
    }

    response, _ = await handler.handle_request(unsubscribe_msg, session_id)

    # Should not produce an error
    assert response["result"] == {}


@pytest.mark.asyncio
async def test_notify_resource_updated(handler):
    """Test that notify_resource_updated sends correct notification."""
    session_id = await _create_session(handler)

    # Subscribe to a resource
    subscribe_msg = {
        "jsonrpc": "2.0",
        "id": 2,
        "method": "resources/subscribe",
        "params": {"uri": "config://settings"},
    }
    await handler.handle_request(subscribe_msg, session_id)

    # Mock _send_to_client
    with patch.object(handler, "_send_to_client", new_callable=AsyncMock) as mock_send:
        await handler.notify_resource_updated("config://settings")

        assert mock_send.called
        call_args = mock_send.call_args[0][0]

        assert call_args["jsonrpc"] == "2.0"
        assert call_args["method"] == "notifications/resources/updated"
        assert "params" in call_args
        assert call_args["params"]["uri"] == "config://settings"
        assert "id" not in call_args  # Notifications don't have IDs


@pytest.mark.asyncio
async def test_subscription_tracking(handler):
    """Test that _resource_subscriptions dict is correctly updated."""
    session_id = await _create_session(handler)

    # Initially should be empty
    assert session_id not in handler._resource_subscriptions

    # Subscribe to first resource
    subscribe_msg_1 = {
        "jsonrpc": "2.0",
        "id": 2,
        "method": "resources/subscribe",
        "params": {"uri": "config://settings"},
    }
    await handler.handle_request(subscribe_msg_1, session_id)

    assert session_id in handler._resource_subscriptions
    assert "config://settings" in handler._resource_subscriptions[session_id]
    assert len(handler._resource_subscriptions[session_id]) == 1

    # Subscribe to second resource
    subscribe_msg_2 = {
        "jsonrpc": "2.0",
        "id": 3,
        "method": "resources/subscribe",
        "params": {"uri": "file:///data.json"},
    }
    await handler.handle_request(subscribe_msg_2, session_id)

    assert len(handler._resource_subscriptions[session_id]) == 2
    assert "config://settings" in handler._resource_subscriptions[session_id]
    assert "file:///data.json" in handler._resource_subscriptions[session_id]

    # Unsubscribe from first resource
    unsubscribe_msg = {
        "jsonrpc": "2.0",
        "id": 4,
        "method": "resources/unsubscribe",
        "params": {"uri": "config://settings"},
    }
    await handler.handle_request(unsubscribe_msg, session_id)

    assert len(handler._resource_subscriptions[session_id]) == 1
    assert "config://settings" not in handler._resource_subscriptions[session_id]
    assert "file:///data.json" in handler._resource_subscriptions[session_id]

    # Unsubscribe from second resource
    unsubscribe_msg_2 = {
        "jsonrpc": "2.0",
        "id": 5,
        "method": "resources/unsubscribe",
        "params": {"uri": "file:///data.json"},
    }
    await handler.handle_request(unsubscribe_msg_2, session_id)

    assert len(handler._resource_subscriptions[session_id]) == 0
