#!/usr/bin/env python3
"""
Integration tests for MCP sampling support.

Tests the full flow: tool calls create_message() -> protocol -> transport -> mock client.
"""

import pytest

from chuk_mcp_server.context import clear_all, create_message, set_session_id
from chuk_mcp_server.protocol import MCPProtocolHandler
from chuk_mcp_server.types import ServerInfo, ToolHandler, create_server_capabilities


@pytest.fixture(autouse=True)
def cleanup():
    clear_all()
    yield
    clear_all()


class TestSamplingIntegration:
    """End-to-end sampling integration tests."""

    @pytest.mark.asyncio
    async def test_full_sampling_flow(self):
        """Full flow: tool calls create_message, protocol routes through transport."""
        # Set up protocol handler
        handler = MCPProtocolHandler(
            ServerInfo(name="test", version="1.0"),
            create_server_capabilities(tools=True),
        )

        # Mock transport
        async def mock_send(request):
            # Simulate client LLM response
            return {
                "jsonrpc": "2.0",
                "id": request["id"],
                "result": {
                    "role": "assistant",
                    "content": {"type": "text", "text": "This terrain shows a valley."},
                    "model": "claude-3.5-sonnet",
                    "stopReason": "endTurn",
                },
            }

        handler._send_to_client = mock_send

        # Register a tool that uses sampling
        async def interpret_tool():
            result = await create_message(
                messages=[
                    {
                        "role": "user",
                        "content": {"type": "text", "text": "Describe this terrain"},
                    },
                ],
                max_tokens=500,
                system_prompt="You are a terrain expert.",
            )
            return f"Interpretation: {result['content']['text']}"

        tool = ToolHandler.from_function(interpret_tool, name="interpret")
        handler.register_tool(tool)

        # Initialize with sampling capability
        init_params = {
            "clientInfo": {"name": "test-client"},
            "protocolVersion": "2025-03-26",
            "capabilities": {"sampling": {}},
        }
        _, session_id = await handler._handle_initialize(init_params, 1)
        set_session_id(session_id)

        # Call the tool
        response, _ = await handler._handle_tools_call({"name": "interpret", "arguments": {}}, 2)

        assert "result" in response
        content = response["result"]["content"]
        assert any("valley" in str(c) for c in content)

    @pytest.mark.asyncio
    async def test_sampling_not_available_error(self):
        """Tool gets RuntimeError when client doesn't support sampling."""
        handler = MCPProtocolHandler(
            ServerInfo(name="test", version="1.0"),
            create_server_capabilities(tools=True),
        )

        async def tool_that_samples():
            return await create_message(
                messages=[{"role": "user", "content": {"type": "text", "text": "hello"}}],
            )

        tool = ToolHandler.from_function(tool_that_samples, name="sampler")
        handler.register_tool(tool)

        # Initialize WITHOUT sampling
        init_params = {
            "clientInfo": {"name": "test-client"},
            "protocolVersion": "2025-03-26",
            "capabilities": {},
        }
        _, session_id = await handler._handle_initialize(init_params, 1)
        set_session_id(session_id)

        # Tool execution should fail gracefully
        response, _ = await handler._handle_tools_call({"name": "sampler", "arguments": {}}, 2)
        assert "error" in response
        assert "sampling" in response["error"]["message"].lower()

    @pytest.mark.asyncio
    async def test_multiple_sampling_calls(self):
        """Tool can make multiple sampling calls in a single execution."""
        handler = MCPProtocolHandler(
            ServerInfo(name="test", version="1.0"),
            create_server_capabilities(tools=True),
        )

        call_count = {"n": 0}

        async def mock_send(request):
            call_count["n"] += 1
            return {
                "jsonrpc": "2.0",
                "id": request["id"],
                "result": {
                    "role": "assistant",
                    "content": {"type": "text", "text": f"Response {call_count['n']}"},
                    "model": "test-model",
                },
            }

        handler._send_to_client = mock_send

        async def multi_sample_tool():
            r1 = await create_message(messages=[{"role": "user", "content": {"type": "text", "text": "q1"}}])
            r2 = await create_message(messages=[{"role": "user", "content": {"type": "text", "text": "q2"}}])
            return f"{r1['content']['text']} + {r2['content']['text']}"

        tool = ToolHandler.from_function(multi_sample_tool, name="multi")
        handler.register_tool(tool)

        init_params = {
            "clientInfo": {"name": "test-client"},
            "protocolVersion": "2025-03-26",
            "capabilities": {"sampling": {}},
        }
        _, session_id = await handler._handle_initialize(init_params, 1)
        set_session_id(session_id)

        response, _ = await handler._handle_tools_call({"name": "multi", "arguments": {}}, 2)
        assert "result" in response
        assert call_count["n"] == 2

    @pytest.mark.asyncio
    async def test_client_error_propagates(self):
        """Error response from client propagates to tool."""
        handler = MCPProtocolHandler(
            ServerInfo(name="test", version="1.0"),
            create_server_capabilities(tools=True),
        )

        async def mock_send(request):
            return {
                "jsonrpc": "2.0",
                "id": request["id"],
                "error": {
                    "code": -32600,
                    "message": "User denied sampling request",
                },
            }

        handler._send_to_client = mock_send

        async def tool_that_samples():
            return await create_message(
                messages=[{"role": "user", "content": {"type": "text", "text": "hello"}}],
            )

        tool = ToolHandler.from_function(tool_that_samples, name="sampler")
        handler.register_tool(tool)

        init_params = {
            "clientInfo": {"name": "test-client"},
            "protocolVersion": "2025-03-26",
            "capabilities": {"sampling": {}},
        }
        _, session_id = await handler._handle_initialize(init_params, 1)
        set_session_id(session_id)

        response, _ = await handler._handle_tools_call({"name": "sampler", "arguments": {}}, 2)
        assert "error" in response
        assert "denied" in response["error"]["message"].lower()
