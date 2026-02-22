#!/usr/bin/env python3
"""Tests for MCP structured tool output (outputSchema, structuredContent)."""

import pytest

from chuk_mcp_server.protocol import MCPProtocolHandler
from chuk_mcp_server.types.tools import ToolHandler


@pytest.fixture
def handler():
    from chuk_mcp_server.types.base import ServerCapabilities, ServerInfo

    info = ServerInfo(name="test", version="1.0")
    caps = ServerCapabilities()
    return MCPProtocolHandler(info, caps)


USER_SCHEMA = {
    "type": "object",
    "properties": {
        "id": {"type": "string"},
        "name": {"type": "string"},
        "age": {"type": "integer"},
    },
    "required": ["id", "name"],
}


class TestOutputSchemaInToolDefinition:
    """Test that outputSchema appears in tool listings."""

    def test_output_schema_in_mcp_format(self):
        def get_user(user_id: str) -> dict:
            return {"id": user_id, "name": "Alice"}

        handler = ToolHandler.from_function(get_user, output_schema=USER_SCHEMA)
        fmt = handler.to_mcp_format()
        assert "outputSchema" in fmt
        assert fmt["outputSchema"]["type"] == "object"
        assert "id" in fmt["outputSchema"]["properties"]

    def test_no_output_schema_by_default(self):
        def simple(x: int) -> int:
            return x

        handler = ToolHandler.from_function(simple)
        fmt = handler.to_mcp_format()
        assert "outputSchema" not in fmt

    @pytest.mark.asyncio
    async def test_output_schema_in_tools_list(self, handler):
        def get_user(user_id: str) -> dict:
            return {"id": user_id, "name": "Alice"}

        handler.register_tool(ToolHandler.from_function(get_user, output_schema=USER_SCHEMA))
        response, _ = await handler._handle_tools_list({}, "req-1")
        tools = response["result"]["tools"]
        assert len(tools) == 1
        assert tools[0]["outputSchema"] == USER_SCHEMA


class TestStructuredContentInToolResults:
    """Test that structuredContent appears in tool call responses."""

    @pytest.mark.asyncio
    async def test_dict_result_with_output_schema(self, handler):
        """When a tool with output_schema returns a dict, structuredContent should be set."""

        def get_user(user_id: str) -> dict:
            return {"id": user_id, "name": "Alice", "age": 30}

        handler.register_tool(ToolHandler.from_function(get_user, output_schema=USER_SCHEMA))

        params = {"name": "get_user", "arguments": {"user_id": "u123"}}
        response, _ = await handler._handle_tools_call(params, "req-1")
        result = response["result"]

        # Should have both content (backwards compat) and structuredContent
        assert "content" in result
        assert "structuredContent" in result
        assert result["structuredContent"] == {"id": "u123", "name": "Alice", "age": 30}

    @pytest.mark.asyncio
    async def test_no_structured_content_without_schema(self, handler):
        """Tools without output_schema should not include structuredContent."""

        def simple_tool(x: str) -> str:
            return f"hello {x}"

        handler.register_tool(ToolHandler.from_function(simple_tool))

        params = {"name": "simple_tool", "arguments": {"x": "world"}}
        response, _ = await handler._handle_tools_call(params, "req-1")
        result = response["result"]

        assert "content" in result
        assert "structuredContent" not in result

    @pytest.mark.asyncio
    async def test_dict_result_without_schema_no_structured(self, handler):
        """A dict return without output_schema should NOT include structuredContent."""

        def data_tool(key: str) -> dict:
            return {"key": key, "value": 42}

        handler.register_tool(ToolHandler.from_function(data_tool))

        params = {"name": "data_tool", "arguments": {"key": "test"}}
        response, _ = await handler._handle_tools_call(params, "req-1")
        result = response["result"]

        assert "content" in result
        assert "structuredContent" not in result

    @pytest.mark.asyncio
    async def test_string_result_with_schema_no_structured(self, handler):
        """A string return even with output_schema should NOT include structuredContent."""

        schema = {"type": "object", "properties": {"msg": {"type": "string"}}}

        def msg_tool(x: str) -> str:
            return x

        handler.register_tool(ToolHandler.from_function(msg_tool, output_schema=schema))

        params = {"name": "msg_tool", "arguments": {"x": "hello"}}
        response, _ = await handler._handle_tools_call(params, "req-1")
        result = response["result"]

        assert "content" in result
        # String result is not a dict, so no structuredContent
        assert "structuredContent" not in result

    @pytest.mark.asyncio
    async def test_pydantic_model_result_with_schema(self, handler):
        """Pydantic model returns should produce structuredContent."""
        from pydantic import BaseModel

        class User(BaseModel):
            id: str
            name: str

        schema = {"type": "object", "properties": {"id": {"type": "string"}, "name": {"type": "string"}}}

        def get_user(user_id: str) -> User:
            return User(id=user_id, name="Bob")

        handler.register_tool(ToolHandler.from_function(get_user, output_schema=schema))

        params = {"name": "get_user", "arguments": {"user_id": "u1"}}
        response, _ = await handler._handle_tools_call(params, "req-1")
        result = response["result"]

        assert "structuredContent" in result
        assert result["structuredContent"]["id"] == "u1"
        assert result["structuredContent"]["name"] == "Bob"


class TestOutputSchemaViaDecorator:
    """Test output_schema passed through the decorator chain."""

    def test_global_tool_decorator(self):
        from chuk_mcp_server.decorators import clear_global_registry, get_global_tools, tool

        clear_global_registry()

        schema = {"type": "object", "properties": {"result": {"type": "integer"}}}

        @tool(output_schema=schema)
        def compute(x: int) -> dict:
            return {"result": x * 2}

        tools = get_global_tools()
        assert len(tools) == 1
        assert tools[0].output_schema == schema
        assert tools[0].to_mcp_format()["outputSchema"] == schema

        clear_global_registry()
