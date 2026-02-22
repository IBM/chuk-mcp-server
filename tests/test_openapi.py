"""Tests for OpenAPI specification generation."""

from chuk_mcp_server.openapi import generate_openapi_spec
from chuk_mcp_server.protocol import MCPProtocolHandler
from chuk_mcp_server.types import ServerInfo, ToolHandler, create_server_capabilities


def _make_protocol_with_tools():
    """Create a protocol handler with test tools."""
    server_info = ServerInfo(name="test-api", version="2.0.0")
    caps = create_server_capabilities()
    protocol = MCPProtocolHandler(server_info, caps)

    def add(a: int, b: int) -> int:
        """Add two numbers together."""
        return a + b

    def greet(name: str) -> str:
        """Greet someone by name."""
        return f"Hello, {name}!"

    protocol.register_tool(ToolHandler.from_function(add, name="add", description="Add two numbers together."))
    protocol.register_tool(ToolHandler.from_function(greet, name="greet", description="Greet someone by name."))
    return protocol


class TestGenerateOpenAPISpec:
    """Tests for generate_openapi_spec."""

    def test_openapi_version(self):
        """Spec has correct OpenAPI version."""
        protocol = _make_protocol_with_tools()
        spec = generate_openapi_spec(protocol)
        assert spec["openapi"] == "3.1.0"

    def test_info_from_server(self):
        """Spec info comes from server info."""
        protocol = _make_protocol_with_tools()
        spec = generate_openapi_spec(protocol)
        assert spec["info"]["title"] == "test-api"
        assert spec["info"]["version"] == "2.0.0"

    def test_paths_for_each_tool(self):
        """Each tool gets a path entry."""
        protocol = _make_protocol_with_tools()
        spec = generate_openapi_spec(protocol)
        assert "/tools/add" in spec["paths"]
        assert "/tools/greet" in spec["paths"]

    def test_tool_has_post_operation(self):
        """Each tool path has a POST operation."""
        protocol = _make_protocol_with_tools()
        spec = generate_openapi_spec(protocol)
        assert "post" in spec["paths"]["/tools/add"]

    def test_tool_operation_has_summary(self):
        """POST operation has description as summary."""
        protocol = _make_protocol_with_tools()
        spec = generate_openapi_spec(protocol)
        assert spec["paths"]["/tools/add"]["post"]["summary"] == "Add two numbers together."

    def test_tool_has_request_body_schema(self):
        """POST operation has requestBody with input schema."""
        protocol = _make_protocol_with_tools()
        spec = generate_openapi_spec(protocol)
        schema = spec["paths"]["/tools/add"]["post"]["requestBody"]["content"]["application/json"]["schema"]
        assert "properties" in schema
        assert "a" in schema["properties"]
        assert "b" in schema["properties"]

    def test_empty_tools(self):
        """Spec with no tools has empty paths."""
        server_info = ServerInfo(name="empty", version="1.0.0")
        caps = create_server_capabilities()
        protocol = MCPProtocolHandler(server_info, caps)
        spec = generate_openapi_spec(protocol)
        assert spec["paths"] == {}

    def test_operation_id_matches_tool_name(self):
        """operationId matches the tool name."""
        protocol = _make_protocol_with_tools()
        spec = generate_openapi_spec(protocol)
        assert spec["paths"]["/tools/add"]["post"]["operationId"] == "add"
        assert spec["paths"]["/tools/greet"]["post"]["operationId"] == "greet"
