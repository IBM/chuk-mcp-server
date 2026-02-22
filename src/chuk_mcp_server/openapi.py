"""
OpenAPI specification generation from MCP tool schemas.

Auto-generates an OpenAPI 3.1.0 spec from registered tools,
making it easy to document and explore the server's API.
"""

from typing import Any

from .protocol import MCPProtocolHandler


def generate_openapi_spec(protocol: MCPProtocolHandler) -> dict[str, Any]:
    """Generate an OpenAPI 3.1.0 specification from registered tools.

    Args:
        protocol: The protocol handler with registered tools.

    Returns:
        OpenAPI specification dict.
    """
    paths: dict[str, Any] = {}

    for name, handler in protocol.tools.items():
        schema = handler.to_mcp_format().get("inputSchema", {})
        paths[f"/tools/{name}"] = {
            "post": {
                "operationId": name,
                "summary": handler.description or name,
                "requestBody": {
                    "required": True,
                    "content": {
                        "application/json": {
                            "schema": schema,
                        }
                    },
                },
                "responses": {
                    "200": {
                        "description": "Tool execution result",
                        "content": {
                            "application/json": {
                                "schema": {
                                    "type": "object",
                                    "properties": {
                                        "content": {
                                            "type": "array",
                                            "items": {
                                                "type": "object",
                                                "properties": {
                                                    "type": {"type": "string"},
                                                    "text": {"type": "string"},
                                                },
                                            },
                                        }
                                    },
                                }
                            }
                        },
                    },
                    "400": {"description": "Invalid parameters"},
                    "500": {"description": "Tool execution error"},
                },
            }
        }

    return {
        "openapi": "3.1.0",
        "info": {
            "title": protocol.server_info.name,
            "version": protocol.server_info.version,
        },
        "paths": paths,
    }
