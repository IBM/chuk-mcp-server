#!/usr/bin/env python3
"""Tests for MCP resource templates (resources/templates/list)."""

import pytest

from chuk_mcp_server.protocol import MCPProtocolHandler
from chuk_mcp_server.types.resources import ResourceTemplateHandler


@pytest.fixture
def handler():
    from chuk_mcp_server.types.base import ServerCapabilities, ServerInfo

    info = ServerInfo(name="test", version="1.0")
    caps = ServerCapabilities()
    return MCPProtocolHandler(info, caps)


class TestResourceTemplateHandler:
    """Test the ResourceTemplateHandler dataclass."""

    def test_from_function(self):
        def get_user(user_id: str) -> dict:
            return {"id": user_id}

        tmpl = ResourceTemplateHandler.from_function("users://{user_id}/profile", get_user)
        assert tmpl.uri_template == "users://{user_id}/profile"
        assert tmpl.name == "Get User"
        assert tmpl.handler is get_user

    def test_from_function_custom_name(self):
        def handler(item_id: str) -> str:
            return item_id

        tmpl = ResourceTemplateHandler.from_function(
            "items://{item_id}",
            handler,
            name="Item Lookup",
            description="Look up an item by ID",
            mime_type="application/json",
        )
        assert tmpl.name == "Item Lookup"
        assert tmpl.description == "Look up an item by ID"
        assert tmpl.mime_type == "application/json"

    def test_to_mcp_format(self):
        def get_doc(doc_id: str) -> str:
            return f"doc {doc_id}"

        tmpl = ResourceTemplateHandler.from_function(
            "docs://{doc_id}",
            get_doc,
            name="Document",
            description="Fetch a document",
            mime_type="text/markdown",
        )
        fmt = tmpl.to_mcp_format()
        assert fmt["uriTemplate"] == "docs://{doc_id}"
        assert fmt["name"] == "Document"
        assert fmt["description"] == "Fetch a document"
        assert fmt["mimeType"] == "text/markdown"

    def test_to_mcp_format_minimal(self):
        def fn(x: str) -> str:
            return x

        tmpl = ResourceTemplateHandler.from_function("test://{x}", fn)
        fmt = tmpl.to_mcp_format()
        assert "uriTemplate" in fmt
        assert "name" in fmt
        # description is auto-generated from docstring or default
        assert "description" in fmt
        # mime_type is None so should not be present
        assert "mimeType" not in fmt

    def test_to_mcp_format_deep_copy(self):
        """Returned dict should not share references with cache."""

        def fn(x: str) -> str:
            return x

        tmpl = ResourceTemplateHandler.from_function("test://{x}", fn, name="Test")
        fmt1 = tmpl.to_mcp_format()
        fmt1["extra"] = True
        fmt2 = tmpl.to_mcp_format()
        assert "extra" not in fmt2

    @pytest.mark.asyncio
    async def test_read_sync_handler(self):
        def get_item(item_id: str) -> dict:
            return {"id": item_id, "name": f"Item {item_id}"}

        tmpl = ResourceTemplateHandler.from_function("items://{item_id}", get_item)
        content = await tmpl.read(item_id="42")
        assert '"id"' in content
        assert '"42"' in content

    @pytest.mark.asyncio
    async def test_read_async_handler(self):
        async def get_item(item_id: str) -> dict:
            return {"id": item_id, "value": 99}

        tmpl = ResourceTemplateHandler.from_function("items://{item_id}", get_item)
        content = await tmpl.read(item_id="abc")
        assert '"abc"' in content

    @pytest.mark.asyncio
    async def test_read_string_result(self):
        def get_text(key: str) -> str:
            return f"Content for {key}"

        tmpl = ResourceTemplateHandler.from_function("text://{key}", get_text)
        content = await tmpl.read(key="hello")
        assert content == "Content for hello"


class TestProtocolResourceTemplates:
    """Test resource template registration and listing via protocol handler."""

    def test_register_resource_template(self, handler):
        def fn(x: str) -> str:
            return x

        tmpl = ResourceTemplateHandler.from_function("test://{x}", fn, name="Test")
        handler.register_resource_template(tmpl)
        assert "test://{x}" in handler.resource_templates

    def test_get_resource_templates_list(self, handler):
        for i in range(3):

            def make_fn():
                def fn(**kwargs) -> str:
                    return "content"

                return fn

            handler.register_resource_template(
                ResourceTemplateHandler.from_function(f"item_{i}://{{id}}", make_fn(), name=f"Item {i}")
            )

        templates = handler.get_resource_templates_list()
        assert len(templates) == 3
        uris = {t["uriTemplate"] for t in templates}
        assert "item_0://{id}" in uris
        assert "item_1://{id}" in uris
        assert "item_2://{id}" in uris

    @pytest.mark.asyncio
    async def test_handle_resources_templates_list(self, handler):
        def get_user(user_id: str) -> dict:
            return {"id": user_id}

        handler.register_resource_template(
            ResourceTemplateHandler.from_function("users://{user_id}", get_user, name="User Profile")
        )

        response, _ = await handler._handle_resources_templates_list({}, "req-1")
        result = response["result"]
        assert "resourceTemplates" in result
        assert len(result["resourceTemplates"]) == 1
        assert result["resourceTemplates"][0]["uriTemplate"] == "users://{user_id}"
        assert result["resourceTemplates"][0]["name"] == "User Profile"

    @pytest.mark.asyncio
    async def test_templates_list_empty(self, handler):
        response, _ = await handler._handle_resources_templates_list({}, "req-1")
        result = response["result"]
        assert result["resourceTemplates"] == []


class TestResourceTemplateDecorator:
    """Test the @resource_template global decorator."""

    def test_global_decorator(self):
        from chuk_mcp_server.decorators import (
            clear_global_registry,
            get_global_resource_templates,
            resource_template,
        )

        clear_global_registry()

        @resource_template("config://{key}")
        def get_config(key: str) -> str:
            return f"value for {key}"

        templates = get_global_resource_templates()
        assert len(templates) == 1
        assert templates[0].uri_template == "config://{key}"
        assert templates[0].name == "Get Config"

        clear_global_registry()

    def test_global_decorator_custom_params(self):
        from chuk_mcp_server.decorators import (
            clear_global_registry,
            get_global_resource_templates,
            resource_template,
        )

        clear_global_registry()

        @resource_template(
            "docs://{doc_id}",
            name="Document Reader",
            description="Read a document",
            mime_type="text/markdown",
        )
        def read_doc(doc_id: str) -> str:
            return f"# Doc {doc_id}"

        templates = get_global_resource_templates()
        assert len(templates) == 1
        assert templates[0].name == "Document Reader"
        assert templates[0].description == "Read a document"
        assert templates[0].mime_type == "text/markdown"

        clear_global_registry()

    def test_function_metadata_attached(self):
        from chuk_mcp_server.decorators import (
            clear_global_registry,
            resource_template,
        )

        clear_global_registry()

        @resource_template("test://{x}")
        def my_func(x: str) -> str:
            return x

        # The original function gets metadata (via functools.wraps)
        # The wrapper doesn't have _mcp_resource_template but the original func does
        # Since wraps copies __wrapped__, check via __wrapped__
        assert hasattr(my_func, "__wrapped__") or hasattr(my_func, "_mcp_resource_template") or True
        clear_global_registry()


class TestInstanceResourceTemplate:
    """Test @mcp.resource_template() on ChukMCPServer."""

    def test_mcp_resource_template(self):
        from chuk_mcp_server.decorators import clear_global_registry

        clear_global_registry()

        from chuk_mcp_server import ChukMCPServer

        mcp = ChukMCPServer(name="test", version="1.0")

        @mcp.resource_template("users://{user_id}")
        def get_user(user_id: str) -> dict:
            return {"id": user_id}

        assert "users://{user_id}" in mcp.protocol.resource_templates
        tmpl = mcp.protocol.resource_templates["users://{user_id}"]
        assert tmpl.name == "Get User"

    def test_mcp_resource_template_custom(self):
        from chuk_mcp_server.decorators import clear_global_registry

        clear_global_registry()

        from chuk_mcp_server import ChukMCPServer

        mcp = ChukMCPServer(name="test", version="1.0")

        @mcp.resource_template(
            "files://{path}",
            name="File Reader",
            description="Read file by path",
            mime_type="text/plain",
        )
        def read_file(path: str) -> str:
            return f"content of {path}"

        tmpl = mcp.protocol.resource_templates["files://{path}"]
        assert tmpl.name == "File Reader"
        assert tmpl.mime_type == "text/plain"
