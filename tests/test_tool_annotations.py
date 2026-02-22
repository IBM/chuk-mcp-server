#!/usr/bin/env python3
"""Tests for MCP tool annotations (readOnlyHint, destructiveHint, etc.)."""

from chuk_mcp_server.types.tools import ToolHandler


class TestToolAnnotations:
    """Test tool annotation hints in ToolHandler."""

    def test_no_annotations_by_default(self):
        """Tools without annotations should not include annotations in MCP format."""

        def simple_tool(x: int) -> int:
            return x

        handler = ToolHandler.from_function(simple_tool)
        fmt = handler.to_mcp_format()
        assert "annotations" not in fmt

    def test_read_only_hint(self):
        """readOnlyHint should appear in MCP format."""

        def lookup(key: str) -> str:
            return key

        handler = ToolHandler.from_function(lookup, read_only_hint=True)
        fmt = handler.to_mcp_format()
        assert fmt["annotations"] == {"readOnlyHint": True}

    def test_destructive_hint(self):
        """destructiveHint should appear in MCP format."""

        def delete_item(item_id: str) -> str:
            return f"Deleted {item_id}"

        handler = ToolHandler.from_function(delete_item, destructive_hint=True)
        fmt = handler.to_mcp_format()
        assert fmt["annotations"]["destructiveHint"] is True

    def test_idempotent_hint(self):
        """idempotentHint should appear in MCP format."""

        def set_value(key: str, value: str) -> str:
            return "ok"

        handler = ToolHandler.from_function(set_value, idempotent_hint=True)
        fmt = handler.to_mcp_format()
        assert fmt["annotations"]["idempotentHint"] is True

    def test_open_world_hint(self):
        """openWorldHint should appear in MCP format."""

        def fetch_url(url: str) -> str:
            return "content"

        handler = ToolHandler.from_function(fetch_url, open_world_hint=True)
        fmt = handler.to_mcp_format()
        assert fmt["annotations"]["openWorldHint"] is True

    def test_multiple_annotations(self):
        """Multiple annotation hints can be set together."""

        def safe_read(key: str) -> str:
            return key

        handler = ToolHandler.from_function(
            safe_read,
            read_only_hint=True,
            destructive_hint=False,
            idempotent_hint=True,
            open_world_hint=False,
        )
        fmt = handler.to_mcp_format()
        assert fmt["annotations"] == {
            "readOnlyHint": True,
            "destructiveHint": False,
            "idempotentHint": True,
            "openWorldHint": False,
        }

    def test_false_annotations_included(self):
        """Explicitly set False values should be included (not filtered out)."""

        def my_tool(x: str) -> str:
            return x

        handler = ToolHandler.from_function(my_tool, read_only_hint=False)
        fmt = handler.to_mcp_format()
        assert fmt["annotations"] == {"readOnlyHint": False}

    def test_annotations_in_bytes_cache(self):
        """Annotations should be included in orjson bytes cache."""
        import orjson

        def my_tool(x: str) -> str:
            return x

        handler = ToolHandler.from_function(my_tool, read_only_hint=True)
        data = orjson.loads(handler.to_mcp_bytes())
        assert data["annotations"] == {"readOnlyHint": True}

    def test_cache_invalidation_preserves_annotations(self):
        """After cache invalidation and rebuild, annotations should persist."""

        def my_tool(x: str) -> str:
            return x

        handler = ToolHandler.from_function(my_tool, destructive_hint=True)
        handler.invalidate_cache()
        fmt = handler.to_mcp_format()
        assert fmt["annotations"]["destructiveHint"] is True

    def test_annotations_do_not_mutate_original(self):
        """to_mcp_format() returns a copy; mutations should not affect the handler."""

        def my_tool(x: str) -> str:
            return x

        handler = ToolHandler.from_function(my_tool, read_only_hint=True)
        fmt = handler.to_mcp_format()
        fmt["annotations"]["extra"] = True
        # Original should be unaffected
        assert "extra" not in handler.to_mcp_format()["annotations"]


class TestToolAnnotationsViaHandler:
    """Test annotations field on the ToolHandler dataclass directly."""

    def test_annotations_field_stored(self):
        """The annotations dict should be stored on the handler."""

        def my_tool(x: str) -> str:
            return x

        handler = ToolHandler.from_function(my_tool, read_only_hint=True, idempotent_hint=True)
        assert handler.annotations == {"readOnlyHint": True, "idempotentHint": True}

    def test_no_annotations_field_is_none(self):
        """Without annotations, the field should be None."""

        def my_tool(x: str) -> str:
            return x

        handler = ToolHandler.from_function(my_tool)
        assert handler.annotations is None


class TestOutputSchemaOnToolHandler:
    """Test output_schema field on ToolHandler (structured output support)."""

    def test_output_schema_in_mcp_format(self):
        """outputSchema should appear in MCP format when set."""

        def get_user(user_id: str) -> dict:
            return {"id": user_id, "name": "Alice"}

        schema = {
            "type": "object",
            "properties": {"id": {"type": "string"}, "name": {"type": "string"}},
            "required": ["id", "name"],
        }
        handler = ToolHandler.from_function(get_user, output_schema=schema)
        fmt = handler.to_mcp_format()
        assert fmt["outputSchema"] == schema

    def test_no_output_schema_by_default(self):
        """Tools without output_schema should not include it."""

        def simple(x: int) -> int:
            return x

        handler = ToolHandler.from_function(simple)
        fmt = handler.to_mcp_format()
        assert "outputSchema" not in fmt

    def test_output_schema_in_bytes(self):
        """outputSchema should be in orjson bytes."""
        import orjson

        schema = {"type": "object", "properties": {"count": {"type": "integer"}}}

        def counter() -> dict:
            return {"count": 42}

        handler = ToolHandler.from_function(counter, output_schema=schema)
        data = orjson.loads(handler.to_mcp_bytes())
        assert data["outputSchema"] == schema
