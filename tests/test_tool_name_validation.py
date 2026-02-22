"""
Tests for MCP 2025-11-25 tool name validation.

Tool names must match ^[a-zA-Z0-9_\\-\\.]{1,128}$ per the MCP specification.
"""

import re

import pytest

from chuk_mcp_server.constants import TOOL_NAME_PATTERN
from chuk_mcp_server.types.tools import ToolHandler

# ---------------------------------------------------------------------------
# Helper: a trivial function to wrap
# ---------------------------------------------------------------------------


def _dummy(x: str) -> str:
    """A dummy tool."""
    return x


async def _async_dummy(x: str) -> str:
    """An async dummy tool."""
    return x


# ===================================================================
# 1. TOOL_NAME_PATTERN constant sanity checks
# ===================================================================


class TestToolNamePattern:
    """Verify the regex constant itself behaves correctly."""

    def test_pattern_matches_simple_alpha(self):
        assert re.fullmatch(TOOL_NAME_PATTERN, "myTool")

    def test_pattern_matches_digits(self):
        assert re.fullmatch(TOOL_NAME_PATTERN, "tool123")

    def test_pattern_matches_underscores(self):
        assert re.fullmatch(TOOL_NAME_PATTERN, "my_tool")

    def test_pattern_matches_hyphens(self):
        assert re.fullmatch(TOOL_NAME_PATTERN, "my-tool")

    def test_pattern_matches_dots(self):
        assert re.fullmatch(TOOL_NAME_PATTERN, "my.tool")

    def test_pattern_matches_mixed(self):
        assert re.fullmatch(TOOL_NAME_PATTERN, "my-tool_v2.0")

    def test_pattern_rejects_space(self):
        assert re.fullmatch(TOOL_NAME_PATTERN, "my tool") is None

    def test_pattern_rejects_empty(self):
        assert re.fullmatch(TOOL_NAME_PATTERN, "") is None

    def test_pattern_rejects_129_chars(self):
        assert re.fullmatch(TOOL_NAME_PATTERN, "a" * 129) is None

    def test_pattern_accepts_128_chars(self):
        assert re.fullmatch(TOOL_NAME_PATTERN, "a" * 128)

    def test_pattern_accepts_single_char(self):
        assert re.fullmatch(TOOL_NAME_PATTERN, "a")


# ===================================================================
# 2. Valid tool names via ToolHandler.from_function
# ===================================================================


class TestValidToolNames:
    """ToolHandler.from_function should succeed for valid names."""

    @pytest.mark.parametrize(
        "name",
        [
            "a",  # single char
            "Z",  # single uppercase
            "5",  # single digit
            "myTool",  # camelCase
            "my_tool",  # snake_case
            "my-tool",  # kebab-case
            "my.tool",  # dotted
            "My.Tool-v2_final",  # mixed separators
            "ALLCAPS",  # uppercase
            "tool123",  # trailing digits
            "123tool",  # leading digits
            "a" * 128,  # max length (128)
            "_private",  # leading underscore
            "-leading",  # leading hyphen
            ".hidden",  # leading dot
            "trailing_",  # trailing underscore
            "trailing-",  # trailing hyphen
            "trailing.",  # trailing dot
            "a.b.c.d.e",  # multiple dots
            "a-b-c-d-e",  # multiple hyphens
            "a_b_c_d_e",  # multiple underscores
            "AbC-123_dEf.456",  # complex mix
        ],
    )
    def test_valid_name_accepted(self, name):
        handler = ToolHandler.from_function(_dummy, name=name)
        assert handler.name == name

    def test_valid_name_from_function_name(self):
        """When no explicit name is given, the function name is used."""

        def valid_func_name(x: str) -> str:
            """doc"""
            return x

        handler = ToolHandler.from_function(valid_func_name)
        assert handler.name == "valid_func_name"

    def test_valid_async_function(self):
        handler = ToolHandler.from_function(_async_dummy, name="async_tool")
        assert handler.name == "async_tool"

    def test_exact_128_characters(self):
        name = "a" * 128
        handler = ToolHandler.from_function(_dummy, name=name)
        assert handler.name == name
        assert len(handler.name) == 128


# ===================================================================
# 3. Invalid tool names via ToolHandler.from_function
# ===================================================================


class TestInvalidToolNames:
    """ToolHandler.from_function should raise ValueError for invalid names."""

    @pytest.mark.parametrize(
        "name,reason",
        [
            ("my tool", "contains space"),
            ("my\ttool", "contains tab"),
            ("my\ntool", "contains newline"),
            ("tool!", "contains exclamation"),
            ("tool@name", "contains at sign"),
            ("tool#1", "contains hash"),
            ("tool$var", "contains dollar"),
            ("tool%20", "contains percent"),
            ("tool^2", "contains caret"),
            ("tool&name", "contains ampersand"),
            ("tool*", "contains asterisk"),
            ("tool(1)", "contains parentheses"),
            ("tool+name", "contains plus"),
            ("tool=name", "contains equals"),
            ("tool[0]", "contains brackets"),
            ("tool{name}", "contains braces"),
            ("tool|name", "contains pipe"),
            ("tool\\name", "contains backslash"),
            ("tool/name", "contains forward slash"),
            ("tool:name", "contains colon"),
            ("tool;name", "contains semicolon"),
            ("tool'name", "contains single quote"),
            ('tool"name', "contains double quote"),
            ("tool<name>", "contains angle brackets"),
            ("tool,name", "contains comma"),
            ("tool?name", "contains question mark"),
            ("tool`name", "contains backtick"),
            ("tool~name", "contains tilde"),
            ("a" * 129, "129 chars, one over limit"),
            ("a" * 200, "200 chars, well over limit"),
            ("a" * 1000, "1000 chars, far over limit"),
        ],
    )
    def test_invalid_name_rejected(self, name, reason):
        with pytest.raises(ValueError, match="[Ii]nvalid tool name"):
            ToolHandler.from_function(_dummy, name=name)

    def test_empty_name_falls_back_to_function_name(self):
        """An empty name= override is falsy and falls back to function name."""
        handler = ToolHandler.from_function(_dummy, name="")
        assert handler.name == "_dummy"

    def test_unicode_rejected(self):
        with pytest.raises(ValueError, match="[Ii]nvalid tool name"):
            ToolHandler.from_function(_dummy, name="tool_\u00e9")

    def test_emoji_rejected(self):
        with pytest.raises(ValueError, match="[Ii]nvalid tool name"):
            ToolHandler.from_function(_dummy, name="tool_\U0001f600")

    def test_only_whitespace_rejected(self):
        with pytest.raises(ValueError, match="[Ii]nvalid tool name"):
            ToolHandler.from_function(_dummy, name="   ")


# ===================================================================
# 4. Decorator-based registration validates names
# ===================================================================


class TestDecoratorValidation:
    """The @tool decorator should also validate names."""

    def test_decorator_valid_name(self):
        """Importing and using the tool decorator with a valid name works."""
        from chuk_mcp_server import tool

        @tool(name="valid_decorated_tool")
        def my_func(x: str) -> str:
            """A decorated tool."""
            return x

        # The decorator returns a ToolHandler (or registers one).
        # It should not raise for a valid name.

    def test_decorator_invalid_name(self):
        """The @tool decorator should reject an invalid name."""
        from chuk_mcp_server import tool

        with pytest.raises(ValueError, match="[Ii]nvalid tool name"):

            @tool(name="invalid name with spaces")
            def my_func(x: str) -> str:
                """A decorated tool."""
                return x

    def test_decorator_invalid_special_chars(self):
        from chuk_mcp_server import tool

        with pytest.raises(ValueError, match="[Ii]nvalid tool name"):

            @tool(name="bad!name")
            def my_func(x: str) -> str:
                """Doc."""
                return x


# ===================================================================
# 5. Name override validation in from_function
# ===================================================================


class TestNameOverride:
    """When from_function receives a name= override it should validate that."""

    def test_override_replaces_function_name(self):
        def original_name(x: str) -> str:
            """doc"""
            return x

        handler = ToolHandler.from_function(original_name, name="overridden")
        assert handler.name == "overridden"

    def test_override_with_invalid_name_raises(self):
        def original_name(x: str) -> str:
            """doc"""
            return x

        with pytest.raises(ValueError, match="[Ii]nvalid tool name"):
            ToolHandler.from_function(original_name, name="bad name!")

    def test_override_too_long_raises(self):
        def ok_name(x: str) -> str:
            """doc"""
            return x

        with pytest.raises(ValueError, match="[Ii]nvalid tool name"):
            ToolHandler.from_function(ok_name, name="x" * 129)


# ===================================================================
# 6. Edge cases
# ===================================================================


class TestEdgeCases:
    """Boundary and edge-case scenarios."""

    def test_single_char_names(self):
        for ch in "aAzZ09_-.":
            handler = ToolHandler.from_function(_dummy, name=ch)
            assert handler.name == ch

    def test_boundary_128_valid(self):
        name = "a" * 128
        handler = ToolHandler.from_function(_dummy, name=name)
        assert len(handler.name) == 128

    def test_boundary_129_invalid(self):
        name = "a" * 129
        with pytest.raises(ValueError):
            ToolHandler.from_function(_dummy, name=name)

    def test_all_valid_characters_together(self):
        # Construct a name using every valid character class
        name = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-."
        handler = ToolHandler.from_function(_dummy, name=name)
        assert handler.name == name

    def test_error_message_includes_name(self):
        """The error message should mention the offending name."""
        bad = "no spaces allowed"
        with pytest.raises(ValueError) as exc_info:
            ToolHandler.from_function(_dummy, name=bad)
        assert bad in str(exc_info.value) or "Invalid tool name" in str(exc_info.value)

    def test_handler_preserves_description(self):
        """Validation should not interfere with other handler properties."""
        handler = ToolHandler.from_function(_dummy, name="good_name")
        assert handler.description == "A dummy tool."
