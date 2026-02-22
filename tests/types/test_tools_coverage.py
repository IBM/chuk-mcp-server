#!/usr/bin/env python3
# tests/types/test_tools_coverage.py
"""
Additional tests for tools.py to achieve 90%+ coverage.
Focuses on OAuth paths, edge cases, and uncovered lines.
"""

import pytest


def test_tool_handler_from_function_with_oauth_metadata():
    """Test ToolHandler.from_function with OAuth metadata (lines 46-48)."""
    from chuk_mcp_server.types.tools import ToolHandler

    def oauth_tool(name: str) -> str:
        """A tool that requires OAuth."""
        return f"Hello, {name}!"

    # Set OAuth metadata on the function (simulating @requires_auth decorator)
    oauth_tool._requires_auth = True
    oauth_tool._auth_scopes = ["read", "write"]

    handler = ToolHandler.from_function(oauth_tool)

    assert handler.requires_auth is True
    assert handler.auth_scopes == ["read", "write"]


def test_tool_handler_from_function_skips_external_access_token():
    """Test that _external_access_token parameter is skipped (line 59-60)."""
    from chuk_mcp_server.types.tools import ToolHandler

    def oauth_aware_tool(name: str, _external_access_token: str = None) -> str:
        """A tool with OAuth token parameter that should be skipped."""
        return f"Hello, {name}!"

    handler = ToolHandler.from_function(oauth_aware_tool)

    # Should only have 'name' parameter, not '_external_access_token'
    assert len(handler.parameters) == 1
    assert handler.parameters[0].name == "name"


def test_tool_handler_from_function_no_oauth_metadata():
    """Test ToolHandler.from_function without OAuth metadata (defaults)."""
    from chuk_mcp_server.types.tools import ToolHandler

    def simple_tool(value: str) -> str:
        return value

    handler = ToolHandler.from_function(simple_tool)

    # Should default to no auth required
    assert handler.requires_auth is False
    assert handler.auth_scopes is None


@pytest.mark.asyncio
async def test_tool_handler_execute_with_default_none_values():
    """Test execute with None default values (line 149)."""
    from chuk_mcp_server.types.tools import ToolHandler

    def tool_with_optional(name: str, metadata: dict = None) -> dict:
        return {"name": name, "metadata": metadata}

    handler = ToolHandler.from_function(tool_with_optional)

    # Execute without providing optional parameter
    result = await handler.execute({"name": "test"})

    assert result["name"] == "test"
    assert result["metadata"] is None


@pytest.mark.asyncio
async def test_tool_handler_convert_type_integer_from_string_float():
    """Test _convert_type integer conversion from float string (lines 178-186)."""
    from chuk_mcp_server.types.tools import ToolHandler

    def int_tool(count: int) -> int:
        return count * 2

    handler = ToolHandler.from_function(int_tool)

    # Test conversion from float string like "42.0"
    result = await handler.execute({"count": "42.0"})
    assert result == 84

    # Test invalid float string conversion
    from chuk_mcp_server.types.errors import ParameterValidationError

    with pytest.raises(ParameterValidationError):
        await handler.execute({"count": "42.7"})


@pytest.mark.asyncio
async def test_tool_handler_convert_type_integer_from_other_type():
    """Test _convert_type integer conversion fallback (line 189-190)."""
    from chuk_mcp_server.types.tools import ToolHandler

    def int_tool(value: int) -> int:
        return value

    handler = ToolHandler.from_function(int_tool)

    # Test direct int conversion from bool
    result = await handler.execute({"value": True})
    assert result == 1


@pytest.mark.asyncio
async def test_tool_handler_convert_type_number_from_string_error():
    """Test _convert_type number conversion error (lines 195-199)."""
    from chuk_mcp_server.types.errors import ParameterValidationError
    from chuk_mcp_server.types.tools import ToolHandler

    def number_tool(ratio: float) -> float:
        return ratio * 2

    handler = ToolHandler.from_function(number_tool)

    # Test invalid string to float conversion
    with pytest.raises(ParameterValidationError):
        await handler.execute({"ratio": "not_a_number"})


@pytest.mark.asyncio
async def test_tool_handler_convert_type_number_from_other_type():
    """Test _convert_type number conversion fallback (line 200-201)."""
    from chuk_mcp_server.types.tools import ToolHandler

    def number_tool(value: float) -> float:
        return value

    handler = ToolHandler.from_function(number_tool)

    # Test direct float conversion
    result = await handler.execute({"value": True})
    assert result == 1.0


@pytest.mark.asyncio
async def test_tool_handler_convert_type_boolean_empty_string():
    """Test _convert_type boolean conversion from empty string (lines 210-212)."""
    from chuk_mcp_server.types.tools import ToolHandler

    def bool_tool(enabled: bool = True) -> bool:
        return enabled

    handler = ToolHandler.from_function(bool_tool)

    # Test empty string uses default
    result = await handler.execute({"enabled": ""})
    assert result is True

    # Test "null" string uses default
    result2 = await handler.execute({"enabled": "null"})
    assert result2 is True


@pytest.mark.asyncio
async def test_tool_handler_convert_type_boolean_unrecognized_string():
    """Test _convert_type boolean conversion from unrecognized string (lines 217-220)."""
    from chuk_mcp_server.types.tools import ToolHandler

    def bool_tool(enabled: bool = False) -> bool:
        return enabled

    handler = ToolHandler.from_function(bool_tool)

    # Test unrecognized string uses default
    result = await handler.execute({"enabled": "maybe"})
    assert result is False


@pytest.mark.asyncio
async def test_tool_handler_convert_type_boolean_none_value():
    """Test _convert_type boolean conversion from None (lines 224-226)."""
    from chuk_mcp_server.types.parameters import ToolParameter
    from chuk_mcp_server.types.tools import ToolHandler

    def bool_tool(enabled: bool = True) -> bool:
        return enabled

    handler = ToolHandler.from_function(bool_tool)

    # Create a parameter with None value and explicit default
    param = ToolParameter(name="enabled", type="boolean", required=False, default=True)

    # Test conversion with None value
    result = handler._convert_type(None, param)
    assert result is True


@pytest.mark.asyncio
async def test_tool_handler_convert_type_boolean_exception():
    """Test _convert_type boolean conversion exception (lines 229-232)."""
    from chuk_mcp_server.types.parameters import ToolParameter
    from chuk_mcp_server.types.tools import ToolHandler

    def dummy_fn():
        return None

    handler = ToolHandler.from_function(dummy_fn)
    param = ToolParameter(name="enabled", type="boolean", required=True)

    # Create an object that can't be converted to bool
    class UnconvertibleType:
        def __bool__(self):
            raise RuntimeError("Cannot convert")

    with pytest.raises(ValueError) as exc_info:
        handler._convert_type(UnconvertibleType(), param)

    assert "Cannot convert" in str(exc_info.value)


@pytest.mark.asyncio
async def test_tool_handler_convert_type_string_from_other_types():
    """Test _convert_type string conversion (lines 234-239)."""
    from chuk_mcp_server.types.tools import ToolHandler

    def string_tool(value: str) -> str:
        return value

    handler = ToolHandler.from_function(string_tool)

    # Test conversion from various types
    result1 = await handler.execute({"value": 42})
    assert result1 == "42"

    result2 = await handler.execute({"value": True})
    assert result2 == "True"

    result3 = await handler.execute({"value": [1, 2, 3]})
    assert result3 == "[1, 2, 3]"


@pytest.mark.asyncio
async def test_tool_handler_convert_type_array_from_tuple_set():
    """Test _convert_type array conversion from tuple/set (lines 244-245)."""
    from chuk_mcp_server.types.parameters import ToolParameter
    from chuk_mcp_server.types.tools import ToolHandler

    handler = ToolHandler.from_function(lambda: None, name="dummy")
    param = ToolParameter(name="items", type="array", required=True)

    # Test tuple conversion
    result1 = handler._convert_type((1, 2, 3), param)
    assert result1 == [1, 2, 3]

    # Test set conversion
    result2 = handler._convert_type({1, 2, 3}, param)
    assert set(result2) == {1, 2, 3}


@pytest.mark.asyncio
async def test_tool_handler_convert_type_array_invalid_json():
    """Test _convert_type array conversion with invalid JSON (lines 252-255)."""
    from chuk_mcp_server.types.parameters import ToolParameter
    from chuk_mcp_server.types.tools import ToolHandler

    handler = ToolHandler.from_function(lambda: None, name="dummy")
    param = ToolParameter(name="items", type="array", required=True)

    # Test invalid JSON string
    with pytest.raises(ValueError) as exc_info:
        handler._convert_type("not valid json", param)

    assert "Cannot convert string" in str(exc_info.value)

    # Test non-array JSON
    with pytest.raises(ValueError) as exc_info:
        handler._convert_type('{"key": "value"}', param)

    assert "does not represent an array" in str(exc_info.value)


@pytest.mark.asyncio
async def test_tool_handler_convert_type_array_invalid_type():
    """Test _convert_type array conversion with invalid type (line 256-257)."""
    from chuk_mcp_server.types.parameters import ToolParameter
    from chuk_mcp_server.types.tools import ToolHandler

    handler = ToolHandler.from_function(lambda: None, name="dummy")
    param = ToolParameter(name="items", type="array", required=True)

    # Test invalid type
    with pytest.raises(ValueError) as exc_info:
        handler._convert_type(42, param)

    assert "Cannot convert" in str(exc_info.value)


@pytest.mark.asyncio
async def test_tool_handler_convert_type_object_invalid_json():
    """Test _convert_type object conversion with invalid JSON (lines 268-271)."""
    from chuk_mcp_server.types.parameters import ToolParameter
    from chuk_mcp_server.types.tools import ToolHandler

    handler = ToolHandler.from_function(lambda: None, name="dummy")
    param = ToolParameter(name="config", type="object", required=True)

    # Test invalid JSON string
    with pytest.raises(ValueError) as exc_info:
        handler._convert_type("not valid json", param)

    assert "Cannot convert string" in str(exc_info.value)

    # Test non-object JSON
    with pytest.raises(ValueError) as exc_info:
        handler._convert_type("[1, 2, 3]", param)

    assert "does not represent an object" in str(exc_info.value)


@pytest.mark.asyncio
async def test_tool_handler_convert_type_object_invalid_type():
    """Test _convert_type object conversion with invalid type (lines 272-273)."""
    from chuk_mcp_server.types.parameters import ToolParameter
    from chuk_mcp_server.types.tools import ToolHandler

    handler = ToolHandler.from_function(lambda: None, name="dummy")
    param = ToolParameter(name="config", type="object", required=True)

    # Test invalid type
    with pytest.raises(ValueError) as exc_info:
        handler._convert_type(42, param)

    assert "Cannot convert" in str(exc_info.value)


def test_tool_handler_convert_type_enum_invalid():
    """Test _convert_type enum validation (lines 276-277).

    Note: Lines 276-277 contain enum validation code, but it's unreachable in practice
    because all type handlers (string, integer, etc.) return early. The enum check
    would only execute for unknown types. This test verifies the code exists but
    acknowledges it's currently dead code that should be refactored.
    """
    from chuk_mcp_server.types.parameters import ToolParameter
    from chuk_mcp_server.types.tools import ToolHandler

    handler = ToolHandler.from_function(lambda: None, name="dummy")

    # Create a parameter with an unknown type to reach the enum check
    param = ToolParameter(name="level", type="unknown_type", required=True, enum=["low", "medium", "high"])

    # For unknown types, the code falls through to the enum check
    # Test valid enum value
    result = handler._convert_type("low", param)
    assert result == "low"

    # Test invalid enum value - should trigger the enum check at lines 276-277
    with pytest.raises(ValueError) as exc_info:
        handler._convert_type("invalid", param)

    assert "must be one of" in str(exc_info.value)


@pytest.mark.asyncio
async def test_tool_handler_execute_validation_error_reraise():
    """Test execute re-raises ValidationError (line 291-293)."""
    from chuk_mcp_server.types.tools import ToolHandler

    def strict_tool(count: int) -> int:
        return count

    handler = ToolHandler.from_function(strict_tool)

    # ValidationError should be re-raised as-is
    from chuk_mcp_server.types.errors import ParameterValidationError

    with pytest.raises(ParameterValidationError):
        await handler.execute({"count": None})


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
