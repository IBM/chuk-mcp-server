#!/usr/bin/env python3
# src/chuk_mcp_server/types/prompts.py
"""
Prompts - PromptHandler with orjson optimization and argument parsing

This module provides the PromptHandler class with performance optimizations
including caching, orjson serialization, and prompt argument validation.
"""

import inspect
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import orjson

from .base import MCPError
from .errors import ParameterValidationError
from .parameters import ToolParameter

# ============================================================================
# MCP Prompt Data Class (Since chuk_mcp doesn't provide one)
# ============================================================================


@dataclass
class MCPPrompt:
    """MCP Prompt data class following the MCP specification."""

    name: str
    description: str | None = None
    arguments: list[dict[str, Any]] | None = None

    def model_dump(self, exclude_none: bool = False) -> dict[str, Any]:
        """Convert to dictionary format for MCP protocol."""
        result = {"name": self.name, "description": self.description, "arguments": self.arguments}

        if exclude_none:
            return {k: v for k, v in result.items() if v is not None}
        return result


# ============================================================================
# PromptHandler with Maximum Performance Optimization
# ============================================================================


@dataclass
class PromptHandler:
    """Framework prompt handler with orjson optimization and argument validation."""

    mcp_prompt: MCPPrompt  # The MCP prompt definition
    handler: Callable[..., Any]
    parameters: list[ToolParameter]  # Reuse ToolParameter for prompt arguments
    _cached_mcp_format: dict[str, Any] | None = None  # Cache the MCP format dict
    _cached_mcp_bytes: bytes | None = None  # 🚀 Cache orjson-serialized bytes
    icons: list[dict[str, Any]] | None = None  # MCP icons (2025-11-25)

    @classmethod
    def from_function(
        cls,
        func: Callable[..., Any],
        name: str | None = None,
        description: str | None = None,
        icons: list[dict[str, Any]] | None = None,
    ) -> "PromptHandler":
        """Create PromptHandler from a function with orjson optimization."""
        prompt_name = name or func.__name__
        prompt_description = description or func.__doc__ or f"Prompt: {prompt_name}"

        # Extract parameters from function signature (for prompt arguments)
        sig = inspect.signature(func)
        parameters = []
        arguments = []

        for param_name, param in sig.parameters.items():
            if param_name == "self":  # Skip self parameter for methods
                continue

            tool_param = ToolParameter.from_annotation(
                name=param_name,
                annotation=param.annotation if param.annotation != inspect.Parameter.empty else str,
                default=param.default,
            )
            parameters.append(tool_param)

            # Build argument schema for MCP format
            arg_schema = {"name": param_name, "description": f"Parameter {param_name}", "required": tool_param.required}

            # Add type information
            if tool_param.type:
                arg_schema["type"] = tool_param.type

            # Add enum if specified
            if tool_param.enum:
                arg_schema["enum"] = tool_param.enum

            arguments.append(arg_schema)

        # Create the MCP prompt
        mcp_prompt = MCPPrompt(
            name=prompt_name, description=prompt_description, arguments=arguments if arguments else None
        )

        # Create instance and pre-cache both dict and orjson formats
        instance = cls(
            mcp_prompt=mcp_prompt,
            handler=func,
            parameters=parameters,
            _cached_mcp_format=None,  # Will be computed immediately
            _cached_mcp_bytes=None,  # Will be computed immediately
            icons=icons,
        )

        # Pre-compute and cache both formats during creation for maximum performance
        instance._ensure_cached_formats()

        return instance

    def _ensure_cached_formats(self) -> None:
        """Ensure both dict and orjson formats are cached."""
        if self._cached_mcp_format is None:
            # Cache the expensive schema generation once
            fmt = self.mcp_prompt.model_dump(exclude_none=True)
            # Add icons if present (MCP 2025-11-25)
            if self.icons:
                fmt["icons"] = [icon.copy() for icon in self.icons]
            self._cached_mcp_format = fmt

        if self._cached_mcp_bytes is None:
            # Pre-serialize with orjson for maximum speed
            self._cached_mcp_bytes = orjson.dumps(self._cached_mcp_format)

    @property
    def name(self) -> str:
        """Get the prompt name."""
        return self.mcp_prompt.name

    @property
    def description(self) -> str | None:
        """Get the prompt description."""
        return self.mcp_prompt.description

    @property
    def arguments(self) -> list[dict[str, Any]] | None:
        """Get the prompt arguments schema."""
        return self.mcp_prompt.arguments

    def to_mcp_format(self) -> dict[str, Any]:
        """Convert to MCP prompt format using cached version for maximum performance."""
        if self._cached_mcp_format is None:
            self._ensure_cached_formats()
        assert self._cached_mcp_format is not None  # Type guard
        return self._cached_mcp_format.copy()  # Return copy to prevent mutation

    def to_mcp_bytes(self) -> bytes:
        """🚀 Get orjson-serialized MCP format bytes for ultimate performance."""
        if self._cached_mcp_bytes is None:
            self._ensure_cached_formats()
        assert self._cached_mcp_bytes is not None  # Type guard
        return self._cached_mcp_bytes

    def invalidate_cache(self) -> None:
        """Invalidate all cached formats (if schema changes at runtime)."""
        self._cached_mcp_format = None
        self._cached_mcp_bytes = None

    def _validate_and_convert_arguments(self, arguments: dict[str, Any]) -> dict[str, Any]:
        """Validate and convert arguments with specific error types."""
        validated_args = {}

        for param in self.parameters:
            value = arguments.get(param.name)

            if value is None:
                if param.required:
                    raise ParameterValidationError(param.name, param.type, None)
                value = param.default

            # Skip validation if value is still None after default assignment
            if value is not None:
                try:
                    validated_value = self._convert_type(value, param)
                    validated_args[param.name] = validated_value
                except (ValueError, TypeError) as e:
                    raise ParameterValidationError(param.name, param.type, value) from e

        return validated_args

    def _convert_type(self, value: Any, param: ToolParameter) -> Any:
        """Convert value to the expected parameter type with orjson optimization."""
        # Reuse the same type conversion logic from ToolHandler
        if param.type == "integer":
            if isinstance(value, int):
                return value
            elif isinstance(value, float):
                if value.is_integer():
                    return int(value)
                else:
                    raise ValueError(f"Cannot convert float {value} to integer without precision loss")
            elif isinstance(value, str):
                try:
                    return int(value)
                except ValueError:
                    try:
                        float_val = float(value)
                        if float_val.is_integer():
                            return int(float_val)
                        else:
                            raise ValueError(f"Cannot convert string '{value}' to integer without precision loss")
                    except ValueError as e:
                        raise ValueError(f"Cannot convert string '{value}' to integer") from e
            else:
                return int(value)

        elif param.type == "number":
            if isinstance(value, int | float):
                return float(value)
            elif isinstance(value, str):
                try:
                    return float(value)
                except ValueError as e:
                    raise ValueError(f"Cannot convert string '{value}' to number") from e
            else:
                return float(value)

        elif param.type == "boolean":
            if isinstance(value, bool):
                return value
            elif isinstance(value, str):
                # Handle string representations of booleans more robustly
                lower_val = value.lower().strip()
                # Handle empty string and null string - use parameter default or False
                if lower_val == "" or lower_val == "null":
                    # For empty/null strings, use the parameter default if available, otherwise False
                    return param.default if param.default is not None else False
                elif lower_val in ("true", "1", "yes", "on", "t", "y"):
                    return True
                elif lower_val in ("false", "0", "no", "off", "f", "n"):
                    return False
                else:
                    # For unrecognized strings, raise an error to ensure valid input
                    raise ValueError(f"Cannot convert string '{value}' to boolean")
            elif isinstance(value, int | float):
                return bool(value)
            elif value is None:
                # Handle None explicitly
                return False
            else:
                # Try converting to bool as last resort
                try:
                    return bool(value)
                except Exception as e:
                    raise ValueError(f"Cannot convert {type(value).__name__} '{value}' to boolean") from e

        elif param.type == "string":
            if isinstance(value, str):
                return value
            else:
                return str(value)

        elif param.type == "array":
            if isinstance(value, list):
                return value
            elif isinstance(value, tuple | set):
                return list(value)
            elif isinstance(value, str):
                try:
                    parsed = orjson.loads(value)
                    if isinstance(parsed, list):
                        return parsed
                    else:
                        raise ValueError(f"String '{value}' does not represent an array")
                except orjson.JSONDecodeError as e:
                    raise ValueError(f"Cannot convert string '{value}' to array") from e
            else:
                raise ValueError(f"Cannot convert {type(value).__name__} to array")

        elif param.type == "object":
            if isinstance(value, dict):
                return value
            elif isinstance(value, str):
                try:
                    parsed = orjson.loads(value)
                    if isinstance(parsed, dict):
                        return parsed
                    else:
                        raise ValueError(f"String '{value}' does not represent an object")
                except orjson.JSONDecodeError as e:
                    raise ValueError(f"Cannot convert string '{value}' to object") from e
            else:
                raise ValueError(f"Cannot convert {type(value).__name__} to object")

        # Check enum values if specified
        if param.enum and value not in param.enum:
            raise ValueError(f"Value '{value}' must be one of {param.enum}")

        return value

    async def get_prompt(self, arguments: dict[str, Any] | None = None) -> str | dict[str, Any]:
        """
        Get the prompt content with optional arguments.

        This is the main method called by the MCP protocol when a prompt is requested.
        """
        try:
            validated_args = self._validate_and_convert_arguments(arguments) if arguments else {}

            if inspect.iscoroutinefunction(self.handler):
                result = await self.handler(**validated_args)
            else:
                result = self.handler(**validated_args)

            # Ensure we return the correct type
            return result  # type: ignore[no-any-return]

        except ParameterValidationError:
            # Re-raise validation errors as-is
            raise
        except Exception as e:
            # Wrap other errors in MCPError
            raise MCPError(f"Failed to generate prompt '{self.name}': {str(e)}", code=-32603) from e


# ============================================================================
# Prompt Creation Utilities
# ============================================================================


def create_prompt_from_function(
    func: Callable[..., Any], name: str | None = None, description: str | None = None
) -> PromptHandler:
    """Create a PromptHandler from a function - convenience function."""
    return PromptHandler.from_function(func, name=name, description=description)


# ============================================================================
# Exports
# ============================================================================

__all__ = [
    "MCPPrompt",
    "PromptHandler",
    "create_prompt_from_function",
]
