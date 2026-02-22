#!/usr/bin/env python3
"""
Errors - Custom error classes for ChukMCPServer

This module provides specialized error classes with enhanced error reporting
and debugging information for tool execution and parameter validation.
"""

from typing import Any

from .base import MCPError, ValidationError


class ParameterValidationError(ValidationError):  # type: ignore[misc]
    """Specific error for parameter validation."""

    def __init__(self, parameter: str, expected_type: str, received: Any):
        message = f"Invalid parameter '{parameter}': expected {expected_type}, got {type(received).__name__}"
        data = {"parameter": parameter, "expected": expected_type, "received": type(received).__name__}
        super().__init__(message, data=data)


class ToolExecutionError(MCPError):  # type: ignore[misc]
    """Error during tool execution."""

    def __init__(self, tool_name: str, error: Exception):
        # Handle KeyError specially to match test expectations
        if isinstance(error, KeyError) and error.args:
            key = error.args[0]
            # Use the key exactly as it is - don't add extra quotes
            main_error_msg = str(key)
            # For the data field, use the key as-is too
            data_error_msg = key
        else:
            main_error_msg = str(error)
            data_error_msg = str(error)

        message = f"Tool '{tool_name}' execution failed: {main_error_msg}"

        data = {"tool": tool_name, "error_type": type(error).__name__, "error_message": data_error_msg}
        # MCPError expects a code parameter - use -32603 for internal error
        super().__init__(message, code=-32603, data=data)


class URLElicitationRequiredError(Exception):
    """Raised by a tool to indicate the user must visit an external URL.

    MCP 2025-11-25 URL mode elicitation. When raised, the protocol handler
    returns JSON-RPC error -32042 with the URL in the error data.

    Args:
        url: The URL the user must visit.
        description: Optional human-readable description of what the URL is for.
        mime_type: Optional MIME type hint for the URL content.
    """

    def __init__(
        self,
        url: str,
        description: str | None = None,
        mime_type: str | None = None,
    ):
        self.url = url
        self.description = description
        self.mime_type = mime_type
        super().__init__(f"URL elicitation required: {url}")


__all__ = ["ParameterValidationError", "ToolExecutionError", "URLElicitationRequiredError"]
