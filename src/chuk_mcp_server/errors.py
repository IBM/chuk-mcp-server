"""
Structured error types for ChukMCPServer.

Provides actionable errors with fix suggestions and documentation links.
"""

from difflib import get_close_matches
from typing import Any


class MCPError(Exception):
    """Structured MCP error with fix suggestions."""

    def __init__(
        self,
        message: str,
        code: int = -32603,
        suggestion: str | None = None,
        docs_url: str | None = None,
    ):
        self.code = code
        self.suggestion = suggestion
        self.docs_url = docs_url
        super().__init__(message)

    def to_message(self) -> str:
        """Format the error with suggestion."""
        parts = [str(self)]
        if self.suggestion:
            parts.append(f"Suggestion: {self.suggestion}")
        if self.docs_url:
            parts.append(f"Docs: {self.docs_url}")
        return " | ".join(parts)


def suggest_tool_name(tool_name: str, available_tools: list[str]) -> str | None:
    """Find the closest matching tool name using fuzzy matching.

    Args:
        tool_name: The unknown tool name.
        available_tools: List of registered tool names.

    Returns:
        The closest match, or None if no good match found.
    """
    matches = get_close_matches(tool_name, available_tools, n=1, cutoff=0.6)
    return matches[0] if matches else None


def format_unknown_tool_error(tool_name: str, available_tools: list[str]) -> str:
    """Create an error message for an unknown tool with suggestions.

    Args:
        tool_name: The requested tool name.
        available_tools: List of registered tool names.

    Returns:
        Error message string, potentially with a suggestion.
    """
    suggestion = suggest_tool_name(tool_name, available_tools)
    if suggestion:
        return f"Unknown tool: '{tool_name}'. Did you mean '{suggestion}'?"
    if available_tools:
        names = ", ".join(sorted(available_tools)[:10])
        suffix = "..." if len(available_tools) > 10 else ""
        return f"Unknown tool: '{tool_name}'. Available tools: {names}{suffix}"
    return f"Unknown tool: '{tool_name}'. No tools are registered."


def format_missing_argument_error(tool_name: str, param_name: str, schema: dict[str, Any] | None = None) -> str:
    """Create an error message for a missing required argument.

    Args:
        tool_name: The tool name.
        param_name: The missing parameter name.
        schema: Optional JSON schema for the parameter.

    Returns:
        Error message string.
    """
    msg = f"Tool '{tool_name}': missing required argument '{param_name}'."
    if schema and param_name in schema.get("properties", {}):
        prop = schema["properties"][param_name]
        prop_type = prop.get("type", "any")
        desc = prop.get("description", "")
        if desc:
            msg += f" ({prop_type}: {desc})"
        else:
            msg += f" (type: {prop_type})"
    return msg
