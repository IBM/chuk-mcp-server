#!/usr/bin/env python3
# src/chuk_mcp_server/types/capabilities.py
"""
Capabilities - Server capability creation and management

This module provides helpers for creating and managing MCP server capabilities
with clean APIs and backward compatibility.
"""

from typing import Any

from .base import (
    CompletionCapability,
    LoggingCapability,
    PromptsCapability,
    ResourcesCapability,
    ServerCapabilities,
    ToolsCapability,
)


class _FilteredServerCapabilities(ServerCapabilities):  # type: ignore[misc]
    """ServerCapabilities subclass with filtered model_dump method"""

    def __init__(
        self,
        *args: Any,
        _filter_kwargs: dict[str, Any] | None = None,
        _experimental: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self._filter_kwargs: dict[str, Any] = _filter_kwargs or {}
        self._experimental: dict[str, Any] | None = _experimental

    def model_dump(self, **dump_kwargs: Any) -> dict[str, Any]:
        """Filter out unwanted fields from model_dump"""
        result = super().model_dump(**dump_kwargs)
        # Remove None fields and unwanted default fields
        filtered: dict[str, Any] = {}
        for key, value in result.items():
            # Only include fields we explicitly set
            if key in self._filter_kwargs or (key == "experimental" and self._experimental is not None):
                # Special handling for empty capability objects
                # Keep logging and experimental even if empty (they're valid empty capabilities)
                if isinstance(value, dict) and not value and key not in ("logging", "experimental", "completion"):
                    # Skip empty capability objects (except logging and experimental)
                    # This prevents MCP Inspector UI issues
                    continue
                filtered[key] = value
        return filtered


def create_server_capabilities(
    tools: bool = True,
    resources: bool = True,
    prompts: bool = False,
    logging: bool = False,
    completions: bool = False,
    experimental: dict[str, Any] | None = None,
) -> ServerCapabilities:
    """Create server capabilities using chuk_mcp types directly."""
    # Build only enabled capabilities
    kwargs: dict[str, Any] = {}

    if tools:
        kwargs["tools"] = ToolsCapability(listChanged=True)

    if resources:
        kwargs["resources"] = ResourcesCapability(listChanged=True, subscribe=True)

    if prompts:
        kwargs["prompts"] = PromptsCapability(listChanged=True)

    if logging:
        kwargs["logging"] = LoggingCapability()

    if completions:
        kwargs["completion"] = CompletionCapability()

    # Handle experimental features
    if experimental is not None:
        if experimental == {}:
            kwargs["experimental"] = experimental
        else:
            # Try to include experimental features
            try:
                kwargs["experimental"] = experimental
                caps = _FilteredServerCapabilities(_filter_kwargs=kwargs, _experimental=experimental, **kwargs)
            except Exception:
                # Create without experimental first, then set it manually
                caps = _FilteredServerCapabilities(
                    _filter_kwargs={k: v for k, v in kwargs.items() if k != "experimental"},
                    _experimental=experimental,
                    **{k: v for k, v in kwargs.items() if k != "experimental"},
                )
                object.__setattr__(caps, "experimental", experimental)
                return caps

    # Create the capabilities object with our subclass
    caps = _FilteredServerCapabilities(_filter_kwargs=kwargs, _experimental=experimental, **kwargs)

    return caps


__all__ = ["create_server_capabilities"]
