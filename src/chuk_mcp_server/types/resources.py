#!/usr/bin/env python3
# src/chuk_mcp_server/types/resources.py
"""
Resources - ResourceHandler with orjson optimization and content caching

This module provides the ResourceHandler class with content caching,
orjson serialization, and MIME type-aware content formatting.
"""

import inspect
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, cast

import orjson
from pydantic import BaseModel

# types
from .base import MCPError, MCPResource

# ============================================================================
# ResourceHandler with orjson Optimization
# ============================================================================


@dataclass
class ResourceHandler:
    """Framework resource handler with orjson optimization and caching."""

    mcp_resource: MCPResource  # The actual MCP resource
    handler: Callable[..., Any]
    cache_ttl: int | None = None  # Cache TTL in seconds
    _cached_mcp_format: dict[str, Any] | None = None  # Cache the MCP format dict
    _cached_mcp_bytes: bytes | None = None  # 🚀 Cache orjson-serialized bytes
    icons: list[dict[str, Any]] | None = None  # MCP icons (2025-11-25)

    def __post_init__(self) -> None:
        self._cached_content: str | None = None
        self._cache_timestamp: float | None = None
        # Pre-cache both dict and orjson formats for resources
        if self._cached_mcp_format is None:
            fmt = self.mcp_resource.model_dump(exclude_none=True)
            # Add icons if present (MCP 2025-11-25)
            if self.icons:
                fmt["icons"] = [icon.copy() for icon in self.icons]
            self._cached_mcp_format = fmt
        if self._cached_mcp_bytes is None:
            self._cached_mcp_bytes = orjson.dumps(self._cached_mcp_format)

    @classmethod
    def from_function(
        cls,
        uri: str,
        func: Callable[..., Any],
        name: str | None = None,
        description: str | None = None,
        mime_type: str = "text/plain",
        cache_ttl: int | None = None,
        icons: list[dict[str, Any]] | None = None,
    ) -> "ResourceHandler":
        """Create ResourceHandler from a function."""
        resource_name = name or func.__name__.replace("_", " ").title()
        resource_description = description or func.__doc__ or f"Resource: {uri}"

        # Create the MCP resource
        mcp_resource = MCPResource(uri=uri, name=resource_name, description=resource_description, mimeType=mime_type)

        return cls(
            mcp_resource=mcp_resource,
            handler=func,
            cache_ttl=cache_ttl,
            _cached_mcp_format=None,  # Will be computed in __post_init__
            _cached_mcp_bytes=None,  # Will be computed in __post_init__
            icons=icons,
        )

    @property
    def uri(self) -> str:
        """Get the resource URI."""
        return cast(str, self.mcp_resource.uri)

    @property
    def name(self) -> str:
        """Get the resource name."""
        return cast(str, self.mcp_resource.name)

    @property
    def description(self) -> str | None:
        """Get the resource description."""
        return cast("str | None", self.mcp_resource.description)

    @property
    def mime_type(self) -> str | None:
        """Get the resource MIME type."""
        return cast("str | None", self.mcp_resource.mimeType)

    def to_mcp_format(self) -> dict[str, Any]:
        """Convert to MCP resource format using cached version."""
        if self._cached_mcp_format is None:
            self._cached_mcp_format = self.mcp_resource.model_dump(exclude_none=True)
        return self._cached_mcp_format.copy()  # Return copy to prevent mutation

    def to_mcp_bytes(self) -> bytes:
        """🚀 Get orjson-serialized MCP format bytes for ultimate performance."""
        if self._cached_mcp_bytes is None:
            if self._cached_mcp_format is None:
                self._cached_mcp_format = self.mcp_resource.model_dump(exclude_none=True)
            self._cached_mcp_bytes = orjson.dumps(self._cached_mcp_format)
        return self._cached_mcp_bytes

    async def read(self) -> str:
        """Read the resource content with optional caching."""
        now = time.time()

        # Check cache validity
        if (
            self.cache_ttl
            and self._cached_content
            and self._cache_timestamp
            and now - self._cache_timestamp < self.cache_ttl
        ):
            return self._cached_content

        # Read fresh content
        try:
            if inspect.iscoroutinefunction(self.handler):
                result = await self.handler()
            else:
                result = self.handler()

            # Format content based on MIME type
            content = self._format_content(result)

            # Cache if TTL is set
            if self.cache_ttl:
                self._cached_content = content
                self._cache_timestamp = now

            return content

        except Exception as e:
            # Fix: MCPError requires a code parameter
            raise MCPError(f"Failed to read resource '{self.uri}': {str(e)}", code=-32603) from e

    def _format_content(self, result: Any) -> str:
        """Format content based on MIME type with orjson optimization."""
        mime_type = self.mime_type or "text/plain"

        # Handle Pydantic models by converting to dict first
        if isinstance(result, BaseModel):
            result = result.model_dump()

        if mime_type == "application/json":
            if isinstance(result, dict | list):
                # 🚀 Use orjson for 2-3x faster JSON serialization
                return str(orjson.dumps(result, option=orjson.OPT_INDENT_2).decode())
            else:
                return str(orjson.dumps(result).decode())
        elif mime_type == "text/markdown" or mime_type == "text/plain":
            if isinstance(result, dict | list):
                return str(orjson.dumps(result, option=orjson.OPT_INDENT_2).decode())
            return str(result)
        else:
            # For unknown MIME types, convert to string with orjson
            if isinstance(result, dict | list):
                return str(orjson.dumps(result, option=orjson.OPT_INDENT_2).decode())
            else:
                return str(result)

    def invalidate_cache(self) -> None:
        """Manually invalidate the cached content."""
        self._cached_content = None
        self._cache_timestamp = None

    def invalidate_mcp_cache(self) -> None:
        """Invalidate the cached MCP formats."""
        self._cached_mcp_format = None
        self._cached_mcp_bytes = None

    def is_cached(self) -> bool:
        """Check if content is currently cached and valid."""
        if not self.cache_ttl or not self._cached_content or not self._cache_timestamp:
            return False

        now = time.time()
        return (now - self._cache_timestamp) < self.cache_ttl

    def get_cache_info(self) -> dict[str, Any]:
        """Get cache status information."""
        if not self.cache_ttl:
            return {"caching": False, "reason": "no_ttl_set"}

        if not self._cached_content:
            return {"caching": True, "status": "empty", "ttl": self.cache_ttl}

        now = time.time()
        age = now - self._cache_timestamp if self._cache_timestamp else 0
        remaining = max(0, self.cache_ttl - age) if self._cache_timestamp else 0

        return {
            "caching": True,
            "status": "cached" if self.is_cached() else "expired",
            "ttl": self.cache_ttl,
            "age_seconds": round(age, 2),
            "remaining_seconds": round(remaining, 2),
            "content_length": len(self._cached_content) if self._cached_content else 0,
        }


# ============================================================================
# ResourceTemplateHandler
# ============================================================================


@dataclass
class ResourceTemplateHandler:
    """Handler for URI template-based resources (RFC 6570).

    Resource templates allow clients to discover parameterized resources.
    The handler function receives extracted template parameters as kwargs.
    """

    uri_template: str
    name: str
    handler: Callable[..., Any]
    description: str | None = None
    mime_type: str | None = None
    _cached_mcp_format: dict[str, Any] | None = None
    _cached_mcp_bytes: bytes | None = None
    icons: list[dict[str, Any]] | None = None  # MCP icons (2025-11-25)

    def __post_init__(self) -> None:
        if self._cached_mcp_format is None:
            fmt: dict[str, Any] = {
                "uriTemplate": self.uri_template,
                "name": self.name,
            }
            if self.description is not None:
                fmt["description"] = self.description
            if self.mime_type is not None:
                fmt["mimeType"] = self.mime_type
            # Add icons if present (MCP 2025-11-25)
            if self.icons:
                fmt["icons"] = [icon.copy() for icon in self.icons]
            self._cached_mcp_format = fmt
        if self._cached_mcp_bytes is None:
            self._cached_mcp_bytes = orjson.dumps(self._cached_mcp_format)

    @classmethod
    def from_function(
        cls,
        uri_template: str,
        func: Callable[..., Any],
        name: str | None = None,
        description: str | None = None,
        mime_type: str | None = None,
        icons: list[dict[str, Any]] | None = None,
    ) -> "ResourceTemplateHandler":
        """Create ResourceTemplateHandler from a function."""
        template_name = name or func.__name__.replace("_", " ").title()
        template_description = description or func.__doc__ or f"Resource template: {uri_template}"

        return cls(
            uri_template=uri_template,
            name=template_name,
            handler=func,
            description=template_description,
            mime_type=mime_type,
            icons=icons,
        )

    def to_mcp_format(self) -> dict[str, Any]:
        """Convert to MCP resource template format."""
        if self._cached_mcp_bytes is None:
            self.__post_init__()
        assert self._cached_mcp_bytes is not None
        result: dict[str, Any] = orjson.loads(self._cached_mcp_bytes)
        return result

    def to_mcp_bytes(self) -> bytes:
        """Get orjson-serialized MCP format bytes."""
        if self._cached_mcp_bytes is None:
            self.__post_init__()
        return self._cached_mcp_bytes  # type: ignore[return-value]

    async def read(self, **kwargs: Any) -> str:
        """Read resource content with template parameters."""
        try:
            if inspect.iscoroutinefunction(self.handler):
                result = await self.handler(**kwargs)
            else:
                result = self.handler(**kwargs)

            # Format result
            if isinstance(result, BaseModel):
                result = result.model_dump()
            if isinstance(result, dict | list):
                return str(orjson.dumps(result, option=orjson.OPT_INDENT_2).decode())
            return str(result)
        except Exception as e:
            raise MCPError(f"Failed to read resource template '{self.uri_template}': {str(e)}", code=-32603) from e


# ============================================================================
# Resource Creation Utilities
# ============================================================================


def create_resource_from_function(
    uri: str,
    func: Callable[..., Any],
    name: str | None = None,
    description: str | None = None,
    mime_type: str = "text/plain",
    cache_ttl: int | None = None,
) -> ResourceHandler:
    """Create a ResourceHandler from a function - convenience function."""
    return ResourceHandler.from_function(
        uri=uri, func=func, name=name, description=description, mime_type=mime_type, cache_ttl=cache_ttl
    )


def create_json_resource(
    uri: str,
    func: Callable[..., Any],
    name: str | None = None,
    description: str | None = None,
    cache_ttl: int | None = None,
) -> ResourceHandler:
    """Create a JSON resource handler - convenience function."""
    return ResourceHandler.from_function(
        uri=uri, func=func, name=name, description=description, mime_type="application/json", cache_ttl=cache_ttl
    )


def create_markdown_resource(
    uri: str,
    func: Callable[..., Any],
    name: str | None = None,
    description: str | None = None,
    cache_ttl: int | None = None,
) -> ResourceHandler:
    """Create a Markdown resource handler - convenience function."""
    return ResourceHandler.from_function(
        uri=uri, func=func, name=name, description=description, mime_type="text/markdown", cache_ttl=cache_ttl
    )


# ============================================================================
# Exports
# ============================================================================

__all__ = [
    "ResourceHandler",
    "ResourceTemplateHandler",
    "create_resource_from_function",
    "create_json_resource",
    "create_markdown_resource",
]
