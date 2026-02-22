#!/usr/bin/env python3
# src/chuk_mcp_server/types/content.py
"""
Content - Content formatting with orjson optimization

This module provides content formatting functions using chuk_mcp types
with orjson optimization for maximum performance.
"""

from typing import Any

import orjson
from pydantic import BaseModel

from .base import (
    Annotations,
    AudioContent,
    EmbeddedResource,
    ImageContent,
    TextContent,
    content_to_dict,
    create_text_content,
)


def format_content(content: Any, annotations: Annotations | dict[str, Any] | None = None) -> list[dict[str, Any]]:
    """Format content using chuk_mcp types with orjson optimization.

    Args:
        content: The content to format (str, dict, MCP content types, Pydantic models, lists).
        annotations: Optional MCP content annotations (audience, priority).
    """
    if isinstance(content, str):
        text_content = create_text_content(content)
        result = [content_to_dict(text_content)]
    elif isinstance(content, dict):
        json_str = orjson.dumps(content, option=orjson.OPT_INDENT_2).decode()
        text_content = create_text_content(json_str)
        result = [content_to_dict(text_content)]
    elif isinstance(content, TextContent | ImageContent | AudioContent | EmbeddedResource):
        # Check MCP content types before generic BaseModel
        result = [content_to_dict(content)]
    elif isinstance(content, BaseModel):
        # Handle other Pydantic models by converting to dict first
        json_str = orjson.dumps(content.model_dump(), option=orjson.OPT_INDENT_2).decode()
        text_content = create_text_content(json_str)
        result = [content_to_dict(text_content)]
    elif isinstance(content, list):
        items = []
        for item in content:
            items.extend(format_content(item))
        result = items
    else:
        text_content = create_text_content(str(content))
        result = [content_to_dict(text_content)]

    # Apply annotations to each content item if provided
    if annotations is not None:
        ann_dict = annotations.model_dump(exclude_none=True) if isinstance(annotations, Annotations) else annotations
        if ann_dict:
            for item in result:
                item["annotations"] = ann_dict

    return result


def format_content_as_text(content: Any) -> str:
    """Format any content as plain text."""
    if isinstance(content, str):
        return content
    elif isinstance(content, BaseModel):
        result: str = orjson.dumps(content.model_dump(), option=orjson.OPT_INDENT_2).decode()
        return result
    elif isinstance(content, dict | list):
        result2: str = orjson.dumps(content, option=orjson.OPT_INDENT_2).decode()
        return result2
    else:
        return str(content)


def format_content_as_json(content: Any) -> str:
    """Format any content as JSON string with orjson."""
    if isinstance(content, str):
        # Try to parse and re-format for consistency
        try:
            parsed = orjson.loads(content)
            result: str = orjson.dumps(parsed, option=orjson.OPT_INDENT_2).decode()
            return result
        except orjson.JSONDecodeError:
            # If not valid JSON, wrap in quotes
            result2: str = orjson.dumps(content).decode()
            return result2
    elif isinstance(content, BaseModel):
        result3: str = orjson.dumps(content.model_dump(), option=orjson.OPT_INDENT_2).decode()
        return result3
    else:
        result4: str = orjson.dumps(content, option=orjson.OPT_INDENT_2).decode()
        return result4


def create_annotated_content(
    content: Any,
    audience: list[str] | None = None,
    priority: float | None = None,
) -> list[dict[str, Any]]:
    """Create content with MCP annotations.

    Args:
        content: The content to format.
        audience: Who should see this content (e.g., ["user"], ["assistant"], ["user", "assistant"]).
        priority: Importance from 0.0 (optional) to 1.0 (required).

    Returns:
        List of content dicts with annotations attached.
    """
    ann: dict[str, Any] = {}
    if audience is not None:
        ann["audience"] = audience
    if priority is not None:
        ann["priority"] = priority
    return format_content(content, annotations=ann if ann else None)


def create_resource_link(
    uri: str,
    name: str | None = None,
    description: str | None = None,
    mime_type: str | None = None,
) -> dict[str, Any]:
    """Create an MCP ResourceLink dict for inclusion in tool results.

    Args:
        uri: The resource URI.
        name: Optional human-readable name.
        description: Optional description.
        mime_type: Optional MIME type of the linked resource.

    Returns:
        A dict matching the MCP ResourceLink schema.
    """
    link: dict[str, Any] = {"uri": uri}
    if name is not None:
        link["name"] = name
    if description is not None:
        link["description"] = description
    if mime_type is not None:
        link["mimeType"] = mime_type
    return link


__all__ = [
    "format_content",
    "format_content_as_text",
    "format_content_as_json",
    "create_annotated_content",
    "create_resource_link",
]
