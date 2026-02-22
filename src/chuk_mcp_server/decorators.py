#!/usr/bin/env python3
# src/chuk_mcp_server/decorators.py
"""
Simple decorators for tools and resources
"""

import inspect
import threading
from collections.abc import Callable
from functools import wraps
from typing import Any

from .types import PromptHandler, ResourceHandler, ToolHandler
from .types.resources import ResourceTemplateHandler

# ============================================================================
# Global Registry (for standalone decorators)
# ============================================================================

_registry_lock = threading.Lock()
_global_tools: list[ToolHandler] = []
_global_resources: list[ResourceHandler] = []
_global_prompts: list[PromptHandler] = []
_global_resource_templates: list[ResourceTemplateHandler] = []


def get_global_tools() -> list[ToolHandler]:
    """Get globally registered tools."""
    return _global_tools.copy()


def get_global_resources() -> list[ResourceHandler]:
    """Get globally registered resources."""
    return _global_resources.copy()


def get_global_prompts() -> list[PromptHandler]:
    """Get globally registered prompts."""
    return _global_prompts.copy()


def get_global_resource_templates() -> list[ResourceTemplateHandler]:
    """Get globally registered resource templates."""
    return _global_resource_templates.copy()


def get_global_registry() -> dict[str, list[Any]]:
    """Get the entire global registry."""
    return {
        "tools": _global_tools.copy(),
        "resources": _global_resources.copy(),
        "prompts": _global_prompts.copy(),
        "resource_templates": _global_resource_templates.copy(),
    }


def clear_global_registry() -> None:
    """Clear global registry (useful for testing)."""
    with _registry_lock:
        _global_tools.clear()
        _global_resources.clear()
        _global_prompts.clear()
        _global_resource_templates.clear()


# ============================================================================
# Tool Decorator
# ============================================================================


def tool(
    name: str | None = None,
    description: str | None = None,
    read_only_hint: bool | None = None,
    destructive_hint: bool | None = None,
    idempotent_hint: bool | None = None,
    open_world_hint: bool | None = None,
    output_schema: dict[str, Any] | None = None,
    icons: list[dict[str, Any]] | None = None,
) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """
    Decorator to register a function as an MCP tool.

    Usage:
        @tool
        def hello(name: str) -> str:
            return f"Hello, {name}!"

        @tool(name="custom_name", description="Custom description")
        def my_func(x: int, y: int = 10) -> int:
            return x + y

        @tool(read_only_hint=True, idempotent_hint=True)
        def safe_lookup(key: str) -> str:
            return db[key]
    """

    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        # Create tool from function
        mcp_tool = ToolHandler.from_function(
            func,
            name=name,
            description=description,
            read_only_hint=read_only_hint,
            destructive_hint=destructive_hint,
            idempotent_hint=idempotent_hint,
            open_world_hint=open_world_hint,
            output_schema=output_schema,
            icons=icons,
        )

        # Register globally
        with _registry_lock:
            _global_tools.append(mcp_tool)

        # Add tool metadata to function
        func._mcp_tool = mcp_tool  # type: ignore[attr-defined]

        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            return func(*args, **kwargs)

        return wrapper

    # Handle both @tool and @tool() usage
    if callable(name):
        # @tool usage (no parentheses)
        func = name
        name = None
        return decorator(func)
    else:
        # @tool() or @tool(name="...") usage
        return decorator


# ============================================================================
# Resource Decorator
# ============================================================================


def resource(
    uri: str,
    name: str | None = None,
    description: str | None = None,
    mime_type: str = "text/plain",
    icons: list[dict[str, Any]] | None = None,
) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """
    Decorator to register a function as an MCP resource.

    Usage:
        @resource("config://settings")
        def get_settings() -> dict:
            return {"app": "my_app", "version": "1.0"}

        @resource("file://readme", mime_type="text/markdown")
        def get_readme() -> str:
            return "# My Application\\n\\nThis is awesome!"
    """

    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        # Create resource from function
        mcp_resource = ResourceHandler.from_function(
            uri=uri, func=func, name=name, description=description, mime_type=mime_type, icons=icons
        )

        # Register globally
        with _registry_lock:
            _global_resources.append(mcp_resource)

        # Add resource metadata to function
        func._mcp_resource = mcp_resource  # type: ignore[attr-defined]

        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            return func(*args, **kwargs)

        return wrapper

    return decorator


# ============================================================================
# Prompt Decorator
# ============================================================================


def prompt(
    name: str | None = None,
    description: str | None = None,
    icons: list[dict[str, Any]] | None = None,
) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """
    Decorator to register a function as an MCP prompt.

    Usage:
        @prompt
        def code_review(code: str, language: str = "python") -> str:
            return f"Please review this {language} code:\\n\\n{code}"

        @prompt(name="custom_prompt", description="Custom prompt template")
        def my_prompt(topic: str, style: str = "formal") -> str:
            return f"Write about {topic} in a {style} style"
    """

    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        # Create prompt from function
        mcp_prompt = PromptHandler.from_function(func, name=name, description=description, icons=icons)

        # Register globally
        with _registry_lock:
            _global_prompts.append(mcp_prompt)

        # Add prompt metadata to function
        func._mcp_prompt = mcp_prompt  # type: ignore[attr-defined]

        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            return func(*args, **kwargs)

        return wrapper

    # Handle both @prompt and @prompt() usage
    if callable(name):
        # @prompt usage (no parentheses)
        func = name
        name = None
        return decorator(func)
    else:
        # @prompt() or @prompt(name="...") usage
        return decorator


# ============================================================================
# Resource Template Decorator
# ============================================================================


def resource_template(
    uri_template: str,
    name: str | None = None,
    description: str | None = None,
    mime_type: str | None = None,
    icons: list[dict[str, Any]] | None = None,
) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """
    Decorator to register a function as an MCP resource template (RFC 6570).

    Usage:
        @resource_template("users://{user_id}/profile")
        def get_user_profile(user_id: str) -> dict:
            return {"id": user_id, "name": "Alice"}

        @resource_template(
            "files://{path}",
            name="File Reader",
            mime_type="text/plain",
        )
        def read_file(path: str) -> str:
            return open(path).read()
    """

    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        # Create resource template from function
        mcp_template = ResourceTemplateHandler.from_function(
            uri_template=uri_template,
            func=func,
            name=name,
            description=description,
            mime_type=mime_type,
            icons=icons,
        )

        # Register globally
        with _registry_lock:
            _global_resource_templates.append(mcp_template)

        # Add template metadata to function
        func._mcp_resource_template = mcp_template  # type: ignore[attr-defined]

        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            return func(*args, **kwargs)

        return wrapper

    return decorator


# ============================================================================
# Authorization Decorator
# ============================================================================


def requires_auth(scopes: list[str] | None = None) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """
    Decorator to mark a tool as requiring OAuth authorization.

    The protocol handler will validate the OAuth token before executing
    the tool and inject the external provider's access token.

    Usage:
        @requires_auth()
        async def linkedin_publish(
            visibility: str = "PUBLIC",
            _external_access_token: Optional[str] = None,
        ) -> str:
            # Use _external_access_token to call external API
            pass

        @requires_auth(scopes=["posts.write", "profile.read"])
        async def advanced_tool(_external_access_token: Optional[str] = None) -> str:
            pass

    Args:
        scopes: Optional list of required OAuth scopes

    Returns:
        Decorator function
    """

    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        # Set authorization metadata on the function
        func._requires_auth = True  # type: ignore[attr-defined]
        func._auth_scopes = scopes  # type: ignore[attr-defined]

        @wraps(func)
        async def wrapper(*args: Any, **kwargs: Any) -> Any:
            # Just pass through - actual auth check happens in protocol handler
            if inspect.iscoroutinefunction(func):
                return await func(*args, **kwargs)
            else:
                return func(*args, **kwargs)

        # Copy metadata to wrapper
        wrapper._requires_auth = True  # type: ignore[attr-defined]
        wrapper._auth_scopes = scopes  # type: ignore[attr-defined]

        return wrapper

    # Handle both @requires_auth and @requires_auth() usage
    if callable(scopes):
        # @requires_auth usage (no parentheses)
        func = scopes
        scopes = None
        return decorator(func)
    else:
        # @requires_auth() or @requires_auth(scopes=[...]) usage
        return decorator


# ============================================================================
# Helper Functions
# ============================================================================


def is_tool(func: Callable[..., Any]) -> bool:
    """Check if a function is decorated as a tool."""
    return hasattr(func, "_mcp_tool")


def is_resource(func: Callable[..., Any]) -> bool:
    """Check if a function is decorated as a resource."""
    return hasattr(func, "_mcp_resource")


def is_prompt(func: Callable[..., Any]) -> bool:
    """Check if a function is decorated as a prompt."""
    return hasattr(func, "_mcp_prompt")


def get_tool_from_function(func: Callable[..., Any]) -> ToolHandler | None:
    """Get the tool metadata from a decorated function."""
    return getattr(func, "_mcp_tool", None)


def get_resource_from_function(func: Callable[..., Any]) -> ResourceHandler | None:
    """Get the resource metadata from a decorated function."""
    return getattr(func, "_mcp_resource", None)


def get_prompt_from_function(func: Callable[..., Any]) -> PromptHandler | None:
    """Get the prompt metadata from a decorated function."""
    return getattr(func, "_mcp_prompt", None)
