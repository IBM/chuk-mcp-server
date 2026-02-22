"""
Request context management for MCP servers.

Provides thread-safe, async-safe context storage for request-scoped data including:
- Session ID (MCP session identifier)
- User ID (OAuth user identifier)
- Progress token (for progress notifications)
- Custom metadata
"""

from collections.abc import Callable
from contextvars import ContextVar
from typing import Any

from starlette.types import Scope

# ============================================================================
# Context Variables
# ============================================================================

_session_id: ContextVar[str | None] = ContextVar("session_id", default=None)
_user_id: ContextVar[str | None] = ContextVar("user_id", default=None)
_progress_token: ContextVar[str | int | None] = ContextVar("progress_token", default=None)
_metadata: ContextVar[dict[str, Any] | None] = ContextVar("metadata", default=None)
_http_request: ContextVar[Scope | None] = ContextVar("http_request", default=None)
_sampling_fn: ContextVar[Callable[..., Any] | None] = ContextVar("sampling_fn", default=None)
_elicitation_fn: ContextVar[Callable[..., Any] | None] = ContextVar("elicitation_fn", default=None)
_progress_notify_fn: ContextVar[Callable[..., Any] | None] = ContextVar("progress_notify_fn", default=None)
_roots_fn: ContextVar[Callable[..., Any] | None] = ContextVar("roots_fn", default=None)
_log_fn: ContextVar[Callable[..., Any] | None] = ContextVar("log_fn", default=None)
_resource_links: ContextVar[list[dict[str, Any]] | None] = ContextVar("resource_links", default=None)


# ============================================================================
# Context Manager
# ============================================================================


class RequestContext:
    """
    Async context manager for MCP request context lifecycle.

    Automatically manages context setup and cleanup for request handling.
    Supports nested contexts (inner context takes precedence).
    """

    def __init__(
        self,
        session_id: str | None = None,
        user_id: str | None = None,
        progress_token: str | int | None = None,
        metadata: dict[str, Any] | None = None,
    ):
        """
        Initialize request context.

        Args:
            session_id: MCP session identifier
            user_id: OAuth user identifier
            progress_token: Progress notification token
            metadata: Additional request metadata
        """
        self.session_id = session_id
        self.user_id = user_id
        self.progress_token = progress_token
        self.metadata = metadata or {}

        # Store previous context for restoration
        self._previous_session_id: str | None = None
        self._previous_user_id: str | None = None
        self._previous_progress_token: str | int | None = None
        self._previous_metadata: dict[str, Any] | None = None

    async def __aenter__(self) -> "RequestContext":
        """Enter context - save previous and set new values."""
        # Save previous context
        self._previous_session_id = _session_id.get()
        self._previous_user_id = _user_id.get()
        self._previous_progress_token = _progress_token.get()
        prev_metadata = _metadata.get()
        self._previous_metadata = prev_metadata.copy() if prev_metadata is not None else None

        # Set new context
        if self.session_id is not None:
            _session_id.set(self.session_id)
        if self.user_id is not None:
            _user_id.set(self.user_id)
        if self.progress_token is not None:
            _progress_token.set(self.progress_token)
        if self.metadata:
            _metadata.set(self.metadata)

        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> bool:
        """Exit context - restore previous values."""
        # Restore previous context
        _session_id.set(self._previous_session_id)
        _user_id.set(self._previous_user_id)
        _progress_token.set(self._previous_progress_token)
        _metadata.set(self._previous_metadata)

        return False  # Don't suppress exceptions


# ============================================================================
# Session Context Functions
# ============================================================================


def get_session_id() -> str | None:
    """
    Get current MCP session ID.

    Returns:
        Session ID if set, None otherwise
    """
    return _session_id.get()


def set_session_id(session_id: str | None) -> None:
    """
    Set current MCP session ID.

    Args:
        session_id: Session identifier
    """
    _session_id.set(session_id)


def require_session_id() -> str:
    """
    Require a session ID to be set in context.

    Returns:
        Session ID string

    Raises:
        RuntimeError: If no session is active
    """
    session_id = _session_id.get()
    if not session_id:
        raise RuntimeError("Session context required. This function must be called within an active MCP session.")
    return session_id


# ============================================================================
# User Context Functions
# ============================================================================


def get_user_id() -> str | None:
    """
    Get current OAuth user ID.

    Returns:
        User ID if authenticated, None otherwise
    """
    return _user_id.get()


def set_user_id(user_id: str | None) -> None:
    """
    Set current OAuth user ID.

    Args:
        user_id: User identifier from OAuth
    """
    _user_id.set(user_id)


def require_user_id() -> str:
    """
    Require a user ID to be set in context.

    Useful for tools/resources that require authentication.

    Returns:
        User ID string

    Raises:
        PermissionError: If no user is authenticated
    """
    user_id = _user_id.get()
    if not user_id:
        raise PermissionError(
            "User authentication required. "
            "This operation requires an authenticated user context. "
            "Ensure OAuth authentication is configured and the user is logged in."
        )
    return user_id


# ============================================================================
# Progress Token Functions
# ============================================================================


def get_progress_token() -> str | int | None:
    """
    Get current progress token for notifications.

    Returns:
        Progress token if set, None otherwise
    """
    return _progress_token.get()


def set_progress_token(token: str | int | None) -> None:
    """
    Set current progress token.

    Args:
        token: Progress notification token
    """
    _progress_token.set(token)


# ============================================================================
# Metadata Functions
# ============================================================================


def get_metadata() -> dict[str, Any]:
    """
    Get current request metadata.

    Returns:
        Metadata dictionary (copy to prevent mutation)
    """
    metadata = _metadata.get()
    return metadata.copy() if metadata is not None else {}


def set_metadata(metadata: dict[str, Any]) -> None:
    """
    Set current request metadata.

    Args:
        metadata: Request metadata dictionary
    """
    _metadata.set(metadata.copy())


def update_metadata(key: str, value: Any) -> None:
    """
    Update a single metadata key.

    Args:
        key: Metadata key
        value: Metadata value
    """
    current_meta = _metadata.get()
    current = current_meta.copy() if current_meta is not None else {}
    current[key] = value
    _metadata.set(current)


def clear_metadata() -> None:
    """Clear all metadata."""
    _metadata.set(None)


# ============================================================================
# Context Utilities
# ============================================================================


def clear_all() -> None:
    """
    Clear all context variables.

    Useful for testing or cleanup.
    """
    _session_id.set(None)
    _user_id.set(None)
    _progress_token.set(None)
    _metadata.set(None)
    _http_request.set(None)
    _sampling_fn.set(None)
    _elicitation_fn.set(None)
    _progress_notify_fn.set(None)
    _roots_fn.set(None)
    _log_fn.set(None)
    _resource_links.set(None)


def get_current_context() -> dict[str, Any]:
    """
    Get all current context values.

    Returns:
        Dictionary with all context values
    """
    current_meta = _metadata.get()
    return {
        "session_id": _session_id.get(),
        "user_id": _user_id.get(),
        "progress_token": _progress_token.get(),
        "metadata": current_meta.copy() if current_meta is not None else {},
        "sampling_available": _sampling_fn.get() is not None,
        "elicitation_available": _elicitation_fn.get() is not None,
        "roots_available": _roots_fn.get() is not None,
    }


# ============================================================================
# Sampling Context Functions
# ============================================================================


def get_sampling_fn() -> Callable[..., Any] | None:
    """
    Get the current sampling function.

    Returns:
        Sampling function if set, None otherwise
    """
    return _sampling_fn.get()


def set_sampling_fn(fn: Callable[..., Any] | None) -> None:
    """
    Set the current sampling function.

    Called by transport layers to provide the sampling callback
    that sends requests to the MCP client.

    Args:
        fn: Async callable that sends sampling/createMessage to the client
    """
    _sampling_fn.set(fn)


# ============================================================================
# Elicitation Context Functions
# ============================================================================


def get_elicitation_fn() -> Callable[..., Any] | None:
    """
    Get the current elicitation function.

    Returns:
        Elicitation function if set, None otherwise
    """
    return _elicitation_fn.get()


def set_elicitation_fn(fn: Callable[..., Any] | None) -> None:
    """
    Set the current elicitation function.

    Called by protocol handler to provide the elicitation callback.

    Args:
        fn: Async callable that sends elicitation/create to the client
    """
    _elicitation_fn.set(fn)


# ============================================================================
# Progress Notification Context Functions
# ============================================================================


def get_progress_notify_fn() -> Callable[..., Any] | None:
    """
    Get the current progress notification function.

    Returns:
        Progress notify function if set, None otherwise
    """
    return _progress_notify_fn.get()


def set_progress_notify_fn(fn: Callable[..., Any] | None) -> None:
    """
    Set the current progress notification function.

    Called by protocol handler to provide the progress callback.

    Args:
        fn: Async callable that sends notifications/progress to the client
    """
    _progress_notify_fn.set(fn)


# ============================================================================
# Roots Context Functions
# ============================================================================


def get_roots_fn() -> Callable[..., Any] | None:
    """
    Get the current roots function.

    Returns:
        Roots function if set, None otherwise
    """
    return _roots_fn.get()


def set_roots_fn(fn: Callable[..., Any] | None) -> None:
    """
    Set the current roots function.

    Called by protocol handler to provide the roots callback.

    Args:
        fn: Async callable that sends roots/list to the client
    """
    _roots_fn.set(fn)


# ============================================================================
# Server-to-Client API Functions
# ============================================================================


async def create_elicitation(
    message: str,
    schema: dict[str, Any],
    title: str | None = None,
    description: str | None = None,
) -> dict[str, Any]:
    """
    Request structured user input from the MCP client.

    This sends an elicitation/create request to the client, which presents
    a form or dialog to the user and returns their structured response.

    Args:
        message: Human-readable explanation of what input is needed
        schema: JSON Schema defining the expected response structure
        title: Optional title for the input dialog
        description: Optional longer description

    Returns:
        Dict with the user's structured response

    Raises:
        RuntimeError: If elicitation is not available (client doesn't support it
            or not running within an MCP request context)
    """
    fn = _elicitation_fn.get()
    if fn is None:
        raise RuntimeError(
            "MCP elicitation is not available. The client may not support elicitation, "
            "or this tool is not running within an MCP request context."
        )
    result: dict[str, Any] = await fn(message=message, schema=schema, title=title, description=description)
    return result


async def send_progress(
    progress: float,
    total: float | None = None,
    message: str | None = None,
) -> None:
    """
    Send a progress notification to the MCP client.

    This is a no-op if progress notifications are not available or no
    progress token was provided in the request. Safe to call unconditionally.

    Args:
        progress: Current progress value
        total: Optional total expected progress value
        message: Optional human-readable progress message
    """
    fn = _progress_notify_fn.get()
    if fn is None:
        return  # Progress is optional - silent no-op
    token = _progress_token.get()
    if token is None:
        return  # No progress token requested
    await fn(progress_token=token, progress=progress, total=total, message=message)


async def list_roots() -> list[dict[str, Any]]:
    """
    Request the MCP client's filesystem roots.

    Returns a list of root directories/files that the client has made available.

    Returns:
        List of Root dicts with uri and optional name

    Raises:
        RuntimeError: If roots is not available (client doesn't support it
            or not running within an MCP request context)
    """
    fn = _roots_fn.get()
    if fn is None:
        raise RuntimeError(
            "MCP roots is not available. The client may not support roots, "
            "or this tool is not running within an MCP request context."
        )
    result: list[dict[str, Any]] = await fn()
    return result


async def create_message(
    messages: list[Any],
    max_tokens: int = 1000,
    system_prompt: str | None = None,
    temperature: float | None = None,
    model_preferences: Any = None,
    stop_sequences: list[str] | None = None,
    metadata: dict[str, Any] | None = None,
    include_context: str | None = None,
    tools: list[dict[str, Any]] | None = None,
    tool_choice: dict[str, Any] | str | None = None,
) -> dict[str, Any]:
    """
    Request the MCP client to sample from its LLM.

    This sends a sampling/createMessage request to the client and returns
    the LLM's response. The client must support MCP sampling capability.

    Args:
        messages: List of SamplingMessage dicts with role and content
        max_tokens: Maximum tokens for the LLM response
        system_prompt: Optional system prompt for the LLM
        temperature: Optional sampling temperature
        model_preferences: Optional model selection preferences
        stop_sequences: Optional stop sequences
        metadata: Optional request metadata
        include_context: Optional context inclusion ("none", "thisServer", "allServers")

    Returns:
        Dict with role, content, model, and stopReason from the client's LLM

    Raises:
        RuntimeError: If sampling is not available (client doesn't support it
            or not running within an MCP request context)
    """
    fn = _sampling_fn.get()
    if fn is None:
        raise RuntimeError(
            "MCP sampling is not available. The client may not support sampling, "
            "or this tool is not running within an MCP request context."
        )
    result: dict[str, Any] = await fn(
        messages=messages,
        max_tokens=max_tokens,
        system_prompt=system_prompt,
        temperature=temperature,
        model_preferences=model_preferences,
        stop_sequences=stop_sequences,
        metadata=metadata,
        include_context=include_context,
        tools=tools,
        tool_choice=tool_choice,
    )
    return result


# ============================================================================
# Log Notification Context Functions
# ============================================================================


def get_log_fn() -> Callable[..., Any] | None:
    """Get the current log notification function."""
    return _log_fn.get()


def set_log_fn(fn: Callable[..., Any] | None) -> None:
    """Set the current log notification function."""
    _log_fn.set(fn)


async def send_log(
    level: str,
    data: Any,
    logger_name: str | None = None,
) -> None:
    """
    Send a log message notification to the MCP client.

    This sends a notifications/message to the client. Safe to call
    unconditionally - it's a no-op if logging notifications are not available.

    Args:
        level: Log level (debug, info, notice, warning, error, critical, alert, emergency)
        data: Log data (any JSON-serializable value)
        logger_name: Optional logger name for filtering
    """
    fn = _log_fn.get()
    if fn is None:
        return  # Logging is optional - silent no-op
    await fn(level=level, data=data, logger_name=logger_name)


# ============================================================================
# Resource Links Context Functions
# ============================================================================


def get_resource_links() -> list[dict[str, Any]] | None:
    """Get resource links accumulated during tool execution."""
    return _resource_links.get()


def set_resource_links(links: list[dict[str, Any]] | None) -> None:
    """Set resource links (used by protocol handler for cleanup)."""
    _resource_links.set(links)


def add_resource_link(
    uri: str,
    name: str | None = None,
    description: str | None = None,
    mime_type: str | None = None,
) -> None:
    """
    Add a resource link to be included in the tool result.

    Tools call this to associate resource references with their output.

    Args:
        uri: The resource URI.
        name: Optional human-readable name.
        description: Optional description.
        mime_type: Optional MIME type.
    """
    links = _resource_links.get()
    if links is None:
        links = []
        _resource_links.set(links)
    link: dict[str, Any] = {"uri": uri}
    if name is not None:
        link["name"] = name
    if description is not None:
        link["description"] = description
    if mime_type is not None:
        link["mimeType"] = mime_type
    links.append(link)


# ============================================================================
# HTTP Request Context Functions
# ============================================================================


def get_http_request() -> Scope | None:
    """
    Get current http request

    Returns:
        HTTP Request object
    """
    return _http_request.get()


def set_http_request(request: Scope) -> None:
    """
    Set current http request

    Args:
        request: HTTP Request object
    """
    _http_request.set(request)


# ============================================================================
# Convenience Aliases (for backward compatibility)
# ============================================================================

# These aliases maintain compatibility with existing code
set_current_user = set_user_id
get_current_user_id = require_user_id  # Note: This raises if not set, matching old behavior
