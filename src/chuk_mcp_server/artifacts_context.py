# chuk_mcp_server/artifacts_context.py
"""
Artifact and workspace context management for MCP servers.

Provides context variable management for the unified VFS-backed artifact/workspace
system from chuk-artifacts. This allows any MCP server built on chuk-mcp-server to
access artifact and workspace functionality.

Usage:
    from chuk_mcp_server import (
        get_artifact_store,
        set_artifact_store,
        NamespaceType,
        StorageScope,
    )

    # In your tool
    @tool
    async def store_file(content: bytes, filename: str) -> str:
        store = get_artifact_store()

        # Create blob namespace
        ns = await store.create_namespace(
            type=NamespaceType.BLOB,
            scope=StorageScope.SESSION
        )

        # Write data
        await store.write_namespace(ns.namespace_id, data=content)

        return ns.namespace_id
"""

from __future__ import annotations

import logging
import threading
from contextvars import ContextVar
from typing import TYPE_CHECKING, Any, cast

if TYPE_CHECKING:
    try:
        from chuk_artifacts import ArtifactStore, NamespaceInfo, StorageScope
        from chuk_virtual_fs import AsyncVirtualFileSystem
    except ImportError:
        ArtifactStore = Any
        NamespaceInfo = Any
        StorageScope = Any
        AsyncVirtualFileSystem = Any

logger = logging.getLogger(__name__)

# Context variables for artifact/workspace management
_artifact_store: ContextVar[ArtifactStore | None] = ContextVar("artifact_store", default=None)

# Global singleton store (fallback when context not available).
# RLock lets the same thread re-enter (e.g. during testing teardown).
_global_artifact_store: ArtifactStore | None = None
_global_store_lock = threading.RLock()


def set_artifact_store(store: ArtifactStore) -> None:
    """
    Set the artifact store for the current context.

    Args:
        store: ArtifactStore instance

    Examples:
        >>> from chuk_artifacts import ArtifactStore
        >>> store = ArtifactStore()
        >>> set_artifact_store(store)
    """
    _artifact_store.set(store)
    logger.debug("Set artifact store in context")


def get_artifact_store() -> ArtifactStore:
    """
    Get the artifact store for the current context.

    Returns:
        ArtifactStore instance

    Raises:
        RuntimeError: If no store has been set in context or globally

    Examples:
        >>> from chuk_artifacts import ArtifactStore
        >>> store = ArtifactStore()
        >>> set_artifact_store(store)
        >>> retrieved = get_artifact_store()
    """
    # Try context variable first
    store = _artifact_store.get()
    if store is not None:
        return store

    # Fall back to global singleton
    with _global_store_lock:
        if _global_artifact_store is not None:
            return _global_artifact_store

    # No store available
    raise RuntimeError(
        "No artifact store has been set. Use set_artifact_store() or "
        "set_global_artifact_store() to configure an ArtifactStore instance."
    )


def set_global_artifact_store(store: ArtifactStore) -> None:
    """
    Set the global artifact store (fallback when context not available).

    Args:
        store: ArtifactStore instance

    Examples:
        >>> from chuk_artifacts import ArtifactStore
        >>> store = ArtifactStore(storage_provider="s3", bucket="my-bucket")
        >>> set_global_artifact_store(store)
    """
    global _global_artifact_store
    with _global_store_lock:
        _global_artifact_store = store
    logger.debug("Set global artifact store")


def has_artifact_store() -> bool:
    """
    Check if an artifact store is currently set.

    Returns:
        True if store is set in context or global, False otherwise

    Examples:
        >>> if has_artifact_store():
        ...     store = get_artifact_store()
        ...     # Use store...
    """
    # Check context
    if _artifact_store.get() is not None:
        return True

    # Check global
    return _global_artifact_store is not None


def clear_artifact_store() -> None:
    """
    Clear the artifact store from the current context and reset the global store.

    This is primarily useful for testing to ensure a clean state.

    Examples:
        >>> clear_artifact_store()
        >>> assert not has_artifact_store()
    """
    global _global_artifact_store
    _artifact_store.set(None)
    with _global_store_lock:
        _global_artifact_store = None
    logger.debug("Cleared artifact store from context and global")


# ============================================================================
# Convenience Functions
# ============================================================================


async def create_blob_namespace(
    scope: StorageScope | None = None,
    session_id: str | None = None,
    user_id: str | None = None,
    **kwargs: Any,
) -> NamespaceInfo:
    """
    Convenience function to create a blob namespace.

    ``session_id`` and ``user_id`` are read from the active request context
    when not supplied explicitly — so callers inside an MCP tool handler get
    the right scoping automatically without having to thread identifiers through
    every call site.  Explicit arguments always take precedence over context.

    Args:
        scope: Storage scope (defaults to SESSION)
        session_id: Session ID override; falls back to ``get_session_id()`` from context
        user_id: User ID override; falls back to ``get_user_id()`` from context
        **kwargs: Additional parameters forwarded to ``create_namespace``

    Returns:
        NamespaceInfo for the created blob namespace

    Examples:
        >>> # Session-scoped — session_id injected from request context
        >>> ns = await create_blob_namespace()

        >>> # User-scoped — user_id injected from OAuth context
        >>> ns = await create_blob_namespace(scope=StorageScope.USER)

        >>> # Explicit override (e.g. admin code acting on behalf of a user)
        >>> ns = await create_blob_namespace(scope=StorageScope.USER, user_id="alice")
    """
    from chuk_artifacts import NamespaceType
    from chuk_artifacts import StorageScope as Scope
    from chuk_mcp_server.context import get_session_id, get_user_id

    if scope is None:
        scope = Scope.SESSION

    # Fall back to request-context identifiers when not supplied explicitly
    if session_id is None:
        session_id = get_session_id()
    if user_id is None:
        user_id = get_user_id()

    store = get_artifact_store()
    ns: NamespaceInfo = await store.create_namespace(
        type=NamespaceType.BLOB,
        scope=scope,
        session_id=session_id,
        user_id=user_id,
        **kwargs,
    )
    return ns


async def create_workspace_namespace(
    name: str,
    scope: StorageScope | None = None,
    session_id: str | None = None,
    user_id: str | None = None,
    provider_type: str = "vfs-memory",
    **kwargs: Any,
) -> NamespaceInfo:
    """
    Convenience function to create a workspace namespace.

    ``session_id`` and ``user_id`` are read from the active request context
    when not supplied explicitly — so callers inside an MCP tool handler get
    the right scoping automatically.  Explicit arguments always take precedence.

    Args:
        name: Workspace name
        scope: Storage scope (defaults to SESSION)
        session_id: Session ID override; falls back to ``get_session_id()`` from context
        user_id: User ID override; falls back to ``get_user_id()`` from context
        provider_type: VFS provider (vfs-memory, vfs-filesystem, vfs-s3, vfs-sqlite)
        **kwargs: Additional parameters forwarded to ``create_namespace``

    Returns:
        NamespaceInfo for the created workspace namespace

    Examples:
        >>> # Session-scoped — session_id injected from request context
        >>> ws = await create_workspace_namespace("my-project")

        >>> # User-scoped — user_id injected from OAuth context
        >>> ws = await create_workspace_namespace("my-project", scope=StorageScope.USER)

        >>> # Explicit override
        >>> ws = await create_workspace_namespace(
        ...     name="alice-project",
        ...     scope=StorageScope.USER,
        ...     user_id="alice",
        ...     provider_type="vfs-filesystem",
        ... )
    """
    from chuk_artifacts import NamespaceType
    from chuk_artifacts import StorageScope as Scope
    from chuk_mcp_server.context import get_session_id, get_user_id

    if scope is None:
        scope = Scope.SESSION

    # Fall back to request-context identifiers when not supplied explicitly
    if session_id is None:
        session_id = get_session_id()
    if user_id is None:
        user_id = get_user_id()

    store = get_artifact_store()
    ns: NamespaceInfo = await store.create_namespace(
        type=NamespaceType.WORKSPACE,
        name=name,
        scope=scope,
        session_id=session_id,
        user_id=user_id,
        provider_type=provider_type,
        **kwargs,
    )
    return ns


async def write_blob(namespace_id: str, data: bytes, mime: str | None = None) -> None:
    """
    Convenience function to write data to blob namespace.

    Args:
        namespace_id: Namespace ID
        data: Data to write
        mime: MIME type

    Examples:
        >>> ns = await create_blob_namespace()
        >>> await write_blob(ns.namespace_id, b"Hello World", mime="text/plain")
    """
    store = get_artifact_store()
    await store.write_namespace(namespace_id, data=data, mime=mime)


async def read_blob(namespace_id: str) -> bytes:
    """
    Convenience function to read data from blob namespace.

    Args:
        namespace_id: Namespace ID

    Returns:
        Blob data as bytes

    Examples:
        >>> data = await read_blob(namespace_id)
    """
    store = get_artifact_store()
    return cast(bytes, await store.read_namespace(namespace_id))


async def write_workspace_file(namespace_id: str, path: str, data: bytes) -> None:
    """
    Convenience function to write file to workspace namespace.

    Args:
        namespace_id: Namespace ID
        path: File path within workspace
        data: File data

    Examples:
        >>> ws = await create_workspace_namespace("my-project")
        >>> await write_workspace_file(ws.namespace_id, "/main.py", b"print('hello')")
    """
    store = get_artifact_store()
    await store.write_namespace(namespace_id, path=path, data=data)


async def read_workspace_file(namespace_id: str, path: str) -> bytes:
    """
    Convenience function to read file from workspace namespace.

    Args:
        namespace_id: Namespace ID
        path: File path within workspace

    Returns:
        File data as bytes

    Examples:
        >>> data = await read_workspace_file(ws.namespace_id, "/main.py")
    """
    store = get_artifact_store()
    return cast(bytes, await store.read_namespace(namespace_id, path=path))


def list_my_namespaces(
    scope: StorageScope | None = None,
    session_id: str | None = None,
    user_id: str | None = None,
) -> list[NamespaceInfo]:
    """
    List namespaces belonging to the current user or session.

    Unlike calling ``get_artifact_store().list_namespaces()`` directly, this
    helper automatically pulls ``session_id`` and ``user_id`` from the active
    request context and **refuses to list without at least one scoping
    identifier** (unless ``scope=StorageScope.SANDBOX`` is given).  This
    prevents the common mistake of accidentally returning every namespace in
    the store when user/session context is absent.

    Args:
        scope: Optional scope filter.  When ``StorageScope.SANDBOX`` is passed
            the listing is unfiltered by design (sandbox is shared).
        session_id: Session ID override; falls back to ``get_session_id()``
        user_id: User ID override; falls back to ``get_user_id()``

    Returns:
        List of NamespaceInfo matching the current user / session scope.

    Raises:
        RuntimeError: When both ``session_id`` and ``user_id`` are ``None``
            and the scope is not SANDBOX.  Call
            ``get_artifact_store().list_namespaces()`` directly if you
            intentionally need an unscoped listing.

    Examples:
        >>> # List namespaces for the current session (session_id from context)
        >>> my_ns = list_my_namespaces()

        >>> # List only user-scoped namespaces (user_id from OAuth context)
        >>> from chuk_artifacts import StorageScope
        >>> my_ns = list_my_namespaces(scope=StorageScope.USER)

        >>> # Explicit override — admin acting on behalf of a specific user
        >>> my_ns = list_my_namespaces(user_id="alice")
    """
    from chuk_artifacts import StorageScope as Scope
    from chuk_mcp_server.context import get_session_id, get_user_id

    # Fall back to request-context identifiers when not supplied explicitly
    if session_id is None:
        session_id = get_session_id()
    if user_id is None:
        user_id = get_user_id()

    store = get_artifact_store()

    # SANDBOX scope is shared by design — listing without user/session is fine
    try:
        if scope is not None and scope == Scope.SANDBOX:
            return store.list_namespaces(scope=scope)
    except Exception:
        pass  # scope comparison may fail if Scope is unavailable; fall through

    # Every other scope requires at least one identifier to prevent full-bucket exposure
    if session_id is None and user_id is None:
        raise RuntimeError(
            "list_my_namespaces() requires either a session or user context. "
            "This prevents accidentally listing all namespaces across all users. "
            "Use get_artifact_store().list_namespaces() directly if you intentionally "
            "need an unscoped listing (e.g. for SANDBOX-scope items)."
        )

    kwargs: dict[str, Any] = {}
    if scope is not None:
        kwargs["scope"] = scope
    if user_id is not None:
        kwargs["user_id"] = user_id
    if session_id is not None:
        kwargs["session_id"] = session_id

    return store.list_namespaces(**kwargs)


def get_namespace_vfs(namespace_id: str) -> AsyncVirtualFileSystem:
    """
    Get VFS instance for namespace.

    Args:
        namespace_id: Namespace ID

    Returns:
        AsyncVirtualFileSystem instance

    Examples:
        >>> vfs = get_namespace_vfs(namespace_id)
        >>> await vfs.write_file("/test.txt", b"content")
        >>> entries = await vfs.list_directory("/")
    """
    store = get_artifact_store()
    vfs: AsyncVirtualFileSystem = store.get_namespace_vfs(namespace_id)
    return vfs


__all__ = [
    # Context management
    "set_artifact_store",
    "get_artifact_store",
    "set_global_artifact_store",
    "has_artifact_store",
    # Namespace creation (context-aware)
    "create_blob_namespace",
    "create_workspace_namespace",
    # Namespace listing (context-aware, isolation-safe)
    "list_my_namespaces",
    # Data I/O
    "write_blob",
    "read_blob",
    "write_workspace_file",
    "read_workspace_file",
    "get_namespace_vfs",
]
