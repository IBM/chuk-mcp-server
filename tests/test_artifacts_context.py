"""
Tests for artifacts_context module.

These tests verify the artifact store context management functionality.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# Try to import artifacts dependencies
try:
    from chuk_artifacts import ArtifactStore, NamespaceType, StorageScope

    ARTIFACTS_AVAILABLE = True
except ImportError:
    ARTIFACTS_AVAILABLE = False

# Import from __init__.py for stub-behavior tests (TestArtifactsNotAvailable)
from chuk_mcp_server import (
    get_artifact_store as _pkg_get_artifact_store,
)
from chuk_mcp_server import (
    has_artifact_store as _pkg_has_artifact_store,
)

# Import from artifacts_context directly for all other tests
# (these are the real implementations, not stubs)
from chuk_mcp_server.artifacts_context import (
    clear_artifact_store,
    create_blob_namespace,
    create_workspace_namespace,
    get_artifact_store,
    get_namespace_vfs,
    has_artifact_store,
    read_blob,
    read_workspace_file,
    set_artifact_store,
    set_global_artifact_store,
    write_blob,
    write_workspace_file,
)
from chuk_mcp_server.artifacts_context import list_my_namespaces  # noqa: F401 (used in new tests)


class TestArtifactsNotAvailable:
    """Tests for when chuk-artifacts is not installed."""

    def test_has_artifact_store_false_when_not_available(self):
        """Test has_artifact_store returns False when artifacts not installed."""
        if ARTIFACTS_AVAILABLE:
            pytest.skip("Artifacts is available, skipping unavailable test")
        assert _pkg_has_artifact_store() is False

    def test_get_artifact_store_raises_when_not_available(self):
        """Test get_artifact_store raises when artifacts not installed."""
        if ARTIFACTS_AVAILABLE:
            pytest.skip("Artifacts is available, skipping unavailable test")
        with pytest.raises(RuntimeError, match="chuk-artifacts"):
            _pkg_get_artifact_store()


@pytest.mark.skipif(not ARTIFACTS_AVAILABLE, reason="chuk-artifacts not installed")
class TestArtifactsContext:
    """Tests for artifact store context management."""

    def teardown_method(self):
        """Clean up after each test."""
        clear_artifact_store()

    def test_has_artifact_store_false_initially(self):
        """Test has_artifact_store returns False when no store set."""
        clear_artifact_store()
        assert has_artifact_store() is False

    def test_set_and_get_artifact_store(self):
        """Test setting and getting artifact store from context."""
        store = ArtifactStore()
        set_artifact_store(store)

        assert has_artifact_store() is True
        retrieved = get_artifact_store()
        assert retrieved is store

    def test_set_global_artifact_store(self):
        """Test setting global artifact store."""
        store = ArtifactStore()
        set_global_artifact_store(store)

        # Should be retrievable via get_artifact_store
        retrieved = get_artifact_store()
        assert retrieved is store

    def test_context_store_takes_precedence_over_global(self):
        """Test that context store takes precedence over global store."""
        global_store = ArtifactStore()
        context_store = ArtifactStore()

        set_global_artifact_store(global_store)
        set_artifact_store(context_store)

        retrieved = get_artifact_store()
        assert retrieved is context_store
        assert retrieved is not global_store

    def test_clear_artifact_store(self):
        """Test clearing artifact store from context."""
        store = ArtifactStore()
        set_artifact_store(store)

        assert has_artifact_store() is True

        clear_artifact_store()

        assert has_artifact_store() is False

    def test_get_artifact_store_without_store_raises(self):
        """Test get_artifact_store raises when no store available."""
        clear_artifact_store()

        with pytest.raises(RuntimeError, match="No artifact store"):
            get_artifact_store()


@pytest.mark.skipif(not ARTIFACTS_AVAILABLE, reason="chuk-artifacts not installed")
class TestConvenienceFunctions:
    """Tests for convenience functions."""

    def setup_method(self):
        """Set up artifact store for tests."""
        self.store = ArtifactStore()
        set_artifact_store(self.store)

    def teardown_method(self):
        """Clean up after tests."""
        clear_artifact_store()

    async def test_create_blob_namespace(self):
        """Test create_blob_namespace convenience function."""
        from chuk_mcp_server.artifacts_context import create_blob_namespace

        ns = await create_blob_namespace(scope=StorageScope.SANDBOX)

        assert ns is not None
        assert ns.type == NamespaceType.BLOB
        assert ns.scope == StorageScope.SANDBOX

    async def test_create_workspace_namespace(self):
        """Test create_workspace_namespace convenience function."""
        from chuk_mcp_server.artifacts_context import create_workspace_namespace

        ns = await create_workspace_namespace(name="test-workspace", scope=StorageScope.SANDBOX)

        assert ns is not None
        assert ns.type == NamespaceType.WORKSPACE
        assert ns.scope == StorageScope.SANDBOX
        assert ns.name == "test-workspace"

    async def test_write_blob(self):
        """Test write_blob convenience function."""
        from chuk_mcp_server.artifacts_context import create_blob_namespace, write_blob

        ns = await create_blob_namespace(scope=StorageScope.SANDBOX)
        await write_blob(ns.namespace_id, b"test data", mime="text/plain")

        # Verify it was written
        content = await self.store.read_namespace(ns.namespace_id)
        assert content == b"test data"

    async def test_read_blob(self):
        """Test read_blob convenience function."""
        from chuk_mcp_server.artifacts_context import create_blob_namespace, read_blob

        ns = await create_blob_namespace(scope=StorageScope.SANDBOX)
        await self.store.write_namespace(ns.namespace_id, data=b"test content")

        content = await read_blob(ns.namespace_id)
        assert content == b"test content"

    async def test_write_workspace_file(self):
        """Test write_workspace_file convenience function."""
        from chuk_mcp_server.artifacts_context import (
            create_workspace_namespace,
            write_workspace_file,
        )

        ns = await create_workspace_namespace(name="test-ws", scope=StorageScope.SANDBOX)
        await write_workspace_file(ns.namespace_id, "/test.txt", b"workspace data")

        # Verify it was written
        vfs = self.store.get_namespace_vfs(ns.namespace_id)
        content = await vfs.read_file("/test.txt")
        assert content == b"workspace data"

    async def test_read_workspace_file(self):
        """Test read_workspace_file convenience function."""
        from chuk_mcp_server.artifacts_context import (
            create_workspace_namespace,
            read_workspace_file,
        )

        ns = await create_workspace_namespace(name="test-ws", scope=StorageScope.SANDBOX)
        vfs = self.store.get_namespace_vfs(ns.namespace_id)
        await vfs.write_file("/test.txt", b"file content")

        content = await read_workspace_file(ns.namespace_id, "/test.txt")
        assert content == b"file content"

    def test_get_namespace_vfs(self):
        """Test get_namespace_vfs convenience function."""
        import asyncio

        from chuk_mcp_server.artifacts_context import get_namespace_vfs

        # Create a namespace

        async def create_ns():
            from chuk_mcp_server.artifacts_context import create_workspace_namespace

            return await create_workspace_namespace(name="vfs-test", scope=StorageScope.SANDBOX)

        ns = asyncio.run(create_ns())

        # Get VFS
        vfs = get_namespace_vfs(ns.namespace_id)
        assert vfs is not None


@pytest.mark.skipif(not ARTIFACTS_AVAILABLE, reason="chuk-artifacts not installed")
class TestDefaultScopes:
    """Tests for default scope handling."""

    def setup_method(self):
        """Set up artifact store for tests."""
        self.store = ArtifactStore()
        set_artifact_store(self.store)

    def teardown_method(self):
        """Clean up after tests."""
        clear_artifact_store()

    async def test_create_blob_namespace_default_scope(self):
        """Test create_blob_namespace uses SESSION scope by default."""
        from chuk_mcp_server.artifacts_context import create_blob_namespace

        # Need to allocate a session first
        session_manager = self.store._session_manager
        await session_manager.allocate_session(session_id="test-session")

        ns = await create_blob_namespace(session_id="test-session")

        assert ns.scope == StorageScope.SESSION

    async def test_create_workspace_namespace_default_scope(self):
        """Test create_workspace_namespace uses SESSION scope by default."""
        from chuk_mcp_server.artifacts_context import create_workspace_namespace

        # Need to allocate a session first
        session_manager = self.store._session_manager
        await session_manager.allocate_session(session_id="test-session-2")

        ns = await create_workspace_namespace(name="test-ws", session_id="test-session-2")

        assert ns.scope == StorageScope.SESSION


# ---------------------------------------------------------------------------
# Coverage tests (mock-based, no chuk-artifacts dependency required)
# ---------------------------------------------------------------------------

# Sentinel objects used as mock return values for NamespaceType / StorageScope
_MOCK_NAMESPACE_TYPE_BLOB = "BLOB_SENTINEL"
_MOCK_NAMESPACE_TYPE_WORKSPACE = "WORKSPACE_SENTINEL"
_MOCK_SCOPE_SESSION = "SESSION_SENTINEL"


def _scope_module():
    """Build a tiny fake chuk_artifacts module for patching."""
    mod = MagicMock()
    mod.NamespaceType.BLOB = _MOCK_NAMESPACE_TYPE_BLOB
    mod.NamespaceType.WORKSPACE = _MOCK_NAMESPACE_TYPE_WORKSPACE
    mod.StorageScope.SESSION = _MOCK_SCOPE_SESSION
    return mod


# ---------------------------------------------------------------------------
# set_artifact_store / get_artifact_store (coverage)
# ---------------------------------------------------------------------------
class TestSetAndGetArtifactStore:
    def setup_method(self):
        clear_artifact_store()

    def teardown_method(self):
        clear_artifact_store()

    def test_set_then_get_returns_same_store(self):
        store = MagicMock(name="store")
        set_artifact_store(store)
        assert get_artifact_store() is store

    def test_get_falls_back_to_global(self):
        store = MagicMock(name="global_store")
        set_global_artifact_store(store)
        # Context var is None, so it should fall back to global
        assert get_artifact_store() is store

    def test_get_raises_when_nothing_set(self):
        with pytest.raises(RuntimeError, match="No artifact store has been set"):
            get_artifact_store()

    def test_context_var_takes_precedence_over_global(self):
        ctx_store = MagicMock(name="ctx")
        global_store = MagicMock(name="global")
        set_global_artifact_store(global_store)
        set_artifact_store(ctx_store)
        assert get_artifact_store() is ctx_store


# ---------------------------------------------------------------------------
# set_global_artifact_store (coverage)
# ---------------------------------------------------------------------------
class TestSetGlobalArtifactStore:
    def setup_method(self):
        clear_artifact_store()

    def teardown_method(self):
        clear_artifact_store()

    def test_sets_global(self):
        store = MagicMock(name="g")
        set_global_artifact_store(store)
        # Verify via get (context var is None, should fall back)
        assert get_artifact_store() is store


# ---------------------------------------------------------------------------
# has_artifact_store (coverage)
# ---------------------------------------------------------------------------
class TestHasArtifactStore:
    def setup_method(self):
        clear_artifact_store()

    def teardown_method(self):
        clear_artifact_store()

    def test_false_when_nothing_set(self):
        assert has_artifact_store() is False

    def test_true_when_context_set(self):
        set_artifact_store(MagicMock())
        assert has_artifact_store() is True

    def test_true_when_global_set(self):
        set_global_artifact_store(MagicMock())
        assert has_artifact_store() is True

    def test_true_when_both_set(self):
        set_artifact_store(MagicMock())
        set_global_artifact_store(MagicMock())
        assert has_artifact_store() is True


# ---------------------------------------------------------------------------
# clear_artifact_store (coverage)
# ---------------------------------------------------------------------------
class TestClearArtifactStore:
    def setup_method(self):
        clear_artifact_store()

    def teardown_method(self):
        clear_artifact_store()

    def test_clears_context_and_global(self):
        set_artifact_store(MagicMock())
        set_global_artifact_store(MagicMock())
        clear_artifact_store()
        assert has_artifact_store() is False

    def test_clear_is_idempotent(self):
        clear_artifact_store()
        clear_artifact_store()
        assert has_artifact_store() is False


# ---------------------------------------------------------------------------
# Async convenience functions (coverage)
# ---------------------------------------------------------------------------
class TestCreateBlobNamespace:
    def setup_method(self):
        clear_artifact_store()

    def teardown_method(self):
        clear_artifact_store()

    @pytest.mark.asyncio
    async def test_default_scope(self):
        fake_mod = _scope_module()
        store = MagicMock()
        ns_info = MagicMock(name="ns_info")
        store.create_namespace = AsyncMock(return_value=ns_info)
        set_artifact_store(store)

        with (
            patch(
                "chuk_mcp_server.artifacts_context.get_artifact_store",
                return_value=store,
            ),
            patch.dict(
                "sys.modules",
                {"chuk_artifacts": fake_mod},
            ),
        ):
            result = await create_blob_namespace()

        assert result is ns_info
        store.create_namespace.assert_awaited_once_with(
            type=_MOCK_NAMESPACE_TYPE_BLOB,
            scope=_MOCK_SCOPE_SESSION,
            session_id=None,
            user_id=None,
        )

    @pytest.mark.asyncio
    async def test_explicit_scope_and_ids(self):
        fake_mod = _scope_module()
        store = MagicMock()
        ns_info = MagicMock(name="ns_info")
        store.create_namespace = AsyncMock(return_value=ns_info)
        set_artifact_store(store)

        custom_scope = MagicMock(name="USER_SCOPE")

        with patch.dict("sys.modules", {"chuk_artifacts": fake_mod}):
            result = await create_blob_namespace(
                scope=custom_scope,
                session_id="sess-1",
                user_id="alice",
                extra="kw",
            )

        assert result is ns_info
        store.create_namespace.assert_awaited_once_with(
            type=_MOCK_NAMESPACE_TYPE_BLOB,
            scope=custom_scope,
            session_id="sess-1",
            user_id="alice",
            extra="kw",
        )


class TestCreateWorkspaceNamespace:
    def setup_method(self):
        clear_artifact_store()

    def teardown_method(self):
        clear_artifact_store()

    @pytest.mark.asyncio
    async def test_default_scope(self):
        fake_mod = _scope_module()
        store = MagicMock()
        ns_info = MagicMock(name="ns_info")
        store.create_namespace = AsyncMock(return_value=ns_info)
        set_artifact_store(store)

        with patch.dict("sys.modules", {"chuk_artifacts": fake_mod}):
            result = await create_workspace_namespace("my-ws")

        assert result is ns_info
        store.create_namespace.assert_awaited_once_with(
            type=_MOCK_NAMESPACE_TYPE_WORKSPACE,
            name="my-ws",
            scope=_MOCK_SCOPE_SESSION,
            session_id=None,
            user_id=None,
            provider_type="vfs-memory",
        )

    @pytest.mark.asyncio
    async def test_explicit_params(self):
        fake_mod = _scope_module()
        store = MagicMock()
        ns_info = MagicMock(name="ns_info")
        store.create_namespace = AsyncMock(return_value=ns_info)
        set_artifact_store(store)

        custom_scope = MagicMock(name="GLOBAL_SCOPE")

        with patch.dict("sys.modules", {"chuk_artifacts": fake_mod}):
            result = await create_workspace_namespace(
                "proj",
                scope=custom_scope,
                session_id="s2",
                user_id="bob",
                provider_type="vfs-s3",
                tag="v1",
            )

        assert result is ns_info
        store.create_namespace.assert_awaited_once_with(
            type=_MOCK_NAMESPACE_TYPE_WORKSPACE,
            name="proj",
            scope=custom_scope,
            session_id="s2",
            user_id="bob",
            provider_type="vfs-s3",
            tag="v1",
        )


class TestWriteBlob:
    def setup_method(self):
        clear_artifact_store()

    def teardown_method(self):
        clear_artifact_store()

    @pytest.mark.asyncio
    async def test_write_blob(self):
        store = MagicMock()
        store.write_namespace = AsyncMock()
        set_artifact_store(store)

        await write_blob("ns-1", b"data", mime="text/plain")

        store.write_namespace.assert_awaited_once_with("ns-1", data=b"data", mime="text/plain")

    @pytest.mark.asyncio
    async def test_write_blob_no_mime(self):
        store = MagicMock()
        store.write_namespace = AsyncMock()
        set_artifact_store(store)

        await write_blob("ns-2", b"raw")

        store.write_namespace.assert_awaited_once_with("ns-2", data=b"raw", mime=None)


class TestReadBlob:
    def setup_method(self):
        clear_artifact_store()

    def teardown_method(self):
        clear_artifact_store()

    @pytest.mark.asyncio
    async def test_read_blob(self):
        store = MagicMock()
        store.read_namespace = AsyncMock(return_value=b"hello")
        set_artifact_store(store)

        result = await read_blob("ns-1")

        assert result == b"hello"
        store.read_namespace.assert_awaited_once_with("ns-1")


class TestWriteWorkspaceFile:
    def setup_method(self):
        clear_artifact_store()

    def teardown_method(self):
        clear_artifact_store()

    @pytest.mark.asyncio
    async def test_write_workspace_file(self):
        store = MagicMock()
        store.write_namespace = AsyncMock()
        set_artifact_store(store)

        await write_workspace_file("ns-w", "/main.py", b"print('hi')")

        store.write_namespace.assert_awaited_once_with("ns-w", path="/main.py", data=b"print('hi')")


class TestReadWorkspaceFile:
    def setup_method(self):
        clear_artifact_store()

    def teardown_method(self):
        clear_artifact_store()

    @pytest.mark.asyncio
    async def test_read_workspace_file(self):
        store = MagicMock()
        store.read_namespace = AsyncMock(return_value=b"code")
        set_artifact_store(store)

        result = await read_workspace_file("ns-w", "/main.py")

        assert result == b"code"
        store.read_namespace.assert_awaited_once_with("ns-w", path="/main.py")


class TestGetNamespaceVfs:
    def setup_method(self):
        clear_artifact_store()

    def teardown_method(self):
        clear_artifact_store()

    def test_get_namespace_vfs(self):
        vfs_mock = MagicMock(name="vfs")
        store = MagicMock()
        store.get_namespace_vfs.return_value = vfs_mock
        set_artifact_store(store)

        result = get_namespace_vfs("ns-x")

        assert result is vfs_mock
        store.get_namespace_vfs.assert_called_once_with("ns-x")


# ---------------------------------------------------------------------------
# Context auto-read for create_blob_namespace / create_workspace_namespace
# ---------------------------------------------------------------------------


class TestCreateBlobNamespaceContextRead:
    """Verify that session_id / user_id are read from request context when omitted."""

    def setup_method(self):
        # chuk_artifacts is optional — patch it for every test in this class
        self._fake_mod = _scope_module()
        self._patcher = patch.dict("sys.modules", {"chuk_artifacts": self._fake_mod})
        self._patcher.start()
        clear_artifact_store()
        from chuk_mcp_server.context import clear_all

        clear_all()

    def teardown_method(self):
        self._patcher.stop()
        clear_artifact_store()
        from chuk_mcp_server.context import clear_all

        clear_all()

    @pytest.mark.asyncio
    async def test_session_id_read_from_context(self):
        from chuk_mcp_server.context import set_session_id

        set_session_id("ctx-session-1")

        store = MagicMock()
        ns_info = MagicMock(name="ns_info")
        store.create_namespace = AsyncMock(return_value=ns_info)
        set_artifact_store(store)

        await create_blob_namespace()

        _call_kwargs = store.create_namespace.call_args[1]
        assert _call_kwargs["session_id"] == "ctx-session-1"

    @pytest.mark.asyncio
    async def test_user_id_read_from_context(self):
        from chuk_mcp_server.context import set_user_id

        set_user_id("ctx-alice")

        store = MagicMock()
        ns_info = MagicMock(name="ns_info")
        store.create_namespace = AsyncMock(return_value=ns_info)
        set_artifact_store(store)

        await create_blob_namespace()

        _call_kwargs = store.create_namespace.call_args[1]
        assert _call_kwargs["user_id"] == "ctx-alice"

    @pytest.mark.asyncio
    async def test_explicit_session_overrides_context(self):
        from chuk_mcp_server.context import set_session_id

        set_session_id("ctx-session-ignored")

        store = MagicMock()
        store.create_namespace = AsyncMock(return_value=MagicMock())
        set_artifact_store(store)

        await create_blob_namespace(session_id="explicit-session")

        _call_kwargs = store.create_namespace.call_args[1]
        assert _call_kwargs["session_id"] == "explicit-session"

    @pytest.mark.asyncio
    async def test_explicit_user_overrides_context(self):
        from chuk_mcp_server.context import set_user_id

        set_user_id("ctx-user-ignored")

        store = MagicMock()
        store.create_namespace = AsyncMock(return_value=MagicMock())
        set_artifact_store(store)

        await create_blob_namespace(user_id="explicit-user")

        _call_kwargs = store.create_namespace.call_args[1]
        assert _call_kwargs["user_id"] == "explicit-user"


class TestCreateWorkspaceNamespaceContextRead:
    """Verify that session_id / user_id are read from request context when omitted."""

    def setup_method(self):
        self._fake_mod = _scope_module()
        self._patcher = patch.dict("sys.modules", {"chuk_artifacts": self._fake_mod})
        self._patcher.start()
        clear_artifact_store()
        from chuk_mcp_server.context import clear_all

        clear_all()

    def teardown_method(self):
        self._patcher.stop()
        clear_artifact_store()
        from chuk_mcp_server.context import clear_all

        clear_all()

    @pytest.mark.asyncio
    async def test_session_id_read_from_context(self):
        from chuk_mcp_server.context import set_session_id

        set_session_id("ws-session-1")

        store = MagicMock()
        store.create_namespace = AsyncMock(return_value=MagicMock())
        set_artifact_store(store)

        await create_workspace_namespace("my-ws")

        _call_kwargs = store.create_namespace.call_args[1]
        assert _call_kwargs["session_id"] == "ws-session-1"

    @pytest.mark.asyncio
    async def test_user_id_read_from_context(self):
        from chuk_mcp_server.context import set_user_id

        set_user_id("ws-alice")

        store = MagicMock()
        store.create_namespace = AsyncMock(return_value=MagicMock())
        set_artifact_store(store)

        await create_workspace_namespace("my-ws")

        _call_kwargs = store.create_namespace.call_args[1]
        assert _call_kwargs["user_id"] == "ws-alice"

    @pytest.mark.asyncio
    async def test_explicit_ids_override_context(self):
        from chuk_mcp_server.context import set_session_id, set_user_id

        set_session_id("ctx-ignored-session")
        set_user_id("ctx-ignored-user")

        store = MagicMock()
        store.create_namespace = AsyncMock(return_value=MagicMock())
        set_artifact_store(store)

        await create_workspace_namespace("proj", session_id="explicit-s", user_id="explicit-u")

        _call_kwargs = store.create_namespace.call_args[1]
        assert _call_kwargs["session_id"] == "explicit-s"
        assert _call_kwargs["user_id"] == "explicit-u"


# ---------------------------------------------------------------------------
# list_my_namespaces
# ---------------------------------------------------------------------------


class TestListMyNamespaces:
    """Tests for the isolation-safe list_my_namespaces helper."""

    def setup_method(self):
        self._fake_mod = _scope_module()
        self._patcher = patch.dict("sys.modules", {"chuk_artifacts": self._fake_mod})
        self._patcher.start()
        clear_artifact_store()
        from chuk_mcp_server.context import clear_all

        clear_all()

    def teardown_method(self):
        self._patcher.stop()
        clear_artifact_store()
        from chuk_mcp_server.context import clear_all

        clear_all()

    def _make_store(self, return_value=None):
        store = MagicMock()
        store.list_namespaces.return_value = return_value or []
        set_artifact_store(store)
        return store

    def test_raises_without_session_or_user(self):
        """No context at all → RuntimeError to prevent full-bucket listing."""
        from chuk_mcp_server.artifacts_context import list_my_namespaces

        self._make_store()

        with pytest.raises(RuntimeError, match="list_my_namespaces"):
            list_my_namespaces()

    def test_lists_by_session_from_context(self):
        from chuk_mcp_server.artifacts_context import list_my_namespaces
        from chuk_mcp_server.context import set_session_id

        set_session_id("my-session")
        store = self._make_store(["ns-a"])

        result = list_my_namespaces()

        assert result == ["ns-a"]
        store.list_namespaces.assert_called_once_with(session_id="my-session")

    def test_lists_by_user_from_context(self):
        from chuk_mcp_server.artifacts_context import list_my_namespaces
        from chuk_mcp_server.context import set_user_id

        set_user_id("alice")
        store = self._make_store(["ns-b"])

        result = list_my_namespaces()

        assert result == ["ns-b"]
        store.list_namespaces.assert_called_once_with(user_id="alice")

    def test_lists_by_both_from_context(self):
        from chuk_mcp_server.artifacts_context import list_my_namespaces
        from chuk_mcp_server.context import set_session_id, set_user_id

        set_session_id("sess-1")
        set_user_id("bob")
        store = self._make_store(["ns-c"])

        result = list_my_namespaces()

        assert result == ["ns-c"]
        call_kwargs = store.list_namespaces.call_args[1]
        assert call_kwargs["session_id"] == "sess-1"
        assert call_kwargs["user_id"] == "bob"

    def test_explicit_ids_override_context(self):
        from chuk_mcp_server.artifacts_context import list_my_namespaces
        from chuk_mcp_server.context import set_session_id, set_user_id

        # Context has different values
        set_session_id("ctx-session")
        set_user_id("ctx-user")
        store = self._make_store()

        list_my_namespaces(session_id="override-session", user_id="override-user")

        call_kwargs = store.list_namespaces.call_args[1]
        assert call_kwargs["session_id"] == "override-session"
        assert call_kwargs["user_id"] == "override-user"

    def test_explicit_session_only(self):
        from chuk_mcp_server.artifacts_context import list_my_namespaces

        store = self._make_store()

        list_my_namespaces(session_id="explicit-sess")

        call_kwargs = store.list_namespaces.call_args[1]
        assert call_kwargs["session_id"] == "explicit-sess"
        assert "user_id" not in call_kwargs

    def test_explicit_user_only(self):
        from chuk_mcp_server.artifacts_context import list_my_namespaces

        store = self._make_store()

        list_my_namespaces(user_id="explicit-user")

        call_kwargs = store.list_namespaces.call_args[1]
        assert call_kwargs["user_id"] == "explicit-user"
        assert "session_id" not in call_kwargs

    def test_scope_filter_forwarded(self):
        from chuk_mcp_server.artifacts_context import list_my_namespaces
        from chuk_mcp_server.context import set_user_id

        set_user_id("carol")
        scope_mock = MagicMock(name="USER_SCOPE")
        store = self._make_store()

        list_my_namespaces(scope=scope_mock)

        call_kwargs = store.list_namespaces.call_args[1]
        assert call_kwargs["scope"] is scope_mock
        assert call_kwargs["user_id"] == "carol"

    def test_raises_when_only_explicit_none_passed(self):
        """Passing explicit None should behave same as omitting — raises."""
        from chuk_mcp_server.artifacts_context import list_my_namespaces

        self._make_store()

        with pytest.raises(RuntimeError, match="list_my_namespaces"):
            list_my_namespaces(session_id=None, user_id=None)
