"""
Tests for chuk_mcp_server.artifacts_context — targeting 100 % line coverage.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

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


# ---------------------------------------------------------------------------
# Fixture: always start with a clean slate
# ---------------------------------------------------------------------------
@pytest.fixture(autouse=True)
def _reset_artifact_store():
    """Ensure every test starts and ends with no store configured."""
    clear_artifact_store()
    yield
    clear_artifact_store()


# ---------------------------------------------------------------------------
# set_artifact_store / get_artifact_store
# ---------------------------------------------------------------------------
class TestSetAndGetArtifactStore:
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
# set_global_artifact_store
# ---------------------------------------------------------------------------
class TestSetGlobalArtifactStore:
    def test_sets_global(self):
        store = MagicMock(name="g")
        set_global_artifact_store(store)
        # Verify via get (context var is None, should fall back)
        assert get_artifact_store() is store


# ---------------------------------------------------------------------------
# has_artifact_store
# ---------------------------------------------------------------------------
class TestHasArtifactStore:
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
# clear_artifact_store
# ---------------------------------------------------------------------------
class TestClearArtifactStore:
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
# Async convenience functions
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


class TestCreateBlobNamespace:
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
    @pytest.mark.asyncio
    async def test_read_blob(self):
        store = MagicMock()
        store.read_namespace = AsyncMock(return_value=b"hello")
        set_artifact_store(store)

        result = await read_blob("ns-1")

        assert result == b"hello"
        store.read_namespace.assert_awaited_once_with("ns-1")


class TestWriteWorkspaceFile:
    @pytest.mark.asyncio
    async def test_write_workspace_file(self):
        store = MagicMock()
        store.write_namespace = AsyncMock()
        set_artifact_store(store)

        await write_workspace_file("ns-w", "/main.py", b"print('hi')")

        store.write_namespace.assert_awaited_once_with("ns-w", path="/main.py", data=b"print('hi')")


class TestReadWorkspaceFile:
    @pytest.mark.asyncio
    async def test_read_workspace_file(self):
        store = MagicMock()
        store.read_namespace = AsyncMock(return_value=b"code")
        set_artifact_store(store)

        result = await read_workspace_file("ns-w", "/main.py")

        assert result == b"code"
        store.read_namespace.assert_awaited_once_with("ns-w", path="/main.py")


class TestGetNamespaceVfs:
    def test_get_namespace_vfs(self):
        vfs_mock = MagicMock(name="vfs")
        store = MagicMock()
        store.get_namespace_vfs.return_value = vfs_mock
        set_artifact_store(store)

        result = get_namespace_vfs("ns-x")

        assert result is vfs_mock
        store.get_namespace_vfs.assert_called_once_with("ns-x")
