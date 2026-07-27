"""Tests for optional provider exports."""

import builtins
import importlib

import chuk_mcp_server.oauth.providers as providers


class TestProviderExports:
    """The provider package degrades gracefully without optional dependencies."""

    def test_google_drive_is_exported_when_importable(self):
        module = importlib.reload(providers)

        assert "GoogleDriveOAuthProvider" in module.__all__
        assert "GoogleDriveOAuthClient" in module.__all__

    def test_missing_dependencies_leave_exports_empty(self, monkeypatch):
        real_import = builtins.__import__

        def fail_google_drive(name, globals=None, locals=None, fromlist=(), level=0):
            if "google_drive" in name:
                raise ImportError("httpx is not installed")
            return real_import(name, globals, locals, fromlist, level)

        monkeypatch.setattr(builtins, "__import__", fail_google_drive)
        module = importlib.reload(providers)

        assert module.__all__ == []

    def teardown_method(self):
        # Restore the real exports for any test that runs afterwards.
        importlib.reload(providers)
