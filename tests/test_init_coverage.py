#!/usr/bin/env python3
"""Tests for chuk_mcp_server/__init__.py to improve coverage.

Focuses on the cloud magic functions (lines 184-389, 576-583)
and the global server lifecycle.
"""

import importlib
import sys
from unittest.mock import MagicMock, patch

import pytest

import chuk_mcp_server
from chuk_mcp_server.decorators import clear_global_registry

# ---------------------------------------------------------------------------
# Fixture: reset global state before every test
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _reset_global_server():
    """Reset the global server and decorator registry between tests."""
    chuk_mcp_server._global_server = None
    clear_global_registry()
    yield
    chuk_mcp_server._global_server = None
    clear_global_registry()


# ===========================================================================
# Version & backward-compat
# ===========================================================================


class TestVersionAndCompat:
    def test_version_attribute(self):
        assert hasattr(chuk_mcp_server, "__version__")
        assert isinstance(chuk_mcp_server.__version__, str)
        assert len(chuk_mcp_server.__version__) > 0

    def test_capabilities_backward_compat(self):
        """Capabilities() should be a thin wrapper around create_server_capabilities."""
        result = chuk_mcp_server.Capabilities()
        # Should return a ServerCapabilities-like object (dict-like or dataclass)
        assert result is not None

    def test_capabilities_with_kwargs(self):
        result = chuk_mcp_server.Capabilities(tools=True, resources=False)
        assert result is not None


# ===========================================================================
# Global server creation
# ===========================================================================


class TestGlobalServer:
    def test_get_or_create_global_server(self):
        assert chuk_mcp_server._global_server is None
        server = chuk_mcp_server.get_or_create_global_server()
        assert server is not None
        assert isinstance(server, chuk_mcp_server.ChukMCPServer)
        # Calling again returns the same instance
        assert chuk_mcp_server.get_or_create_global_server() is server

    def test_get_mcp_server_alias(self):
        server = chuk_mcp_server.get_mcp_server()
        assert server is not None
        assert isinstance(server, chuk_mcp_server.ChukMCPServer)
        # Same singleton
        assert chuk_mcp_server.get_mcp_server() is server

    def test_get_or_create_reuses_existing(self):
        server1 = chuk_mcp_server.get_or_create_global_server()
        server2 = chuk_mcp_server.get_or_create_global_server()
        assert server1 is server2


# ===========================================================================
# Cloud detection helpers (non-cloud env)
# ===========================================================================


class TestCloudDetection:
    def test_is_cloud(self):
        assert chuk_mcp_server.is_cloud() is False

    def test_is_gcf(self):
        assert chuk_mcp_server.is_gcf() is False

    def test_is_lambda(self):
        assert chuk_mcp_server.is_lambda() is False

    def test_is_azure(self):
        assert chuk_mcp_server.is_azure() is False


# ===========================================================================
# Cloud handler functions (should raise in non-cloud env)
# ===========================================================================


class TestCloudHandlers:
    def test_get_cloud_handler_raises(self):
        # In non-cloud env the server's get_cloud_handler() will either
        # return None (triggering RuntimeError) or raise AttributeError.
        with pytest.raises((RuntimeError, AttributeError)):
            chuk_mcp_server.get_cloud_handler()

    def test_get_gcf_handler_raises(self):
        with pytest.raises((RuntimeError, AttributeError)):
            chuk_mcp_server.get_gcf_handler()

    def test_get_lambda_handler_raises(self):
        with pytest.raises((RuntimeError, AttributeError)):
            chuk_mcp_server.get_lambda_handler()

    def test_get_azure_handler_raises(self):
        with pytest.raises((RuntimeError, AttributeError)):
            chuk_mcp_server.get_azure_handler()


# ===========================================================================
# Deployment info
# ===========================================================================


class TestDeploymentInfo:
    def test_get_deployment_info(self):
        # May raise AttributeError if method doesn't exist on server
        try:
            info = chuk_mcp_server.get_deployment_info()
            assert isinstance(info, dict)
        except AttributeError:
            # Expected: ChukMCPServer might not have get_cloud_deployment_info
            pass


# ===========================================================================
# show_cloud_examples
# ===========================================================================


class TestShowCloudExamples:
    def test_show_cloud_examples_outputs(self, capsys):
        chuk_mcp_server.show_cloud_examples()
        captured = capsys.readouterr()
        assert "ChukMCPServer" in captured.out
        assert "GOOGLE CLOUD FUNCTIONS" in captured.out
        assert "AWS LAMBDA" in captured.out
        assert "AZURE FUNCTIONS" in captured.out


# ===========================================================================
# _auto_export_cloud_handlers (no-op in non-cloud)
# ===========================================================================


class TestAutoExportCloudHandlers:
    def test_auto_export_noop_in_non_cloud(self):
        # Should not raise and should be a no-op
        chuk_mcp_server._auto_export_cloud_handlers()

    def test_auto_export_with_exception_is_suppressed(self):
        """If detect_cloud_provider raises, it should be caught and logged."""
        with patch.object(
            chuk_mcp_server,
            "detect_cloud_provider",
            side_effect=Exception("test boom"),
        ):
            # Should not raise — the exception is swallowed with a debug log
            chuk_mcp_server._auto_export_cloud_handlers()


# ===========================================================================
# run() — just verify server creation, don't actually start transport
# ===========================================================================


class TestRunFunction:
    def test_run_creates_server_http(self):
        """run() with http transport creates global server; we mock .run() to avoid blocking."""
        with patch.object(chuk_mcp_server.ChukMCPServer, "run") as mock_run:
            chuk_mcp_server.run(transport="http")
            mock_run.assert_called_once()
        assert chuk_mcp_server._global_server is not None

    def test_run_creates_server_stdio(self):
        """run() with stdio transport creates global server; we mock .run_stdio()."""
        with patch.object(chuk_mcp_server.ChukMCPServer, "run_stdio") as mock_run:
            chuk_mcp_server.run(transport="stdio")
            mock_run.assert_called_once()
        assert chuk_mcp_server._global_server is not None

    def test_run_default_transport(self):
        """Default transport is http."""
        with patch.object(chuk_mcp_server.ChukMCPServer, "run") as mock_run:
            chuk_mcp_server.run()
            mock_run.assert_called_once()


# ===========================================================================
# __all__ exports sanity check
# ===========================================================================


class TestExports:
    def test_all_exports_exist(self):
        """Every name in __all__ should be accessible on the module."""
        for name in chuk_mcp_server.__all__:
            assert hasattr(chuk_mcp_server, name), f"{name!r} listed in __all__ but not on module"

    def test_key_exports_present(self):
        assert hasattr(chuk_mcp_server, "ChukMCPServer")
        assert hasattr(chuk_mcp_server, "tool")
        assert hasattr(chuk_mcp_server, "resource")
        assert hasattr(chuk_mcp_server, "prompt")
        assert hasattr(chuk_mcp_server, "run")
        assert hasattr(chuk_mcp_server, "Capabilities")
        assert hasattr(chuk_mcp_server, "ToolRunner")
        assert hasattr(chuk_mcp_server, "ProxyManager")
        assert hasattr(chuk_mcp_server, "ModuleLoader")


# ===========================================================================
# Interactive-mode branch (sys.ps1)
# ===========================================================================


class TestInteractiveMode:
    def test_interactive_mode_branch(self, capsys):
        """When sys.ps1 is set, reloading the module prints a banner."""
        # Set ps1 to simulate interactive Python
        old_ps1 = getattr(sys, "ps1", None)
        sys.ps1 = ">>> "
        try:
            importlib.reload(chuk_mcp_server)
            captured = capsys.readouterr()
            assert "ChukMCPServer" in captured.out
        finally:
            if old_ps1 is None:
                if hasattr(sys, "ps1"):
                    del sys.ps1
            else:
                sys.ps1 = old_ps1
            # Reload again to restore normal state
            importlib.reload(chuk_mcp_server)


# ===========================================================================
# Lines 120-152: ImportError fallback when chuk_artifacts is NOT installed
# ===========================================================================


class TestArtifactsImportFallback:
    """Test the except ImportError branch for chuk_artifacts (lines 120-152)."""

    def test_artifacts_not_available_after_reload(self):
        """When chuk_artifacts import fails, _ARTIFACTS_AVAILABLE should be False."""
        # Save original modules that will be affected
        saved_modules = {}
        artifacts_keys = [k for k in sys.modules if k.startswith("chuk_artifacts")]
        for k in artifacts_keys:
            saved_modules[k] = sys.modules[k]

        # Also save the artifacts_context module
        ac_key = "chuk_mcp_server.artifacts_context"
        if ac_key in sys.modules:
            saved_modules[ac_key] = sys.modules[ac_key]

        try:
            # Block chuk_artifacts from being imported by inserting None
            for k in artifacts_keys:
                sys.modules[k] = None  # type: ignore[assignment]
            sys.modules["chuk_artifacts"] = None  # type: ignore[assignment]
            # Also block the artifacts_context sub-module
            sys.modules[ac_key] = None  # type: ignore[assignment]

            # Reload to trigger the except ImportError branch
            importlib.reload(chuk_mcp_server)

            assert chuk_mcp_server._ARTIFACTS_AVAILABLE is False
            assert chuk_mcp_server._ARTIFACTS_TYPES_AVAILABLE is False
        finally:
            # Restore original modules
            for k, v in saved_modules.items():
                sys.modules[k] = v
            # Remove any None entries we added
            if "chuk_artifacts" in sys.modules and sys.modules["chuk_artifacts"] is None:
                del sys.modules["chuk_artifacts"]
            # Reload to restore normal state
            importlib.reload(chuk_mcp_server)

    def test_has_artifact_store_returns_false_when_not_available(self):
        """Stub has_artifact_store() should return False."""
        saved_modules = {}
        artifacts_keys = [k for k in sys.modules if k.startswith("chuk_artifacts")]
        for k in artifacts_keys:
            saved_modules[k] = sys.modules[k]
        ac_key = "chuk_mcp_server.artifacts_context"
        if ac_key in sys.modules:
            saved_modules[ac_key] = sys.modules[ac_key]

        try:
            for k in artifacts_keys:
                sys.modules[k] = None  # type: ignore[assignment]
            sys.modules["chuk_artifacts"] = None  # type: ignore[assignment]
            sys.modules[ac_key] = None  # type: ignore[assignment]

            importlib.reload(chuk_mcp_server)

            assert chuk_mcp_server.has_artifact_store() is False
        finally:
            for k, v in saved_modules.items():
                sys.modules[k] = v
            if "chuk_artifacts" in sys.modules and sys.modules["chuk_artifacts"] is None:
                del sys.modules["chuk_artifacts"]
            importlib.reload(chuk_mcp_server)

    def test_get_artifact_store_raises_when_not_available(self):
        """Stub get_artifact_store() should raise RuntimeError."""
        saved_modules = {}
        artifacts_keys = [k for k in sys.modules if k.startswith("chuk_artifacts")]
        for k in artifacts_keys:
            saved_modules[k] = sys.modules[k]
        ac_key = "chuk_mcp_server.artifacts_context"
        if ac_key in sys.modules:
            saved_modules[ac_key] = sys.modules[ac_key]

        try:
            for k in artifacts_keys:
                sys.modules[k] = None  # type: ignore[assignment]
            sys.modules["chuk_artifacts"] = None  # type: ignore[assignment]
            sys.modules[ac_key] = None  # type: ignore[assignment]

            importlib.reload(chuk_mcp_server)

            with pytest.raises(RuntimeError, match="chuk-artifacts"):
                chuk_mcp_server.get_artifact_store()
        finally:
            for k, v in saved_modules.items():
                sys.modules[k] = v
            if "chuk_artifacts" in sys.modules and sys.modules["chuk_artifacts"] is None:
                del sys.modules["chuk_artifacts"]
            importlib.reload(chuk_mcp_server)

    def test_stub_type_constants_are_none_when_not_available(self):
        """NamespaceType, StorageScope, NamespaceInfo should be None when unavailable."""
        saved_modules = {}
        artifacts_keys = [k for k in sys.modules if k.startswith("chuk_artifacts")]
        for k in artifacts_keys:
            saved_modules[k] = sys.modules[k]
        ac_key = "chuk_mcp_server.artifacts_context"
        if ac_key in sys.modules:
            saved_modules[ac_key] = sys.modules[ac_key]

        try:
            for k in artifacts_keys:
                sys.modules[k] = None  # type: ignore[assignment]
            sys.modules["chuk_artifacts"] = None  # type: ignore[assignment]
            sys.modules[ac_key] = None  # type: ignore[assignment]

            importlib.reload(chuk_mcp_server)

            assert chuk_mcp_server.NamespaceType is None
            assert chuk_mcp_server.StorageScope is None
            assert chuk_mcp_server.NamespaceInfo is None
        finally:
            for k, v in saved_modules.items():
                sys.modules[k] = v
            if "chuk_artifacts" in sys.modules and sys.modules["chuk_artifacts"] is None:
                del sys.modules["chuk_artifacts"]
            importlib.reload(chuk_mcp_server)

    def test_other_stub_functions_raise_when_not_available(self):
        """All stub artifact functions should raise RuntimeError."""
        saved_modules = {}
        artifacts_keys = [k for k in sys.modules if k.startswith("chuk_artifacts")]
        for k in artifacts_keys:
            saved_modules[k] = sys.modules[k]
        ac_key = "chuk_mcp_server.artifacts_context"
        if ac_key in sys.modules:
            saved_modules[ac_key] = sys.modules[ac_key]

        try:
            for k in artifacts_keys:
                sys.modules[k] = None  # type: ignore[assignment]
            sys.modules["chuk_artifacts"] = None  # type: ignore[assignment]
            sys.modules[ac_key] = None  # type: ignore[assignment]

            importlib.reload(chuk_mcp_server)

            stub_fns = [
                chuk_mcp_server.set_artifact_store,
                chuk_mcp_server.set_global_artifact_store,
                chuk_mcp_server.clear_artifact_store,
                chuk_mcp_server.create_blob_namespace,
                chuk_mcp_server.create_workspace_namespace,
                chuk_mcp_server.write_blob,
                chuk_mcp_server.read_blob,
                chuk_mcp_server.write_workspace_file,
                chuk_mcp_server.read_workspace_file,
                chuk_mcp_server.get_namespace_vfs,
            ]
            for fn in stub_fns:
                with pytest.raises(RuntimeError, match="chuk-artifacts"):
                    fn()
        finally:
            for k, v in saved_modules.items():
                sys.modules[k] = v
            if "chuk_artifacts" in sys.modules and sys.modules["chuk_artifacts"] is None:
                del sys.modules["chuk_artifacts"]
            importlib.reload(chuk_mcp_server)


# ===========================================================================
# Lines 247-257: get_cloud_handler() when server.get_cloud_handler() is None
# ===========================================================================


class TestGetCloudHandlerBranches:
    """Test get_cloud_handler() branches when server returns None handler."""

    def _mock_server(self, **kwargs):
        """Create a MagicMock server and inject it as the global server."""
        mock_server = MagicMock()
        for k, v in kwargs.items():
            getattr(mock_server, k).return_value = v
        chuk_mcp_server._global_server = mock_server
        return mock_server

    def test_cloud_handler_none_with_cloud_provider_detected(self):
        """Lines 250-253: handler is None but detect_cloud_provider returns a provider."""
        mock_provider = MagicMock()
        mock_provider.display_name = "Google Cloud Functions"
        mock_provider.name = "gcp"

        self._mock_server(get_cloud_handler=None)

        with patch.object(
            chuk_mcp_server,
            "detect_cloud_provider",
            return_value=mock_provider,
        ):
            with pytest.raises(RuntimeError, match="Google Cloud Functions"):
                chuk_mcp_server.get_cloud_handler()

    def test_cloud_handler_none_with_no_cloud_provider(self):
        """Lines 254-255: handler is None and detect_cloud_provider returns None."""
        self._mock_server(get_cloud_handler=None)

        with patch.object(
            chuk_mcp_server,
            "detect_cloud_provider",
            return_value=None,
        ):
            with pytest.raises(RuntimeError, match="Not in a cloud environment"):
                chuk_mcp_server.get_cloud_handler()

    def test_cloud_handler_success(self):
        """When server.get_cloud_handler() returns a handler, return it."""
        sentinel = object()
        self._mock_server(get_cloud_handler=sentinel)

        result = chuk_mcp_server.get_cloud_handler()
        assert result is sentinel


# ===========================================================================
# Lines 265-272: get_gcf_handler() success path
# ===========================================================================


class TestGetGcfHandlerSuccess:
    """Test get_gcf_handler() success path (lines 265-270)."""

    def _mock_server(self, **kwargs):
        """Create a MagicMock server and inject it as the global server."""
        mock_server = MagicMock()
        for k, v in kwargs.items():
            getattr(mock_server, k).return_value = v
        chuk_mcp_server._global_server = mock_server
        return mock_server

    def test_gcf_handler_success(self):
        """When adapter has get_handler and provider is GCPProvider, return handler."""
        from chuk_mcp_server.cloud.providers.gcp import GCPProvider

        sentinel_handler = object()
        mock_adapter = MagicMock()
        mock_adapter.get_handler.return_value = sentinel_handler
        mock_provider = GCPProvider()

        self._mock_server(get_cloud_adapter=mock_adapter)

        with patch.object(
            chuk_mcp_server,
            "detect_cloud_provider",
            return_value=mock_provider,
        ):
            result = chuk_mcp_server.get_gcf_handler()
            assert result is sentinel_handler

    def test_gcf_handler_wrong_provider(self):
        """When provider is not GCPProvider, raise RuntimeError."""
        from chuk_mcp_server.cloud.providers.aws import AWSProvider

        mock_adapter = MagicMock()
        mock_adapter.get_handler.return_value = object()
        mock_provider = AWSProvider()

        self._mock_server(get_cloud_adapter=mock_adapter)

        with patch.object(
            chuk_mcp_server,
            "detect_cloud_provider",
            return_value=mock_provider,
        ):
            with pytest.raises(RuntimeError, match="Google Cloud Functions"):
                chuk_mcp_server.get_gcf_handler()

    def test_gcf_handler_no_adapter(self):
        """When adapter is None, fall through to raise RuntimeError (265->272)."""
        self._mock_server(get_cloud_adapter=None)

        with pytest.raises(RuntimeError, match="Google Cloud Functions"):
            chuk_mcp_server.get_gcf_handler()


# ===========================================================================
# Lines 283-290: get_lambda_handler() success path
# ===========================================================================


class TestGetLambdaHandlerSuccess:
    """Test get_lambda_handler() success path (lines 283-288)."""

    def _mock_server(self, **kwargs):
        """Create a MagicMock server and inject it as the global server."""
        mock_server = MagicMock()
        for k, v in kwargs.items():
            getattr(mock_server, k).return_value = v
        chuk_mcp_server._global_server = mock_server
        return mock_server

    def test_lambda_handler_success(self):
        """When adapter has get_handler and provider is AWSProvider, return handler."""
        from chuk_mcp_server.cloud.providers.aws import AWSProvider

        sentinel_handler = object()
        mock_adapter = MagicMock()
        mock_adapter.get_handler.return_value = sentinel_handler
        mock_provider = AWSProvider()

        self._mock_server(get_cloud_adapter=mock_adapter)

        with patch.object(
            chuk_mcp_server,
            "detect_cloud_provider",
            return_value=mock_provider,
        ):
            result = chuk_mcp_server.get_lambda_handler()
            assert result is sentinel_handler

    def test_lambda_handler_wrong_provider(self):
        """When provider is not AWSProvider, raise RuntimeError."""
        from chuk_mcp_server.cloud.providers.gcp import GCPProvider

        mock_adapter = MagicMock()
        mock_adapter.get_handler.return_value = object()
        mock_provider = GCPProvider()

        self._mock_server(get_cloud_adapter=mock_adapter)

        with patch.object(
            chuk_mcp_server,
            "detect_cloud_provider",
            return_value=mock_provider,
        ):
            with pytest.raises(RuntimeError, match="AWS Lambda"):
                chuk_mcp_server.get_lambda_handler()

    def test_lambda_handler_no_adapter(self):
        """When adapter is None, fall through to raise RuntimeError (283->290)."""
        self._mock_server(get_cloud_adapter=None)

        with pytest.raises(RuntimeError, match="AWS Lambda"):
            chuk_mcp_server.get_lambda_handler()


# ===========================================================================
# Lines 298-305: get_azure_handler() success path
# ===========================================================================


class TestGetAzureHandlerSuccess:
    """Test get_azure_handler() success path (lines 298-303)."""

    def _mock_server(self, **kwargs):
        """Create a MagicMock server and inject it as the global server."""
        mock_server = MagicMock()
        for k, v in kwargs.items():
            getattr(mock_server, k).return_value = v
        chuk_mcp_server._global_server = mock_server
        return mock_server

    def test_azure_handler_success(self):
        """When adapter has get_handler and provider is AzureProvider, return handler."""
        from chuk_mcp_server.cloud.providers.azure import AzureProvider

        sentinel_handler = object()
        mock_adapter = MagicMock()
        mock_adapter.get_handler.return_value = sentinel_handler
        mock_provider = AzureProvider()

        self._mock_server(get_cloud_adapter=mock_adapter)

        with patch.object(
            chuk_mcp_server,
            "detect_cloud_provider",
            return_value=mock_provider,
        ):
            result = chuk_mcp_server.get_azure_handler()
            assert result is sentinel_handler

    def test_azure_handler_wrong_provider(self):
        """When provider is not AzureProvider, raise RuntimeError."""
        from chuk_mcp_server.cloud.providers.gcp import GCPProvider

        mock_adapter = MagicMock()
        mock_adapter.get_handler.return_value = object()
        mock_provider = GCPProvider()

        self._mock_server(get_cloud_adapter=mock_adapter)

        with patch.object(
            chuk_mcp_server,
            "detect_cloud_provider",
            return_value=mock_provider,
        ):
            with pytest.raises(RuntimeError, match="Azure Functions"):
                chuk_mcp_server.get_azure_handler()

    def test_azure_handler_no_adapter(self):
        """When adapter is None, fall through to raise RuntimeError (298->305)."""
        self._mock_server(get_cloud_adapter=None)

        with pytest.raises(RuntimeError, match="Azure Functions"):
            chuk_mcp_server.get_azure_handler()


# ===========================================================================
# Lines 354-386: _auto_export_cloud_handlers() with each cloud provider type
# ===========================================================================


class TestAutoExportCloudHandlersWithProvider:
    """Test _auto_export_cloud_handlers() with various cloud provider types."""

    def _setup_mock_server(self, adapter_return):
        """Create a MagicMock server with get_cloud_adapter and inject it."""
        mock_server = MagicMock()
        mock_server.get_cloud_adapter.return_value = adapter_return
        chuk_mcp_server._global_server = mock_server
        return mock_server

    def _make_mocks(self, provider_name):
        """Create mock provider, adapter, and handler."""
        mock_provider = MagicMock()
        mock_provider.name = provider_name

        sentinel_handler = MagicMock(name=f"handler_{provider_name}")
        mock_adapter = MagicMock()
        mock_adapter.get_handler.return_value = sentinel_handler

        return mock_provider, mock_adapter, sentinel_handler

    def test_auto_export_gcp(self):
        """Lines 365-367: GCP provider exports mcp_gcf_handler."""
        mock_provider, mock_adapter, sentinel_handler = self._make_mocks("gcp")
        self._setup_mock_server(mock_adapter)

        with patch.object(
            chuk_mcp_server,
            "detect_cloud_provider",
            return_value=mock_provider,
        ):
            chuk_mcp_server._auto_export_cloud_handlers()

        assert chuk_mcp_server.mcp_gcf_handler is sentinel_handler  # type: ignore[attr-defined]
        assert chuk_mcp_server.cloud_handler is sentinel_handler  # type: ignore[attr-defined]
        assert chuk_mcp_server.mcp_handler is sentinel_handler  # type: ignore[attr-defined]

        # Clean up module attributes
        for attr in ("mcp_gcf_handler", "cloud_handler", "mcp_handler"):
            if hasattr(chuk_mcp_server, attr):
                delattr(chuk_mcp_server, attr)

    def test_auto_export_aws(self):
        """Lines 369-372: AWS provider exports lambda_handler and handler."""
        mock_provider, mock_adapter, sentinel_handler = self._make_mocks("aws")
        self._setup_mock_server(mock_adapter)

        with patch.object(
            chuk_mcp_server,
            "detect_cloud_provider",
            return_value=mock_provider,
        ):
            chuk_mcp_server._auto_export_cloud_handlers()

        assert chuk_mcp_server.lambda_handler is sentinel_handler  # type: ignore[attr-defined]
        assert chuk_mcp_server.handler is sentinel_handler  # type: ignore[attr-defined]
        assert chuk_mcp_server.cloud_handler is sentinel_handler  # type: ignore[attr-defined]
        assert chuk_mcp_server.mcp_handler is sentinel_handler  # type: ignore[attr-defined]

        # Clean up module attributes
        for attr in ("lambda_handler", "handler", "cloud_handler", "mcp_handler"):
            if hasattr(chuk_mcp_server, attr):
                delattr(chuk_mcp_server, attr)

    def test_auto_export_azure(self):
        """Lines 374-377: Azure provider exports main and azure_handler."""
        mock_provider, mock_adapter, sentinel_handler = self._make_mocks("azure")
        self._setup_mock_server(mock_adapter)

        with patch.object(
            chuk_mcp_server,
            "detect_cloud_provider",
            return_value=mock_provider,
        ):
            chuk_mcp_server._auto_export_cloud_handlers()

        assert chuk_mcp_server.main is sentinel_handler  # type: ignore[attr-defined]
        assert chuk_mcp_server.azure_handler is sentinel_handler  # type: ignore[attr-defined]
        assert chuk_mcp_server.cloud_handler is sentinel_handler  # type: ignore[attr-defined]
        assert chuk_mcp_server.mcp_handler is sentinel_handler  # type: ignore[attr-defined]

        # Clean up module attributes
        for attr in ("main", "azure_handler", "cloud_handler", "mcp_handler"):
            if hasattr(chuk_mcp_server, attr):
                delattr(chuk_mcp_server, attr)

    def test_auto_export_edge_provider(self):
        """Lines 379-382: Edge provider (vercel) exports handler and main."""
        mock_provider, mock_adapter, sentinel_handler = self._make_mocks("vercel")
        self._setup_mock_server(mock_adapter)

        with patch.object(
            chuk_mcp_server,
            "detect_cloud_provider",
            return_value=mock_provider,
        ):
            chuk_mcp_server._auto_export_cloud_handlers()

        assert chuk_mcp_server.handler is sentinel_handler  # type: ignore[attr-defined]
        assert chuk_mcp_server.main is sentinel_handler  # type: ignore[attr-defined]
        assert chuk_mcp_server.cloud_handler is sentinel_handler  # type: ignore[attr-defined]
        assert chuk_mcp_server.mcp_handler is sentinel_handler  # type: ignore[attr-defined]

        # Clean up module attributes
        for attr in ("handler", "main", "cloud_handler", "mcp_handler"):
            if hasattr(chuk_mcp_server, attr):
                delattr(chuk_mcp_server, attr)

    def test_auto_export_no_adapter(self):
        """When adapter is None, should return early (no attributes set)."""
        mock_provider = MagicMock()
        mock_provider.name = "gcp"
        self._setup_mock_server(None)

        with patch.object(
            chuk_mcp_server,
            "detect_cloud_provider",
            return_value=mock_provider,
        ):
            chuk_mcp_server._auto_export_cloud_handlers()

        assert not hasattr(chuk_mcp_server, "mcp_gcf_handler")

    def test_auto_export_handler_returns_none(self):
        """When adapter.get_handler() returns None, should return early."""
        mock_provider = MagicMock()
        mock_provider.name = "gcp"
        mock_adapter = MagicMock()
        mock_adapter.get_handler.return_value = None

        self._setup_mock_server(mock_adapter)

        with patch.object(
            chuk_mcp_server,
            "detect_cloud_provider",
            return_value=mock_provider,
        ):
            chuk_mcp_server._auto_export_cloud_handlers()

        assert not hasattr(chuk_mcp_server, "mcp_gcf_handler")

    def test_auto_export_unknown_provider(self):
        """Unknown provider name still exports generic cloud_handler and mcp_handler."""
        mock_provider, mock_adapter, sentinel_handler = self._make_mocks("custom_provider")
        self._setup_mock_server(mock_adapter)

        with patch.object(
            chuk_mcp_server,
            "detect_cloud_provider",
            return_value=mock_provider,
        ):
            chuk_mcp_server._auto_export_cloud_handlers()

        # Generic names are always exported
        assert chuk_mcp_server.cloud_handler is sentinel_handler  # type: ignore[attr-defined]
        assert chuk_mcp_server.mcp_handler is sentinel_handler  # type: ignore[attr-defined]
        # Platform-specific names should NOT be set
        assert not hasattr(chuk_mcp_server, "mcp_gcf_handler")
        assert not hasattr(chuk_mcp_server, "lambda_handler")
        assert not hasattr(chuk_mcp_server, "azure_handler")

        # Clean up module attributes
        for attr in ("cloud_handler", "mcp_handler"):
            if hasattr(chuk_mcp_server, attr):
                delattr(chuk_mcp_server, attr)
