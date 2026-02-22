#!/usr/bin/env python3
# tests/test_config_coverage.py
"""
Tests targeting missing coverage in the config module.

Covers:
  - config/__init__.py  convenience functions
  - config/cloud_detector.py  cached return, ImportError, delegation methods, get_detection_info
  - config/container_detector.py  exception branches in filesystem checks
  - config/environment_detector.py  transport env vars, isatty, cloud exception, ImportError fallback, detection_info
  - config/project_detector.py  Cargo.toml detection, parent dir, package.json with name, config file parsing
  - config/smart_config.py  uncached individual getters, cloud methods, summary with cloud, detailed_info
"""

import io
import json
import os
import sys
from pathlib import Path
from unittest.mock import MagicMock, PropertyMock, patch

import pytest

# ============================================================================
# 1. config/__init__.py  - convenience functions (lines 24, 29, 34, 39)
# ============================================================================


class TestConfigInitConvenience:
    """Exercise the four convenience functions exported from config/__init__.py."""

    def test_get_smart_defaults(self):
        """Call get_smart_defaults() to cover line 24."""
        from chuk_mcp_server.config import get_smart_defaults

        result = get_smart_defaults()
        assert isinstance(result, dict)
        assert "project_name" in result
        assert "environment" in result

    def test_detect_cloud_provider(self):
        """Call detect_cloud_provider() to cover line 29."""
        from chuk_mcp_server.config import detect_cloud_provider

        result = detect_cloud_provider()
        # In test env, typically None
        assert result is None or hasattr(result, "name")

    def test_get_cloud_config(self):
        """Call get_cloud_config() to cover line 34."""
        from chuk_mcp_server.config import get_cloud_config

        result = get_cloud_config()
        assert isinstance(result, dict)

    def test_is_cloud_environment(self):
        """Call is_cloud_environment() to cover line 39."""
        from chuk_mcp_server.config import is_cloud_environment

        result = is_cloud_environment()
        assert isinstance(result, bool)


# ============================================================================
# 2. config/cloud_detector.py
# ============================================================================


class TestCloudDetectorCoverage:
    """Cover missing lines in CloudDetector."""

    def test_detect_cached_return(self):
        """Cover line 26: return from cache when _detected_provider is not None."""
        from chuk_mcp_server.config.cloud_detector import CloudDetector

        detector = CloudDetector()
        # Pre-set the cache with a sentinel value (not None)
        sentinel = "cached_provider"
        detector._detected_provider = sentinel
        result = detector.detect()
        assert result == sentinel

    def test_detect_import_error(self):
        """Cover lines 44-47: ImportError when cloud module is unavailable."""
        from chuk_mcp_server.config.cloud_detector import CloudDetector

        detector = CloudDetector()
        with patch(
            "chuk_mcp_server.config.cloud_detector.CloudDetector.detect",
            wraps=detector.detect,
        ):
            # Force an ImportError from the cloud module import
            import builtins

            real_import = builtins.__import__

            def fake_import(name, *args, **kwargs):
                if "cloud" in name and "cloud_detector" not in name:
                    raise ImportError("Fake cloud import error")
                return real_import(name, *args, **kwargs)

            # Reset so detect() actually runs the import path
            detector._detected_provider = None
            with patch("builtins.__import__", side_effect=fake_import):
                result = detector.detect()
            assert result is None
            assert detector._detected_provider is None

    def test_get_provider_delegates_to_detect(self):
        """Cover line 51: get_provider() calls detect()."""
        from chuk_mcp_server.config.cloud_detector import CloudDetector

        detector = CloudDetector()
        with patch.object(detector, "detect", return_value=None) as mock_detect:
            result = detector.get_provider()
            mock_detect.assert_called_once()
            assert result is None

    def test_is_cloud_environment_false(self):
        """Cover line 60: is_cloud_environment() returns False when no provider."""
        from chuk_mcp_server.config.cloud_detector import CloudDetector

        detector = CloudDetector()
        with patch.object(detector, "detect", return_value=None):
            assert detector.is_cloud_environment() is False

    def test_is_cloud_environment_true(self):
        """Cover line 60: is_cloud_environment() returns True when provider exists."""
        from chuk_mcp_server.config.cloud_detector import CloudDetector

        detector = CloudDetector()
        mock_provider = MagicMock()
        with patch.object(detector, "detect", return_value=mock_provider):
            assert detector.is_cloud_environment() is True

    def test_get_environment_type_none(self):
        """Cover line 69: get_environment_type() returns None when no provider."""
        from chuk_mcp_server.config.cloud_detector import CloudDetector

        detector = CloudDetector()
        with patch.object(detector, "detect", return_value=None):
            assert detector.get_environment_type() is None

    def test_get_service_type_none(self):
        """Cover line 70: get_service_type() returns None when no provider."""
        from chuk_mcp_server.config.cloud_detector import CloudDetector

        detector = CloudDetector()
        with patch.object(detector, "detect", return_value=None):
            assert detector.get_service_type() is None

    def test_get_detection_info_no_provider(self):
        """Cover lines 80-91: get_detection_info() with no provider detected."""
        from chuk_mcp_server.config.cloud_detector import CloudDetector

        detector = CloudDetector()
        with patch.object(detector, "detect", return_value=None):
            info = detector.get_detection_info()
            assert info["detected"] is False
            assert info["provider"] is None
            assert info["display_name"] is None
            assert info["service_type"] is None
            assert info["environment_type"] is None
            assert info["config_overrides"] == {}
            assert isinstance(info["available_providers"], list)
            assert isinstance(info["total_providers"], int)

    def test_get_detection_info_with_provider(self):
        """Cover lines 80-91: get_detection_info() with a mock provider."""
        from chuk_mcp_server.config.cloud_detector import CloudDetector

        detector = CloudDetector()
        mock_provider = MagicMock()
        mock_provider.name = "aws"
        mock_provider.display_name = "AWS"
        mock_provider.get_service_type.return_value = "ec2"
        mock_provider.get_environment_type.return_value = "production"
        mock_provider.get_config_overrides.return_value = {"host": "0.0.0.0"}

        with patch.object(detector, "detect", return_value=mock_provider):
            info = detector.get_detection_info()
            assert info["detected"] is True
            assert info["provider"] == "aws"
            assert info["display_name"] == "AWS"
            assert info["service_type"] == "ec2"
            assert info["environment_type"] == "production"
            assert info["config_overrides"] == {"host": "0.0.0.0"}

    def test_get_detection_info_import_error(self):
        """Cover lines 87-89: get_detection_info() when cloud module import fails."""
        from chuk_mcp_server.config.cloud_detector import CloudDetector

        detector = CloudDetector()

        with patch.object(detector, "detect", return_value=None):
            import builtins

            real_import = builtins.__import__

            def fake_import(name, *args, **kwargs):
                if "chuk_mcp_server.cloud" in name and "cloud_detector" not in name:
                    raise ImportError("Fake import error")
                return real_import(name, *args, **kwargs)

            with patch("builtins.__import__", side_effect=fake_import):
                info = detector.get_detection_info()
                assert info["available_providers"] == []
                assert info["total_providers"] == 0

    def test_detect_provider_found(self):
        """Cover lines 36-38: detect() when cloud_registry.detect_provider returns a provider."""
        from chuk_mcp_server.config.cloud_detector import CloudDetector

        detector = CloudDetector()
        # Ensure cache is clear
        detector._detected_provider = None

        mock_provider = MagicMock()
        mock_provider.display_name = "Google Cloud Platform"

        mock_registry = MagicMock()
        mock_registry.detect_provider.return_value = mock_provider

        with patch(
            "chuk_mcp_server.config.cloud_detector.CloudDetector.detect.__module__",
            create=True,
        ):
            pass

        # Patch the import inside detect() to return our mock registry
        import builtins

        real_import = builtins.__import__

        def fake_import(name, globals=None, locals=None, fromlist=(), level=0):
            if level > 0 and fromlist and "cloud_registry" in fromlist:
                mock_module = MagicMock()
                mock_module.cloud_registry = mock_registry
                return mock_module
            return real_import(name, globals, locals, fromlist, level)

        with patch("builtins.__import__", side_effect=fake_import):
            result = detector.detect()
            assert result == mock_provider
            assert detector._detected_provider == mock_provider

    def test_clear_cache_with_cloud_registry(self):
        """Cover line 76: clear_cache when _cloud_registry is set."""
        from chuk_mcp_server.config.cloud_detector import CloudDetector

        detector = CloudDetector()
        mock_registry = MagicMock()
        detector._cloud_registry = mock_registry
        detector._detected_provider = "some_provider"

        detector.clear_cache()

        assert detector._detected_provider is None
        mock_registry.clear_cache.assert_called_once()

    def test_get_detection_info_import_error_inside(self):
        """Cover lines 87-89 more directly: ImportError in get_detection_info's own import."""
        from chuk_mcp_server.config.cloud_detector import CloudDetector

        detector = CloudDetector()

        # Pre-set _detected_provider to None so detect() returns None quickly
        detector._detected_provider = None

        # Remove the cloud module from sys.modules to force re-import
        saved_modules = {}
        keys_to_remove = [k for k in sys.modules if k.startswith("chuk_mcp_server.cloud")]
        for k in keys_to_remove:
            saved_modules[k] = sys.modules.pop(k)

        import builtins

        real_import = builtins.__import__

        def fake_import(name, globals=None, locals=None, fromlist=(), level=0):
            # Block the cloud module import
            if level > 0 and fromlist and "cloud_registry" in fromlist:
                raise ImportError("cloud module blocked")
            if name.startswith("chuk_mcp_server.cloud"):
                raise ImportError("cloud module blocked")
            return real_import(name, globals, locals, fromlist, level)

        try:
            with patch("builtins.__import__", side_effect=fake_import):
                info = detector.get_detection_info()
                assert info["detected"] is False
                assert info["available_providers"] == []
                assert info["total_providers"] == 0
        finally:
            # Restore modules
            sys.modules.update(saved_modules)


# ============================================================================
# 3. config/container_detector.py  - exception branches
# ============================================================================


class TestContainerDetectorExceptions:
    """Cover exception branches in ContainerDetector file checks."""

    def test_check_docker_env_exception(self):
        """Cover lines 38-40: exception in _check_docker_env."""
        from chuk_mcp_server.config.container_detector import ContainerDetector

        detector = ContainerDetector()
        with patch("pathlib.Path.exists", side_effect=PermissionError("no access")):
            result = detector._check_docker_env()
            assert result is False

    def test_check_cgroup_exception(self):
        """Cover lines 60-62: exception in _check_cgroup_container."""
        from chuk_mcp_server.config.container_detector import ContainerDetector

        detector = ContainerDetector()
        with patch("pathlib.Path.exists", side_effect=OSError("cgroup read error")):
            result = detector._check_cgroup_container()
            assert result is False

    def test_check_mountinfo_exception(self):
        """Cover lines 74-76: exception in _check_mountinfo_container."""
        from chuk_mcp_server.config.container_detector import ContainerDetector

        detector = ContainerDetector()
        with patch("pathlib.Path.exists", side_effect=OSError("mountinfo error")):
            result = detector._check_mountinfo_container()
            assert result is False


# ============================================================================
# 4. config/environment_detector.py
# ============================================================================


class TestEnvironmentDetectorCoverage:
    """Cover missing lines in EnvironmentDetector."""

    @patch.dict(os.environ, {"MCP_TRANSPORT": "STDIO"}, clear=True)
    def test_detect_transport_mode_explicit(self):
        """Cover line 90: explicit MCP_TRANSPORT env var returns lowered value."""
        from chuk_mcp_server.config.environment_detector import EnvironmentDetector

        detector = EnvironmentDetector()
        result = detector.detect_transport_mode()
        assert result == "stdio"

    @patch.dict(os.environ, {"MCP_STDIO": "1"}, clear=True)
    def test_detect_transport_mode_mcp_stdio(self):
        """Cover line 94: MCP_STDIO env var."""
        from chuk_mcp_server.config.environment_detector import EnvironmentDetector

        detector = EnvironmentDetector()
        result = detector.detect_transport_mode()
        assert result == "stdio"

    @patch.dict(os.environ, {"USE_STDIO": "true"}, clear=True)
    def test_detect_transport_mode_use_stdio(self):
        """Cover line 94: USE_STDIO env var."""
        from chuk_mcp_server.config.environment_detector import EnvironmentDetector

        detector = EnvironmentDetector()
        result = detector.detect_transport_mode()
        assert result == "stdio"

    @patch.dict(os.environ, {}, clear=True)
    def test_detect_transport_mode_isatty_false(self):
        """Cover lines 103-104: stdin/stdout not a tty returns stdio."""
        from chuk_mcp_server.config.environment_detector import EnvironmentDetector

        detector = EnvironmentDetector()

        mock_stdin = MagicMock()
        mock_stdin.isatty.return_value = False
        mock_stdout = MagicMock()
        mock_stdout.isatty.return_value = True
        # Remove _pytest_capture attribute so the check doesn't skip
        del mock_stdin._pytest_capture
        del mock_stdout._pytest_capture

        with patch.object(sys, "stdin", mock_stdin), patch.object(sys, "stdout", mock_stdout):
            result = detector.detect_transport_mode()
            assert result == "stdio"

    @patch.dict(os.environ, {}, clear=True)
    def test_detect_transport_mode_isatty_exception(self):
        """Cover lines 105-110: AttributeError in isatty check falls through to http."""
        from chuk_mcp_server.config.environment_detector import EnvironmentDetector

        detector = EnvironmentDetector()

        mock_stdin = MagicMock()
        # Remove _pytest_capture so the code enters the block
        del mock_stdin._pytest_capture
        mock_stdin.isatty.side_effect = io.UnsupportedOperation("not supported")
        mock_stdout = MagicMock()
        del mock_stdout._pytest_capture
        mock_stdout.isatty.return_value = True

        with patch.object(sys, "stdin", mock_stdin), patch.object(sys, "stdout", mock_stdout):
            result = detector.detect_transport_mode()
            assert result == "http"

    @patch.dict(os.environ, {}, clear=True)
    def test_detect_transport_mode_http_default(self):
        """Cover line 110: default to http when stdin/stdout are ttys."""
        from chuk_mcp_server.config.environment_detector import EnvironmentDetector

        detector = EnvironmentDetector()

        mock_stdin = MagicMock()
        mock_stdin.isatty.return_value = True
        mock_stdout = MagicMock()
        mock_stdout.isatty.return_value = True
        del mock_stdin._pytest_capture
        del mock_stdout._pytest_capture

        with patch.object(sys, "stdin", mock_stdin), patch.object(sys, "stdout", mock_stdout):
            result = detector.detect_transport_mode()
            assert result == "http"

    def test_get_cloud_environment_exception(self):
        """Cover lines 136-138: _get_cloud_environment catches exception."""
        from chuk_mcp_server.config.environment_detector import EnvironmentDetector

        detector = EnvironmentDetector()
        mock_cloud = MagicMock()
        mock_cloud.get_environment_type.side_effect = RuntimeError("boom")
        detector._cloud_detector = mock_cloud
        result = detector._get_cloud_environment()
        assert result is None

    def test_is_containerized_import_error_fallback(self):
        """Cover lines 165-167: _is_containerized ImportError fallback.

        The import `from .container_detector import ContainerDetector` happens
        at function call time.  We temporarily remove the already-cached module
        from sys.modules and patch builtins.__import__ so the relative import
        raises ImportError, forcing the fallback branch.
        """
        from chuk_mcp_server.config.environment_detector import EnvironmentDetector

        detector = EnvironmentDetector()

        import builtins

        real_import = builtins.__import__
        module_key = "chuk_mcp_server.config.container_detector"

        def fake_import(name, globals=None, locals=None, fromlist=(), level=0):
            # Intercept only the relative import of container_detector
            if level > 0 and fromlist and "ContainerDetector" in fromlist:
                raise ImportError("faked for test")
            if name == module_key:
                raise ImportError("faked for test")
            return real_import(name, globals, locals, fromlist, level)

        saved = sys.modules.pop(module_key, None)
        try:
            with patch("builtins.__import__", side_effect=fake_import):
                # No Docker env, no K8s, no CONTAINER => fallback returns False
                with patch("pathlib.Path.exists", return_value=False):
                    with patch.dict(os.environ, {}, clear=True):
                        assert detector._is_containerized() is False

            # Restore module so the next call can work
            if saved:
                sys.modules[module_key] = saved
            else:
                sys.modules.pop(module_key, None)

            saved2 = sys.modules.pop(module_key, None)
            with patch("builtins.__import__", side_effect=fake_import):
                # KUBERNETES_SERVICE_HOST set => fallback returns True
                with patch("pathlib.Path.exists", return_value=False):
                    with patch.dict(os.environ, {"KUBERNETES_SERVICE_HOST": "10.0.0.1"}, clear=True):
                        assert detector._is_containerized() is True
        finally:
            # Always restore
            if saved:
                sys.modules[module_key] = saved
            elif saved2:
                sys.modules[module_key] = saved2

    @patch.dict(os.environ, {}, clear=True)
    def test_is_development_environment_with_dev_files_no_port(self, tmp_path):
        """Cover line 183: dev files present without PORT returns True."""
        from chuk_mcp_server.config.environment_detector import EnvironmentDetector

        detector = EnvironmentDetector()

        # Create a pyproject.toml in tmp_path (but no .git)
        (tmp_path / "pyproject.toml").write_text("[tool.poetry]\nname = 'x'")

        with patch("pathlib.Path.cwd", return_value=tmp_path):
            result = detector._is_development_environment()
            assert result is True

    def test_get_detection_info(self):
        """Cover lines 190-191: get_detection_info() calls cloud and returns dict."""
        from chuk_mcp_server.config.environment_detector import EnvironmentDetector

        detector = EnvironmentDetector()
        info = detector.get_detection_info()
        assert isinstance(info, dict)
        assert "environment" in info
        assert "explicit_env_vars" in info
        assert "ci_detected" in info
        assert "serverless_detected" in info
        assert "containerized" in info
        assert "development_indicators" in info
        assert "cloud" in info

    @patch.dict(os.environ, {}, clear=True)
    def test_detect_cloud_environment_returns_value(self):
        """Cover lines 63-64: _get_cloud_environment returns a value from cloud detector."""
        from chuk_mcp_server.config.environment_detector import EnvironmentDetector

        detector = EnvironmentDetector()
        mock_cloud = MagicMock()
        mock_cloud.get_environment_type.return_value = "production"
        detector._cloud_detector = mock_cloud

        result = detector._get_cloud_environment()
        assert result == "production"

    @patch.dict(os.environ, {}, clear=True)
    def test_detect_falls_through_to_cloud_environment(self):
        """Cover lines 63-64 in detect(): cloud environment path."""
        from chuk_mcp_server.config.environment_detector import EnvironmentDetector

        detector = EnvironmentDetector()
        mock_cloud = MagicMock()
        mock_cloud.get_environment_type.return_value = "staging"
        mock_cloud.get_detection_info.return_value = {}
        detector._cloud_detector = mock_cloud

        with patch.object(detector, "_is_ci_environment", return_value=False):
            with patch.object(detector, "_is_serverless_environment", return_value=False):
                result = detector.detect()
                assert result == "staging"


# ============================================================================
# 5. config/project_detector.py
# ============================================================================


class TestProjectDetectorCoverage:
    """Cover missing lines in ProjectDetector."""

    @patch("pathlib.Path.cwd")
    def test_detect_from_cargo_toml_found(self, mock_cwd):
        """Cover line 41: detect_from_cargo_toml returns a name."""
        from chuk_mcp_server.config.project_detector import ProjectDetector

        # Set cwd to a generic dir so _detect_from_directory returns None
        mock_cwd.return_value = Path("/home/user/src")

        cargo_content = '[package]\nname = "my-rust-app"\nversion = "0.1.0"'

        detector = ProjectDetector()

        # Return None for package.json, pyproject.toml, setup.py but content for Cargo.toml
        def mock_safe_read(path):
            if "Cargo.toml" in str(path):
                return cargo_content
            return None

        with patch.object(detector, "safe_file_read", side_effect=mock_safe_read):
            result = detector.detect()
            assert result == "My Rust App MCP Server"

    @patch("pathlib.Path.cwd")
    def test_detect_from_package_json_with_name(self, mock_cwd):
        """Cover line 75->77: package.json with a name field."""
        from chuk_mcp_server.config.project_detector import ProjectDetector

        mock_cwd.return_value = Path("/home/user/src")

        package_data = json.dumps({"name": "my-node-service", "version": "1.0.0"})
        detector = ProjectDetector()

        def mock_safe_read(path):
            if "package.json" in str(path):
                return package_data
            return None

        with patch.object(detector, "safe_file_read", side_effect=mock_safe_read):
            result = detector.detect()
            assert result == "My Node Service MCP Server"

    @patch("pathlib.Path.cwd")
    def test_detect_from_pyproject_toml_parsing(self, mock_cwd):
        """Cover lines 90->85, 92->85: pyproject.toml name parsing."""
        from chuk_mcp_server.config.project_detector import ProjectDetector

        mock_cwd.return_value = Path("/home/user/src")

        toml_content = 'name = "cool-python-lib"\nversion = "2.0.0"'
        detector = ProjectDetector()

        def mock_safe_read(path):
            if "pyproject.toml" in str(path):
                return toml_content
            if "package.json" in str(path):
                return None
            return None

        with patch.object(detector, "safe_file_read", side_effect=mock_safe_read):
            result = detector.detect()
            assert result == "Cool Python Lib MCP Server"

    @patch("pathlib.Path.cwd")
    def test_detect_from_setup_py_parsing(self, mock_cwd):
        """Cover lines 102->110, 106->102, 108->102: setup.py name parsing."""
        from chuk_mcp_server.config.project_detector import ProjectDetector

        mock_cwd.return_value = Path("/home/user/src")

        setup_content = 'from setuptools import setup\nsetup(\n    name="my-setup-lib",\n    version="1.0"\n)'
        detector = ProjectDetector()

        def mock_safe_read(path):
            if "setup.py" in str(path):
                return setup_content
            if "package.json" in str(path) or "pyproject.toml" in str(path):
                return None
            return None

        with patch.object(detector, "safe_file_read", side_effect=mock_safe_read):
            result = detector.detect()
            assert result == "My Setup Lib MCP Server"

    @patch("pathlib.Path.cwd")
    def test_detect_from_parent_directory(self, mock_cwd):
        """Cover lines 117-124: parent directory fallback when in generic dir."""
        from chuk_mcp_server.config.project_detector import ProjectDetector

        mock_cwd.return_value = Path("/home/user/awesome-project/src")

        detector = ProjectDetector()

        # All file reads return None so we fall through to parent directory
        with patch.object(detector, "safe_file_read", return_value=None):
            result = detector.detect()
            assert result == "Awesome Project MCP Server"

    @patch("pathlib.Path.cwd")
    def test_detect_from_parent_directory_exception(self, mock_cwd):
        """Cover lines 133->137, 135-136: parent dir detection exception."""
        from chuk_mcp_server.config.project_detector import ProjectDetector

        # First call to Path.cwd() works for _detect_from_directory
        # but _detect_from_parent_directory should fail
        mock_cwd.return_value = Path("/tmp/random-dir")

        detector = ProjectDetector()

        with patch.object(detector, "safe_file_read", return_value=None):
            # _detect_from_directory returns None for 'random-dir' (it's in the exclusion list)
            # File detectors return None
            # _detect_from_parent_directory: random-dir is not in GENERIC_DIRS, returns None
            result = detector.detect()
            assert result == "Smart MCP Server"

    @patch("pathlib.Path.cwd")
    def test_detect_from_parent_directory_exception_path(self, mock_cwd):
        """Cover lines 135-136: exception in _detect_from_parent_directory."""
        from chuk_mcp_server.config.project_detector import ProjectDetector

        # Make cwd return a mock that raises on .parent.name
        mock_path = MagicMock()
        mock_path.name = "src"  # Generic dir, triggers parent check
        mock_path.parent.name = property(lambda _self: (_ for _ in ()).throw(OSError("no parent")))

        detector = ProjectDetector()

        # For _detect_from_directory, 'src' is in GENERIC_DIRS so returns None
        # For file detectors, return None
        # For _detect_from_parent_directory, cwd().name is 'src' (in GENERIC_DIRS)
        # Then cwd().parent.name should raise
        def side_effect_cwd():
            return mock_path

        mock_cwd.side_effect = side_effect_cwd
        mock_path.__truediv__ = lambda _self, _other: MagicMock(spec=Path)

        with patch.object(detector, "safe_file_read", return_value=None):
            # The parent mock: make parent.name raise
            type(mock_path.parent).name = PropertyMock(side_effect=OSError("no parent"))
            result = detector.detect()
            # Falls through to "Smart MCP Server"
            assert result == "Smart MCP Server"

    @patch("pathlib.Path.cwd")
    def test_cargo_toml_name_with_match_and_no_match(self, mock_cwd):
        """Cover lines 106->102, 108->102: Cargo.toml lines that don't match regex."""
        from chuk_mcp_server.config.project_detector import ProjectDetector

        mock_cwd.return_value = Path("/home/user/src")

        # Cargo.toml with lines that have name= but no matching quote pattern
        cargo_content = 'version = "0.1.0"\nname = badformat\nname = "good-name"'
        detector = ProjectDetector()

        def mock_safe_read(path):
            if "Cargo.toml" in str(path):
                return cargo_content
            return None

        with patch.object(detector, "safe_file_read", side_effect=mock_safe_read):
            result = detector.detect()
            assert result == "Good Name MCP Server"

    @patch("pathlib.Path.cwd")
    def test_package_json_empty_name(self, mock_cwd):
        """Cover line 75->77: package.json with empty name field."""
        from chuk_mcp_server.config.project_detector import ProjectDetector

        mock_cwd.return_value = Path("/home/user/src")

        package_data = json.dumps({"name": "", "version": "1.0.0"})
        detector = ProjectDetector()

        def mock_safe_read(path):
            if "package.json" in str(path):
                return package_data
            return None

        with patch.object(detector, "safe_file_read", side_effect=mock_safe_read):
            # Empty name should not match, falls through
            result = detector.detect()
            # Falls through to parent dir or fallback
            assert "MCP Server" in result

    @patch("pathlib.Path.cwd")
    def test_pyproject_toml_no_name_match(self, mock_cwd):
        """Cover lines 90->85: pyproject.toml where name regex does not match."""
        from chuk_mcp_server.config.project_detector import ProjectDetector

        mock_cwd.return_value = Path("/home/user/src")

        # name = without quotes won't match the regex
        toml_content = 'name = noQuotes\nversion = "1.0"'
        detector = ProjectDetector()

        def mock_safe_read(path):
            if "pyproject.toml" in str(path):
                return toml_content
            return None

        with patch.object(detector, "safe_file_read", side_effect=mock_safe_read):
            result = detector.detect()
            # Falls through, should still get some name from parent or fallback
            assert "MCP Server" in result

    @patch("pathlib.Path.cwd")
    def test_setup_py_name_starts_with_dunder(self, mock_cwd):
        """Cover line 108->102: setup.py name starting with __ is skipped."""
        from chuk_mcp_server.config.project_detector import ProjectDetector

        mock_cwd.return_value = Path("/home/user/src")

        setup_content = 'setup(\n    name="__private_name",\n)'
        detector = ProjectDetector()

        def mock_safe_read(path):
            if "setup.py" in str(path):
                return setup_content
            return None

        with patch.object(detector, "safe_file_read", side_effect=mock_safe_read):
            result = detector.detect()
            # __private_name is skipped, falls through
            assert "MCP Server" in result


# ============================================================================
# 6. config/smart_config.py
# ============================================================================


class TestSmartConfigCoverage:
    """Cover missing lines in SmartConfig."""

    def test_get_host_uncached(self):
        """Cover lines 104->110: get_host() when not in cache."""
        from chuk_mcp_server.config.smart_config import SmartConfig

        config = SmartConfig()
        # Ensure cache is empty
        assert "host" not in config._cache

        with patch.object(config.environment_detector, "detect", return_value="development"):
            with patch.object(config.container_detector, "detect", return_value=False):
                with patch.object(
                    config.network_detector,
                    "detect_network_config",
                    return_value=("localhost", 8000),
                ):
                    with patch.object(
                        config.cloud_detector,
                        "get_config_overrides",
                        return_value={},
                    ):
                        result = config.get_host()
                        assert result == "localhost"
                        assert config._cache["host"] == "localhost"

    def test_get_workers_uncached(self):
        """Cover lines 127->133: get_workers() when not in cache."""
        from chuk_mcp_server.config.smart_config import SmartConfig

        config = SmartConfig()
        assert "workers" not in config._cache

        with patch.object(config.environment_detector, "detect", return_value="production"):
            with patch.object(config.container_detector, "detect", return_value=False):
                with patch.object(
                    config.system_detector,
                    "detect_optimal_workers",
                    return_value=4,
                ):
                    with patch.object(
                        config.cloud_detector,
                        "get_config_overrides",
                        return_value={},
                    ):
                        result = config.get_workers()
                        assert result == 4
                        assert config._cache["workers"] == 4

    def test_get_max_connections_uncached(self):
        """Cover lines 136->142: get_max_connections() when not in cache."""
        from chuk_mcp_server.config.smart_config import SmartConfig

        config = SmartConfig()
        assert "max_connections" not in config._cache

        with patch.object(config.environment_detector, "detect", return_value="production"):
            with patch.object(config.container_detector, "detect", return_value=True):
                with patch.object(
                    config.system_detector,
                    "detect_max_connections",
                    return_value=5000,
                ):
                    with patch.object(
                        config.cloud_detector,
                        "get_config_overrides",
                        return_value={},
                    ):
                        result = config.get_max_connections()
                        assert result == 5000

    def test_should_enable_debug_uncached(self):
        """Cover lines 145->150: should_enable_debug() when not in cache."""
        from chuk_mcp_server.config.smart_config import SmartConfig

        config = SmartConfig()
        assert "debug" not in config._cache

        with patch.object(config.environment_detector, "detect", return_value="development"):
            with patch.object(
                config.system_detector,
                "detect_debug_mode",
                return_value=True,
            ):
                with patch.object(
                    config.cloud_detector,
                    "get_config_overrides",
                    return_value={},
                ):
                    result = config.should_enable_debug()
                    assert result is True

    def test_get_log_level_uncached(self):
        """Cover lines 153->158: get_log_level() when not in cache."""
        from chuk_mcp_server.config.smart_config import SmartConfig

        config = SmartConfig()
        assert "log_level" not in config._cache

        with patch.object(config.environment_detector, "detect", return_value="production"):
            with patch.object(
                config.system_detector,
                "detect_log_level",
                return_value="WARNING",
            ):
                with patch.object(
                    config.cloud_detector,
                    "get_config_overrides",
                    return_value={},
                ):
                    result = config.get_log_level()
                    assert result == "WARNING"

    def test_get_performance_mode_uncached(self):
        """Cover lines 161->167: get_performance_mode() when not in cache."""
        from chuk_mcp_server.config.smart_config import SmartConfig

        config = SmartConfig()
        assert "performance_mode" not in config._cache

        with patch.object(config.environment_detector, "detect", return_value="development"):
            with patch.object(
                config.system_detector,
                "detect_performance_mode",
                return_value="development",
            ):
                with patch.object(
                    config.cloud_detector,
                    "get_config_overrides",
                    return_value={},
                ):
                    result = config.get_performance_mode()
                    assert result == "development"

    def test_get_transport_mode_uncached(self):
        """Cover lines 170-172: get_transport_mode() when not in cache."""
        from chuk_mcp_server.config.smart_config import SmartConfig

        config = SmartConfig()
        assert "transport_mode" not in config._cache

        with patch.object(
            config.environment_detector,
            "detect_transport_mode",
            return_value="http",
        ):
            result = config.get_transport_mode()
            assert result == "http"
            assert config._cache["transport_mode"] == "http"

    def test_get_cloud_config(self):
        """Cover line 182: get_cloud_config() delegates to cloud_detector."""
        from chuk_mcp_server.config.smart_config import SmartConfig

        config = SmartConfig()
        with patch.object(
            config.cloud_detector,
            "get_config_overrides",
            return_value={"host": "0.0.0.0"},
        ):
            result = config.get_cloud_config()
            assert result == {"host": "0.0.0.0"}

    def test_is_cloud_environment(self):
        """Cover line 185: is_cloud_environment() delegates to cloud_detector."""
        from chuk_mcp_server.config.smart_config import SmartConfig

        config = SmartConfig()
        with patch.object(config.cloud_detector, "is_cloud_environment", return_value=False):
            assert config.is_cloud_environment() is False

    def test_get_cloud_summary_with_provider(self):
        """Cover line 191: get_cloud_summary when provider is detected."""
        from chuk_mcp_server.config.smart_config import SmartConfig

        config = SmartConfig()
        mock_provider = MagicMock()
        mock_provider.name = "aws"
        mock_provider.display_name = "Amazon Web Services"
        mock_provider.get_service_type.return_value = "ec2"
        mock_provider.get_environment_type.return_value = "production"

        with patch.object(config, "get_cloud_provider", return_value=mock_provider):
            result = config.get_cloud_summary()
            assert result["detected"] is True
            assert result["provider"] == "aws"
            assert result["display_name"] == "Amazon Web Services"
            assert result["service_type"] == "ec2"
            assert result["environment_type"] == "production"

    def test_get_cloud_summary_no_provider(self):
        """Cover line 190: get_cloud_summary returns {"detected": False} when no provider."""
        from chuk_mcp_server.config.smart_config import SmartConfig

        config = SmartConfig()
        with patch.object(config, "get_cloud_provider", return_value=None):
            result = config.get_cloud_summary()
            assert result == {"detected": False}

    def test_refresh_cloud_detection(self):
        """Cover lines 208-210: refresh_cloud_detection clears cloud-related cache."""
        from chuk_mcp_server.config.smart_config import SmartConfig

        config = SmartConfig()
        config._cache = {
            "cloud_provider": "aws",
            "cloud_config": {},
            "service_type": "ec2",
            "cloud_display_name": "AWS",
            "host": "0.0.0.0",
        }
        with patch.object(config.cloud_detector, "clear_cache"):
            config.refresh_cloud_detection()

        # Cloud keys should be removed, non-cloud keys preserved
        assert "cloud_provider" not in config._cache
        assert "cloud_config" not in config._cache
        assert "service_type" not in config._cache
        assert "cloud_display_name" not in config._cache
        assert "host" in config._cache

    def test_get_summary_with_detected_cloud(self):
        """Cover lines 229-230: get_summary when cloud is detected."""
        from chuk_mcp_server.config.smart_config import SmartConfig

        config = SmartConfig()

        mock_config = {
            "project_name": "Test MCP Server",
            "environment": "production",
            "host": "0.0.0.0",
            "port": 8000,
            "containerized": True,
            "performance_mode": "high_performance",
            "workers": 4,
            "max_connections": 5000,
            "log_level": "WARNING",
            "debug": False,
        }

        cloud_summary = {
            "detected": True,
            "display_name": "Amazon Web Services",
            "service_type": "ecs",
        }

        with patch.object(config, "get_all_defaults", return_value=mock_config):
            with patch.object(config, "get_cloud_summary", return_value=cloud_summary):
                result = config.get_summary()

                assert result["detection_summary"]["cloud"] == "Amazon Web Services"
                assert result["detection_summary"]["service"] == "ecs"
                assert result["cloud_summary"]["detected"] is True

    def test_get_summary_no_cloud(self):
        """Cover lines 232-233: get_summary when no cloud is detected."""
        from chuk_mcp_server.config.smart_config import SmartConfig

        config = SmartConfig()

        mock_config = {
            "project_name": "Test MCP Server",
            "environment": "development",
            "host": "localhost",
            "port": 8000,
            "containerized": False,
            "performance_mode": "development",
            "workers": 2,
            "max_connections": 1000,
            "log_level": "INFO",
            "debug": True,
        }

        with patch.object(config, "get_all_defaults", return_value=mock_config):
            with patch.object(config, "get_cloud_summary", return_value={"detected": False}):
                result = config.get_summary()
                assert result["detection_summary"]["cloud"] == "None detected"
                assert result["detection_summary"]["service"] == "N/A"

    def test_get_detailed_info(self):
        """Cover lines 237-238: get_detailed_info returns comprehensive info."""
        from chuk_mcp_server.config.smart_config import SmartConfig

        config = SmartConfig()

        mock_config = {
            "project_name": "Test MCP Server",
            "environment": "development",
            "host": "localhost",
            "port": 8000,
            "containerized": False,
            "performance_mode": "development",
            "workers": 2,
            "max_connections": 1000,
            "log_level": "INFO",
            "debug": True,
        }

        with patch.object(config, "get_all_defaults", return_value=mock_config):
            with patch.object(config, "get_cloud_summary", return_value={"detected": False}):
                result = config.get_detailed_info()

                assert "detection_summary" in result
                assert "full_config" in result
                assert "cloud_summary" in result
                assert "detectors" in result
                assert "detection_details" in result
                assert "cache_status" in result

                # Verify detectors section
                assert result["detectors"]["project_detector"] == "ProjectDetector"
                assert result["detectors"]["environment_detector"] == "EnvironmentDetector"
                assert result["detectors"]["cloud_detector"] == "CloudDetector"

                # Verify cache status
                assert "cached_keys" in result["cache_status"]
                assert "total_cached" in result["cache_status"]

    def test_get_workers_serverless_uncached(self):
        """Cover line 132: serverless env forces workers=1 in uncached get_workers."""
        from chuk_mcp_server.config.smart_config import SmartConfig

        config = SmartConfig()
        assert "workers" not in config._cache

        with patch.object(config.environment_detector, "detect", return_value="serverless"):
            with patch.object(config.container_detector, "detect", return_value=False):
                with patch.object(
                    config.system_detector,
                    "detect_optimal_workers",
                    return_value=4,
                ):
                    with patch.object(
                        config.cloud_detector,
                        "get_config_overrides",
                        return_value={},
                    ):
                        result = config.get_workers()
                        assert result == 1

    def test_get_max_connections_serverless_uncached(self):
        """Cover line 141: serverless env forces max_connections=100 in uncached path."""
        from chuk_mcp_server.config.smart_config import SmartConfig

        config = SmartConfig()
        assert "max_connections" not in config._cache

        with patch.object(config.environment_detector, "detect", return_value="serverless"):
            with patch.object(config.container_detector, "detect", return_value=False):
                with patch.object(
                    config.system_detector,
                    "detect_max_connections",
                    return_value=5000,
                ):
                    with patch.object(
                        config.cloud_detector,
                        "get_config_overrides",
                        return_value={},
                    ):
                        result = config.get_max_connections()
                        assert result == 100


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
