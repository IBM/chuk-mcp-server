#!/usr/bin/env python3
"""Tests for thread safety of global registry and singleton."""

import threading

from chuk_mcp_server.decorators import (
    _global_prompts,
    _global_resource_templates,
    _global_resources,
    _global_tools,
    _registry_lock,
    clear_global_registry,
)


class TestRegistryLock:
    """Verify _registry_lock exists and protects global lists."""

    def test_registry_lock_exists(self):
        assert isinstance(_registry_lock, type(threading.Lock()))

    def test_clear_uses_clear_not_reassignment(self):
        """clear_global_registry should use .clear(), not = [], to preserve references."""
        # Get references to the original lists
        tools_ref = _global_tools
        resources_ref = _global_resources
        prompts_ref = _global_prompts
        templates_ref = _global_resource_templates

        clear_global_registry()

        # After clear, the original list objects should still be the same
        # (i.e., clear() was used, not reassignment)
        assert _global_tools is tools_ref
        assert _global_resources is resources_ref
        assert _global_prompts is prompts_ref
        assert _global_resource_templates is templates_ref

    def test_concurrent_tool_registration(self):
        """Multiple threads registering tools should not corrupt the list."""
        clear_global_registry()
        from chuk_mcp_server.types import ToolHandler

        errors: list[Exception] = []

        def register_tools(start: int, count: int) -> None:
            try:
                for i in range(start, start + count):

                    def fn(**kwargs):
                        return "ok"

                    tool = ToolHandler.from_function(fn, name=f"tool_{i}")
                    with _registry_lock:
                        _global_tools.append(tool)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=register_tools, args=(i * 100, 100)) for i in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors
        assert len(_global_tools) == 400

        # Cleanup
        clear_global_registry()


class TestSingletonThreadSafety:
    """Verify double-checked locking on _global_server."""

    def test_server_lock_exists(self):
        """_server_lock should be a threading.Lock in __init__.py."""
        import chuk_mcp_server

        assert hasattr(chuk_mcp_server, "_server_lock")

    def test_concurrent_get_or_create(self):
        """Multiple threads calling get_or_create_global_server should get the same instance."""
        import chuk_mcp_server

        # Reset to None for testing
        original = chuk_mcp_server._global_server
        chuk_mcp_server._global_server = None

        results: list[object] = []
        errors: list[Exception] = []

        def get_server() -> None:
            try:
                server = chuk_mcp_server.get_or_create_global_server()
                results.append(server)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=get_server) for _ in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # Restore original
        chuk_mcp_server._global_server = original

        assert not errors
        # All threads should get the same instance
        assert len({id(r) for r in results}) == 1
