#!/usr/bin/env python3
"""Tests to cover missing lines in composition/manager.py (target: 100%)."""

import logging
import types
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from chuk_mcp_server.decorators import clear_global_registry

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_server(name="test-server", version="1.0"):
    """Create a fresh ChukMCPServer with an empty global registry."""
    clear_global_registry()
    from chuk_mcp_server import ChukMCPServer

    return ChukMCPServer(name=name, version=version)


def _register_tool(server, func, name=None, description=None):
    """Register a plain function as a tool on *server*."""
    from chuk_mcp_server.types import ToolHandler

    handler = ToolHandler.from_function(func, name=name, description=description)
    server.protocol.register_tool(handler)
    return handler


def _register_resource(server, func, uri, name=None, description=None, mime_type="text/plain"):
    """Register a plain function as a resource on *server*."""
    from chuk_mcp_server.types import ResourceHandler

    handler = ResourceHandler.from_function(
        uri=uri,
        func=func,
        name=name,
        description=description,
        mime_type=mime_type,
    )
    server.protocol.register_resource(handler)
    return handler


def _register_prompt(server, func, name=None, description=None):
    """Register a plain function as a prompt on *server*."""
    from chuk_mcp_server.types import PromptHandler

    handler = PromptHandler.from_function(func, name=name, description=description)
    server.protocol.register_prompt(handler)
    return handler


# =========================================================================
# import_server  (lines 114-171 -- already partly covered, but we exercise
#   prefix / no-prefix, component filtering, and tag filtering paths)
# =========================================================================


class TestImportServer:
    """Exercise import_server with and without prefix, component filtering."""

    def test_import_all_with_prefix(self):
        """Import all components from source into parent with a prefix."""
        parent = _make_server("parent")
        source = _make_server("source")

        _register_tool(source, lambda x: x, name="greet", description="Greet")
        _register_resource(source, lambda: "hi", uri="data://info", name="Info", description="Info resource")
        _register_prompt(source, lambda topic: f"Write about {topic}", name="writer", description="Write prompt")

        parent.composition.import_server(source, prefix="src")

        # Tools should be prefixed
        assert "src.greet" in parent.protocol.tools

        # Resources should be prefixed
        assert any("src" in uri for uri in parent.protocol.resources)

        # Prompts should be prefixed
        assert "src.writer" in parent.protocol.prompts

        # Stats updated
        assert parent.composition.composition_stats["imported"] == 1

    def test_import_without_prefix(self):
        """Import components with no prefix -- names stay unchanged."""
        parent = _make_server("parent")
        source = _make_server("source")

        _register_tool(source, lambda x: x, name="ping", description="Ping")

        # server_info is a Pydantic model (no .get()), so mock it as a dict
        # to exercise the `prefix or getattr(server, 'server_info', {}).get(...)` path
        source.server_info = {"name": "source"}

        parent.composition.import_server(source, prefix=None)

        assert "ping" in parent.protocol.tools

    def test_import_only_tools(self):
        """Import only tools via component filtering."""
        parent = _make_server("parent")
        source = _make_server("source")

        _register_tool(source, lambda x: x, name="t1", description="T1")
        _register_resource(source, lambda: "r", uri="data://r", name="R1", description="R1")
        _register_prompt(source, lambda: "p", name="p1", description="P1")

        parent.composition.import_server(source, prefix="ns", components=["tools"])

        assert "ns.t1" in parent.protocol.tools
        # Resources and prompts should NOT have been imported
        assert not any("ns" in uri for uri in parent.protocol.resources)
        assert "ns.p1" not in parent.protocol.prompts

    def test_import_only_resources(self):
        """Import only resources via component filtering."""
        parent = _make_server("parent")
        source = _make_server("source")

        _register_tool(source, lambda x: x, name="t1", description="T1")
        _register_resource(source, lambda: "r", uri="data://r", name="R1", description="R1")

        parent.composition.import_server(source, prefix="ns", components=["resources"])

        assert "ns.t1" not in parent.protocol.tools
        assert any("ns" in uri for uri in parent.protocol.resources)

    def test_import_only_prompts(self):
        """Import only prompts via component filtering."""
        parent = _make_server("parent")
        source = _make_server("source")

        _register_prompt(source, lambda: "p", name="p1", description="P1")

        parent.composition.import_server(source, prefix="ns", components=["prompts"])

        assert "ns.p1" in parent.protocol.prompts

    def test_import_resource_without_protocol_in_uri(self):
        """Resource URIs that do NOT contain :// get prefix/path style."""
        parent = _make_server("parent")
        source = _make_server("source")

        _register_resource(source, lambda: "v", uri="simple-path", name="Simple", description="No protocol")

        parent.composition.import_server(source, prefix="pfx", components=["resources"])

        assert "pfx/simple-path" in parent.protocol.resources

    def test_import_resource_without_prefix(self):
        """Resources imported without prefix are registered as-is."""
        parent = _make_server("parent")
        source = _make_server("source")

        _register_resource(source, lambda: "v", uri="data://keep", name="Keep", description="Keep")

        # Mock server_info as dict to work around Pydantic .get() issue
        source.server_info = {"name": "source"}

        parent.composition.import_server(source, prefix=None, components=["resources"])

        assert "data://keep" in parent.protocol.resources


# =========================================================================
# import_from_config  (lines 59-112)
# =========================================================================


class TestImportFromConfig:
    """Cover import_from_config: module branch, stdio/http/sse branch, unsupported type."""

    # ---- module branch (lines 63-83) ------------------------------------

    @pytest.mark.asyncio
    async def test_module_with_colon_attr(self):
        """type='module' with 'module_name:attr_name' format."""
        parent = _make_server("parent")

        fake_server = _make_server("fake-module-server")
        _register_tool(fake_server, lambda: "ok", name="mod_tool", description="mod")

        fake_module = types.ModuleType("fake_mod")
        fake_module.my_server = fake_server  # type: ignore[attr-defined]

        config = {"type": "module", "module": "fake_mod:my_server"}

        with patch("importlib.import_module", return_value=fake_module) as mock_imp:
            await parent.composition.import_from_config("srv", config, prefix="m")

        mock_imp.assert_called_once_with("fake_mod")
        assert "m.mod_tool" in parent.protocol.tools
        assert parent.composition.composition_stats["imported"] == 1

    @pytest.mark.asyncio
    async def test_module_without_colon(self):
        """type='module' without colon -- attr_name defaults to server_name."""
        parent = _make_server("parent")

        fake_server = _make_server("fake2")
        _register_tool(fake_server, lambda: "ok", name="t2", description="t2")
        # Mock server_info as dict so .get() works when prefix is None
        fake_server.server_info = {"name": "fake2"}

        fake_module = types.ModuleType("fake_mod2")
        fake_module.my_srv = fake_server  # type: ignore[attr-defined]

        config = {"type": "module", "module": "fake_mod2"}

        with patch("importlib.import_module", return_value=fake_module) as mock_imp:
            await parent.composition.import_from_config("my_srv", config, prefix=None)

        mock_imp.assert_called_once_with("fake_mod2")
        assert "t2" in parent.protocol.tools

    @pytest.mark.asyncio
    async def test_module_missing_module_key(self):
        """type='module' but 'module' key is missing -> ValueError."""
        parent = _make_server("parent")
        config = {"type": "module"}

        with pytest.raises(ValueError, match="Missing 'module'"):
            await parent.composition.import_from_config("bad", config)

    # ---- stdio / http / sse branch (lines 85-108) -----------------------

    @pytest.mark.asyncio
    async def test_stdio_type_uses_proxy_manager(self):
        """type='stdio' should create a ProxyManager and start_servers."""
        parent = _make_server("parent")
        config = {"type": "stdio", "command": "some-cmd"}

        mock_pm_instance = MagicMock()
        mock_pm_instance.start_servers = AsyncMock()

        with patch(
            "chuk_mcp_server.proxy.manager.ProxyManager",
            return_value=mock_pm_instance,
        ) as mock_pm_cls:
            await parent.composition.import_from_config("stdio_srv", config, prefix="sp")

        mock_pm_cls.assert_called_once()
        mock_pm_instance.start_servers.assert_awaited_once()

        # Verify proxy config was built correctly
        call_args = mock_pm_cls.call_args
        proxy_config = call_args[0][0]
        assert proxy_config["proxy"]["enabled"] is True
        assert proxy_config["proxy"]["namespace"] == "sp"
        assert "stdio_srv" in proxy_config["servers"]

        # Stats incremented
        assert parent.composition.composition_stats["imported"] == 1

        # _proxy_managers list created
        assert hasattr(parent.composition, "_proxy_managers")
        assert mock_pm_instance in parent.composition._proxy_managers

    @pytest.mark.asyncio
    async def test_http_type_uses_proxy_manager(self):
        """type='http' should also go through ProxyManager path."""
        parent = _make_server("parent")
        config = {"type": "http", "url": "http://localhost:9000"}

        mock_pm_instance = MagicMock()
        mock_pm_instance.start_servers = AsyncMock()

        with patch(
            "chuk_mcp_server.proxy.manager.ProxyManager",
            return_value=mock_pm_instance,
        ):
            await parent.composition.import_from_config("http_srv", config, prefix=None)

        # prefix=None => namespace=""
        assert parent.composition.composition_stats["imported"] == 1

    @pytest.mark.asyncio
    async def test_sse_type_uses_proxy_manager(self):
        """type='sse' should also go through ProxyManager path."""
        parent = _make_server("parent")
        config = {"type": "sse", "url": "http://localhost:9001/sse"}

        mock_pm_instance = MagicMock()
        mock_pm_instance.start_servers = AsyncMock()

        with patch(
            "chuk_mcp_server.proxy.manager.ProxyManager",
            return_value=mock_pm_instance,
        ):
            await parent.composition.import_from_config("sse_srv", config, prefix="sse_ns")

        assert parent.composition.composition_stats["imported"] == 1

    @pytest.mark.asyncio
    async def test_proxy_managers_list_accumulates(self):
        """Calling import_from_config twice should accumulate _proxy_managers."""
        parent = _make_server("parent")

        mock_pm_1 = MagicMock()
        mock_pm_1.start_servers = AsyncMock()
        mock_pm_2 = MagicMock()
        mock_pm_2.start_servers = AsyncMock()

        with patch(
            "chuk_mcp_server.proxy.manager.ProxyManager",
            side_effect=[mock_pm_1, mock_pm_2],
        ):
            await parent.composition.import_from_config(
                "s1",
                {"type": "stdio", "command": "c1"},
                prefix="a",
            )
            await parent.composition.import_from_config(
                "s2",
                {"type": "http", "url": "http://x"},
                prefix="b",
            )

        assert len(parent.composition._proxy_managers) == 2

    # ---- unsupported type branch (lines 110-112) -------------------------

    @pytest.mark.asyncio
    async def test_unsupported_type_logs_warning(self, caplog):
        """An unsupported type should log a warning and return without error."""
        parent = _make_server("parent")
        config = {"type": "unknown"}

        with caplog.at_level(logging.WARNING):
            await parent.composition.import_from_config("bad_srv", config)

        assert "not supported" in caplog.text
        # Stats should NOT be incremented
        assert parent.composition.composition_stats["imported"] == 0

    @pytest.mark.asyncio
    async def test_default_type_is_module(self):
        """When 'type' is missing, it defaults to 'module'."""
        parent = _make_server("parent")

        fake_server = _make_server("default-type")
        # Mock server_info as dict so .get() works when prefix is None
        fake_server.server_info = {"name": "default-type"}
        fake_module = types.ModuleType("dflt")
        fake_module.ds = fake_server  # type: ignore[attr-defined]

        config = {"module": "dflt:ds"}  # no "type" key

        with patch("importlib.import_module", return_value=fake_module):
            await parent.composition.import_from_config("ds", config)

        # Should have gone through the module path successfully
        assert parent.composition.composition_stats["imported"] == 1


# =========================================================================
# mount  (lines 173-218) -- both as_proxy=True and as_proxy=False
# Both paths raise NotImplementedError and are caught; stats should NOT
# be incremented.
# =========================================================================


class TestMount:
    """Cover mount() -- NotImplementedError paths and stats tracking."""

    def test_mount_dynamic_not_implemented(self, caplog):
        """mount(as_proxy=False) triggers _mount_dynamic -> NotImplementedError."""
        parent = _make_server("parent")
        source = _make_server("child")

        with caplog.at_level(logging.WARNING):
            parent.composition.mount(source, prefix="dyn", as_proxy=False)

        # Warning should have been logged
        assert "Dynamic mounting is not yet implemented" in caplog.text

        # Stats should NOT be incremented (early return before lines 211-218)
        assert parent.composition.composition_stats["mounted"] == 0
        assert "dyn" not in parent.composition.mounted_servers

    def test_mount_proxy_not_implemented(self, caplog):
        """mount(as_proxy=True) triggers _mount_as_proxy -> NotImplementedError."""
        parent = _make_server("parent")
        source = _make_server("child")

        with caplog.at_level(logging.WARNING):
            parent.composition.mount(source, prefix="prx", as_proxy=True)

        assert "Proxy mounting is not yet implemented" in caplog.text

        # Stats should NOT be incremented
        assert parent.composition.composition_stats["mounted"] == 0
        assert "prx" not in parent.composition.mounted_servers

    def test_mount_server_name_from_server_info(self, caplog):
        """When prefix is None, server_name is derived from server.server_info."""
        parent = _make_server("parent")
        source = _make_server("child-name")
        # Mock server_info as dict so .get() works when prefix is None
        source.server_info = {"name": "child-name"}

        with caplog.at_level(logging.WARNING):
            parent.composition.mount(source, prefix=None, as_proxy=False)

        # NotImplementedError path still fires, so stats not incremented
        assert parent.composition.composition_stats["mounted"] == 0

    def test_mount_dynamic_success_increments_stats(self):
        """If _mount_dynamic were to succeed, stats and mounted_servers are updated."""
        parent = _make_server("parent")
        source = _make_server("child")

        # Monkey-patch _mount_dynamic to succeed (no exception)
        parent.composition._mount_dynamic = MagicMock()

        parent.composition.mount(source, prefix="ok")

        assert parent.composition.composition_stats["mounted"] == 1
        assert "ok" in parent.composition.mounted_servers
        info = parent.composition.mounted_servers["ok"]
        assert info["prefix"] == "ok"
        assert info["as_proxy"] is False

    def test_mount_proxy_success_increments_stats(self):
        """If _mount_as_proxy were to succeed, stats and mounted_servers are updated."""
        parent = _make_server("parent")
        source = _make_server("child")

        # Monkey-patch _mount_as_proxy to succeed (no exception)
        parent.composition._mount_as_proxy = MagicMock()

        parent.composition.mount(source, prefix="ok_proxy", as_proxy=True)

        assert parent.composition.composition_stats["mounted"] == 1
        assert "ok_proxy" in parent.composition.mounted_servers
        info = parent.composition.mounted_servers["ok_proxy"]
        assert info["as_proxy"] is True


# =========================================================================
# get_composition_stats  (lines 374-381)
# =========================================================================


class TestGetCompositionStats:
    """Cover get_composition_stats() return structure."""

    def test_empty_stats(self):
        """Stats on a fresh manager with nothing imported."""
        parent = _make_server("parent")
        stats = parent.composition.get_composition_stats()

        assert stats["stats"]["imported"] == 0
        assert stats["stats"]["mounted"] == 0
        assert stats["stats"]["modules"] == 0
        assert stats["stats"]["proxied"] == 0
        assert stats["imported_servers"] == []
        assert stats["mounted_servers"] == []
        assert stats["total_components"] == 0

    def test_stats_after_import(self):
        """Stats after importing a server with some tools."""
        parent = _make_server("parent")
        source = _make_server("source")

        _register_tool(source, lambda: "a", name="a", description="a")
        _register_tool(source, lambda: "b", name="b", description="b")
        _register_resource(source, lambda: "r", uri="data://r", name="R", description="R")

        parent.composition.import_server(source, prefix="src")

        stats = parent.composition.get_composition_stats()

        assert stats["stats"]["imported"] == 1
        assert "src" in stats["imported_servers"]
        # 2 tools + 1 resource = 3 total components
        assert stats["total_components"] == 3

    def test_stats_after_multiple_imports(self):
        """Stats accumulate correctly across multiple imports."""
        parent = _make_server("parent")

        s1 = _make_server("s1")
        _register_tool(s1, lambda: "1", name="t1", description="T1")

        s2 = _make_server("s2")
        _register_tool(s2, lambda: "2", name="t2", description="T2")
        _register_prompt(s2, lambda: "p", name="p1", description="P1")

        parent.composition.import_server(s1, prefix="ns1")
        parent.composition.import_server(s2, prefix="ns2")

        stats = parent.composition.get_composition_stats()

        assert stats["stats"]["imported"] == 2
        assert set(stats["imported_servers"]) == {"ns1", "ns2"}
        # ns1: 1 tool, ns2: 1 tool + 1 prompt = 3 total
        assert stats["total_components"] == 3

    def test_stats_copy_isolation(self):
        """The returned stats dict should be a copy, not a live reference."""
        parent = _make_server("parent")
        stats1 = parent.composition.get_composition_stats()
        stats1["stats"]["imported"] = 999

        stats2 = parent.composition.get_composition_stats()
        assert stats2["stats"]["imported"] == 0  # unaffected by mutation


# =========================================================================
# _matches_tags  (lines 364-372)
# =========================================================================


class TestMatchesTags:
    """Cover _matches_tags directly and via tag-filtered import_server calls."""

    def test_matches_tags_no_handler_tags(self):
        """Handler without tags attribute returns False."""
        parent = _make_server("parent")
        handler = MagicMock(spec=[])  # no 'tags' attribute
        assert parent.composition._matches_tags(handler, ["foo"]) is False

    def test_matches_tags_empty_handler_tags(self):
        """Handler with empty tags list returns False."""
        parent = _make_server("parent")
        handler = MagicMock()
        handler.tags = []
        assert parent.composition._matches_tags(handler, ["foo"]) is False

    def test_matches_tags_matching(self):
        """Handler with matching tag returns True."""
        parent = _make_server("parent")
        handler = MagicMock()
        handler.tags = ["alpha", "beta"]
        assert parent.composition._matches_tags(handler, ["beta"]) is True

    def test_matches_tags_no_match(self):
        """Handler with non-matching tags returns False."""
        parent = _make_server("parent")
        handler = MagicMock()
        handler.tags = ["alpha"]
        assert parent.composition._matches_tags(handler, ["gamma"]) is False

    def test_import_with_tags_filters_tools(self):
        """Tag filtering skips tools that do not match (line 267 continue)."""
        parent = _make_server("parent")
        source = _make_server("source")

        h1 = _register_tool(source, lambda: "1", name="t_match", description="M")
        h1.tags = ["wanted"]

        h2 = _register_tool(source, lambda: "2", name="t_skip", description="S")
        h2.tags = ["other"]

        parent.composition.import_server(source, prefix="f", tags=["wanted"])

        assert "f.t_match" in parent.protocol.tools
        assert "f.t_skip" not in parent.protocol.tools

    def test_import_with_tags_filters_resources(self):
        """Tag filtering skips resources that do not match (line 297 continue)."""
        parent = _make_server("parent")
        source = _make_server("source")

        h1 = _register_resource(source, lambda: "1", uri="data://yes", name="Y", description="Y")
        h1.tags = ["wanted"]

        h2 = _register_resource(source, lambda: "2", uri="data://no", name="N", description="N")
        h2.tags = ["other"]

        parent.composition.import_server(
            source,
            prefix="f",
            components=["resources"],
            tags=["wanted"],
        )

        assert any("yes" in uri for uri in parent.protocol.resources)
        assert not any("no" in uri for uri in parent.protocol.resources)

    def test_import_with_tags_filters_prompts(self):
        """Tag filtering skips prompts that do not match (line 335 continue)."""
        parent = _make_server("parent")
        source = _make_server("source")

        h1 = _register_prompt(source, lambda: "1", name="p_yes", description="Y")
        h1.tags = ["wanted"]

        h2 = _register_prompt(source, lambda: "2", name="p_no", description="N")
        h2.tags = ["other"]

        parent.composition.import_server(
            source,
            prefix="f",
            components=["prompts"],
            tags=["wanted"],
        )

        assert "f.p_yes" in parent.protocol.prompts
        assert "f.p_no" not in parent.protocol.prompts


# =========================================================================
# load_module  (lines 246-256)
# =========================================================================


class TestLoadModule:
    """Cover load_module() by mocking ModuleLoader."""

    def test_load_module_delegates_to_module_loader(self):
        """load_module wraps config and delegates to ModuleLoader."""
        parent = _make_server("parent")

        mock_loader_instance = MagicMock()
        mock_loader_instance.load_modules.return_value = {
            "math": ["add", "subtract"],
        }

        with patch(
            "chuk_mcp_server.modules.ModuleLoader",
            return_value=mock_loader_instance,
        ) as mock_loader_cls:
            result = parent.composition.load_module({"math": {"enabled": True}})

        # ModuleLoader constructed with proper config wrapping
        call_args = mock_loader_cls.call_args
        assert call_args[0][0] == {"tool_modules": {"math": {"enabled": True}}}
        assert call_args[0][1] is parent  # parent_server passed

        # load_modules() called
        mock_loader_instance.load_modules.assert_called_once()

        # Result forwarded
        assert result == {"math": ["add", "subtract"]}

        # Stats incremented by number of modules
        assert parent.composition.composition_stats["modules"] == 1

    def test_load_module_empty_result(self):
        """load_module with no modules loaded gives stats == 0."""
        parent = _make_server("parent")

        mock_loader_instance = MagicMock()
        mock_loader_instance.load_modules.return_value = {}

        with patch(
            "chuk_mcp_server.modules.ModuleLoader",
            return_value=mock_loader_instance,
        ):
            result = parent.composition.load_module({})

        assert result == {}
        assert parent.composition.composition_stats["modules"] == 0


# =========================================================================
# Edge cases: server without protocol attribute (partial branches)
# Lines 263->286, 293->324, 331->354
# =========================================================================


class TestImportServerWithoutProtocol:
    """Cover branches where server lacks protocol or protocol lacks tools/resources/prompts."""

    def test_import_tools_from_server_without_protocol(self):
        """_import_tools returns 0 if server has no 'protocol' attribute."""
        parent = _make_server("parent")
        # Create a bare object with no protocol
        bare_server = MagicMock(spec=[])  # no attributes at all

        count = parent.composition._import_tools(bare_server, prefix="x", tags=None)
        assert count == 0

    def test_import_resources_from_server_without_protocol(self):
        """_import_resources returns 0 if server has no 'protocol' attribute."""
        parent = _make_server("parent")
        bare_server = MagicMock(spec=[])

        count = parent.composition._import_resources(bare_server, prefix="x", tags=None)
        assert count == 0

    def test_import_prompts_from_server_without_protocol(self):
        """_import_prompts returns 0 if server has no 'protocol' attribute."""
        parent = _make_server("parent")
        bare_server = MagicMock(spec=[])

        count = parent.composition._import_prompts(bare_server, prefix="x", tags=None)
        assert count == 0

    def test_import_tools_from_server_without_tools_on_protocol(self):
        """_import_tools returns 0 if protocol has no 'tools' attribute."""
        parent = _make_server("parent")
        server = MagicMock()
        server.protocol = MagicMock(spec=[])  # protocol exists but has no 'tools'

        count = parent.composition._import_tools(server, prefix="x", tags=None)
        assert count == 0

    def test_import_resources_from_server_without_resources_on_protocol(self):
        """_import_resources returns 0 if protocol has no 'resources' attribute."""
        parent = _make_server("parent")
        server = MagicMock()
        server.protocol = MagicMock(spec=[])

        count = parent.composition._import_resources(server, prefix="x", tags=None)
        assert count == 0

    def test_import_prompts_from_server_without_prompts_on_protocol(self):
        """_import_prompts returns 0 if protocol has no 'prompts' attribute."""
        parent = _make_server("parent")
        server = MagicMock()
        server.protocol = MagicMock(spec=[])

        count = parent.composition._import_prompts(server, prefix="x", tags=None)
        assert count == 0
