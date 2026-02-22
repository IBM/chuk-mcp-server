#!/usr/bin/env python3
# tests/types/test_capabilities.py
"""
Unit tests for chuk_mcp_server.types.capabilities module

Tests server capability creation and management functionality.
"""

import pytest


def test_create_server_capabilities_default():
    """Test creating server capabilities with default settings."""
    from chuk_mcp_server.types.base import ServerCapabilities
    from chuk_mcp_server.types.capabilities import create_server_capabilities

    capabilities = create_server_capabilities()

    assert isinstance(capabilities, ServerCapabilities)

    # Check default values (tools=True, resources=True, prompts=False, logging=False)
    capabilities_dict = capabilities.model_dump()

    assert "tools" in capabilities_dict
    assert "resources" in capabilities_dict
    assert capabilities_dict["tools"]["listChanged"] is True
    assert capabilities_dict["resources"]["listChanged"] is True
    assert capabilities_dict["resources"]["subscribe"] is True


def test_create_server_capabilities_all_enabled():
    """Test creating server capabilities with all capabilities enabled."""
    from chuk_mcp_server.types.capabilities import create_server_capabilities

    capabilities = create_server_capabilities(tools=True, resources=True, prompts=True, logging=True)

    capabilities_dict = capabilities.model_dump()

    assert "tools" in capabilities_dict
    assert "resources" in capabilities_dict
    assert "prompts" in capabilities_dict
    assert "logging" in capabilities_dict


def test_create_server_capabilities_all_disabled():
    """Test creating server capabilities with all capabilities disabled."""
    from chuk_mcp_server.types.capabilities import create_server_capabilities

    capabilities = create_server_capabilities(tools=False, resources=False, prompts=False, logging=False)

    capabilities_dict = capabilities.model_dump()

    # When disabled, capabilities should not be present in the dict
    assert "tools" not in capabilities_dict
    assert "resources" not in capabilities_dict
    assert "prompts" not in capabilities_dict
    assert "logging" not in capabilities_dict


def test_create_server_capabilities_selective():
    """Test creating server capabilities with selective enabling."""
    from chuk_mcp_server.types.capabilities import create_server_capabilities

    # Enable only tools and logging
    capabilities = create_server_capabilities(tools=True, resources=False, prompts=False, logging=True)

    capabilities_dict = capabilities.model_dump()

    assert "tools" in capabilities_dict
    assert "logging" in capabilities_dict
    assert "resources" not in capabilities_dict
    assert "prompts" not in capabilities_dict


def test_create_server_capabilities_with_experimental():
    """Test creating server capabilities with experimental features."""
    from chuk_mcp_server.types.capabilities import create_server_capabilities

    experimental_features = {
        "feature_x": True,
        "feature_y": {"enabled": True, "version": "beta"},
        "feature_z": ["option1", "option2"],
    }

    capabilities = create_server_capabilities(tools=True, experimental=experimental_features)

    capabilities_dict = capabilities.model_dump()

    assert "experimental" in capabilities_dict
    assert capabilities_dict["experimental"]["feature_x"] is True
    assert capabilities_dict["experimental"]["feature_y"]["enabled"] is True
    assert capabilities_dict["experimental"]["feature_z"] == ["option1", "option2"]


def test_tools_capability_details():
    """Test that tools capability has correct details."""
    from chuk_mcp_server.types.capabilities import create_server_capabilities

    capabilities = create_server_capabilities(tools=True)
    capabilities_dict = capabilities.model_dump()

    tools_cap = capabilities_dict["tools"]
    assert tools_cap["listChanged"] is True


def test_resources_capability_details():
    """Test that resources capability has correct details."""
    from chuk_mcp_server.types.capabilities import create_server_capabilities

    capabilities = create_server_capabilities(resources=True)
    capabilities_dict = capabilities.model_dump()

    resources_cap = capabilities_dict["resources"]
    assert resources_cap["listChanged"] is True
    assert resources_cap["subscribe"] is True  # Subscriptions enabled


def test_prompts_capability_details():
    """Test that prompts capability has correct details."""
    from chuk_mcp_server.types.capabilities import create_server_capabilities

    capabilities = create_server_capabilities(prompts=True)
    capabilities_dict = capabilities.model_dump()

    prompts_cap = capabilities_dict["prompts"]
    assert prompts_cap["listChanged"] is True


def test_logging_capability_details():
    """Test that logging capability is created correctly."""
    from chuk_mcp_server.types.capabilities import create_server_capabilities

    capabilities = create_server_capabilities(logging=True)
    capabilities_dict = capabilities.model_dump()

    assert "logging" in capabilities_dict
    # LoggingCapability might be empty or have specific fields
    logging_cap = capabilities_dict["logging"]
    assert isinstance(logging_cap, dict)


def test_capability_types_import():
    """Test that capability types can be imported and used directly."""
    from chuk_mcp_server.types.base import LoggingCapability, PromptsCapability, ResourcesCapability, ToolsCapability

    # Test that we can create individual capabilities
    tools_cap = ToolsCapability(listChanged=True)
    assert tools_cap.listChanged is True

    resources_cap = ResourcesCapability(listChanged=True, subscribe=False)
    assert resources_cap.listChanged is True
    assert resources_cap.subscribe is False

    prompts_cap = PromptsCapability(listChanged=True)
    assert prompts_cap.listChanged is True

    logging_cap = LoggingCapability()
    assert logging_cap is not None


def test_capabilities_serialization():
    """Test that capabilities can be serialized properly."""
    from chuk_mcp_server.types.capabilities import create_server_capabilities

    capabilities = create_server_capabilities(tools=True, resources=True, prompts=True, experimental={"test": True})

    # Test model_dump
    capabilities_dict = capabilities.model_dump()
    assert isinstance(capabilities_dict, dict)

    # Test model_dump with exclude_none
    capabilities_dict_clean = capabilities.model_dump(exclude_none=True)
    assert isinstance(capabilities_dict_clean, dict)

    # Should contain all enabled capabilities
    assert "tools" in capabilities_dict_clean
    assert "resources" in capabilities_dict_clean
    assert "prompts" in capabilities_dict_clean
    assert "experimental" in capabilities_dict_clean


def test_capabilities_empty_experimental():
    """Test creating capabilities with empty experimental dict."""
    from chuk_mcp_server.types.capabilities import create_server_capabilities

    capabilities = create_server_capabilities(tools=True, experimental={})

    capabilities_dict = capabilities.model_dump()
    assert "experimental" in capabilities_dict
    assert capabilities_dict["experimental"] == {}


def test_capabilities_none_experimental():
    """Test creating capabilities with None experimental."""
    from chuk_mcp_server.types.capabilities import create_server_capabilities

    capabilities = create_server_capabilities(tools=True, experimental=None)

    capabilities_dict = capabilities.model_dump()
    # None experimental should not be included
    assert "experimental" not in capabilities_dict


def test_capabilities_edge_cases():
    """Test edge cases in capability creation."""
    from chuk_mcp_server.types.capabilities import create_server_capabilities

    # Test with all False (empty capabilities)
    empty_capabilities = create_server_capabilities(tools=False, resources=False, prompts=False, logging=False)

    empty_dict = empty_capabilities.model_dump()
    # Should be an empty dict or minimal structure
    assert isinstance(empty_dict, dict)

    # Test with complex experimental data
    complex_experimental = {
        "nested": {"deep": {"very_deep": ["a", "b", "c"]}},
        "numbers": [1, 2, 3, 4, 5],
        "mixed": {"bool": True, "string": "test", "number": 42, "null": None},
    }

    complex_capabilities = create_server_capabilities(tools=True, experimental=complex_experimental)

    complex_dict = complex_capabilities.model_dump()
    assert complex_dict["experimental"]["nested"]["deep"]["very_deep"] == ["a", "b", "c"]
    assert complex_dict["experimental"]["numbers"] == [1, 2, 3, 4, 5]
    assert complex_dict["experimental"]["mixed"]["bool"] is True


def test_capabilities_model_type():
    """Test that created capabilities are of correct type."""
    from chuk_mcp_server.types.base import ServerCapabilities
    from chuk_mcp_server.types.capabilities import create_server_capabilities

    capabilities = create_server_capabilities()

    assert isinstance(capabilities, ServerCapabilities)
    assert hasattr(capabilities, "model_dump")
    assert callable(capabilities.model_dump)


def test_module_exports():
    """Test that all expected exports are available."""
    from chuk_mcp_server.types import capabilities

    assert hasattr(capabilities, "__all__")
    assert isinstance(capabilities.__all__, list)

    expected_exports = ["create_server_capabilities"]

    for export in expected_exports:
        assert export in capabilities.__all__
        assert hasattr(capabilities, export)


def test_capabilities_consistency():
    """Test that multiple calls with same parameters produce consistent results."""
    from chuk_mcp_server.types.capabilities import create_server_capabilities

    params = {"tools": True, "resources": True, "prompts": False, "logging": True, "experimental": {"test": "value"}}

    cap1 = create_server_capabilities(**params)
    cap2 = create_server_capabilities(**params)

    # Should produce equivalent results
    dict1 = cap1.model_dump()
    dict2 = cap2.model_dump()

    assert dict1 == dict2


if __name__ == "__main__":
    pytest.main([__file__])
