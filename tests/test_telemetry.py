#!/usr/bin/env python3
"""Tests for the telemetry module (thin OpenTelemetry wrapper)."""

import pytest

from chuk_mcp_server.telemetry import is_telemetry_available, trace_tool_call


class TestIsTelemetryAvailable:
    """Test is_telemetry_available function."""

    def test_returns_bool(self):
        """is_telemetry_available() returns a boolean reflecting otel installation."""
        result = is_telemetry_available()
        assert isinstance(result, bool)

    def test_matches_otel_import(self):
        """is_telemetry_available() matches whether opentelemetry can be imported."""
        try:
            import opentelemetry  # noqa: F401

            expected = True
        except ImportError:
            expected = False
        assert is_telemetry_available() is expected


class TestTraceToolCall:
    """Test trace_tool_call context manager."""

    def test_works_as_context_manager_without_otel(self):
        """trace_tool_call works as a context manager even without OpenTelemetry."""
        with trace_tool_call("test_tool") as ctx:
            assert ctx["tool_name"] == "test_tool"
            assert "start_time" in ctx

    def test_populates_timing_info(self):
        """trace_tool_call populates duration in ctx dict after exiting."""
        with trace_tool_call("timed_tool") as ctx:
            # Simulate some work
            sum(range(1000))

        assert "duration" in ctx
        assert isinstance(ctx["duration"], float)
        assert ctx["duration"] >= 0

    def test_propagates_exceptions(self):
        """trace_tool_call propagates exceptions raised inside the context."""
        with pytest.raises(ValueError, match="test error"):
            with trace_tool_call("failing_tool") as ctx:
                raise ValueError("test error")

        # Duration should still be set even when an exception occurs
        assert "duration" in ctx
        assert ctx["duration"] >= 0

    def test_ctx_can_store_additional_data(self):
        """The ctx dict returned by trace_tool_call can store arbitrary user data."""
        with trace_tool_call("custom_tool") as ctx:
            ctx["result_size"] = 42
            ctx["custom_key"] = "custom_value"

        assert ctx["result_size"] == 42
        assert ctx["custom_key"] == "custom_value"
        assert ctx["tool_name"] == "custom_tool"
        assert "duration" in ctx
