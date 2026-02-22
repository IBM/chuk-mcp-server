#!/usr/bin/env python3
"""Thin OpenTelemetry wrapper - zero overhead when otel is not installed."""

import contextlib
import time
from collections.abc import Generator
from typing import Any

# Try to import OpenTelemetry; fall back to no-ops
try:
    from opentelemetry import trace

    _tracer = trace.get_tracer("chuk_mcp_server")
    _OTEL_AVAILABLE = True
except ImportError:
    _tracer = None
    _OTEL_AVAILABLE = False


def is_telemetry_available() -> bool:
    """Check if OpenTelemetry is installed and available."""
    return _OTEL_AVAILABLE


@contextlib.contextmanager
def trace_tool_call(tool_name: str) -> Generator[dict[str, Any], None, None]:
    """Context manager that traces a tool call.

    When OpenTelemetry is available, creates a span.
    When not available, uses a lightweight timing dict.

    Usage:
        with trace_tool_call("my_tool") as ctx:
            result = await tool.execute(args)
            ctx["result_size"] = len(str(result))
    """
    ctx: dict[str, Any] = {"tool_name": tool_name, "start_time": time.monotonic()}

    if _OTEL_AVAILABLE and _tracer is not None:
        with _tracer.start_as_current_span(
            f"tool.{tool_name}",
            attributes={"tool.name": tool_name},
        ) as span:
            try:
                yield ctx
            except Exception as e:
                span.set_status(trace.Status(trace.StatusCode.ERROR, str(e)))
                raise
            finally:
                ctx["duration"] = time.monotonic() - ctx["start_time"]
                span.set_attribute("tool.duration_ms", ctx["duration"] * 1000)
    else:
        try:
            yield ctx
        finally:
            ctx["duration"] = time.monotonic() - ctx["start_time"]
