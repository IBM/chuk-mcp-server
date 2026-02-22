#!/usr/bin/env python3
"""Tests for graceful shutdown of protocol handler."""

import asyncio
import contextlib

import pytest

from chuk_mcp_server.protocol import MCPProtocolHandler
from chuk_mcp_server.types import ServerInfo, create_server_capabilities


def _make_handler():
    info = ServerInfo(name="test", version="0.1.0")
    caps = create_server_capabilities(tools=True, resources=True, prompts=True)
    return MCPProtocolHandler(server_info=info, capabilities=caps)


class TestProtocolShutdown:
    """Test MCPProtocolHandler.shutdown()."""

    @pytest.mark.asyncio
    async def test_shutdown_clears_sessions(self):
        handler = _make_handler()
        handler.session_manager.create_session({"name": "c1"}, "2025-06-18")
        handler.session_manager.create_session({"name": "c2"}, "2025-06-18")

        await handler.shutdown()

        assert len(handler.session_manager.sessions) == 0

    @pytest.mark.asyncio
    async def test_shutdown_clears_task_store(self):
        handler = _make_handler()
        handler._task_store["t1"] = {"id": "t1", "status": "completed"}

        await handler.shutdown()

        assert len(handler._task_store) == 0

    @pytest.mark.asyncio
    async def test_shutdown_cleans_session_state(self):
        handler = _make_handler()
        sid = handler.session_manager.create_session({"name": "c1"}, "2025-06-18")
        handler._resource_subscriptions[sid] = {"res://a"}
        handler._sse_event_buffers[sid] = [(1, {"data": "x"})]
        handler._sse_event_counters[sid] = 1

        await handler.shutdown()

        assert sid not in handler._resource_subscriptions
        assert sid not in handler._sse_event_buffers
        assert sid not in handler._sse_event_counters

    @pytest.mark.asyncio
    async def test_shutdown_cancels_in_flight_requests(self):
        handler = _make_handler()

        # Create a long-running task
        async def long_task():
            await asyncio.sleep(60)

        task = asyncio.create_task(long_task())
        handler._in_flight_requests["req-1"] = task

        await handler.shutdown(timeout=0.1)

        # Allow the cancellation to propagate
        with contextlib.suppress(asyncio.CancelledError):
            await task

        assert task.cancelled()
        assert len(handler._in_flight_requests) == 0

    @pytest.mark.asyncio
    async def test_shutdown_waits_for_completing_tasks(self):
        handler = _make_handler()
        completed = False

        async def quick_task():
            nonlocal completed
            await asyncio.sleep(0.05)
            completed = True

        task = asyncio.create_task(quick_task())
        handler._in_flight_requests["req-1"] = task

        await handler.shutdown(timeout=2.0)

        assert completed
        assert len(handler._in_flight_requests) == 0

    @pytest.mark.asyncio
    async def test_shutdown_empty_handler(self):
        """Shutdown on a fresh handler should not raise."""
        handler = _make_handler()
        await handler.shutdown()
        assert len(handler.session_manager.sessions) == 0
