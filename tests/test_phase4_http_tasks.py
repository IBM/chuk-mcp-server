#!/usr/bin/env python3
"""
Tests for Phase 4 features: Streamable HTTP Transport Enhancements and Tasks System.

Covers MCP 2025-11-25 features:
  - Session termination via DELETE
  - SSE event IDs
  - SSE event buffering and resumability
  - Session cleanup on terminate
  - Task CRUD (create, update, get, list, cancel, result)
  - Protocol dispatch for tasks/* methods
"""

import time

import pytest
import pytest_asyncio  # noqa: F401 (ensures pytest-asyncio is available)

from chuk_mcp_server.protocol import MCPProtocolHandler, SessionManager
from chuk_mcp_server.types import ServerInfo, create_server_capabilities

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def handler() -> MCPProtocolHandler:
    """Create a fresh MCPProtocolHandler for each test."""
    return MCPProtocolHandler(
        ServerInfo(name="TestServer", version="1.0.0"),
        create_server_capabilities(tools=True),
    )


@pytest.fixture()
def session_manager() -> SessionManager:
    """Create a fresh SessionManager for each test."""
    return SessionManager()


# ===================================================================
# Feature 1: Streamable HTTP Transport Enhancements
# ===================================================================


class TestSessionTermination:
    """Tests for session termination via DELETE (MCP 2025-11-25)."""

    def test_terminate_existing_session(self, handler: MCPProtocolHandler) -> None:
        """Terminating an existing session returns True and removes it."""
        sid = handler.session_manager.create_session({"name": "test-client"}, "2025-06-18")
        assert handler.session_manager.get_session(sid) is not None

        result = handler.terminate_session(sid)
        assert result is True
        assert handler.session_manager.get_session(sid) is None

    def test_terminate_nonexistent_session(self, handler: MCPProtocolHandler) -> None:
        """Terminating a session that does not exist returns False."""
        assert handler.terminate_session("nonexistent-session-id") is False

    def test_terminate_already_terminated_session(self, handler: MCPProtocolHandler) -> None:
        """Terminating a session that was already terminated returns False."""
        sid = handler.session_manager.create_session({"name": "test"}, "2025-06-18")
        assert handler.terminate_session(sid) is True
        assert handler.terminate_session(sid) is False

    def test_terminate_does_not_affect_other_sessions(self, handler: MCPProtocolHandler) -> None:
        """Terminating one session leaves other sessions intact."""
        sid1 = handler.session_manager.create_session({"name": "client-1"}, "2025-06-18")
        sid2 = handler.session_manager.create_session({"name": "client-2"}, "2025-06-18")

        handler.terminate_session(sid1)

        assert handler.session_manager.get_session(sid1) is None
        assert handler.session_manager.get_session(sid2) is not None

    def test_terminate_multiple_sessions_sequentially(self, handler: MCPProtocolHandler) -> None:
        """Multiple sessions can be terminated independently."""
        sids = [handler.session_manager.create_session({"name": f"client-{i}"}, "2025-06-18") for i in range(5)]

        for sid in sids:
            assert handler.terminate_session(sid) is True

        for sid in sids:
            assert handler.session_manager.get_session(sid) is None
            assert handler.terminate_session(sid) is False


class TestSSEEventIDs:
    """Tests for SSE event ID generation."""

    def test_sequential_event_ids(self, handler: MCPProtocolHandler) -> None:
        """Event IDs increment sequentially for a given session."""
        eid1 = handler.next_sse_event_id("session1")
        eid2 = handler.next_sse_event_id("session1")
        eid3 = handler.next_sse_event_id("session1")

        assert eid1 == 1
        assert eid2 == 2
        assert eid3 == 3

    def test_independent_counters_per_session(self, handler: MCPProtocolHandler) -> None:
        """Different sessions have independent event ID counters."""
        eid_a1 = handler.next_sse_event_id("session-a")
        eid_b1 = handler.next_sse_event_id("session-b")
        eid_a2 = handler.next_sse_event_id("session-a")
        eid_b2 = handler.next_sse_event_id("session-b")

        assert eid_a1 == 1
        assert eid_b1 == 1
        assert eid_a2 == 2
        assert eid_b2 == 2

    def test_event_id_starts_at_one(self, handler: MCPProtocolHandler) -> None:
        """First event ID for a new session is 1, not 0."""
        assert handler.next_sse_event_id("brand-new-session") == 1

    def test_counter_persists_across_many_calls(self, handler: MCPProtocolHandler) -> None:
        """Counter correctly increments over many calls."""
        for expected in range(1, 201):
            assert handler.next_sse_event_id("heavy-session") == expected


class TestSSEEventBuffering:
    """Tests for SSE event buffering and resumability."""

    def test_buffer_and_retrieve_events(self, handler: MCPProtocolHandler) -> None:
        """Buffered events can be retrieved."""
        handler.buffer_sse_event("s1", 1, {"data": "first"})
        handler.buffer_sse_event("s1", 2, {"data": "second"})

        missed = handler.get_missed_events("s1", 0)
        assert len(missed) == 2
        assert missed[0] == (1, {"data": "first"})
        assert missed[1] == (2, {"data": "second"})

    def test_get_missed_events_after_last_id(self, handler: MCPProtocolHandler) -> None:
        """get_missed_events returns only events after the given ID."""
        handler.buffer_sse_event("s1", 1, {"data": "first"})
        handler.buffer_sse_event("s1", 2, {"data": "second"})
        handler.buffer_sse_event("s1", 3, {"data": "third"})

        missed = handler.get_missed_events("s1", 1)
        assert len(missed) == 2
        assert missed[0] == (2, {"data": "second"})
        assert missed[1] == (3, {"data": "third"})

    def test_get_missed_events_none_missed(self, handler: MCPProtocolHandler) -> None:
        """Returns empty list when no events are after the given ID."""
        handler.buffer_sse_event("s1", 1, {"data": "first"})
        handler.buffer_sse_event("s1", 2, {"data": "second"})

        missed = handler.get_missed_events("s1", 2)
        assert len(missed) == 0

    def test_get_missed_events_unknown_session(self, handler: MCPProtocolHandler) -> None:
        """Returns empty list for an unknown session."""
        missed = handler.get_missed_events("nonexistent", 0)
        assert missed == []

    def test_buffer_limit_enforced(self, handler: MCPProtocolHandler) -> None:
        """Buffer is capped at 100 events (oldest evicted)."""
        for i in range(105):
            handler.buffer_sse_event("s2", i + 1, {"data": f"event-{i}"})

        assert len(handler._sse_event_buffers["s2"]) == 100

    def test_buffer_limit_keeps_latest_events(self, handler: MCPProtocolHandler) -> None:
        """When buffer overflows, the latest events are retained."""
        for i in range(110):
            handler.buffer_sse_event("s3", i + 1, {"data": f"event-{i}"})

        buf = handler._sse_event_buffers["s3"]
        assert len(buf) == 100
        # First event in buffer should be event_id 11 (110 - 100 + 1)
        assert buf[0][0] == 11
        # Last event should be event_id 110
        assert buf[-1][0] == 110

    def test_buffer_exactly_100_no_truncation(self, handler: MCPProtocolHandler) -> None:
        """Exactly 100 events fit without truncation."""
        for i in range(100):
            handler.buffer_sse_event("exact", i + 1, {"data": f"e-{i}"})

        assert len(handler._sse_event_buffers["exact"]) == 100
        assert handler._sse_event_buffers["exact"][0][0] == 1
        assert handler._sse_event_buffers["exact"][-1][0] == 100

    def test_buffer_independent_per_session(self, handler: MCPProtocolHandler) -> None:
        """Different sessions have independent buffers."""
        handler.buffer_sse_event("sa", 1, {"data": "a1"})
        handler.buffer_sse_event("sb", 1, {"data": "b1"})
        handler.buffer_sse_event("sa", 2, {"data": "a2"})

        assert len(handler._sse_event_buffers["sa"]) == 2
        assert len(handler._sse_event_buffers["sb"]) == 1


class TestSessionCleanupOnTerminate:
    """Tests that terminate_session cleans up all associated resources."""

    def test_cleanup_resource_subscriptions(self, handler: MCPProtocolHandler) -> None:
        """Termination cleans up resource subscriptions."""
        sid = handler.session_manager.create_session({"name": "t"}, "2025-06-18")
        handler._resource_subscriptions[sid] = {"uri://test", "uri://other"}

        handler.terminate_session(sid)

        assert sid not in handler._resource_subscriptions

    def test_cleanup_sse_event_buffers(self, handler: MCPProtocolHandler) -> None:
        """Termination cleans up SSE event buffers."""
        sid = handler.session_manager.create_session({"name": "t"}, "2025-06-18")
        handler._sse_event_buffers[sid] = [(1, {"data": "test"}), (2, {"data": "test2"})]

        handler.terminate_session(sid)

        assert sid not in handler._sse_event_buffers

    def test_cleanup_sse_event_counters(self, handler: MCPProtocolHandler) -> None:
        """Termination cleans up SSE event counters."""
        sid = handler.session_manager.create_session({"name": "t"}, "2025-06-18")
        handler._sse_event_counters[sid] = 42

        handler.terminate_session(sid)

        assert sid not in handler._sse_event_counters

    def test_cleanup_all_state_on_terminate(self, handler: MCPProtocolHandler) -> None:
        """Termination cleans up all per-session state at once."""
        sid = handler.session_manager.create_session({"name": "t"}, "2025-06-18")
        handler._resource_subscriptions[sid] = {"uri://test"}
        handler._sse_event_buffers[sid] = [(1, {})]
        handler._sse_event_counters[sid] = 1

        handler.terminate_session(sid)

        assert sid not in handler._resource_subscriptions
        assert sid not in handler._sse_event_buffers
        assert sid not in handler._sse_event_counters
        assert handler.session_manager.get_session(sid) is None

    def test_cleanup_does_not_affect_other_sessions(self, handler: MCPProtocolHandler) -> None:
        """Cleanup for one session does not impact another session's state."""
        sid1 = handler.session_manager.create_session({"name": "a"}, "2025-06-18")
        sid2 = handler.session_manager.create_session({"name": "b"}, "2025-06-18")

        handler._resource_subscriptions[sid1] = {"uri://1"}
        handler._resource_subscriptions[sid2] = {"uri://2"}
        handler._sse_event_buffers[sid1] = [(1, {"x": 1})]
        handler._sse_event_buffers[sid2] = [(1, {"y": 2})]
        handler._sse_event_counters[sid1] = 5
        handler._sse_event_counters[sid2] = 10

        handler.terminate_session(sid1)

        assert sid2 in handler._resource_subscriptions
        assert handler._resource_subscriptions[sid2] == {"uri://2"}
        assert sid2 in handler._sse_event_buffers
        assert sid2 in handler._sse_event_counters
        assert handler._sse_event_counters[sid2] == 10


# ===================================================================
# Feature 2: Tasks System (MCP 2025-11-25)
# ===================================================================


class TestTaskCRUD:
    """Tests for task creation, update, and querying."""

    def test_create_task(self, handler: MCPProtocolHandler) -> None:
        """Creating a task stores it with correct initial state."""
        task_id = handler._create_task("req-1", "my_tool")

        assert task_id in handler._task_store
        task = handler._task_store[task_id]
        assert task["status"] == "working"
        assert task["toolName"] == "my_tool"
        assert task["requestId"] == "req-1"
        assert task["id"] == task_id
        assert task["result"] is None
        assert task["error"] is None
        assert task["message"] is None
        assert isinstance(task["createdAt"], float)
        assert isinstance(task["updatedAt"], float)

    def test_create_multiple_tasks(self, handler: MCPProtocolHandler) -> None:
        """Multiple tasks can be created with unique IDs."""
        tid1 = handler._create_task("req-1", "tool_a")
        tid2 = handler._create_task("req-2", "tool_b")
        tid3 = handler._create_task("req-3", "tool_a")

        assert tid1 != tid2
        assert tid2 != tid3
        assert tid1 != tid3
        assert len(handler._task_store) == 3

    def test_update_task_status_to_completed(self, handler: MCPProtocolHandler) -> None:
        """Updating task status to completed sets result."""
        task_id = handler._create_task("req-1", "my_tool")
        result_data = {"content": [{"type": "text", "text": "done"}]}

        handler._update_task_status(task_id, "completed", result=result_data)

        task = handler._task_store[task_id]
        assert task["status"] == "completed"
        assert task["result"] == result_data

    def test_update_task_status_to_failed(self, handler: MCPProtocolHandler) -> None:
        """Updating task status to failed sets error."""
        task_id = handler._create_task("req-1", "my_tool")
        error_data = {"code": -1, "message": "Something went wrong"}

        handler._update_task_status(task_id, "failed", error=error_data)

        task = handler._task_store[task_id]
        assert task["status"] == "failed"
        assert task["error"] == error_data

    def test_update_task_status_with_message(self, handler: MCPProtocolHandler) -> None:
        """Updating task status can include a progress message."""
        task_id = handler._create_task("req-1", "my_tool")

        handler._update_task_status(task_id, "working", message="50% complete")

        task = handler._task_store[task_id]
        assert task["status"] == "working"
        assert task["message"] == "50% complete"

    def test_update_task_updates_timestamp(self, handler: MCPProtocolHandler) -> None:
        """Updating a task updates the updatedAt timestamp."""
        task_id = handler._create_task("req-1", "my_tool")
        original_time = handler._task_store[task_id]["updatedAt"]

        # Small sleep to ensure time difference
        time.sleep(0.01)
        handler._update_task_status(task_id, "completed", result={"content": []})

        assert handler._task_store[task_id]["updatedAt"] >= original_time

    def test_update_nonexistent_task_is_noop(self, handler: MCPProtocolHandler) -> None:
        """Updating a task that does not exist does nothing (no exception)."""
        handler._update_task_status("nonexistent-id", "completed")
        assert "nonexistent-id" not in handler._task_store


class TestTasksGetHandler:
    """Tests for the tasks/get request handler."""

    @pytest.mark.asyncio
    async def test_get_existing_task(self, handler: MCPProtocolHandler) -> None:
        """tasks/get returns the task for a valid task ID."""
        task_id = handler._create_task("req-1", "my_tool")

        response, extra = await handler._handle_tasks_get({"id": task_id}, "msg-1")

        assert "result" in response
        assert response["result"]["id"] == task_id
        assert response["result"]["status"] == "working"
        assert extra is None

    @pytest.mark.asyncio
    async def test_get_unknown_task(self, handler: MCPProtocolHandler) -> None:
        """tasks/get returns error for an unknown task ID."""
        response, extra = await handler._handle_tasks_get({"id": "unknown-id"}, "msg-2")

        assert "error" in response
        assert "unknown" in response["error"]["message"].lower() or "Unknown" in response["error"]["message"]
        assert extra is None

    @pytest.mark.asyncio
    async def test_get_task_includes_jsonrpc_fields(self, handler: MCPProtocolHandler) -> None:
        """tasks/get response includes standard JSON-RPC fields."""
        task_id = handler._create_task("req-1", "tool_x")

        response, _ = await handler._handle_tasks_get({"id": task_id}, "msg-id-42")

        assert response["jsonrpc"] == "2.0"
        assert response["id"] == "msg-id-42"


class TestTasksResultHandler:
    """Tests for the tasks/result request handler."""

    @pytest.mark.asyncio
    async def test_result_completed_task(self, handler: MCPProtocolHandler) -> None:
        """tasks/result returns result for a completed task."""
        task_id = handler._create_task("req-1", "my_tool")
        handler._update_task_status(task_id, "completed", result={"content": []})

        response, _ = await handler._handle_tasks_result({"id": task_id}, "msg-3")

        assert "result" in response
        assert response["result"]["status"] == "completed"
        assert response["result"]["result"] == {"content": []}

    @pytest.mark.asyncio
    async def test_result_failed_task(self, handler: MCPProtocolHandler) -> None:
        """tasks/result returns result for a failed task."""
        task_id = handler._create_task("req-1", "my_tool")
        handler._update_task_status(task_id, "failed", error={"code": -1, "message": "boom"})

        response, _ = await handler._handle_tasks_result({"id": task_id}, "msg-f")

        assert "result" in response
        assert response["result"]["status"] == "failed"

    @pytest.mark.asyncio
    async def test_result_working_task_returns_error(self, handler: MCPProtocolHandler) -> None:
        """tasks/result returns error when task is still working."""
        task_id = handler._create_task("req-2", "tool2")

        response, _ = await handler._handle_tasks_result({"id": task_id}, "msg-4")

        assert "error" in response
        assert "not yet complete" in response["error"]["message"].lower() or "not yet" in response["error"]["message"]

    @pytest.mark.asyncio
    async def test_result_unknown_task_returns_error(self, handler: MCPProtocolHandler) -> None:
        """tasks/result returns error for unknown task."""
        response, _ = await handler._handle_tasks_result({"id": "no-such-task"}, "msg-x")

        assert "error" in response

    @pytest.mark.asyncio
    async def test_result_cancelled_task_returns_error(self, handler: MCPProtocolHandler) -> None:
        """tasks/result returns error for a cancelled task (not completed/failed)."""
        task_id = handler._create_task("req-c", "tool_c")
        handler._update_task_status(task_id, "cancelled")

        response, _ = await handler._handle_tasks_result({"id": task_id}, "msg-c")

        assert "error" in response


class TestTasksListHandler:
    """Tests for the tasks/list request handler."""

    @pytest.mark.asyncio
    async def test_list_empty(self, handler: MCPProtocolHandler) -> None:
        """tasks/list returns empty list when no tasks exist."""
        response, _ = await handler._handle_tasks_list({}, "msg-l0")

        assert "result" in response
        assert response["result"]["tasks"] == []

    @pytest.mark.asyncio
    async def test_list_multiple_tasks(self, handler: MCPProtocolHandler) -> None:
        """tasks/list returns all tasks."""
        handler._create_task("req-1", "tool_a")
        handler._create_task("req-2", "tool_b")
        handler._create_task("req-3", "tool_c")

        response, _ = await handler._handle_tasks_list({}, "msg-l1")

        assert "result" in response
        assert len(response["result"]["tasks"]) == 3

    @pytest.mark.asyncio
    async def test_list_includes_mixed_statuses(self, handler: MCPProtocolHandler) -> None:
        """tasks/list includes tasks in different statuses."""
        tid1 = handler._create_task("req-1", "tool_a")
        tid2 = handler._create_task("req-2", "tool_b")
        handler._create_task("req-3", "tool_c")

        handler._update_task_status(tid1, "completed", result={"content": []})
        handler._update_task_status(tid2, "failed", error={"code": -1, "message": "err"})
        # tid3 remains "working"

        response, _ = await handler._handle_tasks_list({}, "msg-l2")

        statuses = {t["status"] for t in response["result"]["tasks"]}
        assert statuses == {"completed", "failed", "working"}


class TestTasksCancelHandler:
    """Tests for the tasks/cancel request handler."""

    @pytest.mark.asyncio
    async def test_cancel_working_task(self, handler: MCPProtocolHandler) -> None:
        """Cancelling a working task sets status to cancelled."""
        task_id = handler._create_task("req-1", "tool_a")

        response, _ = await handler._handle_tasks_cancel({"id": task_id}, "msg-c1")

        assert "result" in response
        assert response["result"]["status"] == "cancelled"
        assert handler._task_store[task_id]["status"] == "cancelled"

    @pytest.mark.asyncio
    async def test_cancel_completed_task_returns_error(self, handler: MCPProtocolHandler) -> None:
        """Cancelling an already completed task returns error."""
        task_id = handler._create_task("req-1", "tool_a")
        handler._update_task_status(task_id, "completed", result={"content": []})

        response, _ = await handler._handle_tasks_cancel({"id": task_id}, "msg-c2")

        assert "error" in response
        assert "terminal" in response["error"]["message"].lower() or "already" in response["error"]["message"].lower()

    @pytest.mark.asyncio
    async def test_cancel_failed_task_returns_error(self, handler: MCPProtocolHandler) -> None:
        """Cancelling a failed task returns error."""
        task_id = handler._create_task("req-1", "tool_a")
        handler._update_task_status(task_id, "failed", error={"code": -1, "message": "err"})

        response, _ = await handler._handle_tasks_cancel({"id": task_id}, "msg-c3")

        assert "error" in response

    @pytest.mark.asyncio
    async def test_cancel_already_cancelled_task_returns_error(self, handler: MCPProtocolHandler) -> None:
        """Cancelling a task that is already cancelled returns error."""
        task_id = handler._create_task("req-1", "tool_a")
        handler._update_task_status(task_id, "cancelled")

        response, _ = await handler._handle_tasks_cancel({"id": task_id}, "msg-c4")

        assert "error" in response

    @pytest.mark.asyncio
    async def test_cancel_unknown_task_returns_error(self, handler: MCPProtocolHandler) -> None:
        """Cancelling a nonexistent task returns error."""
        response, _ = await handler._handle_tasks_cancel({"id": "no-such-task"}, "msg-c5")

        assert "error" in response


class TestTasksProtocolDispatch:
    """Tests that tasks/* methods are correctly routed through handle_request."""

    @pytest.mark.asyncio
    async def test_dispatch_tasks_list(self, handler: MCPProtocolHandler) -> None:
        """tasks/list is dispatched correctly via handle_request."""
        handler._create_task("req-1", "tool_a")

        response, _ = await handler.handle_request(
            {
                "jsonrpc": "2.0",
                "id": "dispatch-1",
                "method": "tasks/list",
                "params": {},
            }
        )

        assert response is not None
        assert "result" in response
        assert "tasks" in response["result"]
        assert len(response["result"]["tasks"]) == 1

    @pytest.mark.asyncio
    async def test_dispatch_tasks_get(self, handler: MCPProtocolHandler) -> None:
        """tasks/get is dispatched correctly via handle_request."""
        task_id = handler._create_task("req-1", "tool_x")

        response, _ = await handler.handle_request(
            {
                "jsonrpc": "2.0",
                "id": "dispatch-2",
                "method": "tasks/get",
                "params": {"id": task_id},
            }
        )

        assert response is not None
        assert "result" in response
        assert response["result"]["id"] == task_id

    @pytest.mark.asyncio
    async def test_dispatch_tasks_result(self, handler: MCPProtocolHandler) -> None:
        """tasks/result is dispatched correctly via handle_request."""
        task_id = handler._create_task("req-1", "tool_y")
        handler._update_task_status(task_id, "completed", result={"content": []})

        response, _ = await handler.handle_request(
            {
                "jsonrpc": "2.0",
                "id": "dispatch-3",
                "method": "tasks/result",
                "params": {"id": task_id},
            }
        )

        assert response is not None
        assert "result" in response
        assert response["result"]["status"] == "completed"

    @pytest.mark.asyncio
    async def test_dispatch_tasks_cancel(self, handler: MCPProtocolHandler) -> None:
        """tasks/cancel is dispatched correctly via handle_request."""
        task_id = handler._create_task("req-1", "tool_z")

        response, _ = await handler.handle_request(
            {
                "jsonrpc": "2.0",
                "id": "dispatch-4",
                "method": "tasks/cancel",
                "params": {"id": task_id},
            }
        )

        assert response is not None
        assert "result" in response
        assert response["result"]["status"] == "cancelled"

    @pytest.mark.asyncio
    async def test_dispatch_tasks_get_error_for_unknown(self, handler: MCPProtocolHandler) -> None:
        """tasks/get via dispatch returns error for unknown task."""
        response, _ = await handler.handle_request(
            {
                "jsonrpc": "2.0",
                "id": "dispatch-5",
                "method": "tasks/get",
                "params": {"id": "does-not-exist"},
            }
        )

        assert response is not None
        assert "error" in response

    @pytest.mark.asyncio
    async def test_dispatch_unknown_method_returns_error(self, handler: MCPProtocolHandler) -> None:
        """An unknown method returns method-not-found error."""
        response, _ = await handler.handle_request(
            {
                "jsonrpc": "2.0",
                "id": "dispatch-6",
                "method": "tasks/nonexistent",
                "params": {},
            }
        )

        assert response is not None
        assert "error" in response
        assert response["error"]["code"] == -32601  # METHOD_NOT_FOUND


class TestTasksFullLifecycle:
    """End-to-end lifecycle tests combining creation, updates, queries, and cancellation."""

    @pytest.mark.asyncio
    async def test_task_lifecycle_working_to_completed(self, handler: MCPProtocolHandler) -> None:
        """Full lifecycle: create -> working -> completed -> get result."""
        # Create
        task_id = handler._create_task("req-lifecycle-1", "analysis_tool")
        assert handler._task_store[task_id]["status"] == "working"

        # tasks/result should fail while working
        resp, _ = await handler._handle_tasks_result({"id": task_id}, "m1")
        assert "error" in resp

        # Complete
        handler._update_task_status(task_id, "completed", result={"content": [{"type": "text", "text": "result"}]})

        # tasks/get should show completed
        resp, _ = await handler._handle_tasks_get({"id": task_id}, "m2")
        assert resp["result"]["status"] == "completed"

        # tasks/result should succeed
        resp, _ = await handler._handle_tasks_result({"id": task_id}, "m3")
        assert "result" in resp
        assert resp["result"]["status"] == "completed"
        assert resp["result"]["result"]["content"][0]["text"] == "result"

        # Cancel should fail (already completed)
        resp, _ = await handler._handle_tasks_cancel({"id": task_id}, "m4")
        assert "error" in resp

    @pytest.mark.asyncio
    async def test_task_lifecycle_working_to_cancelled(self, handler: MCPProtocolHandler) -> None:
        """Full lifecycle: create -> working -> cancelled."""
        task_id = handler._create_task("req-lifecycle-2", "slow_tool")
        assert handler._task_store[task_id]["status"] == "working"

        # Cancel
        resp, _ = await handler._handle_tasks_cancel({"id": task_id}, "m1")
        assert resp["result"]["status"] == "cancelled"

        # Result should fail (cancelled is not completed/failed)
        resp, _ = await handler._handle_tasks_result({"id": task_id}, "m2")
        assert "error" in resp

        # Cancel again should fail
        resp, _ = await handler._handle_tasks_cancel({"id": task_id}, "m3")
        assert "error" in resp

    @pytest.mark.asyncio
    async def test_task_lifecycle_working_to_failed(self, handler: MCPProtocolHandler) -> None:
        """Full lifecycle: create -> working -> failed -> get result."""
        task_id = handler._create_task("req-lifecycle-3", "failing_tool")

        handler._update_task_status(task_id, "failed", error={"code": -1, "message": "timeout"})

        # tasks/result should succeed for failed task
        resp, _ = await handler._handle_tasks_result({"id": task_id}, "m1")
        assert "result" in resp
        assert resp["result"]["status"] == "failed"
        assert resp["result"]["error"]["message"] == "timeout"

        # Cancel should fail (already terminal)
        resp, _ = await handler._handle_tasks_cancel({"id": task_id}, "m2")
        assert "error" in resp


# ===================================================================
# SessionManager unit tests
# ===================================================================


class TestSessionManager:
    """Unit tests for the SessionManager itself."""

    def test_create_session_returns_unique_ids(self, session_manager: SessionManager) -> None:
        """Each created session has a unique ID."""
        ids = {session_manager.create_session({"name": f"c{i}"}, "2025-06-18") for i in range(20)}
        assert len(ids) == 20

    def test_get_session_returns_none_for_unknown(self, session_manager: SessionManager) -> None:
        """get_session returns None for unknown IDs."""
        assert session_manager.get_session("does-not-exist") is None

    def test_session_stores_client_info(self, session_manager: SessionManager) -> None:
        """Session stores the provided client info."""
        sid = session_manager.create_session({"name": "MyClient", "version": "2.0"}, "2025-06-18")
        session = session_manager.get_session(sid)
        assert session is not None
        assert session["client_info"]["name"] == "MyClient"
        assert session["client_info"]["version"] == "2.0"

    def test_session_stores_protocol_version(self, session_manager: SessionManager) -> None:
        """Session stores the protocol version."""
        sid = session_manager.create_session({"name": "c"}, "2025-11-25")
        session = session_manager.get_session(sid)
        assert session is not None
        assert session["protocol_version"] == "2025-11-25"

    def test_update_activity_updates_timestamp(self, session_manager: SessionManager) -> None:
        """update_activity refreshes the last_activity timestamp."""
        sid = session_manager.create_session({"name": "c"}, "2025-06-18")
        original = session_manager.get_session(sid)["last_activity"]

        time.sleep(0.01)
        session_manager.update_activity(sid)

        assert session_manager.get_session(sid)["last_activity"] >= original

    def test_max_sessions_eviction(self) -> None:
        """When max_sessions is reached, the oldest session is evicted."""
        sm = SessionManager(max_sessions=3)
        sid1 = sm.create_session({"name": "c1"}, "v1")
        time.sleep(0.01)
        sid2 = sm.create_session({"name": "c2"}, "v1")
        time.sleep(0.01)
        sid3 = sm.create_session({"name": "c3"}, "v1")

        # This should evict sid1 (oldest)
        sid4 = sm.create_session({"name": "c4"}, "v1")

        assert sm.get_session(sid1) is None
        assert sm.get_session(sid2) is not None
        assert sm.get_session(sid3) is not None
        assert sm.get_session(sid4) is not None

    def test_cleanup_expired(self) -> None:
        """cleanup_expired removes sessions older than max_age."""
        sm = SessionManager()
        sid = sm.create_session({"name": "old"}, "v1")

        # Manually age the session
        sm.sessions[sid]["last_activity"] = time.time() - 7200  # 2 hours old

        sm.cleanup_expired(max_age=3600)

        assert sm.get_session(sid) is None
