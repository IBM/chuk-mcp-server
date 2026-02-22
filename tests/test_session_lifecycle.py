#!/usr/bin/env python3
"""Tests for session lifecycle, eviction callbacks, and resource cleanup."""

import time

from chuk_mcp_server.protocol import MCPProtocolHandler, SessionManager
from chuk_mcp_server.types import ServerInfo, create_server_capabilities

# ============================================================================
# Helpers
# ============================================================================


def _make_handler(**kwargs):
    """Create a protocol handler with default test config."""
    info = ServerInfo(name="test", version="0.1.0")
    caps = create_server_capabilities(tools=True, resources=True, prompts=True)
    return MCPProtocolHandler(server_info=info, capabilities=caps, **kwargs)


def _make_manager(**kwargs):
    """Create a SessionManager with small limits for testing."""
    return SessionManager(max_sessions=3, cleanup_interval=100, **kwargs)


# ============================================================================
# SessionManager: on_evict callback
# ============================================================================


class TestSessionManagerOnEvict:
    """Verify the on_evict callback fires on eviction and expiry."""

    def test_on_evict_called_on_capacity_eviction(self):
        """When max_sessions is reached, on_evict fires for the oldest session."""
        evicted: list[str] = []
        mgr = _make_manager(on_evict=lambda sid: evicted.append(sid))

        # Fill to capacity
        s1 = mgr.create_session({"name": "c1"}, "2025-06-18")
        s2 = mgr.create_session({"name": "c2"}, "2025-06-18")
        s3 = mgr.create_session({"name": "c3"}, "2025-06-18")

        # Make s1 the oldest
        mgr.sessions[s1]["last_activity"] = time.time() - 1000
        mgr.sessions[s2]["last_activity"] = time.time() - 500
        mgr.sessions[s3]["last_activity"] = time.time()

        # Creating a 4th session should evict s1
        _s4 = mgr.create_session({"name": "c4"}, "2025-06-18")

        assert evicted == [s1]
        assert s1 not in mgr.sessions

    def test_on_evict_called_on_expiry_cleanup(self):
        """cleanup_expired fires on_evict for each expired session."""
        evicted: list[str] = []
        mgr = _make_manager(on_evict=lambda sid: evicted.append(sid))

        s1 = mgr.create_session({"name": "c1"}, "2025-06-18")
        s2 = mgr.create_session({"name": "c2"}, "2025-06-18")

        # Make s1 expired
        mgr.sessions[s1]["last_activity"] = time.time() - 7200  # 2h ago

        mgr.cleanup_expired(max_age=3600)

        assert evicted == [s1]
        assert s1 not in mgr.sessions
        assert s2 in mgr.sessions

    def test_no_callback_when_none(self):
        """No crash when on_evict is None."""
        mgr = _make_manager(on_evict=None)

        s1 = mgr.create_session({"name": "c1"}, "2025-06-18")
        mgr.create_session({"name": "c2"}, "2025-06-18")
        mgr.create_session({"name": "c3"}, "2025-06-18")
        mgr.sessions[s1]["last_activity"] = time.time() - 1000

        # Should not raise
        mgr.create_session({"name": "c4"}, "2025-06-18")
        assert s1 not in mgr.sessions

    def test_eviction_multiple_at_capacity(self):
        """Multiple creations at capacity evict one session each time."""
        evicted: list[str] = []
        mgr = _make_manager(on_evict=lambda sid: evicted.append(sid))

        ids = []
        for i in range(3):
            sid = mgr.create_session({"name": f"c{i}"}, "2025-06-18")
            mgr.sessions[sid]["last_activity"] = time.time() - (300 - i * 100)
            ids.append(sid)

        # Two more creations should evict the two oldest
        mgr.create_session({"name": "c3"}, "2025-06-18")
        mgr.create_session({"name": "c4"}, "2025-06-18")

        assert len(evicted) == 2
        assert ids[0] in evicted
        assert ids[1] in evicted


# ============================================================================
# SessionManager: protected_sessions
# ============================================================================


class TestSessionManagerProtectedSessions:
    """Verify that protected sessions are not evicted."""

    def test_protected_session_not_evicted(self):
        """Protected sessions are skipped during eviction."""
        evicted: list[str] = []
        protected: set[str] = set()

        mgr = _make_manager(
            on_evict=lambda sid: evicted.append(sid),
            protected_sessions=lambda: protected,
        )

        s1 = mgr.create_session({"name": "c1"}, "2025-06-18")
        s2 = mgr.create_session({"name": "c2"}, "2025-06-18")
        mgr.create_session({"name": "c3"}, "2025-06-18")

        # Make s1 the oldest but protect it
        mgr.sessions[s1]["last_activity"] = time.time() - 1000
        mgr.sessions[s2]["last_activity"] = time.time() - 500
        protected.add(s1)

        # Should evict s2 (next oldest non-protected) instead of s1
        mgr.create_session({"name": "c4"}, "2025-06-18")

        assert s1 in mgr.sessions  # Protected, still alive
        assert s2 not in mgr.sessions  # Evicted instead
        assert evicted == [s2]

    def test_all_protected_no_eviction(self):
        """If all sessions are protected, no eviction occurs (exceeds max)."""
        protected: set[str] = set()
        mgr = _make_manager(
            protected_sessions=lambda: protected,
        )

        s1 = mgr.create_session({"name": "c1"}, "2025-06-18")
        s2 = mgr.create_session({"name": "c2"}, "2025-06-18")
        s3 = mgr.create_session({"name": "c3"}, "2025-06-18")
        protected.update({s1, s2, s3})

        # All protected; no candidates to evict. New session still created (over capacity).
        s4 = mgr.create_session({"name": "c4"}, "2025-06-18")
        assert len(mgr.sessions) == 4  # Over capacity but nothing evictable
        assert s4 in mgr.sessions


# ============================================================================
# MCPProtocolHandler: _cleanup_session_state
# ============================================================================


class TestCleanupSessionState:
    """Verify _cleanup_session_state removes all per-session data."""

    def test_cleans_subscriptions(self):
        handler = _make_handler()
        handler._resource_subscriptions["sess1"] = {"res://a", "res://b"}
        handler._cleanup_session_state("sess1")
        assert "sess1" not in handler._resource_subscriptions

    def test_cleans_sse_buffers(self):
        handler = _make_handler()
        handler._sse_event_buffers["sess1"] = [(1, {"data": "x"})]
        handler._sse_event_counters["sess1"] = 1
        handler._cleanup_session_state("sess1")
        assert "sess1" not in handler._sse_event_buffers
        assert "sess1" not in handler._sse_event_counters

    def test_cleans_all_state_at_once(self):
        handler = _make_handler()
        handler._resource_subscriptions["sess1"] = {"res://a"}
        handler._sse_event_buffers["sess1"] = [(1, {"data": "x"})]
        handler._sse_event_counters["sess1"] = 1

        handler._cleanup_session_state("sess1")

        assert "sess1" not in handler._resource_subscriptions
        assert "sess1" not in handler._sse_event_buffers
        assert "sess1" not in handler._sse_event_counters

    def test_cleanup_nonexistent_session_no_error(self):
        """Cleaning up a session that has no state should not raise."""
        handler = _make_handler()
        handler._cleanup_session_state("nonexistent")  # No error


# ============================================================================
# MCPProtocolHandler: terminate_session delegates to _cleanup_session_state
# ============================================================================


class TestTerminateSessionCleanup:
    """Verify terminate_session cleans up all state."""

    def test_terminate_cleans_subscriptions_and_buffers(self):
        handler = _make_handler()
        sid = handler.session_manager.create_session({"name": "t"}, "2025-06-18")

        handler._resource_subscriptions[sid] = {"res://x"}
        handler._sse_event_buffers[sid] = [(1, {"d": "y"})]
        handler._sse_event_counters[sid] = 1

        result = handler.terminate_session(sid)

        assert result is True
        assert sid not in handler._resource_subscriptions
        assert sid not in handler._sse_event_buffers
        assert sid not in handler._sse_event_counters
        assert sid not in handler.session_manager.sessions

    def test_terminate_nonexistent_returns_false(self):
        handler = _make_handler()
        assert handler.terminate_session("nonexistent") is False


# ============================================================================
# Integration: eviction triggers cleanup in protocol handler
# ============================================================================


class TestEvictionIntegration:
    """End-to-end: session eviction cleans up protocol handler state."""

    def test_eviction_cleans_handler_state(self):
        """When SessionManager evicts, protocol handler state is cleaned."""
        handler = _make_handler()
        handler.session_manager.max_sessions = 2

        # Create two sessions and populate state (subscriptions only, no SSE counters
        # since SSE counters mark sessions as protected from eviction)
        s1 = handler.session_manager.create_session({"name": "c1"}, "2025-06-18")
        s2 = handler.session_manager.create_session({"name": "c2"}, "2025-06-18")

        handler._resource_subscriptions[s1] = {"res://a"}

        # Make s1 oldest
        handler.session_manager.sessions[s1]["last_activity"] = time.time() - 1000

        # Creating a 3rd session should evict s1 and clean its state
        _s3 = handler.session_manager.create_session({"name": "c3"}, "2025-06-18")

        assert s1 not in handler.session_manager.sessions
        assert s1 not in handler._resource_subscriptions

        # s2 state untouched
        assert s2 in handler.session_manager.sessions

    def test_eviction_skips_protected_sessions_with_sse(self):
        """Sessions with SSE counters are protected from eviction."""
        handler = _make_handler()
        handler.session_manager.max_sessions = 2

        s1 = handler.session_manager.create_session({"name": "c1"}, "2025-06-18")
        s2 = handler.session_manager.create_session({"name": "c2"}, "2025-06-18")

        # s1 is oldest but has SSE activity (protected)
        handler.session_manager.sessions[s1]["last_activity"] = time.time() - 1000
        handler._sse_event_counters[s1] = 1
        handler._resource_subscriptions[s1] = {"res://a"}

        # Creating a 3rd session should evict s2 (not s1, since s1 is protected)
        _s3 = handler.session_manager.create_session({"name": "c3"}, "2025-06-18")

        assert s1 in handler.session_manager.sessions  # Protected
        assert s2 not in handler.session_manager.sessions  # Evicted

    def test_expiry_cleans_handler_state(self):
        """When SessionManager expires sessions, protocol handler state is cleaned."""
        handler = _make_handler()

        s1 = handler.session_manager.create_session({"name": "c1"}, "2025-06-18")
        handler._resource_subscriptions[s1] = {"res://a"}
        handler._sse_event_buffers[s1] = [(1, {"data": "x"})]
        handler._sse_event_counters[s1] = 1

        # Make expired
        handler.session_manager.sessions[s1]["last_activity"] = time.time() - 7200

        handler.session_manager.cleanup_expired(max_age=3600)

        assert s1 not in handler.session_manager.sessions
        assert s1 not in handler._resource_subscriptions
        assert s1 not in handler._sse_event_buffers
        assert s1 not in handler._sse_event_counters


# ============================================================================
# _get_protected_sessions
# ============================================================================


class TestGetProtectedSessions:
    """Verify _get_protected_sessions logic."""

    def test_no_sse_counters_returns_empty(self):
        handler = _make_handler()
        assert handler._get_protected_sessions() == set()

    def test_sessions_with_sse_counters_are_protected(self):
        handler = _make_handler()
        sid = handler.session_manager.create_session({"name": "c1"}, "2025-06-18")
        handler._sse_event_counters[sid] = 5

        protected = handler._get_protected_sessions()
        assert sid in protected

    def test_stale_sse_counter_not_protected(self):
        """SSE counter for a session no longer in session_manager is not returned."""
        handler = _make_handler()
        handler._sse_event_counters["stale-sid"] = 3

        protected = handler._get_protected_sessions()
        assert "stale-sid" not in protected
