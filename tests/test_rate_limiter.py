#!/usr/bin/env python3
"""Tests for the token bucket rate limiter and its integration with protocol handler."""

import time

import pytest

from chuk_mcp_server.protocol import MCPProtocolHandler
from chuk_mcp_server.rate_limiter import TokenBucketRateLimiter
from chuk_mcp_server.types import ServerInfo, create_server_capabilities

# ============================================================================
# Unit tests for TokenBucketRateLimiter
# ============================================================================


class TestTokenBucketRateLimiter:
    """Test the TokenBucketRateLimiter class."""

    def test_new_session_is_allowed(self):
        """A brand-new session should be allowed on its first request."""
        limiter = TokenBucketRateLimiter(rate=10.0, burst=20.0)
        assert limiter.allow("session-1") is True

    def test_burst_capacity(self):
        """Requests should be allowed up to the burst capacity."""
        burst = 5.0
        limiter = TokenBucketRateLimiter(rate=1.0, burst=burst)

        # First call creates the bucket with burst-1 tokens remaining
        assert limiter.allow("s1") is True  # tokens left: burst - 1 = 4

        # Should allow burst-1 more requests (total burst requests)
        for _ in range(int(burst) - 1):
            assert limiter.allow("s1") is True

    def test_rate_limit_kicks_in_after_burst_exhaustion(self):
        """After burst is exhausted, further requests should be denied."""
        limiter = TokenBucketRateLimiter(rate=1000.0, burst=3.0)

        # Exhaust all 3 burst tokens
        assert limiter.allow("s1") is True  # tokens: 3 - 1 = 2
        assert limiter.allow("s1") is True  # tokens: ~2 - 1 = ~1
        assert limiter.allow("s1") is True  # tokens: ~1 - 1 = ~0

        # Next request should be denied (not enough time to refill 1 token)
        assert limiter.allow("s1") is False

    def test_token_refill_over_time(self):
        """Tokens should refill over time allowing more requests."""
        limiter = TokenBucketRateLimiter(rate=10000.0, burst=1.0)

        # Use the single token
        assert limiter.allow("s1") is True  # tokens: 1 - 1 = 0
        assert limiter.allow("s1") is False  # no tokens

        # Wait enough time for at least 1 token to refill
        # rate=10000 tokens/sec, so 1ms should refill 10 tokens
        time.sleep(0.002)

        assert limiter.allow("s1") is True

    def test_cleanup_removes_session_bucket(self):
        """cleanup() should remove the bucket for a given session."""
        limiter = TokenBucketRateLimiter()
        limiter.allow("s1")
        limiter.allow("s2")
        assert limiter.session_count == 2

        limiter.cleanup("s1")
        assert limiter.session_count == 1

        # Cleaning up non-existent session should not error
        limiter.cleanup("s-nonexistent")
        assert limiter.session_count == 1

    def test_cleanup_stale_removes_old_buckets(self):
        """cleanup_stale() should remove buckets idle longer than max_idle."""
        limiter = TokenBucketRateLimiter()

        # Create two sessions
        limiter.allow("s-old")
        limiter.allow("s-new")

        # Manually backdate the 's-old' bucket's last_refill time
        tokens, _ = limiter._buckets["s-old"]
        limiter._buckets["s-old"] = (tokens, time.monotonic() - 7200)  # 2 hours ago

        limiter.cleanup_stale(max_idle=3600.0)

        assert "s-old" not in limiter._buckets
        assert "s-new" in limiter._buckets
        assert limiter.session_count == 1

    def test_session_count_property(self):
        """session_count should reflect the number of active buckets."""
        limiter = TokenBucketRateLimiter()
        assert limiter.session_count == 0

        limiter.allow("s1")
        assert limiter.session_count == 1

        limiter.allow("s2")
        assert limiter.session_count == 2

        limiter.allow("s1")  # same session, no new bucket
        assert limiter.session_count == 2

    def test_independent_sessions(self):
        """Each session should have its own independent bucket."""
        limiter = TokenBucketRateLimiter(rate=1000.0, burst=2.0)

        # Exhaust s1
        limiter.allow("s1")
        limiter.allow("s1")
        assert limiter.allow("s1") is False

        # s2 should still be allowed
        assert limiter.allow("s2") is True


# ============================================================================
# Integration tests: rate limiter wired into MCPProtocolHandler
# ============================================================================


def _create_handler(rate_limit_rps=None):
    """Helper to create an MCPProtocolHandler."""
    info = ServerInfo(name="test", version="0.1.0")
    caps = create_server_capabilities(tools=True)
    return MCPProtocolHandler(server_info=info, capabilities=caps, rate_limit_rps=rate_limit_rps)


class TestRateLimiterIntegration:
    """Test rate limiter integration with MCPProtocolHandler."""

    def test_rate_limiter_disabled_by_default(self):
        """When rate_limit_rps is not specified, rate limiter should be None."""
        handler = _create_handler()
        assert handler._rate_limiter is None

    def test_rate_limiter_enabled_when_configured(self):
        """When rate_limit_rps is set, rate limiter should be created."""
        handler = _create_handler(rate_limit_rps=10.0)
        assert handler._rate_limiter is not None
        assert handler._rate_limiter.rate == 10.0
        assert handler._rate_limiter.burst == 20.0  # 2x rate

    @pytest.mark.asyncio
    async def test_rate_limited_request_returns_error(self):
        """When rate limit is exceeded, handle_request should return an error."""
        handler = _create_handler(rate_limit_rps=10.0)

        # Initialize a session first
        init_msg = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "initialize",
            "params": {
                "clientInfo": {"name": "test-client"},
                "protocolVersion": "2025-06-18",
            },
        }
        _, session_id = await handler.handle_request(init_msg)
        assert session_id is not None

        # burst = 20.0, so exhaust all tokens
        # The initialize request does not consume a rate limit token because
        # session_id is None when initialize is called.  Subsequent requests
        # pass the session_id, so we get the full burst of 20 tokens.
        for i in range(20):
            ping_msg = {"jsonrpc": "2.0", "id": i + 100, "method": "ping"}
            response, _ = await handler.handle_request(ping_msg, session_id=session_id)
            assert response is not None
            assert "result" in response, f"Request {i} should succeed but got error"

        # Next request should be rate-limited
        ping_msg = {"jsonrpc": "2.0", "id": 999, "method": "ping"}
        response, _ = await handler.handle_request(ping_msg, session_id=session_id)
        assert response is not None
        assert "error" in response
        assert response["error"]["message"] == "Rate limit exceeded"

    @pytest.mark.asyncio
    async def test_no_rate_limit_without_session_id(self):
        """Requests without a session_id should not be rate-limited."""
        handler = _create_handler(rate_limit_rps=10.0)

        # Send ping without session_id -- rate limiter should not trigger
        for i in range(30):
            ping_msg = {"jsonrpc": "2.0", "id": i, "method": "ping"}
            response, _ = await handler.handle_request(ping_msg, session_id=None)
            assert "result" in response

    def test_cleanup_session_state_cleans_rate_limiter(self):
        """_cleanup_session_state should remove the rate limiter bucket."""
        handler = _create_handler(rate_limit_rps=10.0)

        # Simulate a session bucket
        handler._rate_limiter.allow("test-session")
        assert handler._rate_limiter.session_count == 1

        handler._cleanup_session_state("test-session")
        assert handler._rate_limiter.session_count == 0

    def test_cleanup_session_state_no_error_without_rate_limiter(self):
        """_cleanup_session_state should not error when rate limiter is disabled."""
        handler = _create_handler()  # no rate limiter
        # Should not raise
        handler._cleanup_session_state("any-session")
