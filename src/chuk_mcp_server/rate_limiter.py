#!/usr/bin/env python3
"""Token bucket rate limiter for per-session request throttling."""

import time


class TokenBucketRateLimiter:
    """Per-session token bucket rate limiter.

    Each session gets its own bucket that refills at `rate` tokens/second
    with a burst capacity of `burst` tokens.
    """

    def __init__(self, rate: float = 100.0, burst: float = 200.0):
        self.rate = rate
        self.burst = burst
        self._buckets: dict[str, tuple[float, float]] = {}  # session_id -> (tokens, last_refill)

    def allow(self, session_id: str) -> bool:
        """Check if a request is allowed for this session.

        Returns True if allowed, False if rate-limited.
        """
        now = time.monotonic()

        if session_id not in self._buckets:
            # New session: start with full burst capacity minus 1 for this request
            self._buckets[session_id] = (self.burst - 1, now)
            return True

        tokens, last_refill = self._buckets[session_id]

        # Refill tokens based on elapsed time
        elapsed = now - last_refill
        tokens = min(self.burst, tokens + elapsed * self.rate)

        if tokens >= 1:
            self._buckets[session_id] = (tokens - 1, now)
            return True
        else:
            self._buckets[session_id] = (tokens, now)
            return False

    def cleanup(self, session_id: str) -> None:
        """Remove bucket for a session (called on session eviction)."""
        self._buckets.pop(session_id, None)

    def cleanup_stale(self, max_idle: float = 3600.0) -> None:
        """Remove buckets for sessions idle longer than max_idle seconds."""
        now = time.monotonic()
        stale = [sid for sid, (_, last) in self._buckets.items() if now - last > max_idle]
        for sid in stale:
            del self._buckets[sid]

    @property
    def session_count(self) -> int:
        """Number of active rate limit buckets."""
        return len(self._buckets)
