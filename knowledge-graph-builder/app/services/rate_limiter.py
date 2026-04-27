"""In-memory per-key token-bucket rate limiter (STORY-022).

Phase 1: single-process, asyncio-safe in-memory store.
Phase 2: replace with Redis-backed limiter for multi-instance deployments.
"""

import time
from collections import defaultdict


class InMemoryRateLimiter:
    """Token-bucket rate limiter keyed by an arbitrary string.

    Each bucket allows `limit_rpm` requests per 60-second window.
    The window resets from the first request in that window.
    """

    def __init__(self) -> None:
        # bucket_key → (window_start_epoch, request_count)
        self._buckets: dict[str, tuple[float, int]] = defaultdict(lambda: (0.0, 0))

    def check(self, bucket_key: str, limit_rpm: int) -> bool:
        """Return True if the request is within the rate limit; False if exceeded."""
        now = time.monotonic()
        window_start, count = self._buckets[bucket_key]

        if now - window_start >= 60.0:
            # New window
            self._buckets[bucket_key] = (now, 1)
            return True

        if count >= limit_rpm:
            return False

        self._buckets[bucket_key] = (window_start, count + 1)
        return True


# Module-level singleton
rate_limiter = InMemoryRateLimiter()
