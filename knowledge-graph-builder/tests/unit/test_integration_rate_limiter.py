"""Unit tests for InMemoryRateLimiter used in public integration endpoints (STORY-022 / TASK-044)."""

import time

from app.services.rate_limiter import InMemoryRateLimiter


class TestInMemoryRateLimiter:
    def test_first_request_allowed(self):
        limiter = InMemoryRateLimiter()
        assert limiter.check("key1", 10) is True

    def test_requests_within_limit_all_allowed(self):
        limiter = InMemoryRateLimiter()
        for _ in range(5):
            assert limiter.check("key1", 5) is True

    def test_request_exceeding_limit_denied(self):
        limiter = InMemoryRateLimiter()
        for _ in range(5):
            limiter.check("key1", 5)
        assert limiter.check("key1", 5) is False

    def test_different_keys_have_independent_buckets(self):
        limiter = InMemoryRateLimiter()
        for _ in range(3):
            limiter.check("key1", 3)
        # key1 is exhausted; key2 should still pass
        assert limiter.check("key2", 3) is True

    def test_window_reset_allows_requests_again(self):
        limiter = InMemoryRateLimiter()
        for _ in range(2):
            limiter.check("key1", 2)
        assert limiter.check("key1", 2) is False

        # Backdate the window_start so the 60-second window expires
        limiter._buckets["key1"] = (time.monotonic() - 61.0, 2)
        assert limiter.check("key1", 2) is True

    def test_limit_one_allows_exactly_one_request(self):
        limiter = InMemoryRateLimiter()
        assert limiter.check("key1", 1) is True
        assert limiter.check("key1", 1) is False

    def test_count_increments_within_window(self):
        limiter = InMemoryRateLimiter()
        for _ in range(3):
            limiter.check("key1", 10)
        _, count = limiter._buckets["key1"]
        assert count == 3

    def test_window_reset_starts_fresh_count(self):
        limiter = InMemoryRateLimiter()
        limiter.check("key1", 10)
        limiter._buckets["key1"] = (time.monotonic() - 61.0, 10)
        limiter.check("key1", 10)
        _, count = limiter._buckets["key1"]
        assert count == 1
