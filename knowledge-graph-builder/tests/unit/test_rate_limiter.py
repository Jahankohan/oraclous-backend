"""
Unit tests for TASK-030 / STORY-009: per-tenant token bucket rate limiter.

Tests both the flat slowapi limiter and the per-tenant TokenBucketRateLimiter.
All Redis and Neo4j dependencies are mocked — no infrastructure required.

Coverage:
- Token bucket allows burst → then throttles
- After delay, tokens refilled proportionally
- Rate limit exceeded → 429 with error_code: KGB-4029 + Retry-After header
- Two different tenants have independent buckets
- Per-tenant config loaded from mock Neo4j node
"""

from __future__ import annotations

import json
import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_mock_redis(eval_results=None):
    """
    Build an async Redis mock suitable for TokenBucketRateLimiter.

    eval_results: list of return values for successive eval() calls.
                  Each element is [allowed_int, second_value_int].
    """
    r = MagicMock()
    if eval_results:
        r.eval = AsyncMock(side_effect=eval_results)
    else:
        # Default: always allow with 9 remaining tokens
        r.eval = AsyncMock(return_value=[1, 9])
    r.get = AsyncMock(return_value=None)  # no cached config
    r.set = AsyncMock(return_value=True)
    r.delete = AsyncMock(return_value=1)
    return r


# ---------------------------------------------------------------------------
# 1. Flat slowapi limiter (ORA-104 coverage, already in test_rate_limiting.py)
#    Kept here for consolidated STORY-009 verification.
# ---------------------------------------------------------------------------


class TestFlatLimiter:
    """Verify the IP-keyed slowapi limiter is correctly configured."""

    @pytest.mark.unit
    def test_limiter_uses_remote_address_key_func(self):
        from slowapi.util import get_remote_address

        from app.core.rate_limiter import limiter

        assert limiter._key_func is get_remote_address

    @pytest.mark.unit
    def test_limiter_is_singleton(self):
        from app.core.rate_limiter import limiter as limiter_a
        from app.core.rate_limiter import limiter as limiter_b

        assert limiter_a is limiter_b

    @pytest.mark.unit
    def test_rate_limit_exceeded_returns_429(self):
        """When the limiter raises RateLimitExceeded, the response status is 429."""
        from fastapi import FastAPI, Request
        from fastapi.testclient import TestClient
        from slowapi import Limiter, _rate_limit_exceeded_handler
        from slowapi.errors import RateLimitExceeded
        from slowapi.util import get_remote_address

        app = FastAPI()
        test_limiter = Limiter(key_func=get_remote_address)
        app.state.limiter = test_limiter
        app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

        @app.get("/limited")
        @test_limiter.limit("1/minute")
        async def limited_endpoint(request: Request):
            return {"ok": True}

        client = TestClient(app, raise_server_exceptions=False)

        r1 = client.get("/limited", headers={"X-Forwarded-For": "10.0.0.1"})
        assert r1.status_code == 200

        r2 = client.get("/limited", headers={"X-Forwarded-For": "10.0.0.1"})
        assert r2.status_code == 429

    @pytest.mark.unit
    def test_two_different_ips_have_independent_limits(self):
        """Different IPs must not share rate limit buckets."""
        from fastapi import FastAPI, Request
        from fastapi.testclient import TestClient
        from slowapi import Limiter, _rate_limit_exceeded_handler
        from slowapi.errors import RateLimitExceeded

        def ip_from_test_header(request: Request) -> str:
            return request.headers.get("X-IP-Test", "unknown")

        app = FastAPI()
        test_limiter = Limiter(key_func=ip_from_test_header)
        app.state.limiter = test_limiter
        app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

        @app.get("/limited")
        @test_limiter.limit("1/minute")
        async def limited_endpoint(request: Request):
            return {"ok": True}

        client = TestClient(app, raise_server_exceptions=False)

        # IP A exhausts its limit
        client.get("/limited", headers={"X-IP-Test": "10.0.0.1"})
        r_a_blocked = client.get("/limited", headers={"X-IP-Test": "10.0.0.1"})
        assert r_a_blocked.status_code == 429

        # IP B still within its limit
        r_b = client.get("/limited", headers={"X-IP-Test": "10.0.0.2"})
        assert r_b.status_code == 200


# ---------------------------------------------------------------------------
# 2. TokenBucketRateLimiter — burst + throttle behaviour
# ---------------------------------------------------------------------------


class TestTokenBucketBurstAndThrottle:
    """Test token bucket burst allowance and throttling behaviour."""

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_token_bucket_allows_requests_within_burst(self):
        """
        The token bucket must allow requests while tokens remain.
        Simulate a scenario where the Lua script returns allowed=1 (token available).
        """
        from app.core.rate_limiter import TokenBucketRateLimiter

        # Lua returns [1, 9] → allowed, 9 tokens remaining
        r = _make_mock_redis(eval_results=[[1, 9], [1, 8], [1, 7]])
        limiter = TokenBucketRateLimiter(r, "tenant-burst", "write")

        for _ in range(3):
            allowed, headers = await limiter.is_allowed()
            assert allowed is True

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_token_bucket_throttles_when_tokens_exhausted(self):
        """
        When the Lua script returns allowed=0, is_allowed() must return False
        and include Retry-After in the headers.
        """
        from app.core.rate_limiter import TokenBucketRateLimiter

        # Lua returns [0, 12] → rejected, retry in 12 seconds
        r = _make_mock_redis(eval_results=[[0, 12]])
        limiter = TokenBucketRateLimiter(r, "tenant-throttle", "write")

        allowed, headers = await limiter.is_allowed()
        assert allowed is False
        assert "Retry-After" in headers
        assert headers["Retry-After"] == "12"

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_token_bucket_burst_then_throttle(self):
        """
        Burst is allowed until tokens run out, then requests are rejected.
        """
        from app.core.rate_limiter import TokenBucketRateLimiter

        # First 2 calls allowed, third rejected
        r = _make_mock_redis(eval_results=[[1, 9], [1, 0], [0, 2]])
        limiter = TokenBucketRateLimiter(r, "tenant-burst-throttle", "write")

        r1_allowed, _ = await limiter.is_allowed()
        r2_allowed, _ = await limiter.is_allowed()
        r3_allowed, r3_headers = await limiter.is_allowed()

        assert r1_allowed is True
        assert r2_allowed is True
        assert r3_allowed is False
        assert "Retry-After" in r3_headers

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_allowed_response_has_ratelimit_headers(self):
        """
        When a request is allowed, the headers dict must include
        X-RateLimit-Limit, X-RateLimit-Remaining, and X-RateLimit-Reset.
        """
        from app.core.rate_limiter import TokenBucketRateLimiter

        r = _make_mock_redis(eval_results=[[1, 7]])
        limiter = TokenBucketRateLimiter(r, "tenant-headers", "read")

        allowed, headers = await limiter.is_allowed()
        assert allowed is True
        assert "X-RateLimit-Limit" in headers
        assert "X-RateLimit-Remaining" in headers
        assert "X-RateLimit-Reset" in headers
        assert headers["X-RateLimit-Remaining"] == "7"

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_rejected_response_has_retry_after_and_ratelimit_headers(self):
        """
        When a request is rejected, the headers dict must include
        Retry-After, X-RateLimit-Limit, X-RateLimit-Remaining=0, X-RateLimit-Reset.
        """
        from app.core.rate_limiter import TokenBucketRateLimiter

        r = _make_mock_redis(eval_results=[[0, 30]])
        limiter = TokenBucketRateLimiter(r, "tenant-rejected", "write")

        allowed, headers = await limiter.is_allowed()
        assert allowed is False
        assert headers["Retry-After"] == "30"
        assert headers["X-RateLimit-Remaining"] == "0"
        assert "X-RateLimit-Limit" in headers
        assert "X-RateLimit-Reset" in headers


# ---------------------------------------------------------------------------
# 3. Token refill after delay
# ---------------------------------------------------------------------------


class TestTokenRefill:
    """Tokens are refilled continuously; after a delay tokens must be restored."""

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_tokens_refilled_proportionally_after_delay(self):
        """
        Simulate token refill: the Lua script receives the current epoch time
        and computes the refill. Here we verify the Lua script is called with
        a `now` argument close to the current time (within 2 seconds tolerance).
        """
        from app.core.rate_limiter import TokenBucketRateLimiter

        r = MagicMock()
        r.get = AsyncMock(return_value=None)

        captured_args = []

        async def capture_eval(script, num_keys, *args):
            captured_args.extend(args)
            return [1, 9]

        r.eval = AsyncMock(side_effect=capture_eval)

        limiter = TokenBucketRateLimiter(r, "tenant-refill", "write")
        before_call = time.time()
        await limiter.is_allowed()
        after_call = time.time()

        # The Lua script is called as: eval(script, 1, bucket_key, rate, burst, now)
        # In the side_effect capture_eval(script, num_keys, *args):
        #   args = (bucket_key, rate_str, burst_str, now_str)
        # So now is at index 3.
        assert len(captured_args) >= 4
        now_arg = float(captured_args[3])  # 4th arg (0-indexed: [3]) is `now`
        assert before_call - 1.0 <= now_arg <= after_call + 1.0

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_refill_allows_request_after_delay(self):
        """
        Simulate: first request rejected (tokens=0), then after delay tokens
        refill and next request is allowed.
        """
        from app.core.rate_limiter import TokenBucketRateLimiter

        # First call: rejected; second call: allowed (tokens refilled)
        r = _make_mock_redis(eval_results=[[0, 1], [1, 9]])
        limiter = TokenBucketRateLimiter(r, "tenant-refill-seq", "write")

        allowed1, _ = await limiter.is_allowed()
        allowed2, _ = await limiter.is_allowed()

        assert allowed1 is False
        assert allowed2 is True


# ---------------------------------------------------------------------------
# 4. Two different tenants have independent buckets
# ---------------------------------------------------------------------------


class TestTenantIsolation:
    """Each tenant must have its own independent rate limit bucket."""

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_two_tenants_have_independent_buckets(self):
        """
        Tenant A's bucket state must not affect tenant B's bucket state.
        Verified by checking that each limiter uses a distinct Redis key.
        """
        from app.core.rate_limiter import TokenBucketRateLimiter

        r_a = _make_mock_redis(eval_results=[[1, 9]])
        r_b = _make_mock_redis(eval_results=[[1, 9]])

        limiter_a = TokenBucketRateLimiter(r_a, "tenant-AAA", "write")
        limiter_b = TokenBucketRateLimiter(r_b, "tenant-BBB", "write")

        # Keys must be different
        assert limiter_a._bucket_key != limiter_b._bucket_key
        assert "tenant-AAA" in limiter_a._bucket_key
        assert "tenant-BBB" in limiter_b._bucket_key

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_tenant_a_throttled_tenant_b_still_allowed(self):
        """
        When tenant A is rate-limited, tenant B must still be allowed.
        """
        from app.core.rate_limiter import TokenBucketRateLimiter

        r = MagicMock()
        r.get = AsyncMock(return_value=None)

        bucket_states: dict[str, list] = {
            "ratelimit:tenant-A:write": [0, 5],  # rejected
            "ratelimit:tenant-B:write": [1, 9],  # allowed
        }

        async def per_bucket_eval(script, num_keys, bucket_key, *args):
            return bucket_states.get(bucket_key, [1, 9])

        r.eval = AsyncMock(side_effect=per_bucket_eval)

        limiter_a = TokenBucketRateLimiter(r, "tenant-A", "write")
        limiter_b = TokenBucketRateLimiter(r, "tenant-B", "write")

        allowed_a, _ = await limiter_a.is_allowed()
        allowed_b, _ = await limiter_b.is_allowed()

        assert allowed_a is False
        assert allowed_b is True

    @pytest.mark.unit
    def test_bucket_key_format_is_ratelimit_tenant_endpoint(self):
        """
        Bucket key format must be ratelimit:{tenant_id}:{endpoint_type}
        to guarantee tenant isolation at the Redis key level.
        """
        from app.core.rate_limiter import TokenBucketRateLimiter

        r = MagicMock()
        limiter = TokenBucketRateLimiter(r, "my-tenant", "read")

        assert limiter._bucket_key == "ratelimit:my-tenant:read"

    @pytest.mark.unit
    def test_invalid_endpoint_type_raises_value_error(self):
        """
        Constructing a TokenBucketRateLimiter with an unknown endpoint_type
        must raise ValueError immediately.
        """
        from app.core.rate_limiter import TokenBucketRateLimiter

        r = MagicMock()
        with pytest.raises(ValueError, match="endpoint_type"):
            TokenBucketRateLimiter(r, "tenant", "invalid_type")


# ---------------------------------------------------------------------------
# 5. Per-tenant config loaded from mock Neo4j node
# ---------------------------------------------------------------------------


class TestPerTenantConfig:
    """Per-tenant rate limit config can be loaded from Neo4j and cached in Redis."""

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_per_tenant_config_loaded_from_neo4j(self):
        """
        When Redis config cache is empty, _get_rate_and_burst must query Neo4j
        and return the tenant-specific rate/burst values.
        """
        from app.core.rate_limiter import TokenBucketRateLimiter

        tenant_config = {"write": {"rate": 2.0, "burst": 20.0}}

        r = MagicMock()
        r.get = AsyncMock(return_value=None)  # cache miss
        r.set = AsyncMock(return_value=True)
        r.eval = AsyncMock(return_value=[1, 19])

        # Mock Neo4j session returning the config
        mock_record = MagicMock()
        mock_record.__getitem__ = MagicMock(return_value=json.dumps(tenant_config))

        mock_result = AsyncMock()
        mock_result.single = AsyncMock(return_value=mock_record)

        mock_session = AsyncMock()
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=False)
        mock_session.run = AsyncMock(return_value=mock_result)

        mock_neo4j = MagicMock()
        mock_neo4j.session = MagicMock(return_value=mock_session)

        limiter = TokenBucketRateLimiter(
            r, "neo4j-tenant", "write", neo4j_driver=mock_neo4j
        )
        rate, burst = await limiter._get_rate_and_burst()

        assert rate == 2.0
        assert burst == 20.0

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_per_tenant_config_cached_in_redis_after_neo4j_load(self):
        """
        After loading the config from Neo4j, it must be stored in Redis
        with the correct config key and TTL.
        """
        from app.core.rate_limiter import _CONFIG_CACHE_TTL, TokenBucketRateLimiter

        tenant_config = {"write": {"rate": 1.5, "burst": 15.0}}

        r = MagicMock()
        r.get = AsyncMock(return_value=None)  # cache miss
        r.set = AsyncMock(return_value=True)
        r.eval = AsyncMock(return_value=[1, 14])

        mock_record = MagicMock()
        mock_record.__getitem__ = MagicMock(return_value=json.dumps(tenant_config))

        mock_result = AsyncMock()
        mock_result.single = AsyncMock(return_value=mock_record)

        mock_session = AsyncMock()
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=False)
        mock_session.run = AsyncMock(return_value=mock_result)

        mock_neo4j = MagicMock()
        mock_neo4j.session = MagicMock(return_value=mock_session)

        limiter = TokenBucketRateLimiter(
            r, "cache-tenant", "write", neo4j_driver=mock_neo4j
        )
        await limiter._get_rate_and_burst()

        # Redis.set must have been called to cache the config
        r.set.assert_called()
        set_call = r.set.call_args
        assert set_call.kwargs.get("ex") == _CONFIG_CACHE_TTL or (
            len(set_call.args) >= 3 and set_call.args[2] == _CONFIG_CACHE_TTL
        )

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_per_tenant_config_read_from_redis_cache(self):
        """
        When the config is already in Redis, Neo4j must NOT be queried.
        """
        from app.core.rate_limiter import TokenBucketRateLimiter

        cached_config = {"read": {"rate": 3.0, "burst": 30.0}}

        r = MagicMock()
        r.get = AsyncMock(return_value=json.dumps(cached_config))
        r.eval = AsyncMock(return_value=[1, 29])

        mock_neo4j = MagicMock()
        mock_neo4j.session = MagicMock()

        limiter = TokenBucketRateLimiter(
            r, "redis-cached-tenant", "read", neo4j_driver=mock_neo4j
        )
        rate, burst = await limiter._get_rate_and_burst()

        assert rate == 3.0
        assert burst == 30.0
        # Neo4j session must NOT have been called
        mock_neo4j.session.assert_not_called()

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_falls_back_to_defaults_when_no_config(self):
        """
        When neither Redis cache nor Neo4j has a config, defaults must be used.
        """
        from app.core.rate_limiter import _DEFAULTS, TokenBucketRateLimiter

        r = MagicMock()
        r.get = AsyncMock(return_value=None)  # no cache
        r.eval = AsyncMock(return_value=[1, 9])

        # Neo4j returns no record
        mock_result = AsyncMock()
        mock_result.single = AsyncMock(return_value=None)

        mock_session = AsyncMock()
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=False)
        mock_session.run = AsyncMock(return_value=mock_result)

        mock_neo4j = MagicMock()
        mock_neo4j.session = MagicMock(return_value=mock_session)

        limiter = TokenBucketRateLimiter(
            r, "default-tenant", "write", neo4j_driver=mock_neo4j
        )
        rate, burst = await limiter._get_rate_and_burst()

        assert rate == _DEFAULTS["write"]["rate"]
        assert burst == _DEFAULTS["write"]["burst"]

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_redis_unavailable_fails_open(self):
        """
        When Redis is unavailable during is_allowed(), the limiter must fail open
        (allow the request) rather than blocking all traffic.
        """
        import redis.exceptions

        from app.core.rate_limiter import TokenBucketRateLimiter

        r = MagicMock()
        r.get = AsyncMock(return_value=None)
        r.eval = AsyncMock(side_effect=redis.exceptions.ConnectionError("refused"))

        limiter = TokenBucketRateLimiter(r, "redis-down-tenant", "write")
        allowed, headers = await limiter.is_allowed()

        # Fail-open: must allow the request, not crash
        assert allowed is True


# ---------------------------------------------------------------------------
# 6. KGB-4029 error code verification
# ---------------------------------------------------------------------------


class TestRateLimitErrorCode:
    """Rate limit exceeded must use error_code: KGB-4029."""

    @pytest.mark.unit
    def test_kgb_rate_limit_exceeded_code_is_4029(self):
        from app.core.errors import KGBError

        code, message = KGBError.RATE_LIMIT_EXCEEDED
        assert code == "KGB-4029"

    @pytest.mark.unit
    def test_rate_limit_error_message_is_descriptive(self):
        from app.core.errors import KGBError

        _, message = KGBError.RATE_LIMIT_EXCEEDED
        assert len(message) > 0
        assert "rate" in message.lower() or "limit" in message.lower()

    @pytest.mark.unit
    def test_main_app_http_exception_handler_maps_429_to_kgb_4029(self):
        """
        The HTTPException handler in main.py must map status 429 → KGB-4029.
        """
        from unittest.mock import AsyncMock

        from fastapi import HTTPException
        from fastapi.testclient import TestClient

        with (
            patch("app.core.neo4j_client.neo4j_client.connect", new=AsyncMock()),
            patch("app.core.database.create_tables", new=AsyncMock()),
            patch("app.core.telemetry.setup_telemetry"),
            patch("app.core.telemetry.instrument_fastapi"),
        ):
            from app.main import app

        @app.get("/test-429")
        async def raise_429():
            raise HTTPException(
                status_code=429,
                detail="Too Many Requests",
                headers={"Retry-After": "10"},
            )

        client = TestClient(app, raise_server_exceptions=False)
        response = client.get("/test-429")

        assert response.status_code == 429
        body = response.json()
        assert body.get("error_code") == "KGB-4029"

    @pytest.mark.unit
    def test_main_app_http_exception_handler_maps_404_to_kgb_4001(self):
        """
        The HTTPException handler in main.py must map status 404 → KGB-4001.
        """
        from unittest.mock import AsyncMock

        from fastapi import HTTPException
        from fastapi.testclient import TestClient

        with (
            patch("app.core.neo4j_client.neo4j_client.connect", new=AsyncMock()),
            patch("app.core.database.create_tables", new=AsyncMock()),
            patch("app.core.telemetry.setup_telemetry"),
            patch("app.core.telemetry.instrument_fastapi"),
        ):
            from app.main import app

        @app.get("/test-404")
        async def raise_404():
            raise HTTPException(status_code=404, detail="Graph not found")

        client = TestClient(app, raise_server_exceptions=False)
        response = client.get("/test-404")

        assert response.status_code == 404
        body = response.json()
        assert body.get("error_code") == "KGB-4001"
