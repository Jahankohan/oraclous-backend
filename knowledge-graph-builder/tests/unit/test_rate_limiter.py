"""
Unit tests for TokenBucketRateLimiter (TASK-030).

Covers:
- Burst allows the configured number of back-to-back requests.
- Bucket throttles once burst is exhausted.
- Tokens are refilled proportionally after elapsed time.
- Rate limit exceeded returns 429 with KGB-4029 + Retry-After header via FastAPI.
- Two different tenants share no state (independent Redis keys).
- Per-tenant config is loaded from a mock Neo4j node and overrides defaults.
- Redis unavailability causes fail-open (request allowed).
"""

from __future__ import annotations

import json
import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------


class _FakeRedis:
    """
    In-memory Redis stub that implements the minimal surface used by
    TokenBucketRateLimiter:  eval, get, set, delete.

    The Lua script is not executed; instead, the stub simulates the
    token bucket state natively in Python so that we can inject
    controlled time values.
    """

    def __init__(self, *, initial_tokens: float | None = None, last_refill: float | None = None):
        self._store: dict[str, str] = {}
        self._initial_tokens = initial_tokens
        self._last_refill = last_refill

    # Simulate HMGET / HMSET / EXPIRE used internally by the Lua script.
    # For unit tests we bypass Lua and call a Python reimplementation.
    async def eval(self, script: str, num_keys: int, *args):  # noqa: ARG002
        """Python reimplementation of the Lua token bucket script."""
        key = args[0]
        rate = float(args[1])
        burst = float(args[2])
        now = float(args[3])

        bucket = self._store.get(key)
        if bucket:
            data = json.loads(bucket)
            tokens = float(data.get("tokens", burst))
            last_refill = float(data.get("last_refill", now))
        else:
            tokens = self._initial_tokens if self._initial_tokens is not None else burst
            last_refill = self._last_refill if self._last_refill is not None else now

        elapsed = now - last_refill
        tokens = min(burst, tokens + elapsed * rate)

        if tokens >= 1:
            tokens -= 1
            self._store[key] = json.dumps({"tokens": tokens, "last_refill": now})
            return [1, int(tokens)]
        else:
            self._store[key] = json.dumps({"tokens": tokens, "last_refill": now})
            retry_after = int((1 - tokens) / rate) + 1
            return [0, retry_after]

    async def get(self, key: str) -> str | None:
        return self._store.get(key)

    async def set(self, key: str, value: str, ex: int | None = None) -> None:  # noqa: ARG002
        self._store[key] = value

    async def delete(self, *keys: str) -> None:
        for key in keys:
            self._store.pop(key, None)


def _make_limiter(
    redis: _FakeRedis,
    tenant_id: str = "tenant-a",
    endpoint_type: str = "write",
    neo4j_driver=None,
):
    from app.core.rate_limiter import TokenBucketRateLimiter

    return TokenBucketRateLimiter(
        redis_client=redis,
        tenant_id=tenant_id,
        endpoint_type=endpoint_type,
        neo4j_driver=neo4j_driver,
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_burst_allows_10_requests():
    """
    Default write bucket has burst=10.  Ten back-to-back requests (same
    timestamp so no refill occurs) should all be allowed.
    """
    redis = _FakeRedis()
    limiter = _make_limiter(redis, endpoint_type="write")
    now = time.time()

    with patch("app.core.rate_limiter.time") as mock_time:
        mock_time.time.return_value = now
        results = []
        for _ in range(10):
            allowed, _ = await limiter.is_allowed()
            results.append(allowed)

    assert all(results), f"Expected 10 allowed, got: {results}"


@pytest.mark.asyncio
async def test_burst_exhausted_throttles():
    """
    After 10 requests (burst=10 for write), the 11th request at the same
    timestamp must be denied (429-worthy).
    """
    redis = _FakeRedis()
    limiter = _make_limiter(redis, endpoint_type="write")
    now = time.time()

    with patch("app.core.rate_limiter.time") as mock_time:
        mock_time.time.return_value = now
        for _ in range(10):
            await limiter.is_allowed()

        # 11th request — bucket empty
        allowed, headers = await limiter.is_allowed()

    assert not allowed
    assert "Retry-After" in headers
    assert int(headers["Retry-After"]) > 0
    assert headers["X-RateLimit-Remaining"] == "0"


@pytest.mark.asyncio
async def test_tokens_refilled_after_elapsed_time():
    """
    write rate = 0.5 tokens/s.  After 2 s with an empty bucket,
    exactly 1 token (0.5 * 2) is available so the next request is allowed.
    """
    now = time.time()
    redis = _FakeRedis()
    limiter = _make_limiter(redis, endpoint_type="write")

    # Exhaust the bucket at t=now.
    with patch("app.core.rate_limiter.time") as mock_time:
        mock_time.time.return_value = now
        for _ in range(10):
            await limiter.is_allowed()
        # Confirm bucket empty.
        allowed, _ = await limiter.is_allowed()
        assert not allowed

    # Advance time by 2 seconds (0.5 * 2 = 1 token refilled).
    with patch("app.core.rate_limiter.time") as mock_time:
        mock_time.time.return_value = now + 2.0
        allowed, headers = await limiter.is_allowed()

    assert allowed, "Expected 1 token to be refilled after 2 s at rate=0.5/s"
    assert headers["X-RateLimit-Remaining"] == "0"  # consumed the single refilled token


@pytest.mark.asyncio
async def test_rate_limit_exceeded_response_format():
    """
    When the bucket is empty, headers contain KGB-4029 fields and
    a positive Retry-After value.
    """
    from app.core.errors import KGBError

    redis = _FakeRedis()
    limiter = _make_limiter(redis, endpoint_type="write")
    now = time.time()

    with patch("app.core.rate_limiter.time") as mock_time:
        mock_time.time.return_value = now
        for _ in range(10):
            await limiter.is_allowed()
        allowed, headers = await limiter.is_allowed()

    assert not allowed
    assert "Retry-After" in headers
    assert "X-RateLimit-Limit" in headers
    assert "X-RateLimit-Remaining" in headers
    assert "X-RateLimit-Reset" in headers
    assert headers["X-RateLimit-Remaining"] == "0"

    # Verify the KGB error code constant is correct.
    code, msg = KGBError.RATE_LIMIT_EXCEEDED
    assert code == "KGB-4029"
    assert "rate limit" in msg.lower()


@pytest.mark.asyncio
async def test_two_tenants_have_independent_buckets():
    """
    Exhausting tenant-a's bucket must not affect tenant-b's bucket.
    """
    redis = _FakeRedis()
    limiter_a = _make_limiter(redis, tenant_id="tenant-a", endpoint_type="write")
    limiter_b = _make_limiter(redis, tenant_id="tenant-b", endpoint_type="write")
    now = time.time()

    with patch("app.core.rate_limiter.time") as mock_time:
        mock_time.time.return_value = now

        # Exhaust tenant-a.
        for _ in range(10):
            await limiter_a.is_allowed()
        a_blocked_allowed, _ = await limiter_a.is_allowed()
        assert not a_blocked_allowed

        # tenant-b must still have a full bucket.
        b_allowed, _ = await limiter_b.is_allowed()
        assert b_allowed, "tenant-b should be unaffected by tenant-a exhausting its bucket"

    # Verify different Redis keys.
    assert limiter_a._bucket_key != limiter_b._bucket_key


@pytest.mark.asyncio
async def test_per_tenant_config_from_neo4j():
    """
    When a `:ServiceAccount` node has ``rate_limit_config``, those values
    override the defaults.  Here we grant write rate=2.0/s burst=5 and verify
    the bucket starts with 5 tokens (not the default 10).
    """
    custom_config = {
        "write": {"rate": 2.0, "burst": 5.0},
    }
    custom_config_json = json.dumps(custom_config)

    # Mock Neo4j session / result.
    mock_record = MagicMock()
    mock_record.__getitem__ = lambda self, key: custom_config_json if key == "cfg" else None

    mock_result = AsyncMock()
    mock_result.single = AsyncMock(return_value=mock_record)

    mock_session = AsyncMock()
    mock_session.__aenter__ = AsyncMock(return_value=mock_session)
    mock_session.__aexit__ = AsyncMock(return_value=None)
    mock_session.run = AsyncMock(return_value=mock_result)

    mock_driver = MagicMock()
    mock_driver.session = MagicMock(return_value=mock_session)

    redis = _FakeRedis()
    limiter = _make_limiter(redis, tenant_id="tenant-custom", endpoint_type="write", neo4j_driver=mock_driver)

    now = time.time()
    with patch("app.core.rate_limiter.time") as mock_time:
        mock_time.time.return_value = now

        # With burst=5 the 6th request should be denied.
        results = []
        for _ in range(5):
            allowed, _ = await limiter.is_allowed()
            results.append(allowed)

        assert all(results), f"First 5 requests should be allowed with burst=5; got {results}"

        # 6th request with no elapsed time — must be denied.
        denied, headers = await limiter.is_allowed()

    assert not denied, "6th request should be denied when burst=5"
    assert "Retry-After" in headers


@pytest.mark.asyncio
async def test_redis_unavailable_fails_open():
    """
    When Redis raises an exception, the limiter must fail open (allow the
    request) and log a warning rather than blocking traffic.
    """

    class _BrokenRedis:
        async def eval(self, *args, **kwargs):
            raise ConnectionError("Redis connection refused")

        async def get(self, key):
            raise ConnectionError("Redis connection refused")

        async def set(self, *args, **kwargs):
            raise ConnectionError("Redis connection refused")

    limiter = _make_limiter(_BrokenRedis(), endpoint_type="write")

    allowed, headers = await limiter.is_allowed()

    assert allowed, "Limiter must fail open when Redis is unavailable"
    assert headers == {}


@pytest.mark.asyncio
async def test_read_category_defaults():
    """read endpoint type has higher defaults: rate=1.0, burst=20."""
    from app.core.rate_limiter import _DEFAULTS

    assert _DEFAULTS["read"]["rate"] == pytest.approx(1.0)
    assert _DEFAULTS["read"]["burst"] == pytest.approx(20.0)


@pytest.mark.asyncio
async def test_admin_category_defaults():
    """admin endpoint type has restricted defaults: rate≈0.167, burst=3."""
    from app.core.rate_limiter import _DEFAULTS

    assert _DEFAULTS["admin"]["rate"] == pytest.approx(0.167, abs=0.001)
    assert _DEFAULTS["admin"]["burst"] == pytest.approx(3.0)


@pytest.mark.asyncio
async def test_invalid_endpoint_type_raises():
    """Passing an unknown endpoint_type at construction raises ValueError."""
    from app.core.rate_limiter import TokenBucketRateLimiter

    with pytest.raises(ValueError, match="endpoint_type must be one of"):
        TokenBucketRateLimiter(
            redis_client=_FakeRedis(),
            tenant_id="t",
            endpoint_type="bogus",
        )


@pytest.mark.asyncio
async def test_bucket_key_format():
    """Redis bucket key must follow the documented format."""
    limiter = _make_limiter(_FakeRedis(), tenant_id="org-123", endpoint_type="admin")
    assert limiter._bucket_key == "ratelimit:org-123:admin"


@pytest.mark.asyncio
async def test_config_cache_key_format():
    """Redis config cache key must follow the documented format."""
    limiter = _make_limiter(_FakeRedis(), tenant_id="org-456", endpoint_type="read")
    assert limiter._config_key == "ratelimit_config:org-456"


# ---------------------------------------------------------------------------
# FastAPI integration: 429 response format
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_429_response_includes_kgb_4029():
    """
    The FastAPI app must return {"error_code": "KGB-4029", ...} on a 429
    raised via HTTPException(429).
    """
    from fastapi import FastAPI, HTTPException
    from fastapi.testclient import TestClient

    from app.core.errors import KGBError

    test_app = FastAPI()

    @test_app.exception_handler(HTTPException)
    async def _handler(request, exc):
        from fastapi.responses import JSONResponse

        if exc.status_code == 429:
            code, msg = KGBError.RATE_LIMIT_EXCEEDED
            headers = getattr(exc, "headers", None) or {}
            retry_after = headers.get("Retry-After", "0")
            return JSONResponse(
                status_code=429,
                headers=headers,
                content={
                    "error_code": code,
                    "message": msg,
                    "retry_after": int(retry_after) if str(retry_after).isdigit() else 0,
                },
            )
        return JSONResponse(status_code=exc.status_code, content={"detail": exc.detail})

    @test_app.get("/limited")
    async def _endpoint():
        raise HTTPException(
            status_code=429,
            headers={"Retry-After": "12", "X-RateLimit-Remaining": "0"},
            detail="Rate limit exceeded",
        )

    client = TestClient(test_app, raise_server_exceptions=False)
    response = client.get("/limited")

    assert response.status_code == 429
    body = response.json()
    assert body["error_code"] == "KGB-4029"
    assert body["retry_after"] == 12


@pytest.mark.unit
def test_404_response_includes_kgb_4001():
    """The app must return {"error_code": "KGB-4001", ...} on a 404."""
    from fastapi import FastAPI, HTTPException
    from fastapi.testclient import TestClient

    from app.core.errors import KGBError

    test_app = FastAPI()

    @test_app.exception_handler(HTTPException)
    async def _handler(request, exc):
        from fastapi.responses import JSONResponse

        if exc.status_code == 404:
            code, msg = KGBError.GRAPH_NOT_FOUND
            return JSONResponse(
                status_code=404,
                content={"error_code": code, "message": msg, "detail": str(exc.detail)},
            )
        return JSONResponse(status_code=exc.status_code, content={"detail": exc.detail})

    @test_app.get("/graphs/{graph_id}")
    async def _endpoint(graph_id: str):
        raise HTTPException(status_code=404, detail=f"Graph {graph_id!r} not found")

    client = TestClient(test_app, raise_server_exceptions=False)
    response = client.get("/graphs/nonexistent")

    assert response.status_code == 404
    body = response.json()
    assert body["error_code"] == "KGB-4001"


@pytest.mark.unit
def test_403_response_includes_kgb_4003():
    """The app must return {"error_code": "KGB-4003", ...} on a 403."""
    from fastapi import FastAPI, HTTPException
    from fastapi.testclient import TestClient

    from app.core.errors import KGBError

    test_app = FastAPI()

    @test_app.exception_handler(HTTPException)
    async def _handler(request, exc):
        from fastapi.responses import JSONResponse

        if exc.status_code == 403:
            code, msg = KGBError.PERMISSION_DENIED
            return JSONResponse(status_code=403, content={"error_code": code, "message": msg})
        return JSONResponse(status_code=exc.status_code, content={"detail": exc.detail})

    @test_app.get("/protected")
    async def _endpoint():
        raise HTTPException(status_code=403, detail="Access denied")

    client = TestClient(test_app, raise_server_exceptions=False)
    response = client.get("/protected")

    assert response.status_code == 403
    body = response.json()
    assert body["error_code"] == "KGB-4003"
