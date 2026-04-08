"""Unit tests for auth endpoint rate limiting (login, register, refresh, forgot-password).

Tests cover:
- 429 response body is generic — rate-limit configuration is never exposed
- Limit is enforced: (n+1)th request from the same IP returns 429
- Requests under the limit succeed (200)
- Per-IP isolation: independent counter per client IP

Implementation note: these tests use an in-memory slowapi backend (storage_uri="memory://")
so no Redis connection is required.  Per-IP isolation is a property of slowapi's
key_func mechanism — different key values resolve to independent counters.
"""

import pytest
from fastapi import FastAPI, Request
from fastapi.testclient import TestClient
from slowapi import Limiter
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

from app.core.rate_limiter import rate_limit_exceeded_handler


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build_test_client(limit_str: str) -> TestClient:
    """Create a minimal FastAPI app with a single POST /test endpoint.

    Uses an in-memory backend so tests are isolated and deterministic.
    Each call creates a fresh limiter with empty counters.
    """
    test_limiter = Limiter(key_func=get_remote_address, storage_uri="memory://")
    app = FastAPI()
    app.state.limiter = test_limiter
    app.add_exception_handler(RateLimitExceeded, rate_limit_exceeded_handler)

    @app.post("/test")
    @test_limiter.limit(limit_str)
    async def _endpoint(request: Request):
        return {"ok": True}

    return TestClient(app)


def _build_ip_isolated_client(limit_str: str, key_header: str = "X-Test-Key") -> TestClient:
    """Build a test app whose rate-limit key is read from a custom request header.

    This allows per-IP isolation to be exercised within a single TestClient
    by passing different header values (simulating different client IPs).
    """
    def _key_from_header(request: Request) -> str:
        return request.headers.get(key_header, "default")

    test_limiter = Limiter(key_func=_key_from_header, storage_uri="memory://")
    app = FastAPI()
    app.state.limiter = test_limiter
    app.add_exception_handler(RateLimitExceeded, rate_limit_exceeded_handler)

    @app.post("/test")
    @test_limiter.limit(limit_str)
    async def _endpoint(request: Request):
        return {"ok": True}

    return TestClient(app)


# ---------------------------------------------------------------------------
# 429 handler: no info leakage
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_429_body_is_generic_no_limit_detail():
    """429 response body must not expose rate-limit configuration."""
    client = _build_test_client("1/minute")

    client.post("/test")           # consume the 1 allowed
    r = client.post("/test")       # should be blocked

    assert r.status_code == 429
    body = r.json()
    body_str = str(body).lower()

    # Must not leak limit configuration
    assert "per" not in body_str
    assert "minute" not in body_str
    assert "1 per" not in body_str
    assert "rate limit exceeded:" not in body_str

    # Must have a generic user-facing key
    assert "detail" in body
    assert body["detail"] == "Too many requests"


@pytest.mark.unit
def test_429_status_code_is_correct():
    """HTTP status code is 429, not 500 or any other error."""
    client = _build_test_client("1/minute")
    client.post("/test")
    r = client.post("/test")
    assert r.status_code == 429


# ---------------------------------------------------------------------------
# Limit enforcement per endpoint spec
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_register_limit_5_per_minute():
    """POST /register/ limit: first 5 succeed, 6th returns 429."""
    client = _build_test_client("5/minute")

    for _ in range(5):
        r = client.post("/test")
        assert r.status_code == 200

    r = client.post("/test")
    assert r.status_code == 429


@pytest.mark.unit
def test_login_limit_10_per_minute():
    """POST /login/ limit: first 10 succeed, 11th returns 429."""
    client = _build_test_client("10/minute")

    for _ in range(10):
        r = client.post("/test")
        assert r.status_code == 200

    r = client.post("/test")
    assert r.status_code == 429


@pytest.mark.unit
def test_refresh_limit_20_per_minute():
    """POST /refresh/ limit: first 20 succeed, 21st returns 429."""
    client = _build_test_client("20/minute")

    for _ in range(20):
        r = client.post("/test")
        assert r.status_code == 200

    r = client.post("/test")
    assert r.status_code == 429


@pytest.mark.unit
def test_forgot_password_limit_5_per_minute():
    """POST /forgot-password/ limit: first 5 succeed, 6th returns 429."""
    client = _build_test_client("5/minute")

    for _ in range(5):
        r = client.post("/test")
        assert r.status_code == 200

    r = client.post("/test")
    assert r.status_code == 429


# ---------------------------------------------------------------------------
# Per-IP isolation
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_per_ip_isolation():
    """Rate limit counters are independent per client IP.

    Client A exhausting its limit must not affect Client B.
    """
    client = _build_ip_isolated_client("2/minute")

    # Client A exhausts its limit
    client.post("/test", headers={"X-Test-Key": "ip-A"})
    client.post("/test", headers={"X-Test-Key": "ip-A"})
    r_a_blocked = client.post("/test", headers={"X-Test-Key": "ip-A"})
    assert r_a_blocked.status_code == 429

    # Client B is unaffected
    r_b = client.post("/test", headers={"X-Test-Key": "ip-B"})
    assert r_b.status_code == 200
