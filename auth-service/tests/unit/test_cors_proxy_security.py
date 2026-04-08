"""Unit tests for CORS configuration and proxy-aware IP rate limiting.

Covers ORA-135 security fixes:
  Bug 1 — CORS must not use wildcard origin with allow_credentials=True.
           Explicit ALLOWED_ORIGINS must gate cross-origin requests.
  Bug 2 — Rate-limit key func must reflect real client IP, not proxy IP.
           ProxyHeadersMiddleware rewrites request.client only for trusted proxies.
"""

import pytest
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.testclient import TestClient
from slowapi import Limiter
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address
from uvicorn.middleware.proxy_headers import ProxyHeadersMiddleware

from app.core.rate_limiter import rate_limit_exceeded_handler


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build_cors_app(allowed_origins: list[str]) -> TestClient:
    """Minimal app with CORSMiddleware using explicit allowed_origins."""
    app = FastAPI()

    app.add_middleware(
        CORSMiddleware,
        allow_origins=allowed_origins,
        allow_credentials=True,
        allow_methods=["GET", "POST"],
        allow_headers=["Authorization", "Content-Type"],
    )

    @app.get("/ping")
    async def ping():
        return {"ok": True}

    return TestClient(app, raise_server_exceptions=True)


def _build_proxy_rate_limit_app(trusted_hosts: list[str]) -> TestClient:
    """App with ProxyHeadersMiddleware + slowapi, using get_remote_address.

    Simulates the production middleware stack: proxy middleware rewrites
    request.client, then slowapi reads the rewritten value.

    Note: Starlette TestClient presents as host "testclient" (not 127.0.0.1).
    Tests that want ProxyHeadersMiddleware to trust the direct connection must
    pass "testclient" in trusted_hosts to match the TestClient source IP.
    """
    test_limiter = Limiter(key_func=get_remote_address, storage_uri="memory://")
    app = FastAPI()
    app.state.limiter = test_limiter
    app.add_exception_handler(RateLimitExceeded, rate_limit_exceeded_handler)

    @app.post("/test")
    @test_limiter.limit("2/minute")
    async def _endpoint(request: Request):
        return {"client": request.client.host}

    app.add_middleware(ProxyHeadersMiddleware, trusted_hosts=trusted_hosts)

    return TestClient(app, raise_server_exceptions=False)


# TestClient source IP as seen by ASGI scope["client"][0]
_TESTCLIENT_HOST = "testclient"


# ---------------------------------------------------------------------------
# Bug 1 — CORS: no wildcard + credentials
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_cors_allowed_origin_receives_header():
    """Requests from an explicitly allowed origin get Access-Control-Allow-Origin."""
    client = _build_cors_app(["https://app.example.com"])

    r = client.get("/ping", headers={"Origin": "https://app.example.com"})

    assert r.status_code == 200
    assert r.headers.get("access-control-allow-origin") == "https://app.example.com"


@pytest.mark.unit
def test_cors_disallowed_origin_no_header():
    """Requests from an origin not in ALLOWED_ORIGINS do not get the ACAO header."""
    client = _build_cors_app(["https://app.example.com"])

    r = client.get("/ping", headers={"Origin": "https://evil.example.com"})

    # Request still succeeds (CORS is a browser enforcement), but no ACAO header
    assert r.status_code == 200
    assert "access-control-allow-origin" not in r.headers


@pytest.mark.unit
def test_cors_wildcard_is_not_configured():
    """Production CORS config must not use wildcard origin.

    Wildcard + allow_credentials=True is rejected by browsers and violates
    the CORS spec.  This test guards against that combination being re-introduced.
    """
    from app.core.config import settings

    assert "*" not in settings.ALLOWED_ORIGINS, (
        "Wildcard origin in ALLOWED_ORIGINS is forbidden when allow_credentials=True"
    )


@pytest.mark.unit
def test_cors_preflight_returns_correct_methods():
    """OPTIONS preflight for allowed origin returns only the permitted methods."""
    client = _build_cors_app(["https://app.example.com"])

    r = client.options(
        "/ping",
        headers={
            "Origin": "https://app.example.com",
            "Access-Control-Request-Method": "POST",
            "Access-Control-Request-Headers": "Content-Type",
        },
    )

    assert r.status_code == 200
    allowed = r.headers.get("access-control-allow-methods", "")
    # DELETE, PUT, PATCH must not be listed — auth service only needs GET/POST
    for forbidden in ("DELETE", "PUT", "PATCH"):
        assert forbidden not in allowed, f"{forbidden} must not be in allowed methods"


# ---------------------------------------------------------------------------
# Bug 2 — Proxy-aware rate limiting
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_proxy_trusted_uses_forwarded_for_ip():
    """When connection is from a trusted proxy, X-Forwarded-For is used as client IP.

    Rate-limit bucket is keyed on the real client IP, not the proxy IP.
    """
    # Starlette TestClient presents as "testclient" in scope["client"][0].
    # Declare that host as trusted so ProxyHeadersMiddleware honours XFF headers.
    client = _build_proxy_rate_limit_app(trusted_hosts=[_TESTCLIENT_HOST])

    r = client.post("/test", headers={"X-Forwarded-For": "1.2.3.4"})

    assert r.status_code == 200
    # ProxyHeadersMiddleware rewrites request.client to the XFF value
    assert r.json()["client"] == "1.2.3.4"


@pytest.mark.unit
def test_proxy_untrusted_ignores_forwarded_for():
    """When connection is NOT from a trusted proxy, X-Forwarded-For is ignored.

    Protects against IP spoofing by untrusted callers who inject XFF headers.
    """
    # No trusted hosts configured — direct connections only
    client = _build_proxy_rate_limit_app(trusted_hosts=[])

    r = client.post("/test", headers={"X-Forwarded-For": "9.9.9.9"})

    assert r.status_code == 200
    # Real client IP (TestClient host), NOT the spoofed XFF value
    assert r.json()["client"] != "9.9.9.9"


@pytest.mark.unit
def test_proxy_rate_limit_buckets_by_real_ip():
    """Two clients behind the same proxy have independent rate-limit buckets."""
    client = _build_proxy_rate_limit_app(trusted_hosts=[_TESTCLIENT_HOST])

    # Client A: exhaust 2/minute limit
    client.post("/test", headers={"X-Forwarded-For": "10.0.0.1"})
    client.post("/test", headers={"X-Forwarded-For": "10.0.0.1"})
    r_a_blocked = client.post("/test", headers={"X-Forwarded-For": "10.0.0.1"})
    assert r_a_blocked.status_code == 429

    # Client B: independent bucket — should still succeed
    r_b = client.post("/test", headers={"X-Forwarded-For": "10.0.0.2"})
    assert r_b.status_code == 200
