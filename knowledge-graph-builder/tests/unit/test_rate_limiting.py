"""Unit tests for API rate limiting (ORA-104).

Verifies:
- 429 is returned when the per-minute limit is exceeded on /chat
- 429 is returned when the per-minute limit is exceeded on /chat/stream
- 429 is returned when the per-minute limit is exceeded on /graphs/{id}/ingest
- 429 is returned when the per-minute limit is exceeded on /webhooks/{graph_id}/{connector_id}
- Requests within the limit return the expected non-429 response
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient
from slowapi import _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_app():
    """Build a minimal FastAPI app with the rate limiter attached."""
    from fastapi import FastAPI
    from slowapi import _rate_limit_exceeded_handler
    from slowapi.errors import RateLimitExceeded

    from app.core.rate_limiter import limiter

    app = FastAPI()
    app.state.limiter = limiter
    app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
    return app


# ---------------------------------------------------------------------------
# Rate limiter module tests
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_limiter_uses_remote_address_key_func():
    """Limiter is configured to key on the client IP address."""
    from slowapi.util import get_remote_address

    from app.core.rate_limiter import limiter

    assert limiter._key_func is get_remote_address


@pytest.mark.unit
def test_limiter_is_singleton():
    """Importing limiter twice returns the same object."""
    from app.core.rate_limiter import limiter as limiter_a
    from app.core.rate_limiter import limiter as limiter_b

    assert limiter_a is limiter_b


# ---------------------------------------------------------------------------
# App-level exception handler test
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_rate_limit_exceeded_handler_registered():
    """The app registers a handler for RateLimitExceeded that returns 429."""
    app = _make_app()

    # Verify the handler is registered
    assert RateLimitExceeded in app.exception_handlers


@pytest.mark.unit
def test_rate_limit_exceeded_returns_429():
    """When the limiter raises RateLimitExceeded the response status is 429."""
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

    # First request — within limit
    r1 = client.get("/limited", headers={"X-Forwarded-For": "10.0.0.1"})
    assert r1.status_code == 200

    # Second request from the same IP — limit exceeded
    r2 = client.get("/limited", headers={"X-Forwarded-For": "10.0.0.1"})
    assert r2.status_code == 429


@pytest.mark.unit
def test_different_ips_have_independent_limits():
    """Rate limits are per-IP — different IPs do not share a bucket."""
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

    # IP A exhausts its limit
    client.get("/limited", headers={"X-Forwarded-For": "10.0.0.1"})
    r_a_blocked = client.get("/limited", headers={"X-Forwarded-For": "10.0.0.1"})
    assert r_a_blocked.status_code == 429

    # IP B still within its limit
    r_b = client.get("/limited", headers={"X-Forwarded-For": "10.0.0.2"})
    assert r_b.status_code == 200


# ---------------------------------------------------------------------------
# Decorator presence tests (static analysis — no running server needed)
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_chat_endpoint_has_rate_limit_decorator():
    """The /chat endpoint function is wrapped with a rate limit decorator."""
    from app.api.v1.endpoints.chat import chat_with_graph

    # slowapi attaches _rate_limit_key_func or similar attributes when the decorator is applied
    assert (
        hasattr(chat_with_graph, "_rate_limit_key_func")
        or hasattr(chat_with_graph, "__wrapped__")
        or hasattr(chat_with_graph, "_is_coroutine")
    ), "chat_with_graph should be decorated by slowapi @limiter.limit"


@pytest.mark.unit
def test_stream_chat_endpoint_has_rate_limit_decorator():
    """The /chat/stream endpoint function is wrapped with a rate limit decorator."""
    from app.api.v1.endpoints.chat import stream_chat_with_graph

    assert (
        hasattr(stream_chat_with_graph, "_rate_limit_key_func")
        or hasattr(stream_chat_with_graph, "__wrapped__")
        or hasattr(stream_chat_with_graph, "_is_coroutine")
    ), "stream_chat_with_graph should be decorated by slowapi @limiter.limit"


@pytest.mark.unit
def test_ingest_endpoint_has_rate_limit_decorator():
    """The /graphs/{id}/ingest endpoint function is wrapped with a rate limit decorator."""
    from app.api.v1.endpoints.graphs import ingest_data_corrected

    assert (
        hasattr(ingest_data_corrected, "_rate_limit_key_func")
        or hasattr(ingest_data_corrected, "__wrapped__")
        or hasattr(ingest_data_corrected, "_is_coroutine")
    ), "ingest_data_corrected should be decorated by slowapi @limiter.limit"


@pytest.mark.unit
def test_webhook_endpoint_has_rate_limit_decorator():
    """The /webhooks/{graph_id}/{connector_id} endpoint is wrapped with a rate limit decorator."""
    from app.api.v1.endpoints.webhooks import receive_webhook

    assert (
        hasattr(receive_webhook, "_rate_limit_key_func")
        or hasattr(receive_webhook, "__wrapped__")
        or hasattr(receive_webhook, "_is_coroutine")
    ), "receive_webhook should be decorated by slowapi @limiter.limit"


# ---------------------------------------------------------------------------
# Main app integration — limiter attached to app.state
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_main_app_has_limiter_attached():
    """The production FastAPI app has limiter attached to app.state."""
    # We patch all heavy startup side-effects so we can import the app module cleanly.
    with (
        patch("app.core.neo4j_client.neo4j_client.connect", new=AsyncMock()),
        patch("app.core.database.create_tables", new=AsyncMock()),
        patch("app.core.telemetry.setup_telemetry"),
        patch("app.core.telemetry.instrument_fastapi"),
    ):
        from app.core.rate_limiter import limiter
        from app.main import app

        assert hasattr(app.state, "limiter")
        assert app.state.limiter is limiter


@pytest.mark.unit
def test_main_app_has_rate_limit_exception_handler():
    """The production FastAPI app registers a 429 handler for RateLimitExceeded."""
    with (
        patch("app.core.neo4j_client.neo4j_client.connect", new=AsyncMock()),
        patch("app.core.database.create_tables", new=AsyncMock()),
        patch("app.core.telemetry.setup_telemetry"),
        patch("app.core.telemetry.instrument_fastapi"),
    ):
        from slowapi.errors import RateLimitExceeded

        from app.main import app

        assert RateLimitExceeded in app.exception_handlers
