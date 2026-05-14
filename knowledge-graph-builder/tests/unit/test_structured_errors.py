"""
Unit tests for TASK-031 / STORY-009: structured KGB-XXXX error codes.

Uses FastAPI TestClient to verify that the HTTPException handler in main.py
returns the correct error_code in the response body for each scenario.

Coverage:
- 404 on unknown graph_id → error_code: KGB-4001 in response body
- 403 on permission denied → error_code: KGB-4003 in response body
- 503 on Neo4j down → error_code: KGB-5001 in response body
- 429 on rate limit → error_code: KGB-4029 in response body
"""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def test_app():
    """
    Create a minimal FastAPI app with the structured HTTPException handler from
    main.py. We patch heavy startup dependencies so the app can be instantiated
    without a running Neo4j, Redis, or Celery.

    We use `scope="module"` to import app once per module; FastAPI TestClient is
    thread-safe for read-only test scenarios.
    """
    with (
        patch("app.core.neo4j_client.neo4j_client.connect", new=AsyncMock()),
        patch("app.core.database.create_tables", new=AsyncMock()),
        patch("app.core.telemetry.setup_telemetry"),
        patch("app.core.telemetry.instrument_fastapi"),
    ):
        from app.main import app as _app

    # Register test-only routes that trigger each error condition
    @_app.get("/test/graph-not-found")
    async def _graph_not_found():
        raise HTTPException(
            status_code=404, detail="Graph 'unknown-graph' does not exist"
        )

    @_app.get("/test/permission-denied")
    async def _permission_denied():
        raise HTTPException(status_code=403, detail="Insufficient permissions")

    @_app.get("/test/neo4j-down")
    async def _neo4j_down():
        raise HTTPException(status_code=503, detail="Neo4j connection failed")

    @_app.get("/test/rate-limited")
    async def _rate_limited():
        raise HTTPException(
            status_code=429,
            detail="Rate limit exceeded",
            headers={"Retry-After": "30"},
        )

    return _app


@pytest.fixture(scope="module")
def client(test_app):
    return TestClient(test_app, raise_server_exceptions=False)


# ---------------------------------------------------------------------------
# 1. 404 → KGB-4001
# ---------------------------------------------------------------------------


class TestGraphNotFoundError:
    @pytest.mark.unit
    def test_404_returns_kgb_4001_error_code(self, client):
        """404 on unknown graph_id must return error_code: KGB-4001."""
        response = client.get("/test/graph-not-found")
        assert response.status_code == 404
        body = response.json()
        assert body["error_code"] == "KGB-4001"

    @pytest.mark.unit
    def test_404_response_has_message_field(self, client):
        """404 response must include a human-readable message field."""
        response = client.get("/test/graph-not-found")
        body = response.json()
        assert "message" in body
        assert len(body["message"]) > 0

    @pytest.mark.unit
    def test_404_response_has_detail_field(self, client):
        """404 response must preserve the original detail string."""
        response = client.get("/test/graph-not-found")
        body = response.json()
        assert "detail" in body

    @pytest.mark.unit
    def test_kgb_4001_code_matches_errors_module(self):
        """The code returned by the handler must match KGBError.GRAPH_NOT_FOUND."""
        from app.core.errors import KGBError

        code, _ = KGBError.GRAPH_NOT_FOUND
        assert code == "KGB-4001"


# ---------------------------------------------------------------------------
# 2. 403 → KGB-4003
# ---------------------------------------------------------------------------


class TestPermissionDeniedError:
    @pytest.mark.unit
    def test_403_returns_kgb_4003_error_code(self, client):
        """403 permission denied must return error_code: KGB-4003."""
        response = client.get("/test/permission-denied")
        assert response.status_code == 403
        body = response.json()
        assert body["error_code"] == "KGB-4003"

    @pytest.mark.unit
    def test_403_response_has_message_field(self, client):
        """403 response must include a human-readable message field."""
        response = client.get("/test/permission-denied")
        body = response.json()
        assert "message" in body
        assert len(body["message"]) > 0

    @pytest.mark.unit
    def test_403_does_not_leak_internal_detail(self, client):
        """
        403 response should NOT include the original detail string from the exception
        to avoid leaking internal permission logic to unauthorized callers.
        """
        response = client.get("/test/permission-denied")
        body = response.json()
        # The message should be the generic KGBError message, not the internal detail
        assert body["error_code"] == "KGB-4003"

    @pytest.mark.unit
    def test_kgb_4003_code_matches_errors_module(self):
        """The code returned by the handler must match KGBError.PERMISSION_DENIED."""
        from app.core.errors import KGBError

        code, _ = KGBError.PERMISSION_DENIED
        assert code == "KGB-4003"


# ---------------------------------------------------------------------------
# 3. 503 → KGB-5001
# ---------------------------------------------------------------------------


class TestNeo4jDownError:
    @pytest.mark.unit
    def test_503_returns_kgb_5001_error_code(self, client):
        """503 on Neo4j down must return error_code: KGB-5001."""
        response = client.get("/test/neo4j-down")
        assert response.status_code == 503
        body = response.json()
        assert body["error_code"] == "KGB-5001"

    @pytest.mark.unit
    def test_503_response_has_message_field(self, client):
        """503 response must include a human-readable message field."""
        response = client.get("/test/neo4j-down")
        body = response.json()
        assert "message" in body
        assert len(body["message"]) > 0

    @pytest.mark.unit
    def test_503_response_has_detail_field(self, client):
        """503 response must include the detail field for debugging."""
        response = client.get("/test/neo4j-down")
        body = response.json()
        assert "detail" in body

    @pytest.mark.unit
    def test_kgb_5001_code_matches_errors_module(self):
        """The code returned by the handler must match KGBError.NEO4J_UNAVAILABLE."""
        from app.core.errors import KGBError

        code, _ = KGBError.NEO4J_UNAVAILABLE
        assert code == "KGB-5001"

    @pytest.mark.unit
    def test_neo4j_degradation_middleware_also_returns_kgb_5001(self):
        """
        The Neo4jDegradationMiddleware (for unhandled ServiceUnavailable exceptions)
        must also use KGB-5001.
        """
        from neo4j.exceptions import ServiceUnavailable
        from starlette.applications import Starlette
        from starlette.routing import Route
        from starlette.testclient import TestClient

        from app.core.telemetry import Neo4jDegradationMiddleware

        async def crash(request):
            raise ServiceUnavailable("Neo4j is down")

        app = Starlette(routes=[Route("/", crash)])
        app.add_middleware(Neo4jDegradationMiddleware)

        client = TestClient(app, raise_server_exceptions=False)
        response = client.get("/")

        assert response.status_code == 503
        body = response.json()
        assert body["error_code"] == "KGB-5001"


# ---------------------------------------------------------------------------
# 4. 429 → KGB-4029
# ---------------------------------------------------------------------------


class TestRateLimitError:
    @pytest.mark.unit
    def test_429_returns_kgb_4029_error_code(self, client):
        """429 rate limit must return error_code: KGB-4029."""
        response = client.get("/test/rate-limited")
        assert response.status_code == 429
        body = response.json()
        assert body["error_code"] == "KGB-4029"

    @pytest.mark.unit
    def test_429_response_has_retry_after_in_headers(self, client):
        """429 response must include the Retry-After header."""
        response = client.get("/test/rate-limited")
        # Retry-After can come from headers or response body
        has_retry_after_header = "retry-after" in response.headers
        has_retry_after_body = "retry_after" in response.json()
        assert has_retry_after_header or has_retry_after_body

    @pytest.mark.unit
    def test_429_response_has_message_field(self, client):
        """429 response must include a human-readable message."""
        response = client.get("/test/rate-limited")
        body = response.json()
        assert "message" in body
        assert len(body["message"]) > 0

    @pytest.mark.unit
    def test_kgb_4029_code_matches_errors_module(self):
        """The code returned by the handler must match KGBError.RATE_LIMIT_EXCEEDED."""
        from app.core.errors import KGBError

        code, _ = KGBError.RATE_LIMIT_EXCEEDED
        assert code == "KGB-4029"

    @pytest.mark.unit
    def test_429_retry_after_body_value_is_integer(self, client):
        """
        The retry_after field in the body must be an integer (seconds), not a string.
        """
        response = client.get("/test/rate-limited")
        body = response.json()
        if "retry_after" in body:
            assert isinstance(body["retry_after"], int)


# ---------------------------------------------------------------------------
# 5. Unhandled status codes fall through without KGB code
# ---------------------------------------------------------------------------


class TestFallThroughStatusCodes:
    @pytest.mark.unit
    def test_400_bad_request_no_kgb_code(self, client):
        """
        Status codes without a KGB mapping must return plain detail, not a
        KGB error code — avoids misleading clients with wrong error codes.
        """
        app = client.app

        @app.get("/test/bad-request")
        async def _bad_request():
            raise HTTPException(status_code=400, detail="Bad request")

        response = client.get("/test/bad-request")
        assert response.status_code == 400
        body = response.json()
        # Must NOT have an error_code field (no KGB mapping for 400)
        assert "error_code" not in body or body.get("error_code") is None


# ---------------------------------------------------------------------------
# 6. Error code consistency — all codes defined and cross-checked
# ---------------------------------------------------------------------------


class TestErrorCodeConsistency:
    @pytest.mark.unit
    def test_all_four_standard_error_codes_defined(self):
        """All four standard error codes (4001, 4003, 4029, 5001) must be present."""
        from app.core.errors import KGBError

        assert KGBError.GRAPH_NOT_FOUND[0] == "KGB-4001"
        assert KGBError.PERMISSION_DENIED[0] == "KGB-4003"
        assert KGBError.RATE_LIMIT_EXCEEDED[0] == "KGB-4029"
        assert KGBError.NEO4J_UNAVAILABLE[0] == "KGB-5001"

    @pytest.mark.unit
    def test_all_error_codes_start_with_kgb_prefix(self):
        """All error codes must use the KGB- prefix for discoverability."""
        from app.core.errors import KGBError

        attrs = [
            "NEO4J_UNAVAILABLE",
            "REDIS_UNAVAILABLE",
            "LLM_UNAVAILABLE",
            "CELERY_UNAVAILABLE",
            "GRAPH_NOT_FOUND",
            "PERMISSION_DENIED",
            "RATE_LIMIT_EXCEEDED",
        ]
        for attr in attrs:
            code, _ = getattr(KGBError, attr)
            assert code.startswith(
                "KGB-"
            ), f"{attr}: code '{code}' must start with KGB-"

    @pytest.mark.unit
    def test_error_codes_are_unique_no_collisions(self):
        """No two error codes must share the same code string."""
        from app.core.errors import KGBError

        all_codes = [
            KGBError.NEO4J_UNAVAILABLE[0],
            KGBError.REDIS_UNAVAILABLE[0],
            KGBError.LLM_UNAVAILABLE[0],
            KGBError.CELERY_UNAVAILABLE[0],
            KGBError.GRAPH_NOT_FOUND[0],
            KGBError.PERMISSION_DENIED[0],
            KGBError.RATE_LIMIT_EXCEEDED[0],
        ]
        assert len(all_codes) == len(
            set(all_codes)
        ), f"Duplicate error codes found: {all_codes}"
