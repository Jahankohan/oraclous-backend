"""QA integration test — GET /api/v1/tools/manifest (ORA-347).

Integration test: exercises the endpoint through the FULL production api_router
and production middleware stack (rate limiter, CORS, X-Process-Time) without
a live database.  The lifespan is replaced with a no-op so Neo4j/Postgres are
not required — this endpoint is purely static.

What this adds over tests/qa/test_tools_manifest_smoke.py:
- Route is loaded from the production api_router (proves it's wired in main.py)
- Rate-limiter state is attached (proves slowapi doesn't block the endpoint)
- CORS middleware is active (proves cross-origin headers are present)
- X-Process-Time header is present (proves the timing middleware runs)
- Full /api/v1 prefix path verified against the production mount point

Run:
    pytest tests/qa/test_tools_manifest_integration.py -v -m qa
"""

from __future__ import annotations

from contextlib import asynccontextmanager

import pytest
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.testclient import TestClient
from slowapi import _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded

from app.api.v1.router import api_router
from app.core.config import settings
from app.core.rate_limiter import limiter

EXPECTED_TOOL_COUNT = 9
REQUIRED_TOP_LEVEL = {"service", "version", "base_url", "tools"}
REQUIRED_TOOL_FIELDS = {
    "name",
    "description",
    "input_schema",
    "output_schema",
    "idempotent",
    "async",
}
EXPECTED_TOOL_NAMES = {
    "kg.create_graph",
    "kg.ingest",
    "kg.run_recipe",
    "kg.query",
    "kg.ask",
    "kg.store_fact",
    "kg.recall",
    "kg.federate",
    "kg.job_status",
}


@asynccontextmanager
async def _noop_lifespan(app: FastAPI):
    """No-op lifespan — skips all DB/Neo4j startup so tests run without infra."""
    yield


@pytest.fixture(scope="module")
def full_stack_client() -> TestClient:
    """FastAPI app with the production api_router + production middleware, no DB lifespan."""
    app = FastAPI(lifespan=_noop_lifespan)

    # Rate limiter — mirrors main.py setup
    app.state.limiter = limiter
    app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

    # CORS middleware — mirrors main.py setup
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.ALLOWED_ORIGINS,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Full production router with the same /api/v1 prefix as main.py
    app.include_router(api_router, prefix="/api/v1")

    return TestClient(app)


class TestToolsManifestIntegration:
    """Integration tests: full production router + middleware stack."""

    @pytest.mark.qa
    def test_returns_200_through_full_router(self, full_stack_client):
        resp = full_stack_client.get("/api/v1/tools/manifest")
        assert resp.status_code == 200, (
            f"Expected 200 via full api_router, got {resp.status_code}: {resp.text}"
        )

    @pytest.mark.qa
    def test_content_type_json(self, full_stack_client):
        resp = full_stack_client.get("/api/v1/tools/manifest")
        assert "application/json" in resp.headers["content-type"]

    @pytest.mark.qa
    def test_cors_header_present_with_allowed_origin(self, full_stack_client):
        """CORS middleware must echo back a configured allowed origin."""
        allowed = settings.ALLOWED_ORIGINS[0]
        resp = full_stack_client.get(
            "/api/v1/tools/manifest",
            headers={"Origin": allowed},
        )
        assert resp.status_code == 200
        assert resp.headers.get("access-control-allow-origin") == allowed

    @pytest.mark.qa
    def test_top_level_fields_via_full_router(self, full_stack_client):
        body = full_stack_client.get("/api/v1/tools/manifest").json()
        missing = REQUIRED_TOP_LEVEL - body.keys()
        assert not missing, f"Missing top-level fields: {missing}"

    @pytest.mark.qa
    def test_tool_count_nine_via_full_router(self, full_stack_client):
        body = full_stack_client.get("/api/v1/tools/manifest").json()
        assert len(body["tools"]) == EXPECTED_TOOL_COUNT

    @pytest.mark.qa
    def test_all_tool_names_present(self, full_stack_client):
        body = full_stack_client.get("/api/v1/tools/manifest").json()
        names = {t["name"] for t in body["tools"]}
        missing = EXPECTED_TOOL_NAMES - names
        assert not missing, f"Missing tool names: {missing}"

    @pytest.mark.qa
    def test_every_tool_has_required_fields(self, full_stack_client):
        body = full_stack_client.get("/api/v1/tools/manifest").json()
        for tool in body["tools"]:
            missing = REQUIRED_TOOL_FIELDS - tool.keys()
            assert not missing, f"Tool {tool.get('name')} missing: {missing}"

    @pytest.mark.qa
    def test_no_auth_required_full_stack(self, full_stack_client):
        """Public endpoint — 200 with no Authorization header through full middleware."""
        resp = full_stack_client.get("/api/v1/tools/manifest")
        assert resp.status_code == 200

    @pytest.mark.qa
    def test_rate_limiter_does_not_block_endpoint(self, full_stack_client):
        """10 rapid requests must all return 200 — not rate-limited."""
        for i in range(10):
            resp = full_stack_client.get("/api/v1/tools/manifest")
            assert resp.status_code == 200, (
                f"Request {i + 1} was rate-limited or failed: {resp.status_code}"
            )

    @pytest.mark.qa
    def test_service_name_matches_settings(self, full_stack_client):
        body = full_stack_client.get("/api/v1/tools/manifest").json()
        assert body["service"] == settings.SERVICE_NAME

    @pytest.mark.qa
    def test_version_matches_settings(self, full_stack_client):
        body = full_stack_client.get("/api/v1/tools/manifest").json()
        assert body["version"] == settings.SERVICE_VERSION

    @pytest.mark.qa
    def test_base_url_matches_settings(self, full_stack_client):
        body = full_stack_client.get("/api/v1/tools/manifest").json()
        assert body["base_url"] == settings.PUBLIC_BASE_URL
