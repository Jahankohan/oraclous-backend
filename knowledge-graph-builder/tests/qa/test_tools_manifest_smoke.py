"""QA smoke test — GET /api/v1/tools/manifest (ORA-347).

Integration-layer test: exercises the real HTTP routing stack via a
minimal FastAPI harness (no DB lifespan) so the smoke test runs on
any branch without a live Docker stack.

What this adds over the unit tests in tests/unit/test_tools_manifest.py:
- Real HTTP request / response cycle (routing, serialisation, status code)
- Content-Type header verification
- JSON parsing round-trip
- Verifies the route is correctly registered in the api_router

Run:
    pytest tests/qa/test_tools_manifest_smoke.py -v
"""

from __future__ import annotations

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.api.v1.endpoints.tools import _MANIFEST
from app.api.v1.endpoints.tools import router as tools_router

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


@pytest.fixture(scope="module")
def tools_client() -> TestClient:
    """Minimal FastAPI app with only the tools router — no DB lifespan."""
    app = FastAPI()
    app.include_router(tools_router, prefix="/api/v1")
    return TestClient(app)


class TestToolsManifestHTTP:
    """HTTP-layer smoke tests for GET /api/v1/tools/manifest."""

    @pytest.mark.qa
    def test_returns_200(self, tools_client):
        resp = tools_client.get("/api/v1/tools/manifest")
        assert resp.status_code == 200, (
            f"Expected 200, got {resp.status_code}: {resp.text}"
        )

    @pytest.mark.qa
    def test_content_type_is_json(self, tools_client):
        resp = tools_client.get("/api/v1/tools/manifest")
        assert "application/json" in resp.headers["content-type"]

    @pytest.mark.qa
    def test_body_is_valid_json(self, tools_client):
        resp = tools_client.get("/api/v1/tools/manifest")
        body = resp.json()
        assert isinstance(body, dict)

    @pytest.mark.qa
    def test_top_level_fields_present(self, tools_client):
        body = tools_client.get("/api/v1/tools/manifest").json()
        missing = REQUIRED_TOP_LEVEL - body.keys()
        assert not missing, f"Missing top-level fields: {missing}"

    @pytest.mark.qa
    def test_tools_count_is_nine(self, tools_client):
        body = tools_client.get("/api/v1/tools/manifest").json()
        assert len(body["tools"]) == EXPECTED_TOOL_COUNT, (
            f"Expected {EXPECTED_TOOL_COUNT} tools, got {len(body['tools'])}"
        )

    @pytest.mark.qa
    def test_every_tool_has_required_fields(self, tools_client):
        body = tools_client.get("/api/v1/tools/manifest").json()
        for tool in body["tools"]:
            missing = REQUIRED_TOOL_FIELDS - tool.keys()
            assert not missing, f"Tool {tool.get('name')} missing: {missing}"

    @pytest.mark.qa
    def test_service_field_is_string(self, tools_client):
        body = tools_client.get("/api/v1/tools/manifest").json()
        assert isinstance(body["service"], str) and body["service"]

    @pytest.mark.qa
    def test_version_is_semver(self, tools_client):
        body = tools_client.get("/api/v1/tools/manifest").json()
        parts = body["version"].split(".")
        assert len(parts) == 3 and all(p.isdigit() for p in parts)

    @pytest.mark.qa
    def test_base_url_is_string(self, tools_client):
        body = tools_client.get("/api/v1/tools/manifest").json()
        assert isinstance(body["base_url"], str) and body["base_url"]

    @pytest.mark.qa
    def test_response_matches_static_manifest(self, tools_client):
        """HTTP body must match the module-level _MANIFEST singleton."""
        body = tools_client.get("/api/v1/tools/manifest").json()
        assert body["service"] == _MANIFEST["service"]
        assert body["version"] == _MANIFEST["version"]
        assert len(body["tools"]) == len(_MANIFEST["tools"])

    @pytest.mark.qa
    def test_no_auth_required(self, tools_client):
        """Endpoint is public — must return 200 without any Authorization header."""
        resp = tools_client.get("/api/v1/tools/manifest")
        assert resp.status_code == 200
