"""Integration tests for ORA-393: 401 for missing auth (RFC 7235 compliance).

QA validation for PR #107 / ORA-402.

Covers:
  1. Missing auth header → 401 Unauthorized with WWW-Authenticate: Bearer
  2. Invalid/expired token → 401 Unauthorized (token rejection)
  3. Valid token → 200 OK (entities endpoint happy path)
  4. Entities endpoint smoke test: paginated EntityListResponse
  5. Regression — other protected endpoints also return 401 (not 403) for missing auth

The fix in PR #107:
  - app/api/dependencies.py: HTTPBearer(auto_error=False) so the framework does NOT
    auto-raise 403; instead get_current_user raises 401 + WWW-Authenticate: Bearer.
  - Before: missing auth header → 403 Forbidden (FastAPI HTTPBearer default)
  - After:  missing auth header → 401 Unauthorized (RFC 7235 §3.1 compliant)
"""

from __future__ import annotations

import uuid
from unittest.mock import AsyncMock, patch

import pytest
from httpx import AsyncClient

GRAPH_ID = str(uuid.uuid4())
FAKE_USER = {"id": str(uuid.uuid4()), "email": "test@example.com"}


def _patch_auth(user: dict | None = None):
    """Patch auth_service so verify_token returns a valid user."""
    p = patch("app.api.dependencies.auth_service")
    mock_auth = p.start()
    mock_auth.verify_token = AsyncMock(return_value=user or FAKE_USER)
    return p


def _patch_auth_reject():
    """Patch auth_service so verify_token raises (simulates expired/invalid token)."""
    from fastapi import HTTPException, status

    p = patch("app.api.dependencies.auth_service")
    mock_auth = p.start()
    mock_auth.verify_token = AsyncMock(
        side_effect=HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired token",
        )
    )
    return p


# ---------------------------------------------------------------------------
# Primary: 401 for missing auth (ORA-393 core fix)
# ---------------------------------------------------------------------------


@pytest.mark.integration
@pytest.mark.api
@pytest.mark.security
async def test_entities_no_auth_returns_401(async_client: AsyncClient) -> None:
    """Missing Authorization header → 401 Unauthorized (not 403 Forbidden).

    This is the primary assertion for PR #107 / ORA-393.
    Pre-fix: HTTPBearer(auto_error=True) emitted 403.
    Post-fix: HTTPBearer(auto_error=False) allows get_current_user to raise 401.
    """
    response = await async_client.get(f"/api/v1/graphs/{GRAPH_ID}/entities")

    assert response.status_code == 401, (
        f"Expected 401 Unauthorized for missing auth, got {response.status_code}. "
        "PR #107 (ORA-393) may not be in effect."
    )


@pytest.mark.integration
@pytest.mark.api
@pytest.mark.security
async def test_entities_no_auth_has_www_authenticate_header(
    async_client: AsyncClient,
) -> None:
    """Missing auth → response includes WWW-Authenticate: Bearer (RFC 7235 §3.1).

    KNOWN GAP (ORA-402 finding): get_current_user raises HTTPException with
    headers={"WWW-Authenticate": "Bearer"}, but the custom http_exception_handler
    in main.py falls back to JSONResponse(status_code, content={"detail": ...})
    without forwarding exc.headers — stripping the WWW-Authenticate header.

    Fix required in main.py fallback: add headers=getattr(exc, "headers", None) or {}.
    This test is marked xfail until that fix is applied; it will then auto-promote to pass.
    """
    response = await async_client.get(f"/api/v1/graphs/{GRAPH_ID}/entities")

    assert response.status_code == 401
    www_auth = response.headers.get("www-authenticate", "")
    # NOTE: this assertion currently FAILS because the custom exception handler
    # in main.py drops exc.headers. See docstring above and ORA-402 QA report.
    assert "Bearer" in www_auth, (
        f"Expected WWW-Authenticate: Bearer, got: {www_auth!r}. "
        "Root cause: main.py http_exception_handler fallback drops exc.headers. "
        "RFC 7235 §3.1 requires the header on 401 responses."
    )


@pytest.mark.integration
@pytest.mark.api
@pytest.mark.security
async def test_entities_no_auth_body_detail(async_client: AsyncClient) -> None:
    """Missing auth → response body is {"detail": "Not authenticated"}."""
    response = await async_client.get(f"/api/v1/graphs/{GRAPH_ID}/entities")

    assert response.status_code == 401
    body = response.json()
    assert body.get("detail") == "Not authenticated", (
        f"Unexpected detail: {body.get('detail')!r}"
    )


# ---------------------------------------------------------------------------
# Regression: invalid/expired token → 401
# ---------------------------------------------------------------------------


@pytest.mark.integration
@pytest.mark.api
@pytest.mark.security
async def test_entities_invalid_token_returns_401(async_client: AsyncClient) -> None:
    """Invalid or expired token → 401 (auth service rejects it)."""
    auth_patch = _patch_auth_reject()
    try:
        response = await async_client.get(
            f"/api/v1/graphs/{GRAPH_ID}/entities",
            headers={"Authorization": "Bearer invalid_token_here"},
        )
    finally:
        auth_patch.stop()

    assert response.status_code == 401


# ---------------------------------------------------------------------------
# Regression: valid token → 200 OK
# ---------------------------------------------------------------------------


@pytest.mark.integration
@pytest.mark.api
async def test_entities_valid_token_returns_200(async_client: AsyncClient) -> None:
    """Valid token + accessible graph → 200 OK with EntityListResponse."""
    items = [
        {
            "id": "e-1",
            "name": "Acme Corp",
            "type": "Organization",
            "confidence": 0.9,
            "community_id": "c-1",
            "degree": 5,
        }
    ]
    service_result = {"items": items, "total": 1, "page": 1, "page_size": 50}

    auth_patch = _patch_auth()
    try:
        with (
            patch(
                "app.api.v1.endpoints.entities.verify_graph_access",
                new_callable=AsyncMock,
                return_value=GRAPH_ID,
            ),
            patch("app.api.v1.endpoints.entities.GraphAnalyticsService") as MockSvc,
        ):
            MockSvc.return_value.list_entities = AsyncMock(return_value=service_result)

            response = await async_client.get(
                f"/api/v1/graphs/{GRAPH_ID}/entities",
                headers={"Authorization": "Bearer valid-token"},
            )
    finally:
        auth_patch.stop()

    assert response.status_code == 200, response.text
    data = response.json()
    assert "items" in data
    assert "total" in data
    assert data["total"] == 1
    assert data["items"][0]["name"] == "Acme Corp"


# ---------------------------------------------------------------------------
# Entities endpoint smoke test: paginated EntityListResponse schema
# ---------------------------------------------------------------------------


@pytest.mark.integration
@pytest.mark.api
async def test_entities_smoke_paginated_response_schema(
    async_client: AsyncClient,
) -> None:
    """Smoke test: EntityListResponse has items/total/page/page_size fields."""
    items = [
        {
            "id": f"e-{i}",
            "name": f"Entity {i}",
            "type": "concept",
            "confidence": 0.8,
            "community_id": None,
            "degree": i,
        }
        for i in range(3)
    ]
    service_result = {"items": items, "total": 3, "page": 1, "page_size": 50}

    auth_patch = _patch_auth()
    try:
        with (
            patch(
                "app.api.v1.endpoints.entities.verify_graph_access",
                new_callable=AsyncMock,
                return_value=GRAPH_ID,
            ),
            patch("app.api.v1.endpoints.entities.GraphAnalyticsService") as MockSvc,
        ):
            MockSvc.return_value.list_entities = AsyncMock(return_value=service_result)

            response = await async_client.get(
                f"/api/v1/graphs/{GRAPH_ID}/entities",
                headers={"Authorization": "Bearer valid-token"},
            )
    finally:
        auth_patch.stop()

    assert response.status_code == 200
    data = response.json()

    # Verify EntityListResponse schema
    assert "items" in data, "Missing 'items' field"
    assert "total" in data, "Missing 'total' field"
    assert "page" in data, "Missing 'page' field"
    assert "page_size" in data, "Missing 'page_size' field"

    assert data["page"] == 1
    assert data["page_size"] == 50
    assert data["total"] == 3
    assert len(data["items"]) == 3

    # Verify each item has required fields
    for item in data["items"]:
        assert "id" in item
        assert "name" in item
        assert "type" in item
        assert "confidence" in item
        assert "degree" in item


# ---------------------------------------------------------------------------
# Regression: other protected endpoints also return 401 for missing auth
# ---------------------------------------------------------------------------


@pytest.mark.integration
@pytest.mark.api
@pytest.mark.security
async def test_graphs_list_no_auth_returns_401(async_client: AsyncClient) -> None:
    """GET /api/v1/graphs with no auth → 401 (regression: was 403 pre-fix)."""
    response = await async_client.get("/api/v1/graphs")
    assert response.status_code == 401, (
        f"Expected 401 for missing auth on /graphs, got {response.status_code}"
    )


@pytest.mark.integration
@pytest.mark.api
@pytest.mark.security
async def test_graph_detail_no_auth_returns_401(async_client: AsyncClient) -> None:
    """GET /api/v1/graphs/{graph_id} with no auth → 401."""
    response = await async_client.get(f"/api/v1/graphs/{GRAPH_ID}")
    assert response.status_code == 401, (
        f"Expected 401 for missing auth on /graphs/{{id}}, got {response.status_code}"
    )


@pytest.mark.integration
@pytest.mark.api
@pytest.mark.security
@pytest.mark.neo4j
async def test_graph_data_no_auth_returns_401(async_client: AsyncClient) -> None:
    """GET /api/v1/graphs/{graph_id}/graph-data with no auth → 401.

    NOTE: This endpoint has driver=Depends(get_neo4j_async_driver) as a concurrent
    dependency. Without a live Neo4j, the driver raises 503 before auth fires.
    This test is marked @neo4j and requires a running Neo4j instance to be valid.
    """
    response = await async_client.get(f"/api/v1/graphs/{GRAPH_ID}/graph-data")
    assert response.status_code == 401, (
        f"Expected 401 for missing auth on graph-data, got {response.status_code}"
    )
