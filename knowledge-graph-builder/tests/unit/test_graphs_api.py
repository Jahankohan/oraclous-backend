"""
Unit tests for the DELETE /api/v1/graphs/{graph_id} endpoint.

All external dependencies (Neo4j, auth, ReBAC) are mocked so these tests
run without any live services.

A minimal FastAPI test app is assembled from the graphs router only — this
avoids importing the full app.main which would pull in unrelated modules.
"""
from __future__ import annotations

import uuid
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import FastAPI, HTTPException, status
from httpx import ASGITransport, AsyncClient

USER_ID = str(uuid.uuid4())
GRAPH_ID = str(uuid.uuid4())


# ---------------------------------------------------------------------------
# Minimal test app — graphs router only
# ---------------------------------------------------------------------------


def _make_test_app() -> FastAPI:
    """Return a minimal FastAPI app that mounts only the graphs router."""
    from app.api.v1.endpoints.graphs import router as graphs_router

    _app = FastAPI()
    _app.include_router(graphs_router, prefix="/api/v1")
    return _app


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _auth_headers() -> dict:
    return {"Authorization": "Bearer fake-token"}


def _patch_auth(user_id: str = USER_ID):
    """Patch authentication so Bearer tokens resolve to user_id."""
    p = patch("app.api.dependencies.auth_service")
    mock_auth = p.start()
    mock_auth.verify_token = AsyncMock(
        return_value={"id": user_id, "email": "test@example.com"}
    )
    return p


def _patch_rebac_ok():
    """Patch verify_graph_access to pass silently."""
    return patch(
        "app.api.v1.endpoints.graphs.verify_graph_access",
        new=AsyncMock(return_value=str(GRAPH_ID)),
    )


def _patch_rebac_denied():
    """Patch verify_graph_access to raise 403."""
    return patch(
        "app.api.v1.endpoints.graphs.verify_graph_access",
        new=AsyncMock(
            side_effect=HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Forbidden",
            )
        ),
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestDeleteGraphEndpoint:
    """Unit tests for DELETE /api/v1/graphs/{graph_id}."""

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_successful_delete_returns_204(self):
        """Owner deletes an existing graph → 204 No Content."""
        app = _make_test_app()
        auth_patch = _patch_auth()
        try:
            with (
                _patch_rebac_ok(),
                patch("app.api.v1.endpoints.graphs.neo4j_client") as mock_neo4j,
                patch(
                    "app.api.v1.endpoints.graphs.GraphNodeService"
                ) as MockService,
            ):
                mock_neo4j.sync_driver = MagicMock()
                MockService.return_value.delete_graph.return_value = True

                async with AsyncClient(
                    transport=ASGITransport(app=app), base_url="http://test"
                ) as client:
                    response = await client.delete(
                        f"/api/v1/graphs/{GRAPH_ID}",
                        headers=_auth_headers(),
                    )
        finally:
            auth_patch.stop()

        assert response.status_code == 204
        MockService.return_value.delete_graph.assert_called_once_with(
            graph_id=str(GRAPH_ID), user_id=USER_ID
        )

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_rebac_denied_returns_403(self):
        """Caller without admin access → 403 (graph_id enumeration-safe)."""
        app = _make_test_app()
        auth_patch = _patch_auth()
        try:
            with _patch_rebac_denied():
                async with AsyncClient(
                    transport=ASGITransport(app=app), base_url="http://test"
                ) as client:
                    response = await client.delete(
                        f"/api/v1/graphs/{GRAPH_ID}",
                        headers=_auth_headers(),
                    )
        finally:
            auth_patch.stop()

        assert response.status_code == 403

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_graph_not_found_returns_404(self):
        """Service returns False (graph not in Neo4j) → 404."""
        app = _make_test_app()
        auth_patch = _patch_auth()
        try:
            with (
                _patch_rebac_ok(),
                patch("app.api.v1.endpoints.graphs.neo4j_client") as mock_neo4j,
                patch(
                    "app.api.v1.endpoints.graphs.GraphNodeService"
                ) as MockService,
            ):
                mock_neo4j.sync_driver = MagicMock()
                MockService.return_value.delete_graph.return_value = False

                async with AsyncClient(
                    transport=ASGITransport(app=app), base_url="http://test"
                ) as client:
                    response = await client.delete(
                        f"/api/v1/graphs/{GRAPH_ID}",
                        headers=_auth_headers(),
                    )
        finally:
            auth_patch.stop()

        assert response.status_code == 404

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_neo4j_unavailable_returns_503(self):
        """sync_driver is None (Neo4j not connected) → 503."""
        app = _make_test_app()
        auth_patch = _patch_auth()
        try:
            with (
                _patch_rebac_ok(),
                patch("app.api.v1.endpoints.graphs.neo4j_client") as mock_neo4j,
            ):
                mock_neo4j.sync_driver = None  # simulate unavailable Neo4j

                async with AsyncClient(
                    transport=ASGITransport(app=app), base_url="http://test"
                ) as client:
                    response = await client.delete(
                        f"/api/v1/graphs/{GRAPH_ID}",
                        headers=_auth_headers(),
                    )
        finally:
            auth_patch.stop()

        assert response.status_code == 503
