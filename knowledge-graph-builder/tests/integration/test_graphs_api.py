"""
Integration tests for Graphs API and Health endpoint.

Covers Suite 1 (API Integration Tests) from the QA test plan:
- GET  /api/v1/health
- POST /api/v1/graphs
- GET  /api/v1/graphs
- GET  /api/v1/graphs/{id}
- PUT  /api/v1/graphs/{id}
- POST /api/v1/graphs/{id}/ingest
- GET  /api/v1/graphs/{id}/jobs
- GET  /api/v1/graphs/{id}/jobs/{job_id}

Auth is bypassed via patching auth_service.verify_token.
Neo4j and DB operations are mocked so no live services are required.
"""

import uuid
from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

USER_A_ID = str(uuid.uuid4())
USER_B_ID = str(uuid.uuid4())
GRAPH_A_ID = str(uuid.uuid4())
GRAPH_B_ID = str(uuid.uuid4())
JOB_ID = str(uuid.uuid4())

FAKE_USER_A = {"id": USER_A_ID, "email": "user-a@example.com"}
FAKE_USER_B = {"id": USER_B_ID, "email": "user-b@example.com"}

_NOW = datetime(2025, 9, 4, 12, 0, 0, tzinfo=UTC).isoformat()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _neo4j_graph(graph_id: str, user_id: str, name: str = "Test Graph") -> dict:
    """Minimal Neo4j graph record as returned by GraphNodeService."""
    return {
        "graph_id": graph_id,
        "user_id": user_id,
        "name": name,
        "description": "A test graph",
        "status": "active",
        "created_at": _NOW,
        "updated_at": _NOW,
        "node_count": 0,
        "relationship_count": 0,
    }


def _patch_auth(user: dict):
    """Patch auth_service to return the given user."""
    p = patch("app.api.dependencies.auth_service")
    mock_auth = p.start()
    mock_auth.verify_token = AsyncMock(return_value=user)
    return p


def _auth_headers() -> dict:
    return {"Authorization": "Bearer fake-token"}


# ---------------------------------------------------------------------------
# Suite 1a — Health endpoint
# ---------------------------------------------------------------------------


class TestHealthEndpoint:

    @pytest.mark.integration
    @pytest.mark.api
    async def test_health_returns_200_when_all_healthy(self, async_client):
        """GET /health → 200 with overall status healthy."""
        with (
            patch("app.api.v1.endpoints.health.neo4j_client") as mock_neo4j,
            patch("app.api.v1.endpoints.health.check_db_health") as mock_pg,
        ):
            mock_neo4j.health_check = AsyncMock(return_value={"status": "healthy"})
            mock_pg.return_value = {"status": "healthy"}

            response = await async_client.get("/api/v1/health")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "service" in data
        assert "version" in data
        assert "timestamp" in data
        assert "dependencies" in data
        assert data["dependencies"]["neo4j"]["status"] == "healthy"
        assert data["dependencies"]["postgres"]["status"] == "healthy"

    @pytest.mark.integration
    @pytest.mark.api
    async def test_health_returns_degraded_when_neo4j_down(self, async_client):
        """GET /health → 200 but status=degraded when Neo4j is unhealthy."""
        with (
            patch("app.api.v1.endpoints.health.neo4j_client") as mock_neo4j,
            patch("app.api.v1.endpoints.health.check_db_health") as mock_pg,
        ):
            mock_neo4j.health_check = AsyncMock(return_value={"status": "unhealthy"})
            mock_pg.return_value = {"status": "healthy"}

            response = await async_client.get("/api/v1/health")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "degraded"

    @pytest.mark.integration
    @pytest.mark.api
    async def test_health_returns_degraded_when_postgres_down(self, async_client):
        """GET /health → 200 but status=degraded when Postgres is unhealthy."""
        with (
            patch("app.api.v1.endpoints.health.neo4j_client") as mock_neo4j,
            patch("app.api.v1.endpoints.health.check_db_health") as mock_pg,
        ):
            mock_neo4j.health_check = AsyncMock(return_value={"status": "healthy"})
            mock_pg.return_value = {"status": "unhealthy"}

            response = await async_client.get("/api/v1/health")

        assert response.status_code == 200
        assert response.json()["status"] == "degraded"

    @pytest.mark.integration
    @pytest.mark.api
    async def test_health_no_auth_required(self, async_client):
        """Health endpoint must be accessible without a token."""
        with (
            patch("app.api.v1.endpoints.health.neo4j_client") as mock_neo4j,
            patch("app.api.v1.endpoints.health.check_db_health") as mock_pg,
        ):
            mock_neo4j.health_check = AsyncMock(return_value={"status": "healthy"})
            mock_pg.return_value = {"status": "healthy"}

            # No Authorization header
            response = await async_client.get("/api/v1/health")

        assert response.status_code == 200


# ---------------------------------------------------------------------------
# Suite 1b — Graphs CRUD
# ---------------------------------------------------------------------------


class TestGraphsCRUD:

    # ---- CREATE ----

    @pytest.mark.integration
    @pytest.mark.api
    async def test_create_graph_returns_201(self, async_client):
        """POST /graphs → 201 with graph data persisted to Neo4j."""
        graph_record = _neo4j_graph(GRAPH_A_ID, USER_A_ID, name="My New Graph")

        auth_patch = _patch_auth(FAKE_USER_A)
        try:
            with (
                patch("app.api.v1.endpoints.graphs.neo4j_client") as mock_neo4j,
                patch("app.api.v1.endpoints.graphs.GraphNodeService") as MockService,
            ):
                mock_neo4j.sync_driver = MagicMock()
                service_instance = MockService.return_value
                service_instance.create_graph.return_value = graph_record

                response = await async_client.post(
                    "/api/v1/graphs",
                    json={"name": "My New Graph", "description": "A test graph"},
                    headers=_auth_headers(),
                )
        finally:
            auth_patch.stop()

        assert response.status_code == 201
        data = response.json()
        assert data["name"] == "My New Graph"
        assert "id" in data
        assert data["status"] == "active"

    @pytest.mark.integration
    @pytest.mark.api
    async def test_create_graph_missing_name_returns_422(self, async_client):
        """POST /graphs without required 'name' field → 422."""
        auth_patch = _patch_auth(FAKE_USER_A)
        try:
            response = await async_client.post(
                "/api/v1/graphs",
                json={"description": "No name provided"},
                headers=_auth_headers(),
            )
        finally:
            auth_patch.stop()

        assert response.status_code == 422

    @pytest.mark.integration
    @pytest.mark.api
    async def test_create_graph_requires_auth(self, async_client):
        """POST /graphs without token → 403 (missing credentials)."""
        response = await async_client.post(
            "/api/v1/graphs",
            json={"name": "No Auth Graph"},
        )
        assert response.status_code in (401, 403)

    @pytest.mark.integration
    @pytest.mark.api
    async def test_create_graph_neo4j_unavailable_returns_503(self, async_client):
        """POST /graphs when Neo4j driver is None → 503."""
        auth_patch = _patch_auth(FAKE_USER_A)
        try:
            with patch("app.api.v1.endpoints.graphs.neo4j_client") as mock_neo4j:
                mock_neo4j.sync_driver = None

                response = await async_client.post(
                    "/api/v1/graphs",
                    json={"name": "Graph Neo4j Down"},
                    headers=_auth_headers(),
                )
        finally:
            auth_patch.stop()

        assert response.status_code == 503

    # ---- LIST ----

    @pytest.mark.integration
    @pytest.mark.api
    async def test_list_graphs_returns_only_users_graphs(self, async_client):
        """GET /graphs → returns only graphs owned by authenticated user."""
        user_graphs = [
            _neo4j_graph(GRAPH_A_ID, USER_A_ID, name="Graph A1"),
            _neo4j_graph(str(uuid.uuid4()), USER_A_ID, name="Graph A2"),
        ]

        auth_patch = _patch_auth(FAKE_USER_A)
        try:
            with (
                patch("app.api.v1.endpoints.graphs.neo4j_client") as mock_neo4j,
                patch("app.api.v1.endpoints.graphs.GraphNodeService") as MockService,
            ):
                mock_neo4j.sync_driver = MagicMock()
                MockService.return_value.list_user_graphs.return_value = user_graphs

                response = await async_client.get(
                    "/api/v1/graphs", headers=_auth_headers()
                )
        finally:
            auth_patch.stop()

        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)
        assert len(data) == 2
        names = {g["name"] for g in data}
        assert names == {"Graph A1", "Graph A2"}

    @pytest.mark.integration
    @pytest.mark.api
    async def test_list_graphs_empty_when_no_graphs(self, async_client):
        """GET /graphs → empty list when user has no graphs."""
        auth_patch = _patch_auth(FAKE_USER_A)
        try:
            with (
                patch("app.api.v1.endpoints.graphs.neo4j_client") as mock_neo4j,
                patch("app.api.v1.endpoints.graphs.GraphNodeService") as MockService,
            ):
                mock_neo4j.sync_driver = MagicMock()
                MockService.return_value.list_user_graphs.return_value = []

                response = await async_client.get(
                    "/api/v1/graphs", headers=_auth_headers()
                )
        finally:
            auth_patch.stop()

        assert response.status_code == 200
        assert response.json() == []

    # ---- GET ----

    @pytest.mark.integration
    @pytest.mark.api
    async def test_get_graph_returns_correct_graph(self, async_client):
        """GET /graphs/{id} → correct graph for authenticated owner."""
        graph_record = _neo4j_graph(GRAPH_A_ID, USER_A_ID, name="Graph A")

        auth_patch = _patch_auth(FAKE_USER_A)
        try:
            with (
                patch("app.api.v1.endpoints.graphs.neo4j_client") as mock_neo4j,
                patch("app.api.v1.endpoints.graphs.GraphNodeService") as MockService,
            ):
                mock_neo4j.sync_driver = MagicMock()
                MockService.return_value.get_graph.return_value = graph_record

                response = await async_client.get(
                    f"/api/v1/graphs/{GRAPH_A_ID}", headers=_auth_headers()
                )
        finally:
            auth_patch.stop()

        assert response.status_code == 200
        data = response.json()
        assert data["name"] == "Graph A"
        assert data["id"] == GRAPH_A_ID

    @pytest.mark.integration
    @pytest.mark.api
    async def test_get_graph_404_for_unknown_id(self, async_client):
        """GET /graphs/{unknown_id} → 404."""
        auth_patch = _patch_auth(FAKE_USER_A)
        try:
            with (
                patch("app.api.v1.endpoints.graphs.neo4j_client") as mock_neo4j,
                patch("app.api.v1.endpoints.graphs.GraphNodeService") as MockService,
            ):
                mock_neo4j.sync_driver = MagicMock()
                MockService.return_value.get_graph.return_value = None  # not found

                response = await async_client.get(
                    f"/api/v1/graphs/{uuid.uuid4()}", headers=_auth_headers()
                )
        finally:
            auth_patch.stop()

        assert response.status_code == 404

    @pytest.mark.integration
    @pytest.mark.api
    async def test_get_graph_403_when_not_owner(self, async_client):
        """GET /graphs/{id} belonging to User B when authenticated as User A → 403."""
        graph_record = _neo4j_graph(GRAPH_B_ID, USER_B_ID, name="Graph B")

        auth_patch = _patch_auth(FAKE_USER_A)  # User A is calling
        try:
            with (
                patch("app.api.v1.endpoints.graphs.neo4j_client") as mock_neo4j,
                patch("app.api.v1.endpoints.graphs.GraphNodeService") as MockService,
            ):
                mock_neo4j.sync_driver = MagicMock()
                MockService.return_value.get_graph.return_value = (
                    graph_record  # belongs to User B
                )

                response = await async_client.get(
                    f"/api/v1/graphs/{GRAPH_B_ID}", headers=_auth_headers()
                )
        finally:
            auth_patch.stop()

        assert response.status_code == 403

    # ---- UPDATE ----

    @pytest.mark.integration
    @pytest.mark.api
    async def test_update_graph_returns_updated_data(self, async_client):
        """PUT /graphs/{id} → 200 with updated fields."""
        existing = _neo4j_graph(GRAPH_A_ID, USER_A_ID, name="Old Name")
        updated = {**existing, "name": "New Name", "description": "Updated desc"}

        auth_patch = _patch_auth(FAKE_USER_A)
        try:
            with (
                patch("app.api.v1.endpoints.graphs.neo4j_client") as mock_neo4j,
                patch("app.api.v1.endpoints.graphs.GraphNodeService") as MockService,
            ):
                mock_neo4j.sync_driver = MagicMock()
                svc = MockService.return_value
                svc.get_graph.return_value = existing
                svc.update_graph.return_value = updated

                response = await async_client.put(
                    f"/api/v1/graphs/{GRAPH_A_ID}",
                    json={"name": "New Name", "description": "Updated desc"},
                    headers=_auth_headers(),
                )
        finally:
            auth_patch.stop()

        assert response.status_code == 200
        data = response.json()
        assert data["name"] == "New Name"

    @pytest.mark.integration
    @pytest.mark.api
    async def test_update_graph_403_when_not_owner(self, async_client):
        """PUT /graphs/{id} on another user's graph → 403."""
        graph_record = _neo4j_graph(GRAPH_B_ID, USER_B_ID, name="Graph B")

        auth_patch = _patch_auth(FAKE_USER_A)
        try:
            with (
                patch("app.api.v1.endpoints.graphs.neo4j_client") as mock_neo4j,
                patch("app.api.v1.endpoints.graphs.GraphNodeService") as MockService,
            ):
                mock_neo4j.sync_driver = MagicMock()
                MockService.return_value.get_graph.return_value = graph_record

                response = await async_client.put(
                    f"/api/v1/graphs/{GRAPH_B_ID}",
                    json={"name": "Hijacked Name"},
                    headers=_auth_headers(),
                )
        finally:
            auth_patch.stop()

        assert response.status_code == 403

    @pytest.mark.integration
    @pytest.mark.api
    async def test_update_graph_404_when_not_found(self, async_client):
        """PUT /graphs/{unknown_id} → 404."""
        auth_patch = _patch_auth(FAKE_USER_A)
        try:
            with (
                patch("app.api.v1.endpoints.graphs.neo4j_client") as mock_neo4j,
                patch("app.api.v1.endpoints.graphs.GraphNodeService") as MockService,
            ):
                mock_neo4j.sync_driver = MagicMock()
                MockService.return_value.get_graph.return_value = None

                response = await async_client.put(
                    f"/api/v1/graphs/{uuid.uuid4()}",
                    json={"name": "Ghost Graph"},
                    headers=_auth_headers(),
                )
        finally:
            auth_patch.stop()

        assert response.status_code == 404


# ---------------------------------------------------------------------------
# Suite 1b' — DELETE /graphs/{id} (TASK-050, soft-delete)
# ---------------------------------------------------------------------------


class TestDeleteGraphEndpoint:
    """DELETE /api/v1/graphs/{graph_id} — admin-level ReBAC, soft-delete only."""

    @pytest.mark.integration
    @pytest.mark.api
    async def test_delete_graph_returns_204_when_admin(self, async_client):
        """DELETE /graphs/{id} → 204 when caller has admin access."""
        graph_record = _neo4j_graph(GRAPH_A_ID, USER_A_ID)
        auth_patch = _patch_auth(FAKE_USER_A)
        try:
            with (
                patch("app.api.v1.endpoints.graphs.verify_graph_access",
                      new_callable=AsyncMock) as mock_vga,
                patch("app.api.v1.endpoints.graphs.neo4j_client") as mock_neo4j,
                patch("app.api.v1.endpoints.graphs.GraphNodeService") as MockService,
            ):
                mock_vga.return_value = GRAPH_A_ID
                mock_neo4j.sync_driver = MagicMock()
                svc = MockService.return_value
                svc.get_graph.return_value = graph_record
                svc.soft_delete_graph.return_value = True

                response = await async_client.delete(
                    f"/api/v1/graphs/{GRAPH_A_ID}", headers=_auth_headers()
                )
        finally:
            auth_patch.stop()

        assert response.status_code == 204
        assert response.content == b""
        # Confirm soft-delete (not hard delete) was called
        svc.soft_delete_graph.assert_called_once_with(GRAPH_A_ID)
        svc.delete_graph.assert_not_called()
        # Confirm admin-level access was required
        called_args = mock_vga.call_args
        assert called_args.args[1] == "admin" or called_args.kwargs.get("required_level") == "admin"

    @pytest.mark.integration
    @pytest.mark.api
    async def test_delete_graph_returns_404_when_unknown(self, async_client):
        """DELETE /graphs/{unknown_id} → 404."""
        auth_patch = _patch_auth(FAKE_USER_A)
        try:
            with (
                patch("app.api.v1.endpoints.graphs.verify_graph_access",
                      new_callable=AsyncMock) as mock_vga,
                patch("app.api.v1.endpoints.graphs.neo4j_client") as mock_neo4j,
                patch("app.api.v1.endpoints.graphs.GraphNodeService") as MockService,
            ):
                mock_vga.return_value = GRAPH_A_ID
                mock_neo4j.sync_driver = MagicMock()
                MockService.return_value.get_graph.return_value = None

                response = await async_client.delete(
                    f"/api/v1/graphs/{uuid.uuid4()}", headers=_auth_headers()
                )
        finally:
            auth_patch.stop()

        assert response.status_code == 404

    @pytest.mark.integration
    @pytest.mark.api
    async def test_delete_graph_returns_403_when_not_admin(self, async_client):
        """DELETE /graphs/{id} → 403 when caller lacks admin permission."""
        from fastapi import HTTPException, status as fastapi_status

        auth_patch = _patch_auth(FAKE_USER_A)
        try:
            with patch(
                "app.api.v1.endpoints.graphs.verify_graph_access",
                new_callable=AsyncMock,
            ) as mock_vga:
                mock_vga.side_effect = HTTPException(
                    status_code=fastapi_status.HTTP_403_FORBIDDEN, detail="Access denied"
                )

                response = await async_client.delete(
                    f"/api/v1/graphs/{GRAPH_A_ID}", headers=_auth_headers()
                )
        finally:
            auth_patch.stop()

        assert response.status_code == 403

    @pytest.mark.integration
    @pytest.mark.api
    async def test_delete_graph_returns_503_when_neo4j_unavailable(self, async_client):
        """DELETE /graphs/{id} → 503 when Neo4j sync driver is None."""
        auth_patch = _patch_auth(FAKE_USER_A)
        try:
            with (
                patch("app.api.v1.endpoints.graphs.verify_graph_access",
                      new_callable=AsyncMock) as mock_vga,
                patch("app.api.v1.endpoints.graphs.neo4j_client") as mock_neo4j,
            ):
                mock_vga.return_value = GRAPH_A_ID
                mock_neo4j.sync_driver = None

                response = await async_client.delete(
                    f"/api/v1/graphs/{GRAPH_A_ID}", headers=_auth_headers()
                )
        finally:
            auth_patch.stop()

        assert response.status_code == 503

    @pytest.mark.integration
    @pytest.mark.api
    async def test_delete_graph_requires_auth(self, async_client):
        """DELETE /graphs/{id} without token → 401/403."""
        response = await async_client.delete(f"/api/v1/graphs/{GRAPH_A_ID}")
        assert response.status_code in (401, 403)


# ---------------------------------------------------------------------------
# Suite 1c — Ingestion
# ---------------------------------------------------------------------------


class TestIngestEndpoint:

    @pytest.mark.integration
    @pytest.mark.api
    async def test_ingest_creates_job_and_returns_pending(self, async_client):
        """POST /graphs/{id}/ingest → 200 with pending job."""
        graph_record = _neo4j_graph(GRAPH_A_ID, USER_A_ID)

        mock_job = MagicMock()
        mock_job.id = uuid.UUID(JOB_ID)
        mock_job.graph_id = uuid.UUID(GRAPH_A_ID)
        mock_job.status = "pending"
        mock_job.progress = 0
        mock_job.source_type = "text"
        mock_job.created_at = datetime.now(UTC)

        auth_patch = _patch_auth(FAKE_USER_A)
        try:
            with (
                patch("app.api.v1.endpoints.graphs.neo4j_client") as mock_neo4j,
                patch("app.api.v1.endpoints.graphs.GraphNodeService") as MockService,
                patch("app.api.v1.endpoints.graphs.background_job_service") as mock_bg,
                patch("app.api.v1.endpoints.graphs.get_database") as mock_db_dep,
            ):

                mock_neo4j.sync_driver = MagicMock()
                MockService.return_value.get_graph.return_value = graph_record
                mock_bg.start_ingestion_job.return_value = {
                    "status": "started",
                    "message": "ok",
                }

                # Mock the database session
                mock_session = AsyncMock()
                mock_session.add = MagicMock()
                mock_session.commit = AsyncMock()
                mock_session.refresh = AsyncMock(
                    side_effect=lambda job: setattr(job, "id", uuid.UUID(JOB_ID))
                    or None
                )

                async def _db_gen():
                    yield mock_session

                mock_db_dep.return_value = _db_gen()

                response = await async_client.post(
                    f"/api/v1/graphs/{GRAPH_A_ID}/ingest",
                    json={
                        "content": "TechNova Corp was founded in 2015 by Alice Smith. It is headquartered in San Francisco."
                    },
                    headers=_auth_headers(),
                )
        finally:
            auth_patch.stop()

        # Should succeed and return a job (200 or 201)
        assert response.status_code in (200, 201)
        data = response.json()
        assert "id" in data
        assert data["status"] == "pending"

    @pytest.mark.integration
    @pytest.mark.api
    async def test_ingest_404_for_nonexistent_graph(self, async_client):
        """POST /graphs/{unknown_id}/ingest → 404."""
        auth_patch = _patch_auth(FAKE_USER_A)
        try:
            with (
                patch("app.api.v1.endpoints.graphs.neo4j_client") as mock_neo4j,
                patch("app.api.v1.endpoints.graphs.GraphNodeService") as MockService,
            ):
                mock_neo4j.sync_driver = MagicMock()
                MockService.return_value.get_graph.return_value = None

                response = await async_client.post(
                    f"/api/v1/graphs/{uuid.uuid4()}/ingest",
                    json={"content": "Some content"},
                    headers=_auth_headers(),
                )
        finally:
            auth_patch.stop()

        assert response.status_code == 404

    @pytest.mark.integration
    @pytest.mark.api
    async def test_ingest_403_for_other_users_graph(self, async_client):
        """POST /graphs/{id}/ingest on another user's graph → 403."""
        graph_record = _neo4j_graph(GRAPH_B_ID, USER_B_ID)

        auth_patch = _patch_auth(FAKE_USER_A)
        try:
            with (
                patch("app.api.v1.endpoints.graphs.neo4j_client") as mock_neo4j,
                patch("app.api.v1.endpoints.graphs.GraphNodeService") as MockService,
            ):
                mock_neo4j.sync_driver = MagicMock()
                MockService.return_value.get_graph.return_value = graph_record

                response = await async_client.post(
                    f"/api/v1/graphs/{GRAPH_B_ID}/ingest",
                    json={"content": "Some content"},
                    headers=_auth_headers(),
                )
        finally:
            auth_patch.stop()

        assert response.status_code == 403

    @pytest.mark.integration
    @pytest.mark.api
    async def test_ingest_422_for_short_content(self, async_client):
        """POST /graphs/{id}/ingest with content < 10 chars → 422."""
        auth_patch = _patch_auth(FAKE_USER_A)
        try:
            response = await async_client.post(
                f"/api/v1/graphs/{GRAPH_A_ID}/ingest",
                json={"content": "short"},
                headers=_auth_headers(),
            )
        finally:
            auth_patch.stop()

        assert response.status_code == 422


# ---------------------------------------------------------------------------
# Suite 1d — Graph Response Fields
# ---------------------------------------------------------------------------


class TestGraphResponseFields:

    @pytest.mark.integration
    @pytest.mark.api
    async def test_graph_response_contains_all_required_fields(self, async_client):
        """GraphResponse must contain all required fields per spec."""
        graph_record = _neo4j_graph(GRAPH_A_ID, USER_A_ID, name="Field Test Graph")

        auth_patch = _patch_auth(FAKE_USER_A)
        try:
            with (
                patch("app.api.v1.endpoints.graphs.neo4j_client") as mock_neo4j,
                patch("app.api.v1.endpoints.graphs.GraphNodeService") as MockService,
            ):
                mock_neo4j.sync_driver = MagicMock()
                MockService.return_value.get_graph.return_value = graph_record

                response = await async_client.get(
                    f"/api/v1/graphs/{GRAPH_A_ID}", headers=_auth_headers()
                )
        finally:
            auth_patch.stop()

        assert response.status_code == 200
        data = response.json()
        required_fields = [
            "id",
            "name",
            "description",
            "user_id",
            "created_at",
            "updated_at",
            "node_count",
            "relationship_count",
            "status",
        ]
        for field in required_fields:
            assert field in data, f"Missing required field: {field}"
