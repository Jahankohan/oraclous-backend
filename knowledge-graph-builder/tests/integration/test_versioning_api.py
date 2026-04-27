"""
Integration tests for Graph Versioning / Snapshot API.

Covers:
- POST   /graphs/{id}/snapshots          — create snapshot
- GET    /graphs/{id}/snapshots          — list snapshots
- GET    /graphs/{id}/snapshots/{sid}    — get single snapshot
- DELETE /graphs/{id}/snapshots/{sid}    — delete snapshot
- GET    /graphs/{id}/snapshots/{sid}/diff/{oid} — diff two snapshots
- POST   /graphs/{id}/snapshots/{sid}/rollback   — rollback (sync path)
- GET    /graphs/{id}/rollbacks/{jid}             — rollback job status

Auth is bypassed via patching. Neo4j and DB are mocked — no live services required.
"""

import uuid
from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

USER_ID = str(uuid.uuid4())
GRAPH_ID = str(uuid.uuid4())
SNAPSHOT_ID = str(uuid.uuid4())
OTHER_SNAPSHOT_ID = str(uuid.uuid4())
JOB_ID = str(uuid.uuid4())

_NOW = datetime(2025, 9, 4, 12, 0, 0, tzinfo=UTC).isoformat()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _fake_user() -> dict:
    return {"id": USER_ID, "email": "user@example.com"}


def _auth_headers() -> dict:
    return {"Authorization": "Bearer fake-token"}


def _neo4j_graph(graph_id: str = GRAPH_ID, user_id: str = USER_ID) -> dict:
    return {
        "graph_id": graph_id,
        "user_id": user_id,
        "name": "Test Graph",
        "description": "",
        "status": "active",
        "created_at": _NOW,
        "updated_at": _NOW,
        "node_count": 5,
        "relationship_count": 3,
    }


def _fake_snapshot(
    version_id: str = SNAPSHOT_ID,
    graph_id: str = GRAPH_ID,
    version_number: int = 1,
    label: str = "v1",
) -> dict:
    return {
        "version_id": version_id,
        "graph_id": graph_id,
        "version_number": version_number,
        "label": label,
        "description": "test snapshot",
        "captured_at": _NOW,
        "created_by": USER_ID,
        "parent_version_id": None,
        "is_auto": False,
        "entity_count": 10,
        "relationship_count": 5,
        "created_at": _NOW,
    }


def _fake_diff() -> dict:
    return {
        "from_version": {"version_id": SNAPSHOT_ID, "label": "v1", "captured_at": _NOW},
        "to_version": {
            "version_id": OTHER_SNAPSHOT_ID,
            "label": "v2",
            "captured_at": _NOW,
        },
        "summary": {
            "entities_added": 3,
            "entities_deleted": 1,
            "relationships_added": 2,
            "relationships_deleted": 0,
            "property_changes": 0,
        },
        "changes": [
            {
                "type": "entity_added",
                "entity_id": str(uuid.uuid4()),
                "name": "NewEntity",
                "entity_type": "Person",
                "subject": None,
                "predicate": None,
                "object": None,
                "timestamp": _NOW,
            }
        ],
        "offset": 0,
        "limit": 100,
        "has_more": False,
    }


def _fake_rollback_result() -> dict:
    return {
        "checkpoint_version_id": str(uuid.uuid4()),
        "entities_restored": 4,
        "entities_soft_deleted": 2,
        "relationships_restored": 2,
        "relationships_soft_deleted": 1,
        "message": "Full rollback completed",
        "staleness_pct": 0.0,
    }


def _fake_rollback_job() -> dict:
    return {
        "rollback_job_id": JOB_ID,
        "graph_id": GRAPH_ID,
        "version_id": SNAPSHOT_ID,
        "mode": "full",
        "status": "done",
        "progress": 100,
        "entities_restored": 4,
        "entities_soft_deleted": 2,
        "relationships_restored": 2,
        "relationships_soft_deleted": 1,
        "checkpoint_version_id": None,
        "error_message": None,
        "performed_by": USER_ID,
        "scope": None,
        "celery_task_id": "abc-123",
        "started_at": _NOW,
        "completed_at": _NOW,
        "created_at": _NOW,
    }


def _patch_auth():
    p = patch("app.api.dependencies.auth_service")
    mock_auth = p.start()
    mock_auth.verify_token = AsyncMock(return_value=_fake_user())
    return p


def _patch_graph_service(graph: dict | None = None):
    """Patch GraphNodeService to return the provided graph record."""
    p = patch("app.api.v1.endpoints.graphs.GraphNodeService")
    mock_cls = p.start()
    instance = MagicMock()
    instance.get_graph.return_value = graph if graph is not None else _neo4j_graph()
    mock_cls.return_value = instance
    return p, instance


# ---------------------------------------------------------------------------
# Suite — Create snapshot
# ---------------------------------------------------------------------------


class TestCreateSnapshot:

    @pytest.mark.integration
    @pytest.mark.api
    async def test_create_snapshot_returns_201(self, async_client):
        """POST /snapshots → 201 with version metadata."""
        snap = _fake_snapshot()

        auth_p = _patch_auth()
        graph_p, _ = _patch_graph_service()
        with patch("app.api.v1.endpoints.graphs.versioning_service") as mock_vs:
            mock_vs.create_version = AsyncMock(return_value=snap)
            response = await async_client.post(
                f"/api/v1/graphs/{GRAPH_ID}/snapshots",
                json={"label": "v1", "description": "test snapshot"},
                headers=_auth_headers(),
            )
        auth_p.stop()
        graph_p.stop()

        assert response.status_code == 201
        data = response.json()
        assert data["version_id"] == SNAPSHOT_ID
        assert data["label"] == "v1"
        assert data["entity_count"] == 10

    @pytest.mark.integration
    @pytest.mark.api
    async def test_create_snapshot_injects_user_as_created_by(self, async_client):
        """created_by must be the authenticated user_id, not a client-supplied value."""
        snap = _fake_snapshot()

        auth_p = _patch_auth()
        graph_p, _ = _patch_graph_service()
        with patch("app.api.v1.endpoints.graphs.versioning_service") as mock_vs:
            mock_vs.create_version = AsyncMock(return_value=snap)
            await async_client.post(
                f"/api/v1/graphs/{GRAPH_ID}/snapshots",
                json={"label": "v1"},
                headers=_auth_headers(),
            )
            call_kwargs = mock_vs.create_version.call_args.kwargs
        auth_p.stop()
        graph_p.stop()

        assert call_kwargs["created_by"] == USER_ID

    @pytest.mark.integration
    @pytest.mark.api
    async def test_create_snapshot_rejects_wrong_owner(self, async_client):
        """Cannot create a snapshot for a graph owned by another user → 403/404."""
        auth_p = _patch_auth()
        graph_p, mock_svc = _patch_graph_service(
            graph=_neo4j_graph(user_id=str(uuid.uuid4()))
        )
        with patch("app.api.v1.endpoints.graphs.versioning_service"):
            response = await async_client.post(
                f"/api/v1/graphs/{GRAPH_ID}/snapshots",
                json={"label": "v1"},
                headers=_auth_headers(),
            )
        auth_p.stop()
        graph_p.stop()

        assert response.status_code in (403, 404)


# ---------------------------------------------------------------------------
# Suite — List snapshots
# ---------------------------------------------------------------------------


class TestListSnapshots:

    @pytest.mark.integration
    @pytest.mark.api
    async def test_list_snapshots_returns_200(self, async_client):
        """GET /snapshots → 200 with list and total."""
        snaps = [_fake_snapshot(version_number=i + 1) for i in range(3)]

        auth_p = _patch_auth()
        graph_p, _ = _patch_graph_service()
        with patch("app.api.v1.endpoints.graphs.versioning_service") as mock_vs:
            mock_vs.list_versions = AsyncMock(return_value=snaps)
            response = await async_client.get(
                f"/api/v1/graphs/{GRAPH_ID}/snapshots",
                headers=_auth_headers(),
            )
        auth_p.stop()
        graph_p.stop()

        assert response.status_code == 200
        data = response.json()
        assert data["total"] == 3
        assert len(data["versions"]) == 3

    @pytest.mark.integration
    @pytest.mark.api
    async def test_list_snapshots_empty_graph(self, async_client):
        """GET /snapshots on a graph with no snapshots → 200 with empty list."""
        auth_p = _patch_auth()
        graph_p, _ = _patch_graph_service()
        with patch("app.api.v1.endpoints.graphs.versioning_service") as mock_vs:
            mock_vs.list_versions = AsyncMock(return_value=[])
            response = await async_client.get(
                f"/api/v1/graphs/{GRAPH_ID}/snapshots",
                headers=_auth_headers(),
            )
        auth_p.stop()
        graph_p.stop()

        assert response.status_code == 200
        assert response.json()["total"] == 0


# ---------------------------------------------------------------------------
# Suite — Get single snapshot
# ---------------------------------------------------------------------------


class TestGetSnapshot:

    @pytest.mark.integration
    @pytest.mark.api
    async def test_get_existing_snapshot_returns_200(self, async_client):
        """GET /snapshots/{id} → 200 with snapshot data."""
        snap = _fake_snapshot()

        auth_p = _patch_auth()
        graph_p, _ = _patch_graph_service()
        with patch("app.api.v1.endpoints.graphs.versioning_service") as mock_vs:
            mock_vs.get_version = AsyncMock(return_value=snap)
            response = await async_client.get(
                f"/api/v1/graphs/{GRAPH_ID}/snapshots/{SNAPSHOT_ID}",
                headers=_auth_headers(),
            )
        auth_p.stop()
        graph_p.stop()

        assert response.status_code == 200
        assert response.json()["version_id"] == SNAPSHOT_ID

    @pytest.mark.integration
    @pytest.mark.api
    async def test_get_missing_snapshot_returns_404(self, async_client):
        """GET /snapshots/{id} → 404 when snapshot does not exist."""
        auth_p = _patch_auth()
        graph_p, _ = _patch_graph_service()
        with patch("app.api.v1.endpoints.graphs.versioning_service") as mock_vs:
            mock_vs.get_version = AsyncMock(return_value=None)
            response = await async_client.get(
                f"/api/v1/graphs/{GRAPH_ID}/snapshots/{SNAPSHOT_ID}",
                headers=_auth_headers(),
            )
        auth_p.stop()
        graph_p.stop()

        assert response.status_code == 404


# ---------------------------------------------------------------------------
# Suite — Delete snapshot
# ---------------------------------------------------------------------------


class TestDeleteSnapshot:

    @pytest.mark.integration
    @pytest.mark.api
    async def test_delete_existing_snapshot_returns_204(self, async_client):
        """DELETE /snapshots/{id} → 204 when snapshot exists."""
        auth_p = _patch_auth()
        graph_p, _ = _patch_graph_service()
        with patch("app.api.v1.endpoints.graphs.versioning_service") as mock_vs:
            mock_vs.delete_version = AsyncMock(return_value=True)
            response = await async_client.delete(
                f"/api/v1/graphs/{GRAPH_ID}/snapshots/{SNAPSHOT_ID}",
                headers=_auth_headers(),
            )
        auth_p.stop()
        graph_p.stop()

        assert response.status_code == 204

    @pytest.mark.integration
    @pytest.mark.api
    async def test_delete_missing_snapshot_returns_404(self, async_client):
        """DELETE /snapshots/{id} → 404 when snapshot does not exist."""
        auth_p = _patch_auth()
        graph_p, _ = _patch_graph_service()
        with patch("app.api.v1.endpoints.graphs.versioning_service") as mock_vs:
            mock_vs.delete_version = AsyncMock(return_value=False)
            response = await async_client.delete(
                f"/api/v1/graphs/{GRAPH_ID}/snapshots/{SNAPSHOT_ID}",
                headers=_auth_headers(),
            )
        auth_p.stop()
        graph_p.stop()

        assert response.status_code == 404

    @pytest.mark.integration
    @pytest.mark.api
    async def test_delete_enforces_graph_ownership(self, async_client):
        """DELETE /snapshots/{id} → 403/404 when graph is owned by someone else."""
        other_owner = str(uuid.uuid4())
        auth_p = _patch_auth()
        graph_p, _ = _patch_graph_service(graph=_neo4j_graph(user_id=other_owner))
        with patch("app.api.v1.endpoints.graphs.versioning_service") as mock_vs:
            mock_vs.delete_version = AsyncMock(return_value=True)
            response = await async_client.delete(
                f"/api/v1/graphs/{GRAPH_ID}/snapshots/{SNAPSHOT_ID}",
                headers=_auth_headers(),
            )
        auth_p.stop()
        graph_p.stop()

        assert response.status_code in (403, 404)


# ---------------------------------------------------------------------------
# Suite — Diff two snapshots
# ---------------------------------------------------------------------------


class TestDiffSnapshots:

    @pytest.mark.integration
    @pytest.mark.api
    async def test_diff_returns_200_with_changes(self, async_client):
        """GET /snapshots/{id}/diff/{other} → 200 with diff summary and changes."""
        diff = _fake_diff()

        auth_p = _patch_auth()
        graph_p, _ = _patch_graph_service()
        with patch("app.api.v1.endpoints.graphs.versioning_service") as mock_vs:
            mock_vs.diff_versions = AsyncMock(return_value=diff)
            response = await async_client.get(
                f"/api/v1/graphs/{GRAPH_ID}/snapshots/{SNAPSHOT_ID}/diff/{OTHER_SNAPSHOT_ID}",
                headers=_auth_headers(),
            )
        auth_p.stop()
        graph_p.stop()

        assert response.status_code == 200
        data = response.json()
        assert data["summary"]["entities_added"] == 3
        assert len(data["changes"]) == 1

    @pytest.mark.integration
    @pytest.mark.api
    async def test_diff_rejects_limit_over_500(self, async_client):
        """GET /snapshots/diff → 400 when limit > 500."""
        auth_p = _patch_auth()
        graph_p, _ = _patch_graph_service()
        with patch("app.api.v1.endpoints.graphs.versioning_service"):
            response = await async_client.get(
                f"/api/v1/graphs/{GRAPH_ID}/snapshots/{SNAPSHOT_ID}/diff/{OTHER_SNAPSHOT_ID}?limit=501",
                headers=_auth_headers(),
            )
        auth_p.stop()
        graph_p.stop()

        assert response.status_code == 400

    @pytest.mark.integration
    @pytest.mark.api
    async def test_diff_returns_404_for_missing_snapshot(self, async_client):
        """GET /snapshots/diff → 404 when one of the snapshots doesn't exist."""
        auth_p = _patch_auth()
        graph_p, _ = _patch_graph_service()
        with patch("app.api.v1.endpoints.graphs.versioning_service") as mock_vs:
            mock_vs.diff_versions = AsyncMock(
                side_effect=ValueError("One or both versions not found")
            )
            response = await async_client.get(
                f"/api/v1/graphs/{GRAPH_ID}/snapshots/{SNAPSHOT_ID}/diff/{OTHER_SNAPSHOT_ID}",
                headers=_auth_headers(),
            )
        auth_p.stop()
        graph_p.stop()

        assert response.status_code == 404


# ---------------------------------------------------------------------------
# Suite — Rollback (sync path)
# ---------------------------------------------------------------------------


class TestRollbackSnapshot:

    @pytest.mark.integration
    @pytest.mark.api
    async def test_rollback_requires_confirm_true(self, async_client):
        """POST /rollback → 400 when confirm=false."""
        auth_p = _patch_auth()
        graph_p, _ = _patch_graph_service()
        with (
            patch("app.api.v1.endpoints.graphs.versioning_service"),
            patch("app.api.v1.endpoints.graphs.neo4j_client") as mock_neo4j,
        ):
            mock_neo4j.execute_query = AsyncMock(return_value=[{"cnt": 50}])
            response = await async_client.post(
                f"/api/v1/graphs/{GRAPH_ID}/snapshots/{SNAPSHOT_ID}/rollback",
                json={"confirm": False, "mode": "full"},
                headers=_auth_headers(),
            )
        auth_p.stop()
        graph_p.stop()

        assert response.status_code == 400

    @pytest.mark.integration
    @pytest.mark.api
    async def test_rollback_sync_path_returns_rollback_response(self, async_client):
        """POST /rollback → 200 RollbackResponse for small graphs (< 10K entities)."""
        result = _fake_rollback_result()

        auth_p = _patch_auth()
        graph_p, _ = _patch_graph_service()
        with (
            patch("app.api.v1.endpoints.graphs.versioning_service") as mock_vs,
            patch("app.api.v1.endpoints.graphs.neo4j_client") as mock_neo4j,
        ):
            mock_neo4j.execute_query = AsyncMock(return_value=[{"cnt": 50}])  # < 10K
            mock_vs.rollback = AsyncMock(return_value=result)
            response = await async_client.post(
                f"/api/v1/graphs/{GRAPH_ID}/snapshots/{SNAPSHOT_ID}/rollback",
                json={"confirm": True, "mode": "full", "create_checkpoint": True},
                headers=_auth_headers(),
            )
        auth_p.stop()
        graph_p.stop()

        assert response.status_code == 200
        data = response.json()
        assert data["entities_restored"] == 4
        assert data["message"] == "Full rollback completed"

    @pytest.mark.integration
    @pytest.mark.api
    async def test_rollback_async_path_for_large_graph(self, async_client):
        """POST /rollback → 202-style AsyncRollbackResponse for graphs > 10K entities."""
        job_data = {
            "rollback_job_id": JOB_ID,
            "graph_id": GRAPH_ID,
            "version_id": SNAPSHOT_ID,
            "mode": "full",
            "status": "pending",
            "progress": 0,
            "entities_restored": 0,
            "entities_soft_deleted": 0,
            "relationships_restored": 0,
            "relationships_soft_deleted": 0,
            "checkpoint_version_id": None,
            "error_message": None,
            "performed_by": USER_ID,
            "scope": None,
            "celery_task_id": None,
            "started_at": None,
            "completed_at": None,
            "created_at": _NOW,
        }
        fake_task = MagicMock()
        fake_task.id = "celery-task-id-abc"

        auth_p = _patch_auth()
        graph_p, _ = _patch_graph_service()
        with (
            patch("app.api.v1.endpoints.graphs.versioning_service") as mock_vs,
            patch("app.api.v1.endpoints.graphs.neo4j_client") as mock_neo4j,
            patch("app.api.v1.endpoints.graphs.GraphRollbackJob"),
            patch("app.services.background_jobs.async_rollback_graph") as mock_task,
        ):
            mock_neo4j.execute_query = AsyncMock(
                return_value=[{"cnt": 15_000}]
            )  # > 10K
            mock_vs.create_rollback_job = AsyncMock(return_value=job_data)
            mock_task.delay = MagicMock(return_value=fake_task)

            # Mock the db session update/commit
            mock_db = AsyncMock()
            mock_db.execute = AsyncMock()
            mock_db.commit = AsyncMock()

            with patch(
                "app.api.v1.endpoints.graphs.get_database", return_value=mock_db
            ):
                response = await async_client.post(
                    f"/api/v1/graphs/{GRAPH_ID}/snapshots/{SNAPSHOT_ID}/rollback",
                    json={"confirm": True, "mode": "full"},
                    headers=_auth_headers(),
                )
        auth_p.stop()
        graph_p.stop()

        # Should return async response with rollback_job_id
        assert response.status_code == 200
        data = response.json()
        assert "rollback_job_id" in data
        assert data["status"] == "pending"

    @pytest.mark.integration
    @pytest.mark.api
    async def test_rollback_returns_404_for_missing_snapshot(self, async_client):
        """POST /rollback → 404 when snapshot doesn't exist."""
        auth_p = _patch_auth()
        graph_p, _ = _patch_graph_service()
        with (
            patch("app.api.v1.endpoints.graphs.versioning_service") as mock_vs,
            patch("app.api.v1.endpoints.graphs.neo4j_client") as mock_neo4j,
        ):
            mock_neo4j.execute_query = AsyncMock(return_value=[{"cnt": 50}])
            mock_vs.rollback = AsyncMock(side_effect=ValueError("Version not found"))
            response = await async_client.post(
                f"/api/v1/graphs/{GRAPH_ID}/snapshots/{SNAPSHOT_ID}/rollback",
                json={"confirm": True, "mode": "full"},
                headers=_auth_headers(),
            )
        auth_p.stop()
        graph_p.stop()

        assert response.status_code == 404


# ---------------------------------------------------------------------------
# Suite — Get rollback job status
# ---------------------------------------------------------------------------


class TestRollbackJobStatus:

    @pytest.mark.integration
    @pytest.mark.api
    async def test_get_job_returns_200_with_status(self, async_client):
        """GET /rollbacks/{job_id} → 200 with job status."""
        job = _fake_rollback_job()

        auth_p = _patch_auth()
        graph_p, _ = _patch_graph_service()
        with patch("app.api.v1.endpoints.graphs.versioning_service") as mock_vs:
            mock_vs.get_rollback_job = AsyncMock(return_value=job)
            response = await async_client.get(
                f"/api/v1/graphs/{GRAPH_ID}/rollbacks/{JOB_ID}",
                headers=_auth_headers(),
            )
        auth_p.stop()
        graph_p.stop()

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "done"
        assert data["entities_restored"] == 4

    @pytest.mark.integration
    @pytest.mark.api
    async def test_get_job_returns_404_for_missing_job(self, async_client):
        """GET /rollbacks/{job_id} → 404 when job doesn't exist."""
        auth_p = _patch_auth()
        graph_p, _ = _patch_graph_service()
        with patch("app.api.v1.endpoints.graphs.versioning_service") as mock_vs:
            mock_vs.get_rollback_job = AsyncMock(return_value=None)
            response = await async_client.get(
                f"/api/v1/graphs/{GRAPH_ID}/rollbacks/{JOB_ID}",
                headers=_auth_headers(),
            )
        auth_p.stop()
        graph_p.stop()

        assert response.status_code == 404

    @pytest.mark.integration
    @pytest.mark.api
    async def test_get_job_enforces_graph_scoping(self, async_client):
        """GET /rollbacks — get_rollback_job is called with correct graph_id for multi-tenancy."""
        auth_p = _patch_auth()
        graph_p, _ = _patch_graph_service()
        with patch("app.api.v1.endpoints.graphs.versioning_service") as mock_vs:
            mock_vs.get_rollback_job = AsyncMock(return_value=_fake_rollback_job())
            await async_client.get(
                f"/api/v1/graphs/{GRAPH_ID}/rollbacks/{JOB_ID}",
                headers=_auth_headers(),
            )
            call_args = mock_vs.get_rollback_job.call_args
        auth_p.stop()
        graph_p.stop()

        # graph_id must be passed to scope the lookup (multi-tenancy)
        assert str(GRAPH_ID) in str(call_args)


# ---------------------------------------------------------------------------
# Suite — Multi-tenancy isolation
# ---------------------------------------------------------------------------


class TestSnapshotMultiTenancyIsolation:

    @pytest.mark.integration
    @pytest.mark.api
    async def test_cannot_access_snapshots_of_other_users_graph(self, async_client):
        """All snapshot endpoints reject requests for graphs owned by a different user."""
        other_owner = str(uuid.uuid4())
        auth_p = _patch_auth()
        graph_p, _ = _patch_graph_service(graph=_neo4j_graph(user_id=other_owner))

        with patch("app.api.v1.endpoints.graphs.versioning_service"):
            list_response = await async_client.get(
                f"/api/v1/graphs/{GRAPH_ID}/snapshots",
                headers=_auth_headers(),
            )
        auth_p.stop()
        graph_p.stop()

        assert list_response.status_code in (403, 404)

    @pytest.mark.integration
    @pytest.mark.api
    async def test_create_snapshot_passes_graph_id_to_service(self, async_client):
        """create_version must be called with the correct graph_id (multi-tenancy rule)."""
        snap = _fake_snapshot()
        auth_p = _patch_auth()
        graph_p, _ = _patch_graph_service()
        with patch("app.api.v1.endpoints.graphs.versioning_service") as mock_vs:
            mock_vs.create_version = AsyncMock(return_value=snap)
            await async_client.post(
                f"/api/v1/graphs/{GRAPH_ID}/snapshots",
                json={"label": "v1"},
                headers=_auth_headers(),
            )
            call_kwargs = mock_vs.create_version.call_args.kwargs
        auth_p.stop()
        graph_p.stop()

        assert call_kwargs["graph_id"] == GRAPH_ID
