"""
Unit tests for RollbackService.

Covers: create_rollback_job, get_rollback_job, rollback (_full_rollback).
All Neo4j calls are mocked — no live database required.
All rollback operations are whole-graph only (Phase 3 scope).
"""

import uuid
from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.services.rollback_service import RollbackService

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

GRAPH_ID = str(uuid.uuid4())
SNAP_ID = str(uuid.uuid4())
_NOW_ISO = "2025-09-04T12:00:00+00:00"
_CAPTURED_AT = "2025-09-04T10:00:00+00:00"


def _make_svc() -> RollbackService:
    return RollbackService()


def _fake_snapshot(version_id: str = SNAP_ID, captured_at: str = _CAPTURED_AT) -> dict:
    return {
        "version_id": version_id,
        "graph_id": GRAPH_ID,
        "version_number": 1,
        "label": "v1",
        "captured_at": captured_at,
        "created_by": "user-1",
        "snapshot_strategy": "pointer",
    }


def _make_count_result(cnt: int):
    return [{"cnt": cnt}]


# ---------------------------------------------------------------------------
# _full_rollback — property naming (must use invalidated_at, not deleted_at)
# ---------------------------------------------------------------------------


@pytest.mark.unit
@pytest.mark.asyncio
async def test_full_rollback_uses_invalidated_at():
    svc = _make_svc()

    with patch("app.services.rollback_service.neo4j_client") as mock_client:
        mock_client.execute_query = AsyncMock(
            side_effect=[
                _make_count_result(2),  # entities soft-deleted
                _make_count_result(1),  # entities restored
                _make_count_result(3),  # rels soft-deleted
                _make_count_result(0),  # rels restored
                _make_count_result(0),  # integrity pass
            ]
        )
        result = await svc._full_rollback(
            graph_id=GRAPH_ID,
            captured_at=_CAPTURED_AT,
            performed_by="user-1",
            now_iso=_NOW_ISO,
            checkpoint_vid=None,
        )

    assert result["entities_soft_deleted"] == 2
    assert result["entities_restored"] == 1
    assert result["relationships_soft_deleted"] == 3
    assert result["relationships_restored"] == 0

    # Verify all queries reference graph_id and use invalidated_at (not deleted_at)
    for call_args in mock_client.execute_query.call_args_list:
        query: str = call_args[0][0]
        params: dict = call_args[0][1]
        assert (
            "deleted_at" not in query
        ), f"Query must use invalidated_at, not deleted_at: {query[:80]}"
        assert params.get("graph_id") == GRAPH_ID


@pytest.mark.unit
@pytest.mark.asyncio
async def test_full_rollback_all_queries_include_graph_id():
    svc = _make_svc()

    with patch("app.services.rollback_service.neo4j_client") as mock_client:
        mock_client.execute_query = AsyncMock(
            side_effect=[
                _make_count_result(0),
                _make_count_result(0),
                _make_count_result(0),
                _make_count_result(0),
                _make_count_result(0),
            ]
        )
        await svc._full_rollback(GRAPH_ID, _CAPTURED_AT, "u", _NOW_ISO, None)

    for i, args in enumerate(mock_client.execute_query.call_args_list):
        assert args[0][1].get("graph_id") == GRAPH_ID, f"Query {i} missing graph_id"


# ---------------------------------------------------------------------------
# rollback — auto-checkpoint logic
# ---------------------------------------------------------------------------


@pytest.mark.unit
@pytest.mark.asyncio
async def test_rollback_creates_checkpoint_by_default():
    svc = _make_svc()
    checkpoint_id = str(uuid.uuid4())

    fake_snap = _fake_snapshot()
    fake_checkpoint = {
        **_fake_snapshot(version_id=checkpoint_id),
        "captured_at": _NOW_ISO,
    }

    with (
        patch("app.services.rollback_service.snapshot_service") as mock_snap_svc,
        patch("app.services.rollback_service.neo4j_client") as mock_neo,
    ):

        mock_snap_svc.get_snapshot = AsyncMock(return_value=fake_snap)
        mock_snap_svc.create_snapshot = AsyncMock(return_value=fake_checkpoint)
        mock_neo.execute_query = AsyncMock(
            side_effect=[
                _make_count_result(0),
                _make_count_result(0),
                _make_count_result(0),
                _make_count_result(0),
                _make_count_result(0),
            ]
        )

        result = await svc.rollback(
            graph_id=GRAPH_ID,
            version_id=SNAP_ID,
            performed_by="user-1",
            create_checkpoint=True,
        )

    mock_snap_svc.create_snapshot.assert_called_once()
    call_kwargs = mock_snap_svc.create_snapshot.call_args[1]
    assert call_kwargs["is_auto"] is True
    assert call_kwargs["parent_version_id"] == SNAP_ID
    assert result["checkpoint_version_id"] == checkpoint_id


@pytest.mark.unit
@pytest.mark.asyncio
async def test_rollback_skips_checkpoint_when_disabled():
    svc = _make_svc()
    fake_snap = _fake_snapshot()

    with (
        patch("app.services.rollback_service.snapshot_service") as mock_snap_svc,
        patch("app.services.rollback_service.neo4j_client") as mock_neo,
    ):

        mock_snap_svc.get_snapshot = AsyncMock(return_value=fake_snap)
        mock_neo.execute_query = AsyncMock(
            side_effect=[
                _make_count_result(0),
                _make_count_result(0),
                _make_count_result(0),
                _make_count_result(0),
                _make_count_result(0),
            ]
        )

        result = await svc.rollback(
            graph_id=GRAPH_ID,
            version_id=SNAP_ID,
            performed_by="user-1",
            create_checkpoint=False,
        )

    mock_snap_svc.create_snapshot.assert_not_called()
    assert result["checkpoint_version_id"] is None


@pytest.mark.unit
@pytest.mark.asyncio
async def test_rollback_raises_when_snapshot_not_found():
    svc = _make_svc()

    with patch("app.services.rollback_service.snapshot_service") as mock_snap_svc:
        mock_snap_svc.get_snapshot = AsyncMock(return_value=None)
        with pytest.raises(ValueError, match="not found"):
            await svc.rollback(
                graph_id=GRAPH_ID,
                version_id=SNAP_ID,
                performed_by="user-1",
            )


# ---------------------------------------------------------------------------
# create_rollback_job — mode is always "full"
# ---------------------------------------------------------------------------


@pytest.mark.unit
@pytest.mark.asyncio
async def test_create_rollback_job_mode_is_full():
    svc = _make_svc()

    fake_job = MagicMock()
    fake_job.id = uuid.uuid4()
    fake_job.graph_id = uuid.UUID(GRAPH_ID)
    fake_job.version_id = SNAP_ID
    fake_job.mode = "full"
    fake_job.status = "pending"
    fake_job.progress = 0
    fake_job.entities_restored = 0
    fake_job.entities_soft_deleted = 0
    fake_job.relationships_restored = 0
    fake_job.relationships_soft_deleted = 0
    fake_job.checkpoint_version_id = None
    fake_job.error_message = None
    fake_job.performed_by = "user-1"
    fake_job.scope = None
    fake_job.celery_task_id = None
    fake_job.started_at = None
    fake_job.completed_at = None
    fake_job.created_at = datetime.now(UTC)

    mock_db = AsyncMock()
    mock_db.refresh = AsyncMock(side_effect=lambda obj: None)


    with (
        patch("app.services.rollback_service.uuid") as mock_uuid,
        patch.object(mock_db, "add") as mock_add,
    ):

        mock_uuid.uuid4.return_value = fake_job.id
        mock_db.add = MagicMock()
        mock_db.commit = AsyncMock()
        mock_db.refresh = AsyncMock()

        # Make db.refresh populate the job attributes by returning None
        async def _refresh(obj):
            obj.id = fake_job.id
            obj.graph_id = fake_job.graph_id
            obj.version_id = fake_job.version_id
            obj.mode = "full"
            obj.status = "pending"
            obj.progress = 0
            obj.entities_restored = 0
            obj.entities_soft_deleted = 0
            obj.relationships_restored = 0
            obj.relationships_soft_deleted = 0
            obj.checkpoint_version_id = None
            obj.error_message = None
            obj.performed_by = "user-1"
            obj.scope = None
            obj.celery_task_id = None
            obj.started_at = None
            obj.completed_at = None
            obj.created_at = fake_job.created_at

        mock_db.refresh.side_effect = _refresh

        result = await svc.create_rollback_job(
            db=mock_db,
            graph_id=GRAPH_ID,
            version_id=SNAP_ID,
            performed_by="user-1",
        )

    assert result["mode"] == "full"
    assert result["status"] == "pending"
