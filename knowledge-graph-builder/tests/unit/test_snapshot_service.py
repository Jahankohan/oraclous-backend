"""
Unit tests for SnapshotService.

Covers: create_snapshot, list_snapshots, get_snapshot, delete_snapshot, diff_snapshots.
All Neo4j calls are mocked — no live database required.
"""
import uuid
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch, call

import pytest

from app.services.snapshot_service import SnapshotService, _snapshot_ts


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

GRAPH_ID = str(uuid.uuid4())
SNAP_ID = str(uuid.uuid4())
OTHER_SNAP_ID = str(uuid.uuid4())
_NOW = datetime(2025, 9, 4, 12, 0, 0, tzinfo=timezone.utc)
_NOW_ISO = _NOW.isoformat()

_EARLIER_ISO = "2025-09-04T10:00:00+00:00"
_LATER_ISO = "2025-09-04T14:00:00+00:00"


def _fake_neo4j_node(version_id: str = SNAP_ID) -> MagicMock:
    node = MagicMock()
    node.items.return_value = [
        ("version_id", version_id),
        ("graph_id", GRAPH_ID),
        ("version_number", 1),
        ("label", "v1"),
        ("description", "test"),
        ("captured_at", _NOW_ISO),
        ("created_by", "user-1"),
        ("parent_version_id", None),
        ("is_auto", False),
        ("snapshot_strategy", "pointer"),
        ("entity_count", 10),
        ("relationship_count", 5),
        ("created_at", _NOW_ISO),
    ]
    return node


def _make_service() -> SnapshotService:
    return SnapshotService()


# ---------------------------------------------------------------------------
# _snapshot_ts helper
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_snapshot_ts_string_passthrough():
    snapshot = {"captured_at": _NOW_ISO}
    assert _snapshot_ts(snapshot) == _NOW_ISO


@pytest.mark.unit
def test_snapshot_ts_missing_raises():
    with pytest.raises(ValueError, match="missing captured_at"):
        _snapshot_ts({})


@pytest.mark.unit
def test_snapshot_ts_neo4j_datetime_object():
    dt_mock = MagicMock()
    dt_mock.iso_format.return_value = _NOW_ISO
    snapshot = {"captured_at": dt_mock}
    assert _snapshot_ts(snapshot) == _NOW_ISO


# ---------------------------------------------------------------------------
# create_snapshot
# ---------------------------------------------------------------------------

@pytest.mark.unit
@pytest.mark.asyncio
async def test_create_snapshot_returns_dict():
    svc = _make_service()
    node = _fake_neo4j_node()

    with patch("app.services.snapshot_service.neo4j_client") as mock_client:
        mock_client.execute_query = AsyncMock(side_effect=[
            # count query
            [{"entity_count": 10, "relationship_count": 5}],
            # version_number query
            [{"next_num": 1}],
            # create query
            [{"v": node}],
        ])
        result = await svc.create_snapshot(
            graph_id=GRAPH_ID,
            label="v1",
            description="test",
            created_by="user-1",
        )

    assert result["graph_id"] == GRAPH_ID
    assert result["snapshot_strategy"] == "pointer"


@pytest.mark.unit
@pytest.mark.asyncio
async def test_create_snapshot_passes_graph_id_to_all_queries():
    svc = _make_service()
    node = _fake_neo4j_node()

    with patch("app.services.snapshot_service.neo4j_client") as mock_client:
        mock_client.execute_query = AsyncMock(side_effect=[
            [{"entity_count": 0, "relationship_count": 0}],
            [{"next_num": 1}],
            [{"v": node}],
        ])
        await svc.create_snapshot(
            graph_id=GRAPH_ID, label=None, description=None, created_by="u"
        )

    for call_args in mock_client.execute_query.call_args_list:
        params = call_args[0][1]
        assert params.get("graph_id") == GRAPH_ID, "graph_id missing from query params"


@pytest.mark.unit
@pytest.mark.asyncio
async def test_create_snapshot_fallback_when_graph_node_missing():
    svc = _make_service()
    node = _fake_neo4j_node()

    with patch("app.services.snapshot_service.neo4j_client") as mock_client:
        mock_client.execute_query = AsyncMock(side_effect=[
            [{"entity_count": 0, "relationship_count": 0}],
            [{"next_num": 1}],
            [],       # MATCH (g:Graph ...) returns nothing
            [{"v": node}],  # fallback CREATE succeeds
        ])
        result = await svc.create_snapshot(
            graph_id=GRAPH_ID, label="v1", description=None, created_by="u"
        )

    assert result["version_id"] == SNAP_ID


# ---------------------------------------------------------------------------
# list_snapshots
# ---------------------------------------------------------------------------

@pytest.mark.unit
@pytest.mark.asyncio
async def test_list_snapshots_returns_list():
    svc = _make_service()
    node = _fake_neo4j_node()

    with patch("app.services.snapshot_service.neo4j_client") as mock_client:
        mock_client.execute_query = AsyncMock(return_value=[{"v": node}])
        result = await svc.list_snapshots(GRAPH_ID)

    assert isinstance(result, list)
    assert len(result) == 1
    assert result[0]["version_id"] == SNAP_ID


@pytest.mark.unit
@pytest.mark.asyncio
async def test_list_snapshots_enforces_graph_id():
    svc = _make_service()

    with patch("app.services.snapshot_service.neo4j_client") as mock_client:
        mock_client.execute_query = AsyncMock(return_value=[])
        await svc.list_snapshots(GRAPH_ID)

    params = mock_client.execute_query.call_args[0][1]
    assert params["graph_id"] == GRAPH_ID


# ---------------------------------------------------------------------------
# get_snapshot
# ---------------------------------------------------------------------------

@pytest.mark.unit
@pytest.mark.asyncio
async def test_get_snapshot_found():
    svc = _make_service()
    node = _fake_neo4j_node()

    with patch("app.services.snapshot_service.neo4j_client") as mock_client:
        mock_client.execute_query = AsyncMock(return_value=[{"v": node}])
        result = await svc.get_snapshot(GRAPH_ID, SNAP_ID)

    assert result is not None
    assert result["version_id"] == SNAP_ID


@pytest.mark.unit
@pytest.mark.asyncio
async def test_get_snapshot_not_found_returns_none():
    svc = _make_service()

    with patch("app.services.snapshot_service.neo4j_client") as mock_client:
        mock_client.execute_query = AsyncMock(return_value=[])
        result = await svc.get_snapshot(GRAPH_ID, SNAP_ID)

    assert result is None


@pytest.mark.unit
@pytest.mark.asyncio
async def test_get_snapshot_scoped_by_graph_id():
    svc = _make_service()

    with patch("app.services.snapshot_service.neo4j_client") as mock_client:
        mock_client.execute_query = AsyncMock(return_value=[])
        await svc.get_snapshot(GRAPH_ID, SNAP_ID)

    params = mock_client.execute_query.call_args[0][1]
    assert params["graph_id"] == GRAPH_ID
    assert params["version_id"] == SNAP_ID


# ---------------------------------------------------------------------------
# delete_snapshot
# ---------------------------------------------------------------------------

@pytest.mark.unit
@pytest.mark.asyncio
async def test_delete_snapshot_returns_true_when_found():
    svc = _make_service()
    node = _fake_neo4j_node()

    with patch("app.services.snapshot_service.neo4j_client") as mock_client:
        # get_snapshot call returns data, delete call returns []
        mock_client.execute_query = AsyncMock(side_effect=[
            [{"v": node}],
            [],
        ])
        result = await svc.delete_snapshot(GRAPH_ID, SNAP_ID)

    assert result is True


@pytest.mark.unit
@pytest.mark.asyncio
async def test_delete_snapshot_returns_false_when_not_found():
    svc = _make_service()

    with patch("app.services.snapshot_service.neo4j_client") as mock_client:
        mock_client.execute_query = AsyncMock(return_value=[])
        result = await svc.delete_snapshot(GRAPH_ID, SNAP_ID)

    assert result is False


# ---------------------------------------------------------------------------
# diff_snapshots
# ---------------------------------------------------------------------------

def _make_count_result(cnt: int):
    return [{"cnt": cnt}]


@pytest.mark.unit
@pytest.mark.asyncio
async def test_diff_snapshots_orders_older_first():
    svc = _make_service()

    older_node = MagicMock()
    older_node.items.return_value = [
        ("version_id", SNAP_ID),
        ("graph_id", GRAPH_ID),
        ("version_number", 1),
        ("captured_at", _EARLIER_ISO),
        ("label", "old"),
    ]
    newer_node = MagicMock()
    newer_node.items.return_value = [
        ("version_id", OTHER_SNAP_ID),
        ("graph_id", GRAPH_ID),
        ("version_number", 2),
        ("captured_at", _LATER_ISO),
        ("label", "new"),
    ]

    with patch("app.services.snapshot_service.neo4j_client") as mock_client:
        mock_client.execute_query = AsyncMock(side_effect=[
            # get_snapshot(SNAP_ID) = older
            [{"v": older_node}],
            # get_snapshot(OTHER_SNAP_ID) = newer
            [{"v": newer_node}],
            # ea count
            _make_count_result(3),
            # ed count
            _make_count_result(1),
            # ra count
            _make_count_result(2),
            # rd count
            _make_count_result(0),
            # changes page
            [],
        ])
        result = await svc.diff_snapshots(
            graph_id=GRAPH_ID,
            snapshot_id=SNAP_ID,
            compare_to=OTHER_SNAP_ID,
        )

    assert result["from_version"]["version_id"] == SNAP_ID
    assert result["to_version"]["version_id"] == OTHER_SNAP_ID
    assert result["summary"]["entities_added"] == 3
    assert result["summary"]["entities_deleted"] == 1


@pytest.mark.unit
@pytest.mark.asyncio
async def test_diff_snapshots_swaps_order_when_v1_newer():
    svc = _make_service()

    older_node = MagicMock()
    older_node.items.return_value = [
        ("version_id", OTHER_SNAP_ID),
        ("graph_id", GRAPH_ID),
        ("version_number", 1),
        ("captured_at", _EARLIER_ISO),
        ("label", "old"),
    ]
    newer_node = MagicMock()
    newer_node.items.return_value = [
        ("version_id", SNAP_ID),
        ("graph_id", GRAPH_ID),
        ("version_number", 2),
        ("captured_at", _LATER_ISO),
        ("label", "new"),
    ]

    with patch("app.services.snapshot_service.neo4j_client") as mock_client:
        mock_client.execute_query = AsyncMock(side_effect=[
            [{"v": newer_node}],   # get_snapshot(SNAP_ID) = newer
            [{"v": older_node}],   # get_snapshot(OTHER_SNAP_ID) = older
            _make_count_result(0),
            _make_count_result(0),
            _make_count_result(0),
            _make_count_result(0),
            [],
        ])
        result = await svc.diff_snapshots(
            graph_id=GRAPH_ID,
            snapshot_id=SNAP_ID,
            compare_to=OTHER_SNAP_ID,
        )

    # Older should be from_version regardless of input order
    assert result["from_version"]["version_id"] == OTHER_SNAP_ID


@pytest.mark.unit
@pytest.mark.asyncio
async def test_diff_snapshots_raises_when_snapshot_missing():
    svc = _make_service()

    with patch("app.services.snapshot_service.neo4j_client") as mock_client:
        mock_client.execute_query = AsyncMock(return_value=[])
        with pytest.raises(ValueError, match="not found"):
            await svc.diff_snapshots(GRAPH_ID, SNAP_ID, OTHER_SNAP_ID)


@pytest.mark.unit
@pytest.mark.asyncio
async def test_diff_snapshots_all_queries_include_graph_id():
    svc = _make_service()

    node1 = MagicMock()
    node1.items.return_value = [
        ("version_id", SNAP_ID), ("graph_id", GRAPH_ID),
        ("captured_at", _EARLIER_ISO), ("version_number", 1), ("label", None),
    ]
    node2 = MagicMock()
    node2.items.return_value = [
        ("version_id", OTHER_SNAP_ID), ("graph_id", GRAPH_ID),
        ("captured_at", _LATER_ISO), ("version_number", 2), ("label", None),
    ]

    with patch("app.services.snapshot_service.neo4j_client") as mock_client:
        mock_client.execute_query = AsyncMock(side_effect=[
            [{"v": node1}], [{"v": node2}],
            _make_count_result(0), _make_count_result(0),
            _make_count_result(0), _make_count_result(0),
            [],
        ])
        await svc.diff_snapshots(GRAPH_ID, SNAP_ID, OTHER_SNAP_ID)

    for i, args in enumerate(mock_client.execute_query.call_args_list):
        params = args[0][1]
        # First two calls are get_snapshot (graph_id in params), rest have graph_id too
        assert "graph_id" in params, f"Call {i} missing graph_id: {params}"
        assert params["graph_id"] == GRAPH_ID
