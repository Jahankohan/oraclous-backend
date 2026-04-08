"""
Unit tests for VersioningService.

Covers: create_version, list_versions, get_version, delete_version, diff_versions, rollback.
All Neo4j calls are mocked — no live database required.
"""

import uuid
from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.services.versioning_service import VersioningService, _version_ts

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

GRAPH_ID = str(uuid.uuid4())
VERSION_ID = str(uuid.uuid4())
OTHER_VERSION_ID = str(uuid.uuid4())
_NOW = datetime(2025, 9, 4, 12, 0, 0, tzinfo=UTC)
_NOW_ISO = _NOW.isoformat()


def _fake_neo4j_node(version_id: str = VERSION_ID) -> MagicMock:
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
        ("entity_count", 10),
        ("relationship_count", 5),
        ("created_at", _NOW_ISO),
    ]
    return node


def _make_service() -> VersioningService:
    return VersioningService()


# ---------------------------------------------------------------------------
# _version_ts helper
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_version_ts_string_passthrough():
    version = {"captured_at": _NOW_ISO}
    assert _version_ts(version) == _NOW_ISO


@pytest.mark.unit
def test_version_ts_raises_if_missing():
    with pytest.raises(ValueError, match="missing captured_at"):
        _version_ts({})


@pytest.mark.unit
def test_version_ts_handles_neo4j_datetime_object():
    neo4j_dt = MagicMock()
    neo4j_dt.iso_format.return_value = _NOW_ISO
    version = {"captured_at": neo4j_dt}
    assert _version_ts(version) == _NOW_ISO


# ---------------------------------------------------------------------------
# create_version
# ---------------------------------------------------------------------------


@pytest.mark.unit
async def test_create_version_passes_graph_id_to_cypher():
    """All Cypher calls in create_version must include graph_id (multi-tenancy rule #4)."""
    svc = _make_service()
    node = _fake_neo4j_node()

    with patch("app.services.versioning_service.neo4j_client") as mock_client:
        mock_client.execute_query = AsyncMock(
            return_value=[{"entity_count": 5, "relationship_count": 3}]
        )

        # Second call for version_number, third for create
        mock_client.execute_query.side_effect = [
            [{"entity_count": 5, "relationship_count": 3}],
            [{"next_num": 1}],
            [{"v": node}],
        ]

        result = await svc.create_version(
            graph_id=GRAPH_ID,
            label="v1",
            description="test",
            created_by="user-1",
        )

    # Verify graph_id was passed in every call
    for c in mock_client.execute_query.call_args_list:
        params = c.args[1] if len(c.args) > 1 else c.kwargs.get("parameters", {})
        assert GRAPH_ID in str(params), f"graph_id missing in call: {c}"


@pytest.mark.unit
async def test_create_version_returns_version_dict():
    """create_version returns a dict with expected keys."""
    svc = _make_service()
    node = _fake_neo4j_node()

    with patch("app.services.versioning_service.neo4j_client") as mock_client:
        mock_client.execute_query = AsyncMock(
            side_effect=[
                [{"entity_count": 5, "relationship_count": 3}],
                [{"next_num": 2}],
                [{"v": node}],
            ]
        )

        result = await svc.create_version(
            graph_id=GRAPH_ID,
            label="v1",
            description=None,
            created_by="user-1",
        )

    assert "version_id" in result
    assert result["graph_id"] == GRAPH_ID


# ---------------------------------------------------------------------------
# list_versions
# ---------------------------------------------------------------------------


@pytest.mark.unit
async def test_list_versions_filters_by_graph_id():
    """list_versions Cypher query must include graph_id filter."""
    svc = _make_service()
    node = _fake_neo4j_node()

    with patch("app.services.versioning_service.neo4j_client") as mock_client:
        mock_client.execute_query = AsyncMock(return_value=[{"v": node}])

        await svc.list_versions(GRAPH_ID)

        params = mock_client.execute_query.call_args.args[1]
        assert params.get("graph_id") == GRAPH_ID


@pytest.mark.unit
async def test_list_versions_empty_returns_empty_list():
    svc = _make_service()
    with patch("app.services.versioning_service.neo4j_client") as mock_client:
        mock_client.execute_query = AsyncMock(return_value=[])
        result = await svc.list_versions(GRAPH_ID)
    assert result == []


# ---------------------------------------------------------------------------
# get_version
# ---------------------------------------------------------------------------


@pytest.mark.unit
async def test_get_version_filters_by_both_graph_id_and_version_id():
    """get_version must scope to both graph_id and version_id."""
    svc = _make_service()
    node = _fake_neo4j_node()

    with patch("app.services.versioning_service.neo4j_client") as mock_client:
        mock_client.execute_query = AsyncMock(return_value=[{"v": node}])

        await svc.get_version(GRAPH_ID, VERSION_ID)

        params = mock_client.execute_query.call_args.args[1]
        assert params.get("graph_id") == GRAPH_ID
        assert params.get("version_id") == VERSION_ID


@pytest.mark.unit
async def test_get_version_returns_none_when_not_found():
    svc = _make_service()
    with patch("app.services.versioning_service.neo4j_client") as mock_client:
        mock_client.execute_query = AsyncMock(return_value=[])
        result = await svc.get_version(GRAPH_ID, VERSION_ID)
    assert result is None


# ---------------------------------------------------------------------------
# delete_version
# ---------------------------------------------------------------------------


@pytest.mark.unit
async def test_delete_version_returns_true_when_found():
    svc = _make_service()
    node = _fake_neo4j_node()

    with patch("app.services.versioning_service.neo4j_client") as mock_client:
        # get_version succeeds, delete succeeds
        mock_client.execute_query = AsyncMock(
            side_effect=[[{"v": node}], []]  # get, then delete
        )
        result = await svc.delete_version(GRAPH_ID, VERSION_ID)

    assert result is True


@pytest.mark.unit
async def test_delete_version_returns_false_when_not_found():
    svc = _make_service()
    with patch("app.services.versioning_service.neo4j_client") as mock_client:
        mock_client.execute_query = AsyncMock(return_value=[])
        result = await svc.delete_version(GRAPH_ID, VERSION_ID)
    assert result is False


@pytest.mark.unit
async def test_delete_version_passes_graph_id_to_cypher():
    """DETACH DELETE query must include graph_id for multi-tenancy."""
    svc = _make_service()
    node = _fake_neo4j_node()

    with patch("app.services.versioning_service.neo4j_client") as mock_client:
        mock_client.execute_query = AsyncMock(side_effect=[[{"v": node}], []])
        await svc.delete_version(GRAPH_ID, VERSION_ID)

    # Both calls (get and delete) should have graph_id
    for c in mock_client.execute_query.call_args_list:
        params = c.args[1] if len(c.args) > 1 else {}
        assert GRAPH_ID in str(params)


# ---------------------------------------------------------------------------
# diff_versions
# ---------------------------------------------------------------------------


@pytest.mark.unit
async def test_diff_versions_returns_summary_and_changes():
    """diff_versions returns expected structure with summary and changes."""
    svc = _make_service()
    v1 = _fake_neo4j_node(VERSION_ID)
    v2 = _fake_neo4j_node(OTHER_VERSION_ID)

    with patch("app.services.versioning_service.neo4j_client") as mock_client:
        mock_client.execute_query = AsyncMock(
            side_effect=[
                [{"v": v1}],  # get v1
                [{"v": v2}],  # get v2
                [{"cnt": 2}],  # ea count
                [{"cnt": 1}],  # ed count
                [{"cnt": 1}],  # ra count
                [{"cnt": 0}],  # rd count
                [  # changes
                    {
                        "type": "entity_added",
                        "id": "e1",
                        "name": "A",
                        "entity_type": "Person",
                        "subject": "",
                        "predicate": "",
                        "object": "",
                        "ts": _NOW_ISO,
                    }
                ],
            ]
        )

        result = await svc.diff_versions(GRAPH_ID, VERSION_ID, OTHER_VERSION_ID)

    assert result["summary"]["entities_added"] == 2
    assert result["summary"]["entities_deleted"] == 1
    assert len(result["changes"]) == 1


@pytest.mark.unit
async def test_diff_versions_raises_on_missing_version():
    """diff_versions raises ValueError when a version is not found."""
    svc = _make_service()
    with patch("app.services.versioning_service.neo4j_client") as mock_client:
        mock_client.execute_query = AsyncMock(return_value=[])  # both versions missing
        with pytest.raises(ValueError, match="not found"):
            await svc.diff_versions(GRAPH_ID, VERSION_ID, OTHER_VERSION_ID)


# ---------------------------------------------------------------------------
# rollback (full path)
# ---------------------------------------------------------------------------


@pytest.mark.unit
async def test_rollback_requires_version_to_exist():
    svc = _make_service()
    with patch("app.services.versioning_service.neo4j_client") as mock_client:
        mock_client.execute_query = AsyncMock(return_value=[])  # version not found
        with pytest.raises(ValueError, match="not found"):
            await svc.rollback(
                graph_id=GRAPH_ID,
                version_id=VERSION_ID,
                mode="full",
                performed_by="user-1",
            )


@pytest.mark.unit
async def test_rollback_all_cypher_queries_include_graph_id():
    """Every Cypher query in _full_rollback must include graph_id (multi-tenancy rule #4)."""
    svc = _make_service()
    node = _fake_neo4j_node()

    with patch("app.services.versioning_service.neo4j_client") as mock_client:
        # Sequence: get_version, create_checkpoint (count, num, create), then 5 rollback queries
        checkpoint_node = _fake_neo4j_node(str(uuid.uuid4()))
        mock_client.execute_query = AsyncMock(
            side_effect=[
                [{"v": node}],  # get_version
                [
                    {"entity_count": 5, "relationship_count": 3}
                ],  # create checkpoint — count
                [{"next_num": 2}],  # checkpoint — version number
                [{"v": checkpoint_node}],  # checkpoint — create node
                [{"cnt": 2}],  # del_new_entities
                [{"cnt": 1}],  # restore_entities
                [{"cnt": 1}],  # del_new_rels
                [{"cnt": 0}],  # restore_rels
                [],  # integrity cascade
            ]
        )

        await svc.rollback(
            graph_id=GRAPH_ID,
            version_id=VERSION_ID,
            mode="full",
            performed_by="user-1",
            create_checkpoint=True,
        )

    for c in mock_client.execute_query.call_args_list:
        params = c.args[1] if len(c.args) > 1 else c.kwargs.get("parameters", {})
        assert GRAPH_ID in str(params), f"graph_id missing in call: {c}"
