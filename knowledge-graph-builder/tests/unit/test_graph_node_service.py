"""
Unit tests for GraphNodeService.

Tests CRUD operations and multi-tenant isolation (user_id + graph_id scoping)
against a mocked Neo4j sync driver.
"""

from unittest.mock import MagicMock

import pytest
from neo4j.exceptions import Neo4jError

from app.services.graph_node_service import GraphNodeService

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_driver(single_return=None, all_records=None):
    """Build a mock Neo4j sync driver whose session().run() returns controllable data."""
    mock_record = MagicMock()
    mock_record.__getitem__ = MagicMock(
        side_effect=lambda key: single_return.get(key) if single_return else None
    )

    mock_result = MagicMock()
    mock_result.single.return_value = mock_record if single_return is not None else None
    mock_result.__iter__ = MagicMock(
        return_value=iter([_wrap_record(r) for r in (all_records or [])])
    )

    mock_session = MagicMock()
    mock_session.run.return_value = mock_result
    mock_session.__enter__ = MagicMock(return_value=mock_session)
    mock_session.__exit__ = MagicMock(return_value=False)

    mock_driver = MagicMock()
    mock_driver.session.return_value = mock_session

    return mock_driver, mock_session, mock_result, mock_record


def _wrap_record(data: dict):
    r = MagicMock()
    r.__getitem__ = MagicMock(side_effect=lambda key: data.get(key))
    return r


def _graph_node_data():
    return {
        "graph_id": "g-123",
        "name": "My Graph",
        "description": "A test graph",
        "user_id": "user-1",
        "created_at": MagicMock(isoformat=lambda: "2025-01-01T00:00:00"),
        "updated_at": MagicMock(isoformat=lambda: "2025-01-01T00:00:00"),
        "node_count": 0,
        "relationship_count": 0,
        "status": "active",
    }


# ---------------------------------------------------------------------------
# Tests: create_graph
# ---------------------------------------------------------------------------


class TestCreateGraph:
    @pytest.mark.unit
    def test_create_graph_success(self):
        mock_driver, mock_session, mock_result, mock_record = _make_driver()

        real_graph_data = {
            "graph_id": "g-123",
            "name": "My Graph",
            "description": "A test graph",
            "user_id": "user-1",
            "created_at": MagicMock(
                isoformat=MagicMock(return_value="2025-01-01T00:00:00")
            ),
            "updated_at": MagicMock(
                isoformat=MagicMock(return_value="2025-01-01T00:00:00")
            ),
            "node_count": 0,
            "relationship_count": 0,
            "status": "active",
        }

        # record["g"] returns a real dict so dict(record["g"]) works
        mock_record.__getitem__ = MagicMock(
            side_effect=lambda k: real_graph_data if k == "g" else None
        )
        mock_result.single.return_value = mock_record

        svc = GraphNodeService(mock_driver)
        result = svc.create_graph("g-123", "My Graph", "user-1", "A test graph")

        assert result["graph_id"] == "g-123"
        assert result["name"] == "My Graph"
        assert result["user_id"] == "user-1"
        assert result["status"] == "active"
        assert result["node_count"] == 0

    @pytest.mark.unit
    def test_create_graph_raises_when_driver_none(self):
        svc = GraphNodeService(None)
        with pytest.raises(Exception, match="Unexpected error creating graph"):
            svc.create_graph("g-1", "name", "user-1")

    @pytest.mark.unit
    def test_create_graph_raises_on_no_record(self):
        mock_driver, mock_session, mock_result, _ = _make_driver()
        mock_result.single.return_value = None

        svc = GraphNodeService(mock_driver)
        with pytest.raises(Exception, match="Unexpected error creating graph"):
            svc.create_graph("g-1", "name", "user-1")

    @pytest.mark.unit
    def test_create_graph_wraps_neo4j_error(self):
        mock_driver = MagicMock()
        mock_session = MagicMock()
        mock_session.__enter__ = MagicMock(return_value=mock_session)
        mock_session.__exit__ = MagicMock(return_value=False)
        mock_session.run.side_effect = Neo4jError("DB error")
        mock_driver.session.return_value = mock_session

        svc = GraphNodeService(mock_driver)
        with pytest.raises(Exception, match="Failed to create graph"):
            svc.create_graph("g-1", "name", "user-1")

    @pytest.mark.unit
    def test_create_graph_passes_graph_id_in_query_params(self):
        mock_driver, mock_session, mock_result, mock_record = _make_driver()
        real_data = {
            "graph_id": "g-123",
            "name": "N",
            "description": "",
            "user_id": "u",
            "created_at": MagicMock(isoformat=MagicMock(return_value="2025-01-01")),
            "updated_at": MagicMock(isoformat=MagicMock(return_value="2025-01-01")),
            "node_count": 0,
            "relationship_count": 0,
            "status": "active",
        }
        mock_record.__getitem__ = MagicMock(
            side_effect=lambda k: real_data if k == "g" else None
        )

        svc = GraphNodeService(mock_driver)
        try:
            svc.create_graph("g-123", "N", "u")
        except Exception:
            pass  # May fail on dict() — we just need to check params

        call_args = mock_session.run.call_args
        params = call_args[0][1] if len(call_args[0]) > 1 else call_args[1]
        assert params.get("graph_id") == "g-123"
        assert params.get("user_id") == "u"


# ---------------------------------------------------------------------------
# Tests: get_graph
# ---------------------------------------------------------------------------


class TestGetGraph:
    @pytest.mark.unit
    def test_get_graph_returns_none_when_not_found(self):
        mock_driver, mock_session, mock_result, _ = _make_driver()
        mock_result.single.return_value = None

        svc = GraphNodeService(mock_driver)
        result = svc.get_graph("nonexistent")
        assert result is None

    @pytest.mark.unit
    def test_get_graph_passes_graph_id_to_query(self):
        mock_driver, mock_session, mock_result, mock_record = _make_driver()
        mock_record.__getitem__ = MagicMock(
            side_effect=lambda k: {"graph_id": "g-1"} if k == "graph" else None
        )

        svc = GraphNodeService(mock_driver)
        svc.get_graph("g-1")

        call_args = mock_session.run.call_args
        params = call_args[0][1] if len(call_args[0]) > 1 else call_args[1]
        assert params.get("graph_id") == "g-1"

    @pytest.mark.unit
    def test_get_graph_wraps_neo4j_error(self):
        mock_driver = MagicMock()
        mock_session = MagicMock()
        mock_session.__enter__ = MagicMock(return_value=mock_session)
        mock_session.__exit__ = MagicMock(return_value=False)
        mock_session.run.side_effect = Neo4jError("Query failed")
        mock_driver.session.return_value = mock_session

        svc = GraphNodeService(mock_driver)
        with pytest.raises(Exception, match="Failed to retrieve Graph node"):
            svc.get_graph("g-1")

    @pytest.mark.unit
    def test_get_graph_raises_when_driver_none(self):
        svc = GraphNodeService(None)
        with pytest.raises(Exception):  # noqa: B017
            svc.get_graph("g-1")


# ---------------------------------------------------------------------------
# Tests: list_user_graphs
# ---------------------------------------------------------------------------


class TestListUserGraphs:
    @pytest.mark.unit
    def test_list_user_graphs_returns_empty_list_when_none(self):
        mock_driver, mock_session, mock_result, _ = _make_driver(all_records=[])
        mock_result.__iter__ = MagicMock(return_value=iter([]))

        svc = GraphNodeService(mock_driver)
        result = svc.list_user_graphs("user-1")
        assert result == []

    @pytest.mark.unit
    def test_list_user_graphs_passes_user_id_to_query(self):
        mock_driver, mock_session, mock_result, _ = _make_driver()
        mock_result.__iter__ = MagicMock(return_value=iter([]))

        svc = GraphNodeService(mock_driver)
        svc.list_user_graphs("user-abc")

        call_args = mock_session.run.call_args
        params = call_args[0][1] if len(call_args[0]) > 1 else call_args[1]
        assert params.get("user_id") == "user-abc"

    @pytest.mark.unit
    def test_list_user_graphs_scopes_to_user(self):
        """The query must always filter by user_id — never return other users' graphs."""
        mock_driver, mock_session, mock_result, _ = _make_driver()
        mock_result.__iter__ = MagicMock(return_value=iter([]))

        svc = GraphNodeService(mock_driver)
        svc.list_user_graphs("user-1")

        run_call_query = mock_session.run.call_args[0][0]
        assert "user_id" in run_call_query

    @pytest.mark.unit
    def test_list_user_graphs_wraps_neo4j_error(self):
        mock_driver = MagicMock()
        mock_session = MagicMock()
        mock_session.__enter__ = MagicMock(return_value=mock_session)
        mock_session.__exit__ = MagicMock(return_value=False)
        mock_session.run.side_effect = Neo4jError("List failed")
        mock_driver.session.return_value = mock_session

        svc = GraphNodeService(mock_driver)
        with pytest.raises(Exception, match="Failed to list user graphs"):
            svc.list_user_graphs("user-1")


# ---------------------------------------------------------------------------
# Tests: delete_graph
# ---------------------------------------------------------------------------


class TestDeleteGraph:
    @pytest.mark.unit
    def test_delete_graph_returns_true_when_deleted(self):
        mock_driver, mock_session, mock_result, mock_record = _make_driver()
        mock_record.__getitem__ = MagicMock(
            side_effect=lambda k: 1 if k == "deleted_count" else None
        )
        mock_result.single.return_value = mock_record

        svc = GraphNodeService(mock_driver)
        result = svc.delete_graph("g-1", "user-1")
        assert result is True

    @pytest.mark.unit
    def test_delete_graph_returns_false_when_not_found(self):
        mock_driver, mock_session, mock_result, mock_record = _make_driver()
        mock_record.__getitem__ = MagicMock(
            side_effect=lambda k: 0 if k == "deleted_count" else None
        )

        svc = GraphNodeService(mock_driver)
        result = svc.delete_graph("g-none", "user-1")
        assert result is False

    @pytest.mark.unit
    def test_delete_graph_requires_user_id_in_query(self):
        """Delete query must include user_id for tenant isolation."""
        mock_driver, mock_session, mock_result, mock_record = _make_driver()
        mock_record.__getitem__ = MagicMock(side_effect=lambda k: 0)

        svc = GraphNodeService(mock_driver)
        svc.delete_graph("g-1", "user-1")

        call_args = mock_session.run.call_args
        params = call_args[0][1] if len(call_args[0]) > 1 else call_args[1]
        assert params.get("user_id") == "user-1"
        assert params.get("graph_id") == "g-1"

    @pytest.mark.unit
    def test_delete_graph_wraps_neo4j_error(self):
        mock_driver = MagicMock()
        mock_session = MagicMock()
        mock_session.__enter__ = MagicMock(return_value=mock_session)
        mock_session.__exit__ = MagicMock(return_value=False)
        mock_session.run.side_effect = Neo4jError("Delete failed")
        mock_driver.session.return_value = mock_session

        svc = GraphNodeService(mock_driver)
        with pytest.raises(Exception, match="Failed to delete graph"):
            svc.delete_graph("g-1", "user-1")


# ---------------------------------------------------------------------------
# Tests: graph_exists
# ---------------------------------------------------------------------------


class TestGraphExists:
    @pytest.mark.unit
    def test_graph_exists_returns_true(self):
        mock_driver, mock_session, mock_result, mock_record = _make_driver()
        mock_record.__getitem__ = MagicMock(
            side_effect=lambda k: True if k == "exists" else None
        )
        mock_result.single.return_value = mock_record

        svc = GraphNodeService(mock_driver)
        assert svc.graph_exists("g-1", "user-1") is True

    @pytest.mark.unit
    def test_graph_exists_returns_false_when_no_record(self):
        mock_driver, mock_session, mock_result, _ = _make_driver()
        mock_result.single.return_value = None

        svc = GraphNodeService(mock_driver)
        assert svc.graph_exists("g-none", "user-1") is False

    @pytest.mark.unit
    def test_graph_exists_requires_both_graph_and_user_ids(self):
        """Existence check must scope by BOTH graph_id AND user_id."""
        mock_driver, mock_session, mock_result, mock_record = _make_driver()
        mock_record.__getitem__ = MagicMock(side_effect=lambda k: False)

        svc = GraphNodeService(mock_driver)
        svc.graph_exists("g-1", "user-1")

        call_args = mock_session.run.call_args
        query = call_args[0][0]
        params = call_args[0][1] if len(call_args[0]) > 1 else call_args[1]
        assert "graph_id" in query
        assert "user_id" in query
        assert params.get("graph_id") == "g-1"
        assert params.get("user_id") == "user-1"


# ---------------------------------------------------------------------------
# Tests: update_graph
# ---------------------------------------------------------------------------


class TestUpdateGraph:
    @pytest.mark.unit
    def test_update_graph_returns_none_when_not_found(self):
        mock_driver, mock_session, mock_result, _ = _make_driver()
        mock_result.single.return_value = None

        svc = GraphNodeService(mock_driver)
        result = svc.update_graph("g-none", "user-1", name="New Name")
        assert result is None

    @pytest.mark.unit
    def test_update_graph_passes_user_id_for_tenant_isolation(self):
        mock_driver, mock_session, mock_result, _ = _make_driver()
        mock_result.single.return_value = None

        svc = GraphNodeService(mock_driver)
        svc.update_graph("g-1", "user-1", name="New Name")

        call_args = mock_session.run.call_args
        params = call_args[0][1] if len(call_args[0]) > 1 else call_args[1]
        assert params.get("user_id") == "user-1"
        assert params.get("graph_id") == "g-1"

    @pytest.mark.unit
    def test_update_graph_wraps_neo4j_error(self):
        mock_driver = MagicMock()
        mock_session = MagicMock()
        mock_session.__enter__ = MagicMock(return_value=mock_session)
        mock_session.__exit__ = MagicMock(return_value=False)
        mock_session.run.side_effect = Neo4jError("Update failed")
        mock_driver.session.return_value = mock_session

        svc = GraphNodeService(mock_driver)
        with pytest.raises(Exception, match="Failed to update graph"):
            svc.update_graph("g-1", "user-1", name="New Name")

    @pytest.mark.unit
    def test_update_graph_federatable_syncs_shadow_node(self):
        """When federatable is updated, the Cypher must also SET shadow.federatable
        so that federation_service reads the correct flag from the ReBAC shadow node.
        """
        mock_driver, mock_session, mock_result, _ = _make_driver()
        mock_result.single.return_value = None  # return value doesn't matter here

        svc = GraphNodeService(mock_driver)
        svc.update_graph("g-1", "user-1", federatable=True)

        call_args = mock_session.run.call_args
        query: str = call_args[0][0] if call_args[0] else ""
        params: dict = call_args[0][1] if len(call_args[0]) > 1 else call_args[1]

        assert (
            'namespace: "__system__"' in query
        ), "Shadow node OPTIONAL MATCH missing from query when federatable is updated"
        assert (
            "shadow.federatable" in query
        ), "SET shadow.federatable missing from query when federatable is updated"
        assert params.get("federatable") is True
        assert params.get("graph_id") == "g-1"

    @pytest.mark.unit
    def test_update_graph_no_shadow_sync_when_federatable_not_provided(self):
        """When federatable is not in the update payload, the shadow node must NOT
        be touched (avoids unnecessary writes and respects separation of concerns).
        """
        mock_driver, mock_session, mock_result, _ = _make_driver()
        mock_result.single.return_value = None

        svc = GraphNodeService(mock_driver)
        svc.update_graph("g-1", "user-1", name="Renamed")

        call_args = mock_session.run.call_args
        query: str = call_args[0][0] if call_args[0] else ""

        assert (
            "shadow" not in query
        ), "Shadow node must not appear in query when federatable is not being updated"


# ---------------------------------------------------------------------------
# Tests: migrate_relationship_properties
# ---------------------------------------------------------------------------


class TestMigrateRelationshipProperties:
    """Tests for the 3-phase migration that moves contextual properties off nodes."""

    def _make_migration_driver(
        self,
        transferred_job_title: int = 0,
        transferred_proficiency: int = 0,
        orphan_records=None,
        cleaned: int = 0,
    ):
        """
        Build a mock driver whose session.run() returns different results
        for each of the 4 sequential queries (phase1a, phase1b, phase2, phase3).
        """
        orphan_records = orphan_records or []

        def _single_result(value: dict):
            rec = MagicMock()
            rec.__getitem__ = MagicMock(side_effect=lambda k: value.get(k))
            result = MagicMock()
            result.single.return_value = rec
            return result

        def _iter_result(records):
            result = MagicMock()
            result.single.return_value = None
            result.__iter__ = MagicMock(return_value=iter(records))
            return result

        def _make_orphan_record(entity_id, name, entity_type, props):
            r = MagicMock()
            r.__getitem__ = MagicMock(
                side_effect=lambda k: {
                    "entity_id": entity_id,
                    "name": name,
                    "type": entity_type,
                    "props": props,
                }.get(k)
            )
            return r

        call_results = [
            _single_result({"transferred": transferred_job_title}),  # phase1a
            _single_result({"transferred": transferred_proficiency}),  # phase1b
            _iter_result([_make_orphan_record(**o) for o in orphan_records]),  # phase2
            _single_result({"cleaned": cleaned}),  # phase3
        ]
        call_count = [0]

        mock_session = MagicMock()
        mock_session.__enter__ = MagicMock(return_value=mock_session)
        mock_session.__exit__ = MagicMock(return_value=False)

        def run_side_effect(query, params=None, **kwargs):
            idx = call_count[0]
            call_count[0] += 1
            return call_results[idx] if idx < len(call_results) else MagicMock()

        mock_session.run.side_effect = run_side_effect

        mock_driver = MagicMock()
        mock_driver.session.return_value = mock_session
        return mock_driver

    @pytest.mark.unit
    def test_returns_graph_id_in_result(self):
        driver = self._make_migration_driver()
        svc = GraphNodeService(driver)
        result = svc.migrate_relationship_properties("g-abc")
        assert result["graph_id"] == "g-abc"

    @pytest.mark.unit
    def test_transferred_counts_are_correct(self):
        driver = self._make_migration_driver(
            transferred_job_title=5, transferred_proficiency=3
        )
        svc = GraphNodeService(driver)
        result = svc.migrate_relationship_properties("g-1")
        assert result["transferred_job_title"] == 5
        assert result["transferred_proficiency"] == 3
        assert result["transferred_total"] == 8

    @pytest.mark.unit
    def test_orphans_detected_count(self):
        orphans = [
            {
                "entity_id": "e-1",
                "name": "Alice",
                "entity_type": "Person",
                "props": ["job_title"],
            },
            {
                "entity_id": "e-2",
                "name": "Bob",
                "entity_type": "Person",
                "props": ["job_title"],
            },
        ]
        driver = self._make_migration_driver(orphan_records=orphans)
        svc = GraphNodeService(driver)
        result = svc.migrate_relationship_properties("g-1")
        assert result["orphans_detected"] == 2

    @pytest.mark.unit
    def test_zero_violations_clean_graph(self):
        driver = self._make_migration_driver(
            transferred_job_title=0,
            transferred_proficiency=0,
            orphan_records=[],
            cleaned=0,
        )
        svc = GraphNodeService(driver)
        result = svc.migrate_relationship_properties("g-clean")
        assert result["transferred_total"] == 0
        assert result["orphans_detected"] == 0
        assert result["nodes_cleaned"] == 0

    @pytest.mark.unit
    def test_nodes_cleaned_count(self):
        driver = self._make_migration_driver(cleaned=7)
        svc = GraphNodeService(driver)
        result = svc.migrate_relationship_properties("g-1")
        assert result["nodes_cleaned"] == 7

    @pytest.mark.unit
    def test_graph_id_passed_to_all_queries(self):
        """graph_id must appear in every query parameter — multi-tenancy enforcement."""
        driver = self._make_migration_driver()
        svc = GraphNodeService(driver)
        svc.migrate_relationship_properties("target-graph")

        session = driver.session.return_value.__enter__.return_value
        for call in session.run.call_args_list:
            params = call[0][1] if len(call[0]) > 1 else call[1].get("parameters", {})
            assert (
                params.get("graph_id") == "target-graph"
            ), f"graph_id not passed in query: {call}"

    @pytest.mark.unit
    def test_wraps_neo4j_error(self):
        mock_driver = MagicMock()
        mock_session = MagicMock()
        mock_session.__enter__ = MagicMock(return_value=mock_session)
        mock_session.__exit__ = MagicMock(return_value=False)
        from neo4j.exceptions import Neo4jError

        mock_session.run.side_effect = Neo4jError("write failed")
        mock_driver.session.return_value = mock_session

        svc = GraphNodeService(mock_driver)
        with pytest.raises(Exception, match="Migration failed"):
            svc.migrate_relationship_properties("g-1")


# ---------------------------------------------------------------------------
# Tests: _TEMPORAL_INDEX_STATEMENTS — ORA-138
# ---------------------------------------------------------------------------


class TestTemporalIndexStatements:
    """Verify the _TEMPORAL_INDEX_STATEMENTS class-level list includes all required indexes."""

    @pytest.mark.unit
    def test_rel_temporal_composite_idx_present(self):
        """Composite (graph_id, valid_from, valid_to) index must be defined."""
        stmts = " ".join(GraphNodeService._TEMPORAL_INDEX_STATEMENTS)
        assert "rel_temporal_idx" in stmts

    @pytest.mark.unit
    def test_standalone_rel_valid_from_idx_present(self):
        """ORA-138: standalone rel_valid_from_idx required for traversal queries."""
        stmts = " ".join(GraphNodeService._TEMPORAL_INDEX_STATEMENTS)
        assert "rel_valid_from_idx" in stmts

    @pytest.mark.unit
    def test_standalone_rel_valid_to_idx_present(self):
        """ORA-138: standalone rel_valid_to_idx required for traversal queries."""
        stmts = " ".join(GraphNodeService._TEMPORAL_INDEX_STATEMENTS)
        assert "rel_valid_to_idx" in stmts

    @pytest.mark.unit
    def test_standalone_indexes_use_if_not_exists(self):
        """All index statements must be idempotent (IF NOT EXISTS)."""
        for stmt in GraphNodeService._TEMPORAL_INDEX_STATEMENTS:
            assert (
                "IF NOT EXISTS" in stmt
            ), f"Index statement missing IF NOT EXISTS (not idempotent): {stmt!r}"

    @pytest.mark.unit
    def test_ensure_temporal_indexes_calls_session_run_for_each_statement(self):
        """_ensure_temporal_indexes must run every statement in _TEMPORAL_INDEX_STATEMENTS."""
        mock_driver = MagicMock()
        mock_session = MagicMock()
        mock_session.__enter__ = MagicMock(return_value=mock_session)
        mock_session.__exit__ = MagicMock(return_value=False)
        mock_driver.session.return_value = mock_session

        svc = GraphNodeService(mock_driver)
        svc._ensure_temporal_indexes()

        assert mock_session.run.call_count == len(
            GraphNodeService._TEMPORAL_INDEX_STATEMENTS
        ), "session.run() call count must match number of index statements"
