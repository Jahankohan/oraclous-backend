"""Unit tests for get_entity_neighborhood() (ORA-358 / ORA-378).

Tests cover:
- 1-hop, 2-hop, 3-hop traversal
- edge_type filter
- >300-node truncation (keep focal + highest-degree)
- entity not found (returns empty response)
- hops validation (out-of-range raises ValueError)
"""

from unittest.mock import AsyncMock, MagicMock

import pytest

from app.services.analytics_service import get_entity_neighborhood

# ---------------------------------------------------------------------------
# Test helpers
# ---------------------------------------------------------------------------


def _make_result(rows: list[dict]) -> AsyncMock:
    """Build a mock Neo4j result cursor."""
    result = AsyncMock()
    result.single = AsyncMock(return_value=rows[0] if rows else None)
    result.data = AsyncMock(return_value=list(rows))
    return result


def _make_driver(*result_groups: list[dict]):
    """Build a mock AsyncDriver whose session.run() returns each group in turn.

    Each positional argument is a list[dict] returned for one session.run() call.
    Calls beyond the supplied groups return an empty list.
    """
    results = [_make_result(g) for g in result_groups]
    run_idx = [0]

    async def fake_run(query, params=None):
        idx = run_idx[0]
        run_idx[0] += 1
        return results[idx] if idx < len(results) else _make_result([])

    mock_session = AsyncMock()
    mock_session.run = fake_run

    mock_driver = MagicMock()
    mock_driver.session.return_value.__aenter__ = AsyncMock(return_value=mock_session)
    mock_driver.session.return_value.__aexit__ = AsyncMock(return_value=None)
    return mock_driver


def _entity_row(
    node_id: str,
    name: str = "Entity",
    degree: int = 1,
    labels: list[str] | None = None,
    node_type: str | None = None,
    community_id: str | None = None,
) -> dict:
    return {
        "node_id": node_id,
        "node_labels": labels or ["__Entity__"],
        "node_type": node_type,
        "degree": degree,
        "community_id": community_id,
        "node_props": {"name": name},
    }


def _edge_row(
    rel_id: str,
    source: str,
    target: str,
    rel_type: str = "RELATES_TO",
    weight: float = 1.0,
) -> dict:
    return {
        "id": rel_id,
        "source": source,
        "target": target,
        "type": rel_type,
        "weight": weight,
    }


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestHopsValidation:
    @pytest.mark.unit
    async def test_hops_0_raises(self):
        driver = _make_driver()
        with pytest.raises(ValueError, match="hops must be"):
            await get_entity_neighborhood(driver, graph_id="g1", entity_id="e1", hops=0)

    @pytest.mark.unit
    async def test_hops_4_raises(self):
        driver = _make_driver()
        with pytest.raises(ValueError, match="hops must be"):
            await get_entity_neighborhood(driver, graph_id="g1", entity_id="e1", hops=4)

    @pytest.mark.unit
    async def test_hops_negative_raises(self):
        driver = _make_driver()
        with pytest.raises(ValueError):
            await get_entity_neighborhood(
                driver, graph_id="g1", entity_id="e1", hops=-1
            )


class TestEntityNotFound:
    @pytest.mark.unit
    async def test_returns_empty_when_focal_not_found(self):
        # Pre-query returns no rows → focal not in this graph
        driver = _make_driver([])  # empty focal result

        result = await get_entity_neighborhood(
            driver, graph_id="g1", entity_id="nonexistent"
        )

        assert result == {"nodes": [], "edges": [], "truncated": False}


class TestOneHop:
    @pytest.mark.unit
    async def test_1_hop_returns_focal_and_direct_neighbors(self):
        focal_row = {"focal_id": "e1"}
        node_rows = [
            _entity_row("e1", name="Focal", degree=2),
            _entity_row("e2", name="Neighbor A", degree=1),
            _entity_row("e3", name="Neighbor B", degree=1),
        ]
        edge_rows = [
            _edge_row("r1", "e1", "e2"),
            _edge_row("r2", "e1", "e3"),
        ]

        driver = _make_driver([focal_row], node_rows, edge_rows)
        result = await get_entity_neighborhood(
            driver, graph_id="g1", entity_id="e1", hops=1
        )

        node_ids = {n["id"] for n in result["nodes"]}
        assert node_ids == {"e1", "e2", "e3"}
        assert len(result["edges"]) == 2
        assert result["truncated"] is False

    @pytest.mark.unit
    async def test_1_hop_focal_only_when_no_neighbors(self):
        focal_row = {"focal_id": "e1"}
        node_rows = [_entity_row("e1", name="Isolated", degree=0)]
        edge_rows: list[dict] = []

        driver = _make_driver([focal_row], node_rows, edge_rows)
        result = await get_entity_neighborhood(
            driver, graph_id="g1", entity_id="e1", hops=1
        )

        assert len(result["nodes"]) == 1
        assert result["nodes"][0]["id"] == "e1"
        assert result["edges"] == []
        assert result["truncated"] is False


class TestTwoHop:
    @pytest.mark.unit
    async def test_2_hop_default_includes_indirect_neighbors(self):
        focal_row = {"focal_id": "e1"}
        # e2 is direct neighbor; e3 is reached via e2 (2 hops)
        node_rows = [
            _entity_row("e1", name="Focal", degree=3),
            _entity_row("e2", name="Direct", degree=2),
            _entity_row("e3", name="Indirect", degree=1),
        ]
        edge_rows = [
            _edge_row("r1", "e1", "e2"),
            _edge_row("r2", "e2", "e3"),
        ]

        driver = _make_driver([focal_row], node_rows, edge_rows)
        result = await get_entity_neighborhood(driver, graph_id="g1", entity_id="e1")

        assert {n["id"] for n in result["nodes"]} == {"e1", "e2", "e3"}
        assert result["truncated"] is False


class TestThreeHop:
    @pytest.mark.unit
    async def test_3_hop_traversal(self):
        focal_row = {"focal_id": "e1"}
        node_rows = [
            _entity_row("e1", degree=1),
            _entity_row("e2", degree=2),
            _entity_row("e3", degree=2),
            _entity_row("e4", degree=1),
        ]
        edge_rows = [
            _edge_row("r1", "e1", "e2"),
            _edge_row("r2", "e2", "e3"),
            _edge_row("r3", "e3", "e4"),
        ]

        driver = _make_driver([focal_row], node_rows, edge_rows)
        result = await get_entity_neighborhood(
            driver, graph_id="g1", entity_id="e1", hops=3
        )

        assert {n["id"] for n in result["nodes"]} == {"e1", "e2", "e3", "e4"}
        assert len(result["edges"]) == 3


class TestEdgeTypeFilter:
    @pytest.mark.unit
    async def test_edge_type_filter_limits_returned_edges(self):
        focal_row = {"focal_id": "e1"}
        node_rows = [
            _entity_row("e1", degree=2),
            _entity_row("e2", degree=1),
            _entity_row("e3", degree=1),
        ]
        # Only RELATES_TO edges survive the filter; HAS_PART is excluded
        edge_rows = [
            _edge_row("r1", "e1", "e2", rel_type="RELATES_TO"),
        ]

        driver = _make_driver([focal_row], node_rows, edge_rows)
        result = await get_entity_neighborhood(
            driver,
            graph_id="g1",
            entity_id="e1",
            edge_types=["RELATES_TO"],
        )

        # All nodes are returned; only RELATES_TO edges are in the response
        assert {n["id"] for n in result["nodes"]} == {"e1", "e2", "e3"}
        assert len(result["edges"]) == 1
        assert result["edges"][0]["type"] == "RELATES_TO"

    @pytest.mark.unit
    async def test_no_edge_type_filter_returns_all_edges(self):
        focal_row = {"focal_id": "e1"}
        node_rows = [_entity_row("e1", degree=2), _entity_row("e2", degree=1)]
        edge_rows = [_edge_row("r1", "e1", "e2", rel_type="OWNS")]

        driver = _make_driver([focal_row], node_rows, edge_rows)
        result = await get_entity_neighborhood(
            driver, graph_id="g1", entity_id="e1", edge_types=None
        )

        assert len(result["edges"]) == 1


class TestTruncation:
    @pytest.mark.unit
    async def test_over_300_nodes_is_truncated(self):
        focal_row = {"focal_id": "focal"}
        # 301 nodes: focal + 300 neighbors, all with degree 1 except focal
        node_rows = [_entity_row("focal", degree=300)]
        node_rows += [_entity_row(f"e{i}", degree=i % 10 + 1) for i in range(300)]
        edge_rows: list[dict] = []

        driver = _make_driver([focal_row], node_rows, edge_rows)
        result = await get_entity_neighborhood(
            driver, graph_id="g1", entity_id="focal", hops=1
        )

        assert result["truncated"] is True
        assert len(result["nodes"]) == 300

    @pytest.mark.unit
    async def test_truncation_keeps_focal_node(self):
        focal_row = {"focal_id": "focal"}
        node_rows = [_entity_row("focal", degree=1)]
        node_rows += [_entity_row(f"e{i}", degree=i) for i in range(300)]
        edge_rows: list[dict] = []

        driver = _make_driver([focal_row], node_rows, edge_rows)
        result = await get_entity_neighborhood(
            driver, graph_id="g1", entity_id="focal", hops=2
        )

        assert result["truncated"] is True
        focal_ids = [n["id"] for n in result["nodes"] if n["id"] == "focal"]
        assert len(focal_ids) == 1, "focal node must always be present after truncation"

    @pytest.mark.unit
    async def test_truncation_keeps_highest_degree_neighbors(self):
        focal_row = {"focal_id": "focal"}
        # 301 nodes total; neighbors have degrees 1..300
        node_rows = [_entity_row("focal", degree=300)]
        node_rows += [_entity_row(f"e{i}", degree=i + 1) for i in range(300)]
        edge_rows: list[dict] = []

        driver = _make_driver([focal_row], node_rows, edge_rows)
        result = await get_entity_neighborhood(
            driver, graph_id="g1", entity_id="focal", hops=1
        )

        # The 299 highest-degree neighbors should be e1..e299 (degree 300..2)
        # e0 (degree 1) should be dropped
        non_focal = [n for n in result["nodes"] if n["id"] != "focal"]
        degrees = [n["degree"] for n in non_focal]
        # All retained neighbors should have degree >= 2
        assert min(degrees) >= 2

    @pytest.mark.unit
    async def test_exactly_300_nodes_not_truncated(self):
        focal_row = {"focal_id": "focal"}
        node_rows = [_entity_row("focal", degree=299)]
        node_rows += [_entity_row(f"e{i}", degree=1) for i in range(299)]
        edge_rows: list[dict] = []

        driver = _make_driver([focal_row], node_rows, edge_rows)
        result = await get_entity_neighborhood(
            driver, graph_id="g1", entity_id="focal", hops=2
        )

        assert result["truncated"] is False
        assert len(result["nodes"]) == 300


class TestResponseShape:
    @pytest.mark.unit
    async def test_node_shape_matches_graph_data_response(self):
        focal_row = {"focal_id": "e1"}
        node_rows = [
            {
                "node_id": "e1",
                "node_labels": ["__Entity__", "Person"],
                "node_type": "Person",
                "degree": 3,
                "community_id": "c-1",
                "node_props": {"name": "Alice", "age": 30},
            }
        ]
        edge_rows: list[dict] = []

        driver = _make_driver([focal_row], node_rows, edge_rows)
        result = await get_entity_neighborhood(
            driver, graph_id="g1", entity_id="e1", hops=1
        )

        node = result["nodes"][0]
        assert node["id"] == "e1"
        assert node["label"] == "Person"  # first non-reserved label
        assert node["type"] == "Person"
        assert node["community_id"] == "c-1"
        assert node["degree"] == 3
        assert "age" in node["properties"]

    @pytest.mark.unit
    async def test_embedding_dropped_from_properties(self):
        focal_row = {"focal_id": "e1"}
        node_rows = [
            {
                "node_id": "e1",
                "node_labels": ["__Entity__"],
                "node_type": None,
                "degree": 0,
                "community_id": None,
                "node_props": {"name": "X", "embedding": [0.1, 0.2], "graph_id": "g1"},
            }
        ]
        driver = _make_driver([focal_row], node_rows, [])
        result = await get_entity_neighborhood(
            driver, graph_id="g1", entity_id="e1", hops=1
        )

        props = result["nodes"][0]["properties"]
        assert "embedding" not in props
        assert "graph_id" not in props
        assert "id" not in props
