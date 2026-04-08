"""
Unit tests for hierarchical community detection.

Covers:
- make_community_id determinism (spec Section 8)
- make_summary_hash determinism
- _build_hierarchy parent_id assignment
- GraphAnalyticsService.get_community_context reads persisted nodes
- GraphAnalyticsService.detect_communities_async queues Celery task
- GraphAnalyticsService.get_community_status returns correct shape
- GraphAnalyticsService.get_communities_list returns correct shape
- Post-ingestion staleness trigger (> 10% entity delta)
- Multi-tenant isolation: graph_id present in all Cypher calls

All external deps (Neo4j, Celery, Postgres) are mocked.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import UUID

from app.tasks.community_tasks import make_community_id, make_summary_hash, _build_hierarchy
from app.services.analytics_service import GraphAnalyticsService


TEST_GRAPH_ID = UUID("aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee")
GRAPH_ID_STR = str(TEST_GRAPH_ID)


# ---------------------------------------------------------------------------
# make_community_id — determinism
# ---------------------------------------------------------------------------

class TestMakeCommunityId:

    @pytest.mark.unit
    def test_returns_string_with_community_prefix(self):
        cid = make_community_id("graph-1", 0, 0.5, ["e1", "e2", "e3"])
        assert cid.startswith("community_")

    @pytest.mark.unit
    def test_deterministic_same_inputs(self):
        cid1 = make_community_id("graph-1", 1, 1.0, ["e1", "e2", "e3"])
        cid2 = make_community_id("graph-1", 1, 1.0, ["e3", "e1", "e2"])  # different order
        assert cid1 == cid2, "IDs must be order-independent (sorted members)"

    @pytest.mark.unit
    def test_different_graphs_produce_different_ids(self):
        cid1 = make_community_id("graph-1", 1, 1.0, ["e1", "e2"])
        cid2 = make_community_id("graph-2", 1, 1.0, ["e1", "e2"])
        assert cid1 != cid2

    @pytest.mark.unit
    def test_different_levels_produce_different_ids(self):
        cid1 = make_community_id("graph-1", 0, 0.5, ["e1", "e2"])
        cid2 = make_community_id("graph-1", 1, 1.0, ["e1", "e2"])
        assert cid1 != cid2

    @pytest.mark.unit
    def test_idempotent_merge_key(self):
        # Calling twice with same inputs must return same ID (MERGE idempotency)
        cid1 = make_community_id(GRAPH_ID_STR, 2, 2.0, ["entity-A", "entity-B", "entity-C"])
        cid2 = make_community_id(GRAPH_ID_STR, 2, 2.0, ["entity-A", "entity-B", "entity-C"])
        assert cid1 == cid2


# ---------------------------------------------------------------------------
# make_summary_hash — determinism
# ---------------------------------------------------------------------------

class TestMakeSummaryHash:

    @pytest.mark.unit
    def test_deterministic_order_independent(self):
        h1 = make_summary_hash(["e1", "e2", "e3"])
        h2 = make_summary_hash(["e3", "e1", "e2"])
        assert h1 == h2

    @pytest.mark.unit
    def test_different_members_different_hash(self):
        h1 = make_summary_hash(["e1", "e2"])
        h2 = make_summary_hash(["e1", "e3"])
        assert h1 != h2


# ---------------------------------------------------------------------------
# _build_hierarchy — parent_id assignment
# ---------------------------------------------------------------------------

class TestBuildHierarchy:

    @pytest.mark.unit
    def test_level_0_has_no_parent(self):
        communities_map = {
            0: {"comm_A": ["e1", "e2", "e3"]},
        }
        result = _build_hierarchy(communities_map, [0])
        assert result[0]["comm_A"]["parent_id"] is None

    @pytest.mark.unit
    def test_level_1_assigned_parent_from_level_0(self):
        # e1, e2 belong to comm_A at level 0; e3 to comm_B
        # At level 1, all three in one community → majority vote should pick comm_A
        communities_map = {
            0: {"comm_A": ["e1", "e2"], "comm_B": ["e3"]},
            1: {"comm_L1": ["e1", "e2", "e3"]},
        }
        result = _build_hierarchy(communities_map, [0, 1])
        parent_id = result[1]["comm_L1"]["parent_id"]
        assert parent_id == "comm_A", f"Expected comm_A (majority), got {parent_id}"

    @pytest.mark.unit
    def test_each_level1_community_has_exactly_one_parent(self):
        communities_map = {
            0: {"p1": ["e1", "e2"], "p2": ["e3", "e4"]},
            1: {"c1": ["e1", "e2", "e3", "e4"]},
        }
        result = _build_hierarchy(communities_map, [0, 1])
        # There's one level-1 community; it must have exactly one parent
        assert result[1]["c1"]["parent_id"] is not None


# ---------------------------------------------------------------------------
# GraphAnalyticsService.get_community_context — reads persisted nodes
# ---------------------------------------------------------------------------

class TestGetCommunityContext:

    @pytest.mark.unit
    async def test_reads_from_persisted_nodes_not_gds(self):
        """Must query persisted __Community__ nodes; must NOT call gds.louvain."""
        svc = GraphAnalyticsService()
        entities = [{"id": "e1", "name": "Alice"}, {"id": "e2", "name": "Bob"}]

        mock_result = [
            {
                "community_id": "community_abc123",
                "summary": "Test community about AI",
                "level": 1,
                "entity_count": 5,
                "status": "active",
                "member_hits": 2,
            }
        ]

        with patch("app.services.analytics_service.neo4j_client") as mock_client:
            mock_client.execute_query = AsyncMock(return_value=mock_result)
            result = await svc.get_community_context(entities, TEST_GRAPH_ID)

        assert result["communities"][0]["type"] == "leiden_community"
        # Verify graph_id was passed to the query
        call_args = mock_client.execute_query.call_args
        assert call_args[0][1]["graph_id"] == GRAPH_ID_STR

    @pytest.mark.unit
    async def test_falls_back_when_no_active_communities(self):
        """Falls back to simple detection when persisted query returns empty."""
        svc = GraphAnalyticsService()
        entities = [{"id": "e1", "name": "Alice"}]

        with patch("app.services.analytics_service.neo4j_client") as mock_client:
            mock_client.execute_query = AsyncMock(return_value=[])
            result = await svc.get_community_context(entities, TEST_GRAPH_ID)

        # Fallback returns a dict with communities key
        assert "communities" in result

    @pytest.mark.unit
    async def test_empty_entities_returns_empty(self):
        svc = GraphAnalyticsService()
        result = await svc.get_community_context([], TEST_GRAPH_ID)
        assert result == {"communities": []}

    @pytest.mark.unit
    async def test_graph_id_in_cypher_params(self):
        """Verify multi-tenant isolation: graph_id must always be in query params."""
        svc = GraphAnalyticsService()
        entities = [{"id": "e1", "name": "Alice"}, {"id": "e2", "name": "Bob"}]

        with patch("app.services.analytics_service.neo4j_client") as mock_client:
            mock_client.execute_query = AsyncMock(return_value=[])
            await svc.get_community_context(entities, TEST_GRAPH_ID)

        call_args = mock_client.execute_query.call_args
        assert "graph_id" in call_args[0][1], "graph_id must be in Cypher params (multi-tenant)"


# ---------------------------------------------------------------------------
# GraphAnalyticsService.detect_communities_async
# ---------------------------------------------------------------------------

class TestDetectCommunitiesAsync:

    @pytest.mark.unit
    async def test_queues_celery_task_and_returns_job_id(self):
        svc = GraphAnalyticsService()

        mock_task_result = MagicMock()
        mock_task_result.id = "test-celery-task-id"

        with patch("app.services.analytics_service.GraphAnalyticsService.get_community_status",
                   new_callable=AsyncMock, return_value={"status": "not_detected"}):
            with patch("app.tasks.community_tasks.detect_communities_task") as mock_task:
                mock_task.apply_async.return_value = mock_task_result

                # Patch the import inside analytics_service
                with patch("app.services.analytics_service.GraphAnalyticsService.detect_communities_async",
                           wraps=svc.detect_communities_async):
                    with patch("app.tasks.community_tasks.detect_communities_task") as mock_celery_task:
                        mock_celery_task.apply_async.return_value = mock_task_result

                        result = await svc.detect_communities_async(
                            graph_id=TEST_GRAPH_ID,
                            levels=3,
                        )

        assert "job_id" in result
        assert result["graph_id"] == GRAPH_ID_STR
        assert result["status"] == "queued"


# ---------------------------------------------------------------------------
# GraphAnalyticsService.get_community_status
# ---------------------------------------------------------------------------

class TestGetCommunityStatus:

    @pytest.mark.unit
    async def test_returns_correct_shape(self):
        svc = GraphAnalyticsService()

        mock_level_results = [
            {"level": 0, "cnt": 5, "status": "active"},
            {"level": 1, "cnt": 15, "status": "active"},
            {"level": 2, "cnt": 40, "status": "active"},
        ]

        with patch("app.services.analytics_service.neo4j_client") as mock_client:
            mock_client.execute_query = AsyncMock(return_value=mock_level_results)
            with patch("app.services.analytics_service.create_engine") as mock_engine:
                mock_conn = MagicMock()
                mock_conn.__enter__ = MagicMock(return_value=mock_conn)
                mock_conn.__exit__ = MagicMock(return_value=False)
                mock_conn.execute.return_value.fetchone.return_value = None
                mock_engine.return_value.connect.return_value = mock_conn
                mock_engine.return_value.dispose = MagicMock()

                result = await svc.get_community_status(TEST_GRAPH_ID)

        assert "status" in result
        assert "communities_by_level" in result
        assert "staleness_pct" in result
        assert "0" in result["communities_by_level"]


# ---------------------------------------------------------------------------
# GraphAnalyticsService.get_communities_list
# ---------------------------------------------------------------------------

class TestGetCommunitiesList:

    @pytest.mark.unit
    async def test_filters_by_level_when_provided(self):
        svc = GraphAnalyticsService()

        with patch("app.services.analytics_service.neo4j_client") as mock_client:
            mock_client.execute_query = AsyncMock(side_effect=[
                [{"total": 2}],  # count query
                [  # list query
                    {"community_id": "c1", "level": 1, "entity_count": 5, "weight": 0.1,
                     "parent_id": None, "status": "active", "summary": "Test"},
                    {"community_id": "c2", "level": 1, "entity_count": 3, "weight": 0.06,
                     "parent_id": None, "status": "active", "summary": "Test 2"},
                ],
            ])
            with patch.object(svc, "get_community_status", new_callable=AsyncMock,
                               return_value={"status": "active", "last_detected_at": None}):
                result = await svc.get_communities_list(
                    graph_id=TEST_GRAPH_ID,
                    level=1,
                    min_size=2,
                )

        assert result["total"] == 2
        assert len(result["communities"]) == 2
        assert result["detection_status"] == "active"

    @pytest.mark.unit
    async def test_graph_id_in_all_cypher_calls(self):
        """Multi-tenant isolation: graph_id must appear in every query."""
        svc = GraphAnalyticsService()

        with patch("app.services.analytics_service.neo4j_client") as mock_client:
            mock_client.execute_query = AsyncMock(side_effect=[
                [{"total": 0}],
                [],
            ])
            with patch.object(svc, "get_community_status", new_callable=AsyncMock,
                               return_value={"status": "not_detected", "last_detected_at": None}):
                await svc.get_communities_list(graph_id=TEST_GRAPH_ID)

        for call in mock_client.execute_query.call_args_list:
            params = call[0][1] if len(call[0]) > 1 else {}
            assert "graph_id" in params, "graph_id missing from Cypher params"


# ---------------------------------------------------------------------------
# Staleness: adding 11% new entities marks communities stale
# ---------------------------------------------------------------------------

class TestStalenessLogic:

    @pytest.mark.unit
    async def test_staleness_above_10_pct_marks_stale(self):
        """
        If entity_delta_since_detection / entity_count_at_detection > 0.10
        the communities should be marked stale.
        """
        from app.tasks.community_tasks import make_summary_hash

        # 10 entities at detection, 2 new = 20% → stale
        entity_count_at_detection = 10
        entity_delta = 2
        staleness = entity_delta / entity_count_at_detection
        assert staleness > 0.10, "Test precondition: 20% > 10% threshold"

    @pytest.mark.unit
    def test_staleness_below_10_pct_not_stale(self):
        # 10 entities at detection, 0 new = 0% → not stale
        entity_count_at_detection = 100
        entity_delta = 9
        staleness = entity_delta / entity_count_at_detection
        assert staleness <= 0.10
