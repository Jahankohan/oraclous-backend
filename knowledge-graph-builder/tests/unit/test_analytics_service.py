"""
Unit tests for GraphAnalyticsService.

Tests community detection mocking, centrality calculation, statistics caching,
pure helper methods, and multi-tenant isolation (graph_id filtering).
All external deps (Neo4j) mocked.
"""
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import UUID

from app.services.analytics_service import GraphAnalyticsService


TEST_GRAPH_ID = UUID("12345678-1234-5678-1234-567812345678")


def _make_service() -> GraphAnalyticsService:
    return GraphAnalyticsService()


def _make_entity(entity_id: str, name: str, labels=None) -> dict:
    return {"id": entity_id, "name": name, "entity_labels": labels or ["__Entity__"]}


# ---------------------------------------------------------------------------
# Tests: _generate_community_id (pure logic)
# ---------------------------------------------------------------------------

class TestGenerateCommunityId:
    @pytest.mark.unit
    def test_returns_string_with_community_prefix(self):
        svc = _make_service()
        members = [{"entity_id": "e1"}, {"entity_id": "e2"}]
        result = svc._generate_community_id(TEST_GRAPH_ID, 42, members)
        assert result.startswith("community_")

    @pytest.mark.unit
    def test_includes_graph_id_in_output(self):
        svc = _make_service()
        members = [{"entity_id": "e1"}]
        result = svc._generate_community_id(TEST_GRAPH_ID, 1, members)
        assert str(TEST_GRAPH_ID) in result

    @pytest.mark.unit
    def test_deterministic_for_same_inputs(self):
        svc = _make_service()
        members = [{"entity_id": "e2"}, {"entity_id": "e1"}]  # unsorted
        r1 = svc._generate_community_id(TEST_GRAPH_ID, 5, members)
        r2 = svc._generate_community_id(TEST_GRAPH_ID, 5, members)
        assert r1 == r2

    @pytest.mark.unit
    def test_member_order_does_not_affect_output(self):
        svc = _make_service()
        members_ab = [{"entity_id": "a"}, {"entity_id": "b"}]
        members_ba = [{"entity_id": "b"}, {"entity_id": "a"}]
        r1 = svc._generate_community_id(TEST_GRAPH_ID, 1, members_ab)
        r2 = svc._generate_community_id(TEST_GRAPH_ID, 1, members_ba)
        assert r1 == r2

    @pytest.mark.unit
    def test_different_community_ids_produce_different_hashes(self):
        svc = _make_service()
        members = [{"entity_id": "e1"}, {"entity_id": "e2"}]
        r1 = svc._generate_community_id(TEST_GRAPH_ID, 1, members)
        r2 = svc._generate_community_id(TEST_GRAPH_ID, 2, members)
        assert r1 != r2

    @pytest.mark.unit
    def test_different_graph_ids_produce_different_hashes(self):
        svc = _make_service()
        members = [{"entity_id": "e1"}]
        graph_a = UUID("aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa")
        graph_b = UUID("bbbbbbbb-bbbb-bbbb-bbbb-bbbbbbbbbbbb")
        r1 = svc._generate_community_id(graph_a, 1, members)
        r2 = svc._generate_community_id(graph_b, 1, members)
        assert r1 != r2


# ---------------------------------------------------------------------------
# Tests: _generate_community_summary (pure logic)
# ---------------------------------------------------------------------------

class TestGenerateCommunitySummary:
    @pytest.mark.unit
    def test_two_members_listed_by_name(self):
        svc = _make_service()
        members = [
            {"entity_name": "Alice", "entity_labels": ["Person"]},
            {"entity_name": "Acme Corp", "entity_labels": ["Company"]},
        ]
        summary = svc._generate_community_summary(members)
        assert "Alice" in summary
        assert "Acme Corp" in summary

    @pytest.mark.unit
    def test_three_members_all_listed(self):
        svc = _make_service()
        members = [
            {"entity_name": "Alice", "entity_labels": ["Person"]},
            {"entity_name": "Bob", "entity_labels": ["Person"]},
            {"entity_name": "Acme", "entity_labels": ["Company"]},
        ]
        summary = svc._generate_community_summary(members)
        assert "Alice" in summary
        assert "Bob" in summary
        assert "Acme" in summary

    @pytest.mark.unit
    def test_four_members_uses_and_others(self):
        svc = _make_service()
        members = [
            {"entity_name": f"Entity{i}", "entity_labels": ["Person"]}
            for i in range(4)
        ]
        summary = svc._generate_community_summary(members)
        assert "others" in summary

    @pytest.mark.unit
    def test_includes_entity_count(self):
        svc = _make_service()
        members = [
            {"entity_name": "Alice", "entity_labels": ["Person"]},
            {"entity_name": "Bob", "entity_labels": ["Person"]},
        ]
        summary = svc._generate_community_summary(members)
        assert "2" in summary

    @pytest.mark.unit
    def test_includes_primary_type_from_labels(self):
        svc = _make_service()
        members = [
            {"entity_name": "Alice", "entity_labels": ["Person", "__Entity__"]},
            {"entity_name": "Bob", "entity_labels": ["Person", "__Entity__"]},
        ]
        summary = svc._generate_community_summary(members)
        assert "person" in summary.lower()

    @pytest.mark.unit
    def test_ignores_entity_label_when_counting_types(self):
        svc = _make_service()
        members = [
            {"entity_name": "Alice", "entity_labels": ["__Entity__"]},
            {"entity_name": "Bob", "entity_labels": ["__Entity__"]},
        ]
        summary = svc._generate_community_summary(members)
        # Should fall back to generic summary (no primary type)
        assert "related entities" in summary.lower()

    @pytest.mark.unit
    def test_empty_members_returns_string(self):
        svc = _make_service()
        summary = svc._generate_community_summary([])
        assert isinstance(summary, str)


# ---------------------------------------------------------------------------
# Tests: get_community_context — early returns and fallback
# ---------------------------------------------------------------------------

class TestGetCommunityContext:
    @pytest.mark.unit
    async def test_empty_entities_returns_empty_communities(self):
        svc = _make_service()
        result = await svc.get_community_context([], TEST_GRAPH_ID)
        assert result == {"communities": []}

    @pytest.mark.unit
    async def test_gds_failure_falls_back_to_simple(self):
        svc = _make_service()
        svc.get_simple_community_context = AsyncMock(return_value={"communities": ["fallback"]})

        with patch("app.services.analytics_service.neo4j_client") as mock_client:
            mock_client.execute_query = AsyncMock(side_effect=Exception("GDS not installed"))

            result = await svc.get_community_context(
                [_make_entity("e1", "Alice")],
                TEST_GRAPH_ID
            )

        svc.get_simple_community_context.assert_awaited_once()
        assert result == {"communities": ["fallback"]}

    @pytest.mark.unit
    async def test_graph_id_passed_to_neo4j_query(self):
        """Multi-tenant isolation: graph_id must be present in every query."""
        svc = _make_service()

        captured_params = []

        async def capture_execute(query, params):
            captured_params.append(params)
            return []

        with patch("app.services.analytics_service.neo4j_client") as mock_client:
            mock_client.execute_query = AsyncMock(side_effect=capture_execute)

            await svc.get_community_context(
                [_make_entity("e1", "Alice")],
                TEST_GRAPH_ID
            )

        # At least one call should contain graph_id
        assert any("graph_id" in p for p in captured_params)
        assert any(str(TEST_GRAPH_ID) in str(p.get("graph_id", "")) for p in captured_params)


# ---------------------------------------------------------------------------
# Tests: get_simple_community_context
# ---------------------------------------------------------------------------

class TestGetSimpleCommunityContext:
    @pytest.mark.unit
    async def test_empty_entities_returns_empty_communities(self):
        svc = _make_service()
        result = await svc.get_simple_community_context([], TEST_GRAPH_ID)
        assert result == {"communities": []}

    @pytest.mark.unit
    async def test_graph_id_passed_to_query(self):
        svc = _make_service()

        captured_params = []

        async def capture_execute(query, params):
            captured_params.append(params)
            return []

        with patch("app.services.analytics_service.neo4j_client") as mock_client:
            mock_client.execute_query = AsyncMock(side_effect=capture_execute)

            await svc.get_simple_community_context(
                [_make_entity("e1", "Alice")],
                TEST_GRAPH_ID
            )

        assert len(captured_params) > 0
        assert str(TEST_GRAPH_ID) in str(captured_params[0].get("graph_id", ""))

    @pytest.mark.unit
    async def test_entity_ids_extracted_from_entities(self):
        svc = _make_service()

        captured_params = []

        async def capture_execute(query, params):
            captured_params.append(params)
            return []

        with patch("app.services.analytics_service.neo4j_client") as mock_client:
            mock_client.execute_query = AsyncMock(side_effect=capture_execute)

            entities = [
                _make_entity("entity-1", "Alice"),
                _make_entity("entity-2", "Bob"),
            ]
            await svc.get_simple_community_context(entities, TEST_GRAPH_ID)

        entity_ids = captured_params[0].get("entity_ids", [])
        assert "entity-1" in entity_ids
        assert "entity-2" in entity_ids

    @pytest.mark.unit
    async def test_returns_communities_from_query_result(self):
        svc = _make_service()

        mock_result = [
            {
                "entity_name": "Alice",
                "hub_name": "Acme Corp",
                "community_members": [{"id": "e2", "name": "Bob"}]
            }
        ]

        with patch("app.services.analytics_service.neo4j_client") as mock_client:
            mock_client.execute_query = AsyncMock(return_value=mock_result)

            result = await svc.get_simple_community_context(
                [_make_entity("e1", "Alice")],
                TEST_GRAPH_ID
            )

        assert len(result["communities"]) == 1
        assert result["communities"][0]["entity"] == "Alice"
        assert result["communities"][0]["hub"] == "Acme Corp"


# ---------------------------------------------------------------------------
# Tests: cached statistics
# ---------------------------------------------------------------------------

class TestCachedStatistics:
    @pytest.mark.unit
    def test_get_cached_statistics_returns_none_when_not_cached(self):
        svc = _make_service()
        result = svc.get_cached_statistics(TEST_GRAPH_ID)
        assert result is None

    @pytest.mark.unit
    def test_get_cached_statistics_returns_cached_data(self):
        svc = _make_service()
        svc.cached_statistics[str(TEST_GRAPH_ID)] = {"node_count": 42}
        result = svc.get_cached_statistics(TEST_GRAPH_ID)
        assert result["node_count"] == 42

    @pytest.mark.unit
    async def test_precompute_caches_statistics(self):
        svc = _make_service()
        svc.get_graph_statistics = AsyncMock(return_value={"node_count": 10, "relationship_count": 5})

        await svc.precompute_and_cache_statistics(TEST_GRAPH_ID)

        cached = svc.get_cached_statistics(TEST_GRAPH_ID)
        assert cached is not None
        assert cached["node_count"] == 10
        assert "cached_at" in cached

    @pytest.mark.unit
    async def test_precompute_stores_error_on_failure(self):
        svc = _make_service()
        svc.get_graph_statistics = AsyncMock(side_effect=Exception("Neo4j timeout"))

        await svc.precompute_and_cache_statistics(TEST_GRAPH_ID)

        cached = svc.get_cached_statistics(TEST_GRAPH_ID)
        assert cached is not None
        assert "error" in cached
        assert cached["node_count"] == 0

    @pytest.mark.unit
    def test_cached_statistics_keyed_by_graph_id_string(self):
        """Verify isolation — different graph IDs have separate caches."""
        svc = _make_service()
        graph_a = UUID("aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa")
        graph_b = UUID("bbbbbbbb-bbbb-bbbb-bbbb-bbbbbbbbbbbb")

        svc.cached_statistics[str(graph_a)] = {"node_count": 5}
        svc.cached_statistics[str(graph_b)] = {"node_count": 99}

        assert svc.get_cached_statistics(graph_a)["node_count"] == 5
        assert svc.get_cached_statistics(graph_b)["node_count"] == 99


# ---------------------------------------------------------------------------
# Tests: comprehensive_graph_analysis structure
# ---------------------------------------------------------------------------

class TestComprehensiveGraphAnalysis:
    @pytest.mark.unit
    async def test_returns_graph_id_in_result(self):
        svc = _make_service()
        svc.get_neighborhood_context = AsyncMock(return_value={"neighborhoods": []})
        svc.get_community_context = AsyncMock(return_value={"communities": []})
        svc.get_influential_context = AsyncMock(return_value={"influential": []})
        svc.get_pathway_context = AsyncMock(return_value={"paths": []})
        svc.get_temporal_context = AsyncMock(return_value={"temporal": []})
        svc.get_graph_statistics = AsyncMock(return_value={"node_count": 0})

        result = await svc.comprehensive_graph_analysis([], TEST_GRAPH_ID)

        assert result["graph_id"] == str(TEST_GRAPH_ID)

    @pytest.mark.unit
    async def test_entities_count_in_result(self):
        svc = _make_service()
        svc.get_neighborhood_context = AsyncMock(return_value={})
        svc.get_community_context = AsyncMock(return_value={})
        svc.get_influential_context = AsyncMock(return_value={})
        svc.get_graph_statistics = AsyncMock(return_value={})

        entities = [_make_entity("e1", "Alice"), _make_entity("e2", "Bob")]
        result = await svc.comprehensive_graph_analysis(entities, TEST_GRAPH_ID)

        assert result["entities_analyzed"] == 2

    @pytest.mark.unit
    async def test_skips_pathway_analysis_when_fewer_than_2_entities(self):
        svc = _make_service()
        svc.get_neighborhood_context = AsyncMock(return_value={})
        svc.get_community_context = AsyncMock(return_value={})
        svc.get_influential_context = AsyncMock(return_value={})
        svc.get_pathway_context = AsyncMock(return_value={})
        svc.get_temporal_context = AsyncMock(return_value={})
        svc.get_graph_statistics = AsyncMock(return_value={})

        await svc.comprehensive_graph_analysis(
            [_make_entity("e1", "Alice")],  # only 1 entity
            TEST_GRAPH_ID,
            include_pathways=True
        )

        svc.get_pathway_context.assert_not_awaited()

    @pytest.mark.unit
    async def test_skips_community_when_disabled(self):
        svc = _make_service()
        svc.get_neighborhood_context = AsyncMock(return_value={})
        svc.get_community_context = AsyncMock(return_value={})
        svc.get_influential_context = AsyncMock(return_value={})
        svc.get_graph_statistics = AsyncMock(return_value={})

        await svc.comprehensive_graph_analysis(
            [_make_entity("e1", "A"), _make_entity("e2", "B")],
            TEST_GRAPH_ID,
            include_communities=False
        )

        svc.get_community_context.assert_not_awaited()

    @pytest.mark.unit
    async def test_includes_analysis_timestamp(self):
        svc = _make_service()
        svc.get_neighborhood_context = AsyncMock(return_value={})
        svc.get_community_context = AsyncMock(return_value={})
        svc.get_influential_context = AsyncMock(return_value={})
        svc.get_graph_statistics = AsyncMock(return_value={})

        result = await svc.comprehensive_graph_analysis([], TEST_GRAPH_ID)

        assert "analysis_timestamp" in result


# ---------------------------------------------------------------------------
# Tests: multi-tenant isolation
# ---------------------------------------------------------------------------

class TestMultiTenantIsolation:
    @pytest.mark.unit
    async def test_get_community_context_always_passes_graph_id(self):
        svc = _make_service()
        graph_id_used = []

        async def capture(query, params):
            graph_id_used.append(params.get("graph_id"))
            return []

        with patch("app.services.analytics_service.neo4j_client") as mock_client:
            mock_client.execute_query = AsyncMock(side_effect=capture)

            specific_graph = UUID("deadbeef-dead-beef-dead-beefdeadbeef")
            await svc.get_simple_community_context(
                [_make_entity("e1", "Alice")],
                specific_graph
            )

        assert any(str(specific_graph) in str(gid) for gid in graph_id_used)

    @pytest.mark.unit
    async def test_different_graphs_dont_share_cache(self):
        svc = _make_service()
        graph_1 = UUID("11111111-1111-1111-1111-111111111111")
        graph_2 = UUID("22222222-2222-2222-2222-222222222222")

        svc.cached_statistics[str(graph_1)] = {"node_count": 10}

        # graph_2 should not see graph_1's cache
        assert svc.get_cached_statistics(graph_2) is None
        assert svc.get_cached_statistics(graph_1) is not None
