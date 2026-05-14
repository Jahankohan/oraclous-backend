"""Unit tests for SimilarityService (STORY-7).

Verifies the SIMILAR_TO edge generator:
- Per-target Cypher uses the right label + index name
- Threshold and top_k passed via params
- Tenant-isolation filter (graph_id) on source, neighbour, AND edge
- Pair-dedup via ``a.id < b.id``
- force_rebuild deletes existing edges first
- Unknown target raises ValueError
- IngestionJob counter only touched when job_id provided
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.services.similarity_service import (
    SimilarityReport,
    SimilarityService,
)


def _make_report_rows(edges: int = 0, processed: int = 0) -> list[dict]:
    """Build the rows a successful build_similarities call returns."""
    return [{"edges_created": edges, "nodes_processed": processed}]


class TestUnknownTarget:
    @pytest.mark.unit
    async def test_unknown_target_raises_value_error(self):
        svc = SimilarityService()
        with pytest.raises(ValueError, match="Unknown similarity target"):
            await svc.build_similarities("g1", target="bogus")


class TestPerTargetCypher:
    @pytest.mark.unit
    async def test_chunks_target_uses_chunk_label_and_text_index(self):
        mock_client = MagicMock()
        mock_client.execute_query = AsyncMock(
            return_value=_make_report_rows(edges=42, processed=100)
        )
        with patch("app.services.similarity_service.neo4j_client", mock_client):
            svc = SimilarityService()
            report = await svc.build_similarities("g1", target="chunks")
        # One execute_query call (the target loop) — no delete because
        # force_rebuild defaults False.
        cypher = mock_client.execute_query.call_args[0][0]
        assert "`Chunk`" in cypher
        assert "text_embeddings_primary" in cypher
        assert report.chunk_edges_created == 42
        assert report.entity_edges_created == 0

    @pytest.mark.unit
    async def test_entities_target_uses_entity_label_and_entity_index(self):
        mock_client = MagicMock()
        mock_client.execute_query = AsyncMock(
            return_value=_make_report_rows(edges=7, processed=18)
        )
        with patch("app.services.similarity_service.neo4j_client", mock_client):
            svc = SimilarityService()
            report = await svc.build_similarities("g1", target="entities")
        cypher = mock_client.execute_query.call_args[0][0]
        assert "`__Entity__`" in cypher
        assert "entity_embeddings" in cypher
        assert report.entity_edges_created == 7
        assert report.chunk_edges_created == 0

    @pytest.mark.unit
    async def test_all_target_runs_both_loops(self):
        mock_client = MagicMock()
        mock_client.execute_query = AsyncMock(
            side_effect=[
                _make_report_rows(edges=10, processed=20),
                _make_report_rows(edges=5, processed=8),
            ]
        )
        with patch("app.services.similarity_service.neo4j_client", mock_client):
            svc = SimilarityService()
            report = await svc.build_similarities("g1", target="all")
        # Two queries — one per target — no deletes
        assert mock_client.execute_query.await_count == 2
        assert report.chunk_edges_created == 10
        assert report.entity_edges_created == 5


class TestTenantIsolation:
    @pytest.mark.unit
    async def test_graph_id_in_params_and_filtered_three_ways(self):
        """graph_id must appear on the source node, the neighbour, AND
        be stored on the edge — three lines of defense for tenant
        isolation since the vector index has no per-graph partition."""
        mock_client = MagicMock()
        mock_client.execute_query = AsyncMock(return_value=_make_report_rows())
        with patch("app.services.similarity_service.neo4j_client", mock_client):
            svc = SimilarityService()
            await svc.build_similarities("my-graph", target="chunks")
        cypher = mock_client.execute_query.call_args[0][0]
        params = mock_client.execute_query.call_args[0][1]
        # Source filter
        assert "a:`Chunk` {graph_id: $gid}" in cypher
        # Neighbour filter
        assert "b.graph_id = $gid" in cypher
        # Edge property
        assert "SIMILAR_TO {graph_id: $gid}" in cypher
        # Graph id parameter
        assert params["gid"] == "my-graph"


class TestPairDedup:
    @pytest.mark.unit
    async def test_filter_uses_element_id_for_pair_dedup(self):
        """One edge per unordered pair: enforce by elementId ordering.
        Two SIMILAR_TO edges between the same nodes is a bug.

        ``elementId()`` is used rather than ``a.id`` because not every
        node label has a user-facing ``id`` property (e.g. :Chunk only
        has ``index`` / ``text`` / ``embedding`` in the current
        deployment), while elementId is always unique within Neo4j.
        """
        mock_client = MagicMock()
        mock_client.execute_query = AsyncMock(return_value=_make_report_rows())
        with patch("app.services.similarity_service.neo4j_client", mock_client):
            svc = SimilarityService()
            await svc.build_similarities("g", target="chunks")
        cypher = mock_client.execute_query.call_args[0][0]
        assert "elementId(a) < elementId(b)" in cypher


class TestEmbeddingDimFilter:
    @pytest.mark.unit
    async def test_source_and_neighbour_filtered_by_expected_dim(self):
        """Both ``a`` and ``b`` must pass the dimension check — a single
        corrupt 6035-dim row on either side would crash the vector
        index call with IllegalArgumentException without this filter."""
        mock_client = MagicMock()
        mock_client.execute_query = AsyncMock(return_value=_make_report_rows())
        with patch("app.services.similarity_service.neo4j_client", mock_client):
            svc = SimilarityService()
            await svc.build_similarities("g", target="chunks")
        cypher = mock_client.execute_query.call_args[0][0]
        params = mock_client.execute_query.call_args[0][1]
        assert "size(a.embedding) = $expected_dim" in cypher
        assert "size(b.embedding) = $expected_dim" in cypher
        assert params["expected_dim"] == 3072


class TestThresholdsAndTopK:
    @pytest.mark.unit
    async def test_default_chunk_threshold_is_085(self):
        mock_client = MagicMock()
        mock_client.execute_query = AsyncMock(return_value=_make_report_rows())
        with patch("app.services.similarity_service.neo4j_client", mock_client):
            svc = SimilarityService()
            await svc.build_similarities("g", target="chunks")
        params = mock_client.execute_query.call_args[0][1]
        assert params["threshold"] == 0.85

    @pytest.mark.unit
    async def test_default_entity_threshold_is_092(self):
        """Entities need tighter — short name embeddings false-positive."""
        mock_client = MagicMock()
        mock_client.execute_query = AsyncMock(return_value=_make_report_rows())
        with patch("app.services.similarity_service.neo4j_client", mock_client):
            svc = SimilarityService()
            await svc.build_similarities("g", target="entities")
        params = mock_client.execute_query.call_args[0][1]
        assert params["threshold"] == 0.92

    @pytest.mark.unit
    async def test_threshold_overrides_take_effect(self):
        mock_client = MagicMock()
        mock_client.execute_query = AsyncMock(
            side_effect=[_make_report_rows(), _make_report_rows()]
        )
        with patch("app.services.similarity_service.neo4j_client", mock_client):
            svc = SimilarityService()
            await svc.build_similarities(
                "g",
                target="all",
                threshold_chunks=0.7,
                threshold_entities=0.99,
            )
        # Two calls — first chunks (0.7), then entities (0.99)
        chunk_call = mock_client.execute_query.call_args_list[0]
        entity_call = mock_client.execute_query.call_args_list[1]
        assert chunk_call.args[1]["threshold"] == 0.7
        assert entity_call.args[1]["threshold"] == 0.99

    @pytest.mark.unit
    async def test_top_k_plus_one_passed_to_vector_index(self):
        """The vector index returns the source node at score=1.0, so we
        request ``top_k+1`` and rely on the id-inequality filter to
        strip it. Off-by-one here would silently drop one real
        neighbour."""
        mock_client = MagicMock()
        mock_client.execute_query = AsyncMock(return_value=_make_report_rows())
        with patch("app.services.similarity_service.neo4j_client", mock_client):
            svc = SimilarityService()
            await svc.build_similarities("g", target="chunks", top_k=5)
        params = mock_client.execute_query.call_args[0][1]
        assert params["top_k_plus_one"] == 6


class TestForceRebuild:
    @pytest.mark.unit
    async def test_force_rebuild_deletes_first_then_rebuilds(self):
        mock_client = MagicMock()
        mock_client.execute_query = AsyncMock(return_value=_make_report_rows(edges=3))
        mock_client.execute_write_query = AsyncMock(return_value=None)
        with patch("app.services.similarity_service.neo4j_client", mock_client):
            svc = SimilarityService()
            await svc.build_similarities("g", target="chunks", force_rebuild=True)
        # One delete + one rebuild
        delete_cypher = mock_client.execute_write_query.call_args[0][0]
        assert "DELETE r" in delete_cypher
        assert "`Chunk`" in delete_cypher

    @pytest.mark.unit
    async def test_no_force_rebuild_no_delete(self):
        mock_client = MagicMock()
        mock_client.execute_query = AsyncMock(return_value=_make_report_rows())
        mock_client.execute_write_query = AsyncMock(return_value=None)
        with patch("app.services.similarity_service.neo4j_client", mock_client):
            svc = SimilarityService()
            await svc.build_similarities("g", target="chunks")
        mock_client.execute_write_query.assert_not_called()


class TestReport:
    @pytest.mark.unit
    def test_total_edges_sums_chunks_and_entities(self):
        report = SimilarityReport(
            target="all", chunk_edges_created=10, entity_edges_created=4
        )
        assert report.total_edges() == 14

    @pytest.mark.unit
    async def test_no_neo4j_response_yields_zero_counts(self):
        """If the driver returns empty rows, the service shouldn't
        crash — it should report zero edges."""
        mock_client = MagicMock()
        mock_client.execute_query = AsyncMock(return_value=[])
        with patch("app.services.similarity_service.neo4j_client", mock_client):
            svc = SimilarityService()
            report = await svc.build_similarities("g", target="chunks")
        assert report.chunk_edges_created == 0
        assert report.chunks_processed == 0
