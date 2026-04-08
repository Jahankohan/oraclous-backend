"""
Unit tests for PipelineService and MultiTenantGraphRAGPipeline.

Tests extraction logic, entity normalization, banned property enforcement,
pipeline caching, and multi-tenant isolation — all external deps mocked.
"""

from unittest.mock import AsyncMock, MagicMock, patch
from uuid import UUID

import pytest

from app.schemas.graph_schemas import BANNED_NODE_PROPERTIES
from app.services.pipeline_service import (
    MultiTenantGraphRAGPipeline,
    PipelineConfig,
    PipelineService,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_node(node_id: str, label: str = "Person", **props):
    node = MagicMock()
    node.id = node_id
    node.label = label
    node.properties = {"name": node_id, **props}
    return node


def _make_relationship(
    start_id: str, end_id: str, rel_type: str = "WORKS_FOR", **props
):
    rel = MagicMock()
    rel.start_node_id = start_id
    rel.end_node_id = end_id
    rel.type = rel_type
    rel.properties = props
    return rel


def _make_graph(nodes=None, relationships=None):
    graph = MagicMock()
    graph.nodes = nodes or []
    graph.relationships = relationships or []
    return graph


def _make_pipeline(graph_id: str = "test-graph") -> MultiTenantGraphRAGPipeline:
    with patch("app.services.pipeline_service.settings") as mock_settings:
        mock_settings.NEO4J_URI = "neo4j://localhost:7687"
        mock_settings.NEO4J_USERNAME = "neo4j"
        mock_settings.NEO4J_PASSWORD = "password"
        mock_settings.NEO4J_DATABASE = "neo4j"
        mock_settings.OPENAI_API_KEY = "test-key"
        return MultiTenantGraphRAGPipeline(graph_id=graph_id)


# ---------------------------------------------------------------------------
# Tests: PipelineConfig
# ---------------------------------------------------------------------------


class TestPipelineConfig:
    @pytest.mark.unit
    def test_config_reads_from_settings(self):
        with patch("app.services.pipeline_service.settings") as mock_settings:
            mock_settings.NEO4J_URI = "neo4j://test:7687"
            mock_settings.NEO4J_USERNAME = "user"
            mock_settings.NEO4J_PASSWORD = "pass"
            mock_settings.NEO4J_DATABASE = "neo4j"
            mock_settings.OPENAI_API_KEY = "key"

            config = PipelineConfig()

            assert config.neo4j_uri == "neo4j://test:7687"
            assert config.neo4j_user == "user"
            assert config.openai_api_key == "key"

    @pytest.mark.unit
    def test_config_defaults_llm_model(self):
        with patch("app.services.pipeline_service.settings") as mock_settings:
            mock_settings.NEO4J_URI = "neo4j://localhost:7687"
            mock_settings.NEO4J_USERNAME = "neo4j"
            mock_settings.NEO4J_PASSWORD = "password"
            mock_settings.NEO4J_DATABASE = "neo4j"
            mock_settings.OPENAI_API_KEY = "key"
            # Simulate settings without LLM_MODEL attr
            del mock_settings.LLM_MODEL

            config = PipelineConfig()
            assert config.llm_model == "gpt-4o"


# ---------------------------------------------------------------------------
# Tests: _model_supports_json_object
# ---------------------------------------------------------------------------


class TestModelSupportsJsonObject:
    @pytest.mark.unit
    def test_gpt4o_supports_json(self):
        pipeline = _make_pipeline()
        assert pipeline._model_supports_json_object("gpt-4o") is True

    @pytest.mark.unit
    def test_gpt4o_mini_supports_json(self):
        pipeline = _make_pipeline()
        assert pipeline._model_supports_json_object("gpt-4o-mini") is True

    @pytest.mark.unit
    def test_gpt4_turbo_supports_json(self):
        pipeline = _make_pipeline()
        assert pipeline._model_supports_json_object("gpt-4-turbo") is True

    @pytest.mark.unit
    def test_old_gpt4_does_not_support_json(self):
        pipeline = _make_pipeline()
        assert pipeline._model_supports_json_object("gpt-4") is False

    @pytest.mark.unit
    def test_gpt35_without_version_does_not_support_json(self):
        pipeline = _make_pipeline()
        assert pipeline._model_supports_json_object("gpt-3.5-turbo") is False

    @pytest.mark.unit
    def test_unknown_model_does_not_support_json(self):
        pipeline = _make_pipeline()
        assert pipeline._model_supports_json_object("claude-opus-4") is False


# ---------------------------------------------------------------------------
# Tests: _strip_banned_node_properties (Critical — property placement enforcement)
# ---------------------------------------------------------------------------


class TestStripBannedNodeProperties:
    @pytest.mark.unit
    def test_strips_job_title_from_node(self):
        node = _make_node("alice", job_title="CEO")
        graph = _make_graph(nodes=[node])
        pipeline = _make_pipeline()

        result_graph, violations, migrated = pipeline._strip_banned_node_properties(
            graph
        )

        assert "job_title" not in result_graph.nodes[0].properties
        assert violations == 1
        assert migrated == 0  # Property is dropped, not migrated

    @pytest.mark.unit
    def test_strips_all_banned_properties(self):
        """All BANNED_NODE_PROPERTIES should be stripped."""
        node = _make_node("entity")
        for prop in BANNED_NODE_PROPERTIES:
            node.properties[prop] = "some_value"

        graph = _make_graph(nodes=[node])
        pipeline = _make_pipeline()

        result_graph, violations, _ = pipeline._strip_banned_node_properties(graph)

        remaining_props = result_graph.nodes[0].properties
        for prop in BANNED_NODE_PROPERTIES:
            assert prop not in remaining_props
        assert violations == len(BANNED_NODE_PROPERTIES)

    @pytest.mark.unit
    def test_preserves_allowed_properties(self):
        node = _make_node("alice", industry="Software", founded_year=2010)
        graph = _make_graph(nodes=[node])
        pipeline = _make_pipeline()

        result_graph, violations, _ = pipeline._strip_banned_node_properties(graph)

        assert result_graph.nodes[0].properties["industry"] == "Software"
        assert result_graph.nodes[0].properties["founded_year"] == 2010
        assert violations == 0

    @pytest.mark.unit
    def test_no_violations_on_clean_node(self):
        node = _make_node("alice")
        graph = _make_graph(nodes=[node])
        pipeline = _make_pipeline()

        _, violations, migrated = pipeline._strip_banned_node_properties(graph)

        assert violations == 0
        assert migrated == 0

    @pytest.mark.unit
    def test_handles_node_with_none_properties(self):
        node = MagicMock()
        node.properties = None
        graph = _make_graph(nodes=[node])
        pipeline = _make_pipeline()

        # Should not raise
        _, violations, _ = pipeline._strip_banned_node_properties(graph)
        assert violations == 0

    @pytest.mark.unit
    def test_handles_empty_graph(self):
        graph = _make_graph(nodes=[])
        pipeline = _make_pipeline()

        result_graph, violations, migrated = pipeline._strip_banned_node_properties(
            graph
        )

        assert violations == 0
        assert migrated == 0

    @pytest.mark.unit
    def test_violation_count_across_multiple_nodes(self):
        node1 = _make_node("alice", job_title="CEO", position="Director")
        node2 = _make_node("bob", role="Engineer")
        graph = _make_graph(nodes=[node1, node2])
        pipeline = _make_pipeline()

        _, violations, _ = pipeline._strip_banned_node_properties(graph)

        assert violations == 3  # job_title + position + role

    @pytest.mark.unit
    def test_relationships_are_unchanged(self):
        node = _make_node("alice", job_title="CEO")
        rel = _make_relationship("alice", "acme", position="CTO")
        graph = _make_graph(nodes=[node], relationships=[rel])
        pipeline = _make_pipeline()

        result_graph, _, _ = pipeline._strip_banned_node_properties(graph)

        # Relationship should be untouched
        assert result_graph.relationships[0].properties["position"] == "CTO"


# ---------------------------------------------------------------------------
# Tests: _normalize_overlapping_entities
# ---------------------------------------------------------------------------


class TestNormalizeOverlappingEntities:
    @pytest.mark.unit
    async def test_deduplicates_nodes_with_same_name(self):
        """Nodes with same name from different chunks should be merged."""
        node1 = _make_node("chunk_1:Alice Chen")
        node1.properties["name"] = "Alice Chen"
        node2 = _make_node("chunk_2:Alice Chen")
        node2.properties["name"] = "Alice Chen"

        graph = _make_graph(nodes=[node1, node2])
        pipeline = _make_pipeline()

        result = await pipeline._normalize_overlapping_entities(graph)

        assert len(result.nodes) == 1

    @pytest.mark.unit
    async def test_updates_relationship_references_after_dedup(self):
        """Relationship start/end IDs must point to canonical IDs after dedup."""
        node1 = _make_node("chunk_1:Alice")
        node1.properties["name"] = "Alice"
        node2 = _make_node("chunk_1:Acme")
        node2.properties["name"] = "Acme"
        rel = _make_relationship("chunk_1:Alice", "chunk_1:Acme")

        graph = _make_graph(nodes=[node1, node2], relationships=[rel])
        pipeline = _make_pipeline()

        result = await pipeline._normalize_overlapping_entities(graph)

        # Relationship IDs should no longer have chunk prefix after normalization
        assert "chunk_1:" not in result.relationships[0].start_node_id
        assert "chunk_1:" not in result.relationships[0].end_node_id

    @pytest.mark.unit
    async def test_returns_graph_unchanged_when_no_duplicates(self):
        node1 = _make_node("alice")
        node1.properties["name"] = "Alice Chen"
        node2 = _make_node("bob")
        node2.properties["name"] = "Bob Smith"

        graph = _make_graph(nodes=[node1, node2])
        pipeline = _make_pipeline()

        result = await pipeline._normalize_overlapping_entities(graph)

        assert len(result.nodes) == 2

    @pytest.mark.unit
    async def test_handles_empty_graph(self):
        graph = _make_graph(nodes=[])
        pipeline = _make_pipeline()

        result = await pipeline._normalize_overlapping_entities(graph)

        assert result is graph

    @pytest.mark.unit
    async def test_handles_nodes_without_name(self):
        node = MagicMock()
        node.id = "some-id"
        node.properties = {}  # No name
        graph = _make_graph(nodes=[node])
        pipeline = _make_pipeline()

        # Should not raise
        result = await pipeline._normalize_overlapping_entities(graph)
        assert len(result.nodes) == 1

    @pytest.mark.unit
    async def test_keeps_node_with_more_properties_on_merge(self):
        """When merging, retain the node that has more properties."""
        node1 = _make_node("chunk_1:Alice")
        node1.properties = {"name": "Alice"}  # fewer props

        node2 = _make_node("chunk_2:Alice")
        node2.properties = {
            "name": "Alice",
            "industry": "Tech",
            "founded": 2020,
        }  # more props

        graph = _make_graph(nodes=[node1, node2])
        pipeline = _make_pipeline()

        result = await pipeline._normalize_overlapping_entities(graph)

        assert len(result.nodes) == 1
        assert "industry" in result.nodes[0].properties


# ---------------------------------------------------------------------------
# Tests: process_documents (routing logic)
# ---------------------------------------------------------------------------


class TestMultiTenantGraphRAGPipelineProcessDocuments:
    @pytest.mark.unit
    async def test_small_doc_set_processed_synchronously(self):
        pipeline = _make_pipeline(graph_id="tenant-1")
        pipeline._initialize_components = AsyncMock()
        pipeline._process_documents_sync = AsyncMock(
            return_value={
                "documents_processed": 3,
                "entities_created": 10,
                "relationships_created": 5,
                "chunks_created": 3,
                "property_violations_detected": 0,
                "property_violations_migrated": 0,
            }
        )

        docs = [{"text": f"doc{i}", "source": f"src{i}"} for i in range(3)]
        result = await pipeline.process_documents(docs)

        assert result["status"] == "completed"
        assert result["graph_id"] == "tenant-1"
        pipeline._process_documents_sync.assert_awaited_once()

    @pytest.mark.unit
    async def test_large_doc_set_uses_background_processing(self):
        pipeline = _make_pipeline(graph_id="tenant-2")
        pipeline._initialize_components = AsyncMock()
        pipeline._process_documents_background = AsyncMock()

        docs = [{"text": f"doc{i}", "source": f"src{i}"} for i in range(11)]
        result = await pipeline.process_documents(docs)

        assert result["status"] == "processing"
        assert result["documents_queued"] == 11

    @pytest.mark.unit
    async def test_exception_returns_failed_status(self):
        pipeline = _make_pipeline(graph_id="tenant-err")
        pipeline._initialize_components = AsyncMock(side_effect=Exception("Neo4j down"))

        result = await pipeline.process_documents([{"text": "doc", "source": "src"}])

        assert result["status"] == "failed"
        assert "error" in result
        assert result["graph_id"] == "tenant-err"

    @pytest.mark.unit
    async def test_graph_id_always_in_result(self):
        pipeline = _make_pipeline(graph_id="isolated-tenant")
        pipeline._initialize_components = AsyncMock()
        pipeline._process_documents_sync = AsyncMock(
            return_value={
                "documents_processed": 1,
                "entities_created": 2,
                "relationships_created": 1,
                "chunks_created": 1,
                "property_violations_detected": 0,
                "property_violations_migrated": 0,
            }
        )

        docs = [{"text": "Hello world document.", "source": "src"}]
        result = await pipeline.process_documents(docs)

        assert result["graph_id"] == "isolated-tenant"


# ---------------------------------------------------------------------------
# Tests: PipelineService — pipeline caching and multi-tenant isolation
# ---------------------------------------------------------------------------


class TestPipelineService:
    @pytest.mark.unit
    def test_get_pipeline_returns_pipeline_for_graph_id(self):
        with patch("app.services.pipeline_service.settings") as mock_settings:
            mock_settings.NEO4J_URI = "neo4j://localhost:7687"
            mock_settings.NEO4J_USERNAME = "neo4j"
            mock_settings.NEO4J_PASSWORD = "password"
            mock_settings.NEO4J_DATABASE = "neo4j"
            mock_settings.OPENAI_API_KEY = "key"

            svc = PipelineService()
            graph_id = UUID("12345678-1234-5678-1234-567812345678")
            pipeline = svc.get_pipeline(graph_id)

            assert pipeline.graph_id == str(graph_id)

    @pytest.mark.unit
    def test_get_pipeline_caches_per_graph_id(self):
        with patch("app.services.pipeline_service.settings") as mock_settings:
            mock_settings.NEO4J_URI = "neo4j://localhost:7687"
            mock_settings.NEO4J_USERNAME = "neo4j"
            mock_settings.NEO4J_PASSWORD = "password"
            mock_settings.NEO4J_DATABASE = "neo4j"
            mock_settings.OPENAI_API_KEY = "key"

            svc = PipelineService()
            graph_id = UUID("12345678-1234-5678-1234-567812345678")

            p1 = svc.get_pipeline(graph_id)
            p2 = svc.get_pipeline(graph_id)

            assert p1 is p2  # Same instance returned from cache

    @pytest.mark.unit
    def test_different_graph_ids_get_different_pipelines(self):
        with patch("app.services.pipeline_service.settings") as mock_settings:
            mock_settings.NEO4J_URI = "neo4j://localhost:7687"
            mock_settings.NEO4J_USERNAME = "neo4j"
            mock_settings.NEO4J_PASSWORD = "password"
            mock_settings.NEO4J_DATABASE = "neo4j"
            mock_settings.OPENAI_API_KEY = "key"

            svc = PipelineService()
            id1 = UUID("11111111-1111-1111-1111-111111111111")
            id2 = UUID("22222222-2222-2222-2222-222222222222")

            p1 = svc.get_pipeline(id1)
            p2 = svc.get_pipeline(id2)

            assert p1 is not p2
            assert p1.graph_id != p2.graph_id

    @pytest.mark.unit
    def test_clear_pipeline_cache_removes_specific_graph(self):
        with patch("app.services.pipeline_service.settings") as mock_settings:
            mock_settings.NEO4J_URI = "neo4j://localhost:7687"
            mock_settings.NEO4J_USERNAME = "neo4j"
            mock_settings.NEO4J_PASSWORD = "password"
            mock_settings.NEO4J_DATABASE = "neo4j"
            mock_settings.OPENAI_API_KEY = "key"

            svc = PipelineService()
            graph_id = UUID("12345678-1234-5678-1234-567812345678")
            svc.get_pipeline(graph_id)

            assert len(svc._pipeline_cache) == 1
            svc.clear_pipeline_cache(graph_id)
            assert len(svc._pipeline_cache) == 0

    @pytest.mark.unit
    def test_clear_all_pipeline_caches(self):
        with patch("app.services.pipeline_service.settings") as mock_settings:
            mock_settings.NEO4J_URI = "neo4j://localhost:7687"
            mock_settings.NEO4J_USERNAME = "neo4j"
            mock_settings.NEO4J_PASSWORD = "password"
            mock_settings.NEO4J_DATABASE = "neo4j"
            mock_settings.OPENAI_API_KEY = "key"

            svc = PipelineService()
            for i in range(3):
                svc.get_pipeline(UUID(f"1234567{i}-1234-5678-1234-567812345678"))

            assert len(svc._pipeline_cache) == 3
            svc.clear_pipeline_cache()
            assert len(svc._pipeline_cache) == 0

    @pytest.mark.unit
    async def test_process_documents_passes_graph_id_to_pipeline(self):
        """Multi-tenant isolation: graph_id must flow from service to pipeline."""
        with patch("app.services.pipeline_service.settings") as mock_settings:
            mock_settings.NEO4J_URI = "neo4j://localhost:7687"
            mock_settings.NEO4J_USERNAME = "neo4j"
            mock_settings.NEO4J_PASSWORD = "password"
            mock_settings.NEO4J_DATABASE = "neo4j"
            mock_settings.OPENAI_API_KEY = "key"

            svc = PipelineService()
            graph_id = UUID("aaaabbbb-cccc-dddd-eeee-ffffaaaabbbb")

            mock_pipeline = AsyncMock()
            mock_pipeline.process_documents = AsyncMock(
                return_value={
                    "status": "completed",
                    "graph_id": str(graph_id),
                    "entities_created": 5,
                }
            )
            svc._pipeline_cache[f"pipeline_{graph_id}"] = mock_pipeline

            docs = [{"text": "test doc content here", "source": "test"}]
            await svc.process_documents(documents=docs, graph_id=graph_id)

            mock_pipeline.process_documents.assert_awaited_once_with(
                docs,
                None,
                None,
                temporal_context=None,
                mode=pytest.approx(None)
                or mock_pipeline.process_documents.await_args.kwargs.get("mode"),
                job_id=None,
            )
            # Core check: graph_id must flow to the pipeline instance (validated by cache key)
            assert f"pipeline_{graph_id}" in svc._pipeline_cache


# ---------------------------------------------------------------------------
# Tests: multi-tenant isolation — graph_id enforcement
# ---------------------------------------------------------------------------


class TestMultiTenantIsolation:
    @pytest.mark.unit
    def test_pipeline_stores_graph_id(self):
        pipeline = _make_pipeline("tenant-abc")
        assert pipeline.graph_id == "tenant-abc"

    @pytest.mark.unit
    def test_pipeline_user_id_stored(self):
        with patch("app.services.pipeline_service.settings") as mock_settings:
            mock_settings.NEO4J_URI = "neo4j://localhost:7687"
            mock_settings.NEO4J_USERNAME = "neo4j"
            mock_settings.NEO4J_PASSWORD = "password"
            mock_settings.NEO4J_DATABASE = "neo4j"
            mock_settings.OPENAI_API_KEY = "key"
            p = MultiTenantGraphRAGPipeline(graph_id="g1", user_id="user-123")
            assert p.user_id == "user-123"

    @pytest.mark.unit
    async def test_process_documents_sync_skips_empty_text(self):
        """Documents with no text content are skipped — no blank ingestion."""
        pipeline = _make_pipeline("tenant-x")
        pipeline._initialize_components = AsyncMock()
        pipeline.driver = MagicMock()

        mock_writer = MagicMock()
        mock_writer.run = AsyncMock()

        with patch(
            "app.services.pipeline_service.create_multi_tenant_kg_writer",
            return_value=mock_writer,
        ):
            docs = [
                {"text": "", "source": "empty"},
                {"content": "", "source": "also-empty"},
            ]
            result = await pipeline._process_documents_sync(docs)

            # Writer should not have been called since both docs are empty
            mock_writer.run.assert_not_awaited()
            assert result["entities_created"] == 0
