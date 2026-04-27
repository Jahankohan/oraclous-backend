"""
Unit tests for Schema Service.

Tests the Neo4j schema extraction, caching, and formatting functionality
in isolation using mocked dependencies.
"""

from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.services.schema_service import (
    GraphSchema,
    Neo4jSchemaManager,
    NodeSchema,
    RelationshipSchema,
    get_text2cypher_schema,
)


class TestNeo4jSchemaManager:
    """Test Neo4j Schema Manager functionality."""

    @pytest.fixture
    def schema_manager(self):
        """Create a fresh schema manager for testing."""
        return Neo4jSchemaManager()

    @pytest.fixture
    def mock_neo4j_records(self):
        """Mock Neo4j query records for node extraction."""
        return [
            {"label": "Company", "prop": "name", "propType": "STRING", "frequency": 5},
            {"label": "Company", "prop": "type", "propType": "STRING", "frequency": 5},
            {"label": "Person", "prop": "name", "propType": "STRING", "frequency": 3},
            {"label": "Person", "prop": "role", "propType": "STRING", "frequency": 3},
        ]

    @pytest.fixture
    def mock_relationship_records(self):
        """Mock Neo4j query records for relationship extraction."""
        return [
            {
                "relType": "CEO_OF",
                "allStartLabels": [["Person"]],
                "allEndLabels": [["Company"]],
                "prop": "since",
                "propType": "DATE",
                "frequency": 2,
            },
            {
                "relType": "WORKS_AT",
                "allStartLabels": [["Person"]],
                "allEndLabels": [["Company"]],
                "prop": "department",
                "propType": "STRING",
                "frequency": 5,
            },
        ]

    @pytest.mark.unit
    @pytest.mark.schema
    async def test_extract_node_schemas(self, schema_manager, mock_neo4j_records):
        """Test node schema extraction."""
        with patch("app.services.schema_service.neo4j_client") as mock_client:
            mock_client.execute_query = AsyncMock(return_value=mock_neo4j_records)

            # Mock count queries
            mock_client.execute_query.side_effect = [
                mock_neo4j_records,  # Main query
                [{"count": 5}],  # Company count
                [{"count": 3}],  # Person count
            ]

            nodes = await schema_manager._extract_node_schemas("test-graph")

            assert len(nodes) == 2
            assert "Company" in nodes
            assert "Person" in nodes

            # Check Company schema
            company = nodes["Company"]
            assert company.label == "Company"
            assert company.sample_count == 5
            assert "name" in company.properties
            assert "type" in company.properties
            assert company.properties["name"] == "STRING"

            # Check Person schema
            person = nodes["Person"]
            assert person.label == "Person"
            assert person.sample_count == 3
            assert "name" in person.properties
            assert "role" in person.properties

    @pytest.mark.unit
    @pytest.mark.schema
    async def test_extract_relationship_schemas(
        self, schema_manager, mock_relationship_records
    ):
        """Test relationship schema extraction."""
        with patch("app.services.schema_service.neo4j_client") as mock_client:
            mock_client.execute_query = AsyncMock(
                return_value=mock_relationship_records
            )

            relationships = await schema_manager._extract_relationship_schemas(
                "test-graph"
            )

            assert len(relationships) == 2
            assert "CEO_OF" in relationships
            assert "WORKS_AT" in relationships

            # Check CEO_OF relationship
            ceo_rel = relationships["CEO_OF"]
            assert ceo_rel.type == "CEO_OF"
            assert ceo_rel.sample_count == 2
            assert "Person" in ceo_rel.start_labels
            assert "Company" in ceo_rel.end_labels
            assert "since" in ceo_rel.properties

            # Check WORKS_AT relationship
            works_rel = relationships["WORKS_AT"]
            assert works_rel.type == "WORKS_AT"
            assert works_rel.sample_count == 5
            assert "Person" in works_rel.start_labels
            assert "Company" in works_rel.end_labels
            assert "department" in works_rel.properties

    @pytest.mark.unit
    @pytest.mark.schema
    async def test_extract_schema_with_cache(self, schema_manager):
        """Test schema extraction with caching."""
        graph_id = "test-graph"

        # Mock the extraction methods
        mock_nodes = {"Company": NodeSchema("Company", {}, 5, [])}
        mock_rels = {
            "CEO_OF": RelationshipSchema("CEO_OF", {}, {"Person"}, {"Company"}, 2)
        }

        with (
            patch.object(
                schema_manager, "_extract_node_schemas", return_value=mock_nodes
            ),
            patch.object(
                schema_manager, "_extract_relationship_schemas", return_value=mock_rels
            ),
            patch.object(schema_manager, "_get_constraints", return_value=[]),
            patch.object(schema_manager, "_get_indexes", return_value=[]),
            patch("app.services.schema_service.neo4j_client") as mock_client,
        ):

            mock_client.connect_async = AsyncMock()
            mock_client.async_driver = MagicMock()

            # First call - should extract
            schema1 = await schema_manager.extract_schema(graph_id)
            assert schema1.graph_id == graph_id
            assert len(schema1.nodes) == 1
            assert len(schema1.relationships) == 1

            # Second call - should use cache
            schema2 = await schema_manager.extract_schema(graph_id)
            assert schema2 is schema1  # Same object from cache

    @pytest.mark.unit
    @pytest.mark.schema
    async def test_extract_schema_force_refresh(self, schema_manager):
        """Test schema extraction with force refresh."""
        graph_id = "test-graph"

        mock_nodes = {"Company": NodeSchema("Company", {}, 5, [])}
        mock_rels = {
            "CEO_OF": RelationshipSchema("CEO_OF", {}, {"Person"}, {"Company"}, 2)
        }

        with (
            patch.object(
                schema_manager, "_extract_node_schemas", return_value=mock_nodes
            ),
            patch.object(
                schema_manager, "_extract_relationship_schemas", return_value=mock_rels
            ),
            patch.object(schema_manager, "_get_constraints", return_value=[]),
            patch.object(schema_manager, "_get_indexes", return_value=[]),
            patch("app.services.schema_service.neo4j_client") as mock_client,
        ):

            mock_client.connect_async = AsyncMock()
            mock_client.async_driver = MagicMock()

            # First call
            schema1 = await schema_manager.extract_schema(graph_id)

            # Second call with force_refresh
            schema2 = await schema_manager.extract_schema(graph_id, force_refresh=True)
            assert schema2 is not schema1  # Different object due to refresh

    @pytest.mark.unit
    @pytest.mark.schema
    def test_format_schema_for_text2cypher(self, schema_manager):
        """Test schema formatting for Text2Cypher."""
        # Create a test schema
        nodes = {
            "Company": NodeSchema(
                label="Company",
                properties={"name": "string", "type": "string"},
                sample_count=5,
                indexes=["name"],
            ),
            "Person": NodeSchema(
                label="Person",
                properties={"name": "string", "role": "string"},
                sample_count=3,
                indexes=[],
            ),
        }

        relationships = {
            "CEO_OF": RelationshipSchema(
                type="CEO_OF",
                properties={"since": "date"},
                start_labels={"Person"},
                end_labels={"Company"},
                sample_count=2,
            )
        }

        schema = GraphSchema(
            graph_id="test-graph",
            nodes=nodes,
            relationships=relationships,
            constraints=[],
            indexes=[],
            last_updated=datetime.now(UTC),
            schema_version="test_v1",
        )

        formatted = schema_manager.format_schema_for_text2cypher(schema)

        assert "Neo4j Knowledge Graph Schema" in formatted
        assert "test-graph" in formatted
        assert "Company (5 nodes)" in formatted
        assert "Person (3 nodes)" in formatted
        assert "CEO_OF (2 relationships)" in formatted
        assert "name: string" in formatted
        assert "graph_id" in formatted
        assert "multi-tenant isolation" in formatted

    @pytest.mark.unit
    @pytest.mark.schema
    def test_cache_management(self, schema_manager):
        """Test cache management functionality."""
        graph_id = "test-graph"

        # Create a test schema
        schema = GraphSchema(
            graph_id=graph_id,
            nodes={},
            relationships={},
            constraints=[],
            indexes=[],
            last_updated=datetime.now(UTC),
            schema_version="test_v1",
        )

        # Add to cache
        schema_manager._schema_cache[graph_id] = schema

        # Test cache stats
        stats = schema_manager.get_cache_stats()
        assert stats["cached_count"] == 1
        assert "cache_ttl_minutes" in stats

        # Test cache details
        details = schema_manager.get_cache_details()
        assert graph_id in details
        assert details[graph_id] is schema

        # Test clear specific cache
        schema_manager.clear_cache(graph_id)
        assert graph_id not in schema_manager._schema_cache

        # Test clear all cache
        schema_manager._schema_cache[graph_id] = schema
        schema_manager.clear_cache()
        assert len(schema_manager._schema_cache) == 0

    @pytest.mark.unit
    @pytest.mark.schema
    async def test_get_text2cypher_schema_convenience_function(self):
        """Test the convenience function for getting Text2Cypher schema."""
        graph_id = "test-graph"

        with patch("app.services.schema_service.schema_manager") as mock_manager:
            mock_manager.get_schema_for_text2cypher = AsyncMock(
                return_value="Mock schema"
            )

            result = await get_text2cypher_schema(graph_id)

            assert result == "Mock schema"
            mock_manager.get_schema_for_text2cypher.assert_called_once_with(graph_id)

    @pytest.mark.unit
    @pytest.mark.schema
    async def test_schema_extraction_error_handling(self, schema_manager):
        """Test error handling in schema extraction."""
        with patch("app.services.schema_service.neo4j_client") as mock_client:
            mock_client.connect_async = AsyncMock(
                side_effect=Exception("Connection failed")
            )

            with pytest.raises(Exception, match="Connection failed"):
                await schema_manager.extract_schema("test-graph")

    @pytest.mark.unit
    @pytest.mark.schema
    async def test_get_schema_for_text2cypher_with_fallback(self, schema_manager):
        """Test Text2Cypher schema generation with fallback."""
        graph_id = "test-graph"

        with patch.object(
            schema_manager, "extract_schema", side_effect=Exception("Failed")
        ):
            result = await schema_manager.get_schema_for_text2cypher(graph_id)

            # Should return fallback schema
            assert "Basic Schema" in result
            assert graph_id in result
            assert "Entity" in result
            assert "graph_id" in result
