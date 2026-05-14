"""
Integration tests for Schema API.

Tests the schema management API endpoints with caching,
error handling, and Docker environment integration.
"""

from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.services.schema_service import GraphSchema, NodeSchema, RelationshipSchema


class TestSchemaAPIIntegration:
    """Test Schema API integration functionality."""

    @pytest.fixture
    def mock_schema(self):
        """Create a mock schema for testing."""
        return GraphSchema(
            graph_id="test-graph-12345",
            nodes={
                "Company": NodeSchema(
                    label="Company",
                    properties={
                        "name": "string",
                        "type": "string",
                        "graph_id": "string",
                    },
                    sample_count=5,
                    indexes=["name"],
                ),
                "Person": NodeSchema(
                    label="Person",
                    properties={
                        "name": "string",
                        "role": "string",
                        "graph_id": "string",
                    },
                    sample_count=3,
                    indexes=["name"],
                ),
            },
            relationships={
                "CEO_OF": RelationshipSchema(
                    type="CEO_OF",
                    properties={"since": "date", "graph_id": "string"},
                    start_labels={"Person"},
                    end_labels={"Company"},
                    sample_count=2,
                )
            },
            constraints=[{"name": "unique_company_name", "type": "UNIQUENESS"}],
            indexes=[{"name": "company_name_index", "type": "BTREE"}],
            last_updated=datetime.now(UTC),
            schema_version="test_v1",
        )

    @pytest.mark.integration
    @pytest.mark.api
    @pytest.mark.schema
    async def test_get_schema_info(self, async_client, mock_schema):
        """Test getting schema information."""
        graph_id = "test-graph-12345"

        with patch("app.api.schema.schema_manager") as mock_manager:
            mock_manager.extract_schema = AsyncMock(return_value=mock_schema)

            response = await async_client.get(f"/api/v1/schema/info/{graph_id}")

            assert response.status_code == 200
            data = response.json()

            # Check required fields
            required_fields = [
                "graph_id",
                "schema_version",
                "last_updated",
                "nodes",
                "relationships",
                "constraints",
                "indexes",
            ]
            for field in required_fields:
                assert field in data

            assert data["graph_id"] == graph_id
            assert data["schema_version"] == "test_v1"
            assert len(data["nodes"]) == 2
            assert len(data["relationships"]) == 1
            assert data["constraints"] == 1
            assert data["indexes"] == 1

            # Check node details
            assert "Company" in data["nodes"]
            assert "Person" in data["nodes"]
            assert data["nodes"]["Company"]["sample_count"] == 5

            # Check relationship details
            assert "CEO_OF" in data["relationships"]
            assert data["relationships"]["CEO_OF"]["sample_count"] == 2

    @pytest.mark.integration
    @pytest.mark.api
    @pytest.mark.schema
    async def test_get_text2cypher_schema(self, async_client, mock_schema):
        """Test getting Text2Cypher formatted schema."""
        graph_id = "test-graph-12345"
        formatted_schema = "# Mock formatted schema for Text2Cypher"

        with patch("app.api.schema.schema_manager") as mock_manager:
            mock_manager.extract_schema = AsyncMock(return_value=mock_schema)
            mock_manager.format_schema_for_text2cypher = MagicMock(
                return_value=formatted_schema
            )

            response = await async_client.get(f"/api/v1/schema/text2cypher/{graph_id}")

            assert response.status_code == 200
            data = response.json()

            assert "graph_id" in data
            assert "schema_version" in data
            assert "formatted_schema" in data
            assert "last_updated" in data

            assert data["graph_id"] == graph_id
            assert data["formatted_schema"] == formatted_schema
            assert data["schema_version"] == "test_v1"

    @pytest.mark.integration
    @pytest.mark.api
    @pytest.mark.schema
    async def test_get_text2cypher_schema_with_force_refresh(
        self, async_client, mock_schema
    ):
        """Test getting Text2Cypher schema with force refresh."""
        graph_id = "test-graph-12345"

        with patch("app.api.schema.schema_manager") as mock_manager:
            mock_manager.extract_schema = AsyncMock(return_value=mock_schema)
            mock_manager.format_schema_for_text2cypher = MagicMock(
                return_value="Refreshed schema"
            )

            response = await async_client.get(
                f"/api/v1/schema/text2cypher/{graph_id}?force_refresh=true"
            )

            assert response.status_code == 200

            # Verify force_refresh was passed
            mock_manager.extract_schema.assert_called_once_with(
                graph_id, force_refresh=True
            )

    @pytest.mark.integration
    @pytest.mark.api
    @pytest.mark.schema
    async def test_refresh_schema(self, async_client, mock_schema):
        """Test schema refresh endpoint."""
        request_data = {"graph_id": "test-graph-12345", "force_refresh": True}

        with patch("app.api.schema.schema_manager") as mock_manager:
            mock_manager.extract_schema = AsyncMock(return_value=mock_schema)

            response = await async_client.post(
                "/api/v1/schema/refresh", json=request_data
            )

            assert response.status_code == 200
            data = response.json()

            # Should return same format as schema info
            assert "graph_id" in data
            assert "schema_version" in data
            assert "nodes" in data
            assert "relationships" in data

            # Verify force refresh was called
            mock_manager.extract_schema.assert_called_once_with(
                request_data["graph_id"], force_refresh=request_data["force_refresh"]
            )

    @pytest.mark.integration
    @pytest.mark.api
    @pytest.mark.schema
    async def test_clear_schema_cache(self, async_client):
        """Test clearing schema cache for specific graph."""
        graph_id = "test-graph-12345"

        with patch("app.api.schema.schema_manager") as mock_manager:
            mock_manager.clear_cache = MagicMock()

            response = await async_client.delete(f"/api/v1/schema/cache/{graph_id}")

            assert response.status_code == 200
            data = response.json()

            assert "message" in data
            assert "graph_id" in data
            assert data["graph_id"] == graph_id
            assert "cleared" in data["message"]

            # Verify clear_cache was called with correct graph_id
            mock_manager.clear_cache.assert_called_once_with(graph_id)

    @pytest.mark.integration
    @pytest.mark.api
    @pytest.mark.schema
    async def test_clear_all_schema_cache(self, async_client):
        """Test clearing all schema caches."""
        with patch("app.api.schema.schema_manager") as mock_manager:
            mock_manager.clear_cache = MagicMock()

            response = await async_client.delete("/api/v1/schema/cache")

            assert response.status_code == 200
            data = response.json()

            assert "message" in data
            assert "All schema caches cleared" in data["message"]

            # Verify clear_cache was called without arguments
            mock_manager.clear_cache.assert_called_once_with()

    @pytest.mark.integration
    @pytest.mark.api
    @pytest.mark.schema
    async def test_get_cache_status(self, async_client, mock_schema):
        """Test getting cache status."""
        with patch("app.api.schema.schema_manager") as mock_manager:
            mock_manager.get_cache_details = MagicMock(
                return_value={"test-graph-12345": mock_schema}
            )

            response = await async_client.get("/api/v1/schema/cache/status")

            assert response.status_code == 200
            data = response.json()

            assert isinstance(data, dict)
            assert "test-graph-12345" in data

            cache_info = data["test-graph-12345"]
            assert "graph_id" in cache_info
            assert "schema_version" in cache_info
            assert "last_updated" in cache_info
            assert "node_count" in cache_info
            assert "relationship_count" in cache_info
            assert "age_minutes" in cache_info

    @pytest.mark.integration
    @pytest.mark.api
    @pytest.mark.schema
    async def test_schema_health_check(self, async_client):
        """Test schema service health check."""
        with patch("app.api.schema.schema_manager") as mock_manager:
            mock_manager.get_cache_stats = MagicMock(
                return_value={"cached_count": 2, "cache_ttl_minutes": 60}
            )

            response = await async_client.get("/api/v1/schema/health")

            assert response.status_code == 200
            data = response.json()

            assert "status" in data
            assert "service" in data
            assert data["status"] == "healthy"
            assert data["service"] == "schema_manager"
            assert "cached_count" in data
            assert "cache_ttl_minutes" in data

    @pytest.mark.integration
    @pytest.mark.api
    @pytest.mark.schema
    async def test_schema_api_error_handling(self, async_client):
        """Test error handling in schema API."""
        graph_id = "test-graph-12345"

        with patch("app.api.schema.schema_manager") as mock_manager:
            mock_manager.extract_schema = AsyncMock(
                side_effect=Exception("Database connection failed")
            )

            response = await async_client.get(f"/api/v1/schema/info/{graph_id}")

            assert response.status_code == 500
            data = response.json()

            assert "detail" in data
            assert "Failed to extract schema" in data["detail"]

    @pytest.mark.integration
    @pytest.mark.api
    @pytest.mark.schema
    async def test_refresh_schema_validation_error(self, async_client):
        """Test schema refresh with validation error."""
        # Missing required graph_id
        request_data = {"force_refresh": True}

        response = await async_client.post("/api/v1/schema/refresh", json=request_data)

        assert response.status_code == 422  # Validation error
        data = response.json()

        assert "detail" in data

    @pytest.mark.integration
    @pytest.mark.api
    @pytest.mark.schema
    async def test_schema_health_check_failure(self, async_client):
        """Test schema health check when service fails."""
        with patch("app.api.schema.schema_manager") as mock_manager:
            mock_manager.get_cache_stats = MagicMock(
                side_effect=Exception("Service failed")
            )

            response = await async_client.get("/api/v1/schema/health")

            assert response.status_code == 200  # Health endpoint should not fail
            data = response.json()

            assert "status" in data
            assert data["status"] == "unhealthy"
            assert "error" in data

    @pytest.mark.integration
    @pytest.mark.api
    @pytest.mark.schema
    async def test_get_nonexistent_graph_schema(self, async_client):
        """Test getting schema for nonexistent graph."""
        graph_id = "nonexistent-graph"

        with patch("app.api.schema.schema_manager") as mock_manager:
            # Mock empty schema for nonexistent graph
            empty_schema = GraphSchema(
                graph_id=graph_id,
                nodes={},
                relationships={},
                constraints=[],
                indexes=[],
                last_updated=datetime.now(UTC),
                schema_version="empty",
            )
            mock_manager.extract_schema = AsyncMock(return_value=empty_schema)

            response = await async_client.get(f"/api/v1/schema/info/{graph_id}")

            assert response.status_code == 200
            data = response.json()

            assert data["graph_id"] == graph_id
            assert len(data["nodes"]) == 0
            assert len(data["relationships"]) == 0

    @pytest.mark.integration
    @pytest.mark.api
    @pytest.mark.schema
    async def test_schema_api_response_format(self, async_client, mock_schema):
        """Test that all schema API responses have correct format."""
        graph_id = "test-graph-12345"

        with patch("app.api.schema.schema_manager") as mock_manager:
            mock_manager.extract_schema = AsyncMock(return_value=mock_schema)
            mock_manager.format_schema_for_text2cypher = MagicMock(
                return_value="Formatted schema"
            )

            # Test schema info response format
            response = await async_client.get(f"/api/v1/schema/info/{graph_id}")
            assert response.status_code == 200

            data = response.json()

            # Check that dates are in ISO format
            from datetime import datetime

            datetime.fromisoformat(data["last_updated"].replace("Z", "+00:00"))

            # Check that node info has expected structure
            for _node_label, node_info in data["nodes"].items():
                assert isinstance(node_info, dict)
                assert "sample_count" in node_info
                assert "property_count" in node_info
                assert "properties" in node_info
                assert isinstance(node_info["properties"], list)

            # Check that relationship info has expected structure
            for _rel_type, rel_info in data["relationships"].items():
                assert isinstance(rel_info, dict)
                assert "sample_count" in rel_info
                assert "property_count" in rel_info
                assert "start_labels" in rel_info
                assert "end_labels" in rel_info
                assert isinstance(rel_info["start_labels"], list)
                assert isinstance(rel_info["end_labels"], list)
