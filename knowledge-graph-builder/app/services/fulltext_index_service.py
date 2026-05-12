"""
Full-text Index Management Service

Service for creating and managing Neo4j full-text indexes required by
HybridRetriever and HybridCypherRetriever with multi-tenant support.
"""

from dataclasses import dataclass
from typing import Any

from neo4j.exceptions import ClientError
from neo4j_graphrag.indexes import create_fulltext_index

from app.core.logging import get_logger
from app.core.neo4j_client import neo4j_client

logger = get_logger(__name__)


@dataclass
class FullTextIndexConfig:
    """Configuration for full-text index creation"""

    name: str
    label: str
    node_properties: list[str]
    graph_id: str | None = None  # For multi-tenant filtering


class FullTextIndexManager:
    """
    Manager for Neo4j full-text indexes with multi-tenant support.

    Features:
    - Automatic index creation and validation
    - Multi-tenant index isolation
    - Index health checking and monitoring
    - Batch index operations
    """

    def __init__(self):
        """Initialize full-text index manager"""
        self._created_indexes: dict[str, FullTextIndexConfig] = {}

    async def _ensure_connections(self):
        """Ensure Neo4j connections are available"""
        await neo4j_client.connect_async()
        neo4j_client.connect_sync()

        if neo4j_client.sync_driver is None:
            raise ConnectionError(
                "Neo4j sync driver not available for index operations"
            )

    async def create_fulltext_index(self, config: FullTextIndexConfig) -> bool:
        """
        Create a full-text index based on configuration.

        Args:
            config: Full-text index configuration

        Returns:
            True if index was created or already exists, False otherwise
        """
        try:
            await self._ensure_connections()

            # Check if index already exists
            if await self.index_exists(config.name):
                logger.info(f"Full-text index '{config.name}' already exists")
                self._created_indexes[config.name] = config
                return True

            # Create the index using neo4j_graphrag helper
            if neo4j_client.sync_driver is None:
                raise ConnectionError("Neo4j sync driver not available")

            create_fulltext_index(
                driver=neo4j_client.sync_driver,
                name=config.name,
                label=config.label,
                node_properties=config.node_properties,
            )

            # Verify index was created
            if await self.index_exists(config.name):
                self._created_indexes[config.name] = config
                logger.info(f"Successfully created full-text index '{config.name}'")
                return True
            else:
                logger.error(
                    f"Failed to create full-text index '{config.name}' - verification failed"
                )
                return False

        except ClientError as e:
            if "already exists" in str(e).lower():
                logger.info(f"Full-text index '{config.name}' already exists")
                self._created_indexes[config.name] = config
                return True
            else:
                logger.error(
                    f"Neo4j client error creating full-text index '{config.name}': {e}"
                )
                return False
        except Exception as e:
            logger.error(f"Failed to create full-text index '{config.name}': {e}")
            return False

    async def index_exists(self, index_name: str) -> bool:
        """
        Check if a full-text index exists.

        Args:
            index_name: Name of the index to check

        Returns:
            True if index exists, False otherwise
        """
        try:
            if neo4j_client.async_driver is None:
                await neo4j_client.connect_async()

            query = "SHOW FULLTEXT INDEXES YIELD name WHERE name = $index_name"
            records = await neo4j_client.execute_query(
                query, {"index_name": index_name}
            )

            return len(records) > 0

        except Exception as e:
            logger.error(f"Error checking if index '{index_name}' exists: {e}")
            return False

    async def get_fulltext_indexes(self) -> list[dict[str, Any]]:
        """
        Get list of all full-text indexes in the database.

        Returns:
            List of index information dictionaries
        """
        try:
            if neo4j_client.async_driver is None:
                await neo4j_client.connect_async()

            query = """
            SHOW FULLTEXT INDEXES
            YIELD name, labelsOrTypes, properties, state, type
            WHERE type = 'FULLTEXT'
            """
            records = await neo4j_client.execute_query(query)

            return [
                {
                    "name": record.get("name"),
                    "labels": record.get("labelsOrTypes"),
                    "properties": record.get("properties"),
                    "state": record.get("state"),
                    "type": record.get("type"),
                }
                for record in records
            ]

        except Exception as e:
            logger.error(f"Error getting full-text indexes: {e}")
            return []

    async def drop_fulltext_index(self, index_name: str) -> bool:
        """
        Drop a full-text index.

        Args:
            index_name: Name of the index to drop

        Returns:
            True if index was dropped, False otherwise
        """
        try:
            if neo4j_client.async_driver is None:
                await neo4j_client.connect_async()

            query = f"DROP INDEX {index_name} IF EXISTS"
            await neo4j_client.execute_query(query)

            # Remove from our tracking
            if index_name in self._created_indexes:
                del self._created_indexes[index_name]

            logger.info(f"Successfully dropped full-text index '{index_name}'")
            return True

        except Exception as e:
            logger.error(f"Error dropping full-text index '{index_name}': {e}")
            return False

    async def ensure_chunk_fulltext_index(self, graph_id: str) -> str:
        """
        Ensure full-text index exists for chunk nodes with multi-tenant support.

        Args:
            graph_id: Graph identifier for multi-tenant isolation

        Returns:
            Name of the created/existing index
        """
        index_name = f"fulltext_chunks_{graph_id[:8]}"  # Short hash for readability

        config = FullTextIndexConfig(
            name=index_name,
            label="Chunk",
            node_properties=["text", "title"],
            graph_id=graph_id,
        )

        success = await self.create_fulltext_index(config)
        if not success:
            # Fallback to global index
            fallback_name = "fulltext_chunks"
            fallback_config = FullTextIndexConfig(
                name=fallback_name, label="Chunk", node_properties=["text", "title"]
            )
            await self.create_fulltext_index(fallback_config)
            return fallback_name

        return index_name

    async def ensure_document_fulltext_index(self, graph_id: str) -> str:
        """
        Ensure full-text index exists for document nodes with multi-tenant support.

        Args:
            graph_id: Graph identifier for multi-tenant isolation

        Returns:
            Name of the created/existing index
        """
        index_name = f"fulltext_documents_{graph_id[:8]}"  # Short hash for readability

        config = FullTextIndexConfig(
            name=index_name,
            label="Document",
            node_properties=["title", "path", "content"],
            graph_id=graph_id,
        )

        success = await self.create_fulltext_index(config)
        if not success:
            # Fallback to global index
            fallback_name = "fulltext_documents"
            fallback_config = FullTextIndexConfig(
                name=fallback_name,
                label="Document",
                node_properties=["title", "path", "content"],
            )
            await self.create_fulltext_index(fallback_config)
            return fallback_name

        return index_name

    async def setup_default_indexes(self, graph_id: str) -> dict[str, str]:
        """
        Set up all default full-text indexes for a graph.

        Args:
            graph_id: Graph identifier for multi-tenant isolation

        Returns:
            Dictionary mapping index types to index names
        """
        try:
            chunk_index = await self.ensure_chunk_fulltext_index(graph_id)
            document_index = await self.ensure_document_fulltext_index(graph_id)

            indexes = {"chunks": chunk_index, "documents": document_index}

            logger.info(
                f"Set up default full-text indexes for graph {graph_id}: {indexes}"
            )
            return indexes

        except Exception as e:
            logger.error(f"Failed to set up default indexes for graph {graph_id}: {e}")
            return {}

    async def validate_hybrid_retriever_requirements(
        self, vector_index_name: str, fulltext_index_name: str
    ) -> dict[str, bool]:
        """
        Validate that both vector and full-text indexes exist for hybrid retrieval.

        Args:
            vector_index_name: Name of the vector index
            fulltext_index_name: Name of the full-text index

        Returns:
            Dictionary with validation results
        """
        try:
            # Check vector index
            vector_exists = await self._check_vector_index_exists(vector_index_name)

            # Check full-text index
            fulltext_exists = await self.index_exists(fulltext_index_name)

            return {
                "vector_index_exists": vector_exists,
                "fulltext_index_exists": fulltext_exists,
                "hybrid_ready": vector_exists and fulltext_exists,
            }

        except Exception as e:
            logger.error(f"Error validating hybrid retriever requirements: {e}")
            return {
                "vector_index_exists": False,
                "fulltext_index_exists": False,
                "hybrid_ready": False,
            }

    async def _check_vector_index_exists(self, index_name: str) -> bool:
        """Check if a vector index exists"""
        try:
            if neo4j_client.async_driver is None:
                await neo4j_client.connect_async()

            query = "SHOW VECTOR INDEXES YIELD name WHERE name = $index_name"
            records = await neo4j_client.execute_query(
                query, {"index_name": index_name}
            )

            return len(records) > 0

        except Exception as e:
            logger.error(f"Error checking vector index '{index_name}': {e}")
            return False

    def get_created_indexes(self) -> dict[str, FullTextIndexConfig]:
        """Get dictionary of indexes created by this manager"""
        return self._created_indexes.copy()


# ==================== GLOBAL MANAGER INSTANCE ====================

# Global manager instance for dependency injection
fulltext_index_manager = FullTextIndexManager()


# ==================== CONVENIENCE FUNCTIONS ====================


async def ensure_fulltext_index(config: FullTextIndexConfig) -> bool:
    """Convenience function to create full-text index"""
    return await fulltext_index_manager.create_fulltext_index(config)


async def setup_hybrid_indexes(graph_id: str) -> dict[str, str]:
    """Convenience function to set up indexes for hybrid retrieval"""
    return await fulltext_index_manager.setup_default_indexes(graph_id)


async def validate_hybrid_setup(
    vector_index_name: str, fulltext_index_name: str
) -> dict[str, bool]:
    """Convenience function to validate hybrid retriever setup"""
    return await fulltext_index_manager.validate_hybrid_retriever_requirements(
        vector_index_name, fulltext_index_name
    )
