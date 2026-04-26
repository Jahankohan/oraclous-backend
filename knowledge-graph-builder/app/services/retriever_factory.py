"""
Retriever Factory Service

Factory for creating and managing all types of Neo4j GraphRAG retrievers
with multi-tenant support and dynamic configuration.
"""

from functools import lru_cache
from typing import cast

from neo4j_graphrag.embeddings import OpenAIEmbeddings
from neo4j_graphrag.llm import OpenAILLM
from neo4j_graphrag.retrievers import (
    HybridCypherRetriever,
    HybridRetriever,
    Text2CypherRetriever,
    VectorCypherRetriever,
    VectorRetriever,
)
from neo4j_graphrag.retrievers.base import Retriever

from app.core.config import settings
from app.core.logging import get_logger
from app.core.neo4j_client import neo4j_client
from app.schemas.retriever_schemas import (
    HybridCypherRetrieverConfig,
    HybridRetrieverConfig,
    MemoryRetrieverConfig,
    RetrieverConfig,
    RetrieverType,
    Text2CypherRetrieverConfig,
    VectorCypherRetrieverConfig,
    VectorRetrieverConfig,
    get_default_retriever_config,
)
from app.services.schema_service import get_text2cypher_schema

logger = get_logger(__name__)


class RetrieverFactory:
    """
    Factory for creating Neo4j GraphRAG retrievers with multi-tenant support.

    Features:
    - Dynamic retriever creation based on configuration
    - Automatic component initialization (embedders, LLMs)
    - Multi-tenant isolation with graph_id injection
    - Connection management and error handling
    - Performance monitoring and caching
    """

    def __init__(self):
        """Initialize retriever factory"""
        self._embedder_cache: dict[str, OpenAIEmbeddings] = {}
        self._llm_cache: dict[str, OpenAILLM] = {}
        self._initialized = False

    async def _ensure_connections(self):
        """Ensure Neo4j connections are available"""
        if not self._initialized:
            # Ensure both sync and async drivers are connected
            await neo4j_client.connect_async()
            neo4j_client.connect_sync()

            if neo4j_client.sync_driver is None:
                raise ConnectionError(
                    "Failed to establish Neo4j sync driver connection for GraphRAG"
                )

            self._initialized = True
            logger.info("RetrieverFactory initialized with Neo4j connections")

    @lru_cache(maxsize=32)  # noqa: B019
    def _get_embedder(self, model: str = "text-embedding-3-large") -> OpenAIEmbeddings:
        """Get cached embedder instance"""
        if model not in self._embedder_cache:
            self._embedder_cache[model] = OpenAIEmbeddings(
                api_key=settings.OPENAI_API_KEY, model=model
            )
            logger.info(f"Created new embedder for model: {model}")

        return self._embedder_cache[model]

    @lru_cache(maxsize=32)  # noqa: B019
    def _get_llm(
        self, model: str = "gpt-4o", temperature: float = 0.1, max_tokens: int = 3000
    ) -> OpenAILLM:
        """Get cached LLM instance"""
        cache_key = f"{model}_{temperature}_{max_tokens}"

        if cache_key not in self._llm_cache:
            # Check if model supports JSON object response format
            json_supported_models = [
                "gpt-4o",
                "gpt-4o-mini",
                "gpt-4-turbo",
                "gpt-4-1106-preview",
                "gpt-4-0125-preview",
                "gpt-3.5-turbo-1106",
                "gpt-3.5-turbo-0125",
            ]

            model_params = {"temperature": temperature, "max_tokens": max_tokens}

            if any(supported in model for supported in json_supported_models):
                model_params["response_format"] = {"type": "json_object"}

            self._llm_cache[cache_key] = OpenAILLM(
                model_name=model,
                api_key=settings.OPENAI_API_KEY,
                model_params=model_params,
            )
            logger.info(f"Created new LLM for model: {model}")

        return self._llm_cache[cache_key]

    async def create_vector_retriever(
        self, config: VectorRetrieverConfig, graph_id: str
    ) -> VectorRetriever:
        """Create VectorRetriever with configuration"""
        await self._ensure_connections()

        embedder = self._get_embedder()

        if neo4j_client.sync_driver is None:
            raise ConnectionError("Neo4j sync driver not available")

        retriever = VectorRetriever(
            driver=neo4j_client.sync_driver,
            index_name=config.index_name,
            embedder=embedder,
            return_properties=config.return_properties,
            neo4j_database=settings.NEO4J_DATABASE,
        )

        logger.info(f"Created VectorRetriever for graph {graph_id}")
        return retriever

    async def create_vector_cypher_retriever(
        self, config: VectorCypherRetrieverConfig, graph_id: str
    ) -> VectorCypherRetriever:
        """Create VectorCypherRetriever with configuration"""
        await self._ensure_connections()

        embedder = self._get_embedder()

        # Inject parameterized graph_id filter into retrieval query
        retrieval_query = self._inject_graph_id_filter(config.retrieval_query)

        if neo4j_client.sync_driver is None:
            raise ConnectionError("Neo4j sync driver not available")

        retriever = VectorCypherRetriever(
            driver=neo4j_client.sync_driver,
            index_name=config.index_name,
            retrieval_query=retrieval_query,
            embedder=embedder,
            neo4j_database=settings.NEO4J_DATABASE,
        )

        logger.info(f"Created VectorCypherRetriever for graph {graph_id}")
        return retriever

    async def create_hybrid_retriever(
        self, config: HybridRetrieverConfig, graph_id: str
    ) -> HybridRetriever:
        """Create HybridRetriever with configuration"""
        await self._ensure_connections()

        embedder = self._get_embedder()

        if neo4j_client.sync_driver is None:
            raise ConnectionError("Neo4j sync driver not available")

        retriever = HybridRetriever(
            driver=neo4j_client.sync_driver,
            vector_index_name=config.vector_index_name,
            fulltext_index_name=config.fulltext_index_name,
            embedder=embedder,
            return_properties=config.return_properties,
            neo4j_database=settings.NEO4J_DATABASE,
        )

        logger.info(f"Created HybridRetriever for graph {graph_id}")
        return retriever

    async def create_hybrid_cypher_retriever(
        self, config: HybridCypherRetrieverConfig, graph_id: str
    ) -> HybridCypherRetriever:
        """Create HybridCypherRetriever with configuration"""
        await self._ensure_connections()

        embedder = self._get_embedder()

        # Inject parameterized graph_id filter into retrieval query
        retrieval_query = self._inject_graph_id_filter(config.retrieval_query)

        if neo4j_client.sync_driver is None:
            raise ConnectionError("Neo4j sync driver not available")

        retriever = HybridCypherRetriever(
            driver=neo4j_client.sync_driver,
            vector_index_name=config.vector_index_name,
            fulltext_index_name=config.fulltext_index_name,
            retrieval_query=retrieval_query,
            embedder=embedder,
            neo4j_database=settings.NEO4J_DATABASE,
        )

        logger.info(f"Created HybridCypherRetriever for graph {graph_id}")
        return retriever

    async def create_text2cypher_retriever(
        self, config: Text2CypherRetrieverConfig, graph_id: str
    ) -> Text2CypherRetriever:
        """Create Text2CypherRetriever with configuration"""
        await self._ensure_connections()

        # Get LLM configuration from config or use defaults
        llm_params = config.llm_params or {}
        llm = self._get_llm(
            model=llm_params.get("model", "gpt-4o"),
            temperature=llm_params.get("temperature", 0.1),
            max_tokens=llm_params.get("max_tokens", 3000),
        )

        # Use provided schema or generate default
        neo4j_schema = config.neo4j_schema or await self._generate_schema_for_graph(
            graph_id
        )

        if neo4j_client.sync_driver is None:
            raise ConnectionError("Neo4j sync driver not available")

        retriever = Text2CypherRetriever(
            driver=neo4j_client.sync_driver,
            llm=llm,
            neo4j_schema=neo4j_schema,
            examples=config.examples,
            custom_prompt=config.custom_prompt,
            neo4j_database=settings.NEO4J_DATABASE,
        )

        logger.info(f"Created Text2CypherRetriever for graph {graph_id}")
        return retriever

    async def create_retriever(
        self, retriever_config: RetrieverConfig, graph_id: str
    ) -> Retriever:
        """
        Create retriever based on configuration.

        Args:
            retriever_config: Unified retriever configuration
            graph_id: Graph identifier for multi-tenant isolation

        Returns:
            Configured retriever instance
        """
        retriever_type = retriever_config.type
        config = retriever_config.config

        try:
            if retriever_type == RetrieverType.VECTOR:
                return await self.create_vector_retriever(
                    cast(VectorRetrieverConfig, config), graph_id
                )

            elif retriever_type == RetrieverType.VECTOR_CYPHER:
                return await self.create_vector_cypher_retriever(
                    cast(VectorCypherRetrieverConfig, config), graph_id
                )

            elif retriever_type == RetrieverType.HYBRID:
                return await self.create_hybrid_retriever(
                    cast(HybridRetrieverConfig, config), graph_id
                )

            elif retriever_type == RetrieverType.HYBRID_CYPHER:
                return await self.create_hybrid_cypher_retriever(
                    cast(HybridCypherRetrieverConfig, config), graph_id
                )

            elif retriever_type == RetrieverType.TEXT2CYPHER:
                return await self.create_text2cypher_retriever(
                    cast(Text2CypherRetrieverConfig, config), graph_id
                )

            elif retriever_type == RetrieverType.MEMORY:
                return await self.create_memory_retriever(
                    cast(MemoryRetrieverConfig, config), graph_id
                )

            else:
                raise ValueError(f"Unsupported retriever type: {retriever_type}")

        except Exception as e:
            logger.error(
                f"Failed to create {retriever_type} retriever for graph {graph_id}: {e}"
            )
            raise

    async def create_default_retriever(
        self, retriever_type: RetrieverType, graph_id: str
    ) -> Retriever:
        """
        Create retriever with default configuration.

        Args:
            retriever_type: Type of retriever to create
            graph_id: Graph identifier for multi-tenant isolation

        Returns:
            Configured retriever instance with defaults
        """
        default_config = get_default_retriever_config(retriever_type, graph_id)

        retriever_config = RetrieverConfig(
            type=retriever_type,
            config=cast(
                VectorRetrieverConfig
                | VectorCypherRetrieverConfig
                | HybridRetrieverConfig
                | HybridCypherRetrieverConfig
                | Text2CypherRetrieverConfig,
                default_config,
            ),
        )

        return await self.create_retriever(retriever_config, graph_id)

    def _inject_graph_id_filter(self, retrieval_query: str) -> str:
        """
        Inject parameterized graph_id filter into retrieval query for multi-tenant isolation.

        Uses $graph_id parameter — never interpolates values directly into Cypher.
        The caller must pass {"graph_id": graph_id_value} as query_params when executing.
        """
        if "$graph_id" in retrieval_query:
            return retrieval_query

        if "WHERE" in retrieval_query.upper():
            return retrieval_query.replace(
                "WHERE",
                "WHERE node.graph_id = $graph_id AND ",
                1,
            )
        else:
            lines = retrieval_query.strip().split("\n")
            if len(lines) > 1:
                lines.insert(1, "WHERE node.graph_id = $graph_id")
                return "\n".join(lines)
            else:
                return f"{retrieval_query}\nWHERE node.graph_id = $graph_id"

    def inject_temporal_filter_into_retrieval_query(
        self,
        retrieval_query: str,
        temporal_filter: str | None,
        retriever_type: "RetrieverType | None" = None,
    ) -> str:
        """Inject a temporal WHERE clause into a Cypher retrieval query.

        This method is used by Cypher-capable retrievers (VECTOR_CYPHER,
        HYBRID_CYPHER) to scope results to a specific time window.
        For vector-only retrievers (VECTOR, HYBRID, TEXT2CYPHER), calling this
        method logs a warning and returns the query unchanged.

        Security: ``temporal_filter`` must be a trusted clause constructed by
        ``ChatService._build_temporal_filter()`` — it is NEVER derived from
        raw user input.  Datetime values must always be passed as Cypher
        parameters (via ``query_params`` at search time), never interpolated.

        Args:
            retrieval_query: The base Cypher retrieval query string.
            temporal_filter: A trusted WHERE-clause fragment, e.g.
                "(r.event_time IS NULL OR r.event_time <= $temporal_at)".
                Pass None or empty string to return the query unchanged.
            retriever_type: The retriever type — used to log a warning for
                vector-only retrievers that cannot apply relationship filters.

        Returns:
            The retrieval query with the temporal clause appended to the WHERE
            block, or the original query if no filter is provided.
        """
        if not temporal_filter:
            return retrieval_query

        # Vector-only retrievers have no relationship traversal — filter is a no-op.
        _vector_only = {RetrieverType.VECTOR, RetrieverType.HYBRID, RetrieverType.TEXT2CYPHER}
        if retriever_type in _vector_only:
            logger.warning(
                "Temporal filter not supported for %s retriever; ignoring",
                retriever_type.value if retriever_type else "unknown",
            )
            return retrieval_query

        # Append the temporal clause after the existing WHERE predicate(s).
        # The clause references relationship alias `r` which is present in
        # all default VECTOR_CYPHER / HYBRID_CYPHER retrieval queries.
        if "WHERE" in retrieval_query.upper():
            # Append with AND to keep existing predicates intact.
            return retrieval_query.rstrip() + f"\n    AND {temporal_filter}"
        else:
            return retrieval_query.rstrip() + f"\nWHERE {temporal_filter}"

    async def create_memory_retriever(
        self,
        config: MemoryRetrieverConfig,
        graph_id: str,
    ) -> "MemoryRetriever":
        """Create MemoryRetriever for agent memory recall."""
        return MemoryRetriever(graph_id=graph_id, config=config)

    async def _generate_schema_for_graph(self, graph_id: str) -> str:
        """
        Generate Neo4j schema description for a specific graph using schema service.

        Falls back to basic schema if schema service fails.
        """
        try:
            # Use schema service for comprehensive schema extraction
            return await get_text2cypher_schema(graph_id)

        except Exception as e:
            logger.warning(
                f"Failed to get schema from schema service for graph {graph_id}: {e}"
            )

            # Fallback to basic schema
            return f"""
            // Multi-tenant Knowledge Graph Schema for graph_id: {graph_id}
            
            // Core entity types
            (:Entity {{name: string, type: string, graph_id: string}})
            (:Chunk {{text: string, graph_id: string, embedding: vector}})
            (:Document {{path: string, title: string, graph_id: string}})
            
            // Entity relationships
            (:Entity)-[:FOUNDED_BY]->(:Entity)
            (:Entity)-[:CEO_OF]->(:Entity)
            (:Entity)-[:LOCATED_IN]->(:Entity)
            (:Entity)-[:PARTNER_WITH]->(:Entity)
            (:Entity)-[:DEVELOPS]->(:Entity)
            
            // Document relationships
            (:Entity)-[:FROM_CHUNK]->(:Chunk)
            (:Chunk)-[:FROM_DOCUMENT]->(:Document)
            
            // All nodes have graph_id property for multi-tenant isolation
            """


# ==================== MEMORY RETRIEVER ====================


class MemoryRetriever:
    """
    Retriever that queries the Agent Memory API instead of the standard
    vector/fulltext indexes.  Drop-in compatible with the Retriever protocol
    used by chat_service: exposes a `search(query)` coroutine returning a
    list of result dicts.
    """

    def __init__(self, graph_id: str, config: MemoryRetrieverConfig) -> None:
        self.graph_id = graph_id
        self.config = config

    async def search(self, query: str) -> list:
        from app.services.memory_service import memory_service

        response = await memory_service.search_memories(
            graph_id=self.graph_id,
            query=query,
            memory_type=None,
            scope=None,
            temporal="current",
            min_confidence=self.config.min_confidence,
            limit=self.config.top_k,
            include_graph_facts=False,
        )
        return [
            {
                "content": m.content,
                "score": m.relevance_score,
                "memory_id": m.memory_id,
                "type": m.type.value,
                "importance_score": m.importance_score,
            }
            for m in response.memories
        ]

    # Sync shim so callers using the sync Retriever protocol still work
    def get_search_results(self, query_text: str, **kwargs) -> list:  # type: ignore[override]
        import asyncio

        return asyncio.run(self.search(query_text))


# ==================== GLOBAL FACTORY INSTANCE ====================

# Global factory instance for dependency injection
retriever_factory = RetrieverFactory()


# ==================== CONVENIENCE FUNCTIONS ====================


async def create_retriever_from_config(
    retriever_config: RetrieverConfig, graph_id: str
) -> Retriever:
    """Convenience function to create retriever from configuration"""
    return await retriever_factory.create_retriever(retriever_config, graph_id)


async def create_default_retriever(
    retriever_type: RetrieverType, graph_id: str
) -> Retriever:
    """Convenience function to create retriever with default configuration"""
    return await retriever_factory.create_default_retriever(retriever_type, graph_id)


def get_supported_retriever_types() -> dict[str, str]:
    """Get mapping of supported retriever types and their descriptions"""
    return {
        RetrieverType.VECTOR: "Simple vector similarity search",
        RetrieverType.VECTOR_CYPHER: "Vector search with graph traversal",
        RetrieverType.HYBRID: "Combined vector and full-text search",
        RetrieverType.HYBRID_CYPHER: "Hybrid search with graph traversal",
        RetrieverType.TEXT2CYPHER: "Natural language to Cypher query generation",
        RetrieverType.MEMORY: "Agent memory recall (episodic, semantic, procedural)",
    }
