# app/components/multi_tenant_components.py
"""
Multi-tenant wrapper components following Neo4j GraphRAG factory patterns.
Simple, maintainable wrappers that inject graph_id for tenant isolation.
"""

from datetime import UTC, datetime

from neo4j import Driver
from neo4j_graphrag.embeddings.base import Embedder
from neo4j_graphrag.experimental.components.kg_writer import Neo4jWriter
from neo4j_graphrag.experimental.components.types import Neo4jGraph
from neo4j_graphrag.retrievers import (
    HybridRetriever,
    VectorCypherRetriever,
    VectorRetriever,
)
from neo4j_graphrag.retrievers.base import Retriever
from neo4j_graphrag.types import RetrieverResult

from app.core.logging import get_logger

logger = get_logger(__name__)


class MultiTenantRetriever(Retriever):
    """
    Base multi-tenant wrapper for any Neo4j GraphRAG retriever.

    DESIGN PRINCIPLES:
    - Simple composition over inheritance
    - Compatible with Neo4j GraphRAG factory patterns
    - Automatic graph_id filtering for perfect tenant isolation
    """

    def __init__(self, base_retriever: Retriever, graph_id: str):
        """
        Args:
            base_retriever: Any Neo4j GraphRAG retriever (VectorRetriever, HybridRetriever, etc.)
            graph_id: Tenant identifier for filtering
        """
        self.base_retriever = base_retriever
        self.graph_id = graph_id
        # Initialize parent with same driver
        super().__init__(base_retriever.driver)

    def get_search_results(
        self, query_vector=None, query_text=None, **kwargs
    ) -> RetrieverResult:
        """Add graph_id filter and delegate to base retriever"""

        # Add multi-tenant filtering via index filters (for VectorRetriever/HybridRetriever)
        filters = kwargs.get("filters", {})
        filters["graph_id"] = self.graph_id
        kwargs["filters"] = filters

        # Inject graph_id as a Cypher query parameter (for VectorCypherRetriever retrieval_query)
        query_params = kwargs.get("query_params", {})
        query_params["graph_id"] = self.graph_id
        kwargs["query_params"] = query_params

        logger.debug(f"Multi-tenant search for graph {self.graph_id}")

        # Delegate to base retriever
        results = self.base_retriever.get_search_results(
            query_vector=query_vector, query_text=query_text, **kwargs
        )

        # Additional safety filtering (in case base retriever doesn't support filters)
        filtered_items = []
        for item in results.items:
            # Check metadata for graph_id match
            if (
                hasattr(item, "metadata")
                and item.metadata.get("graph_id") == self.graph_id
            ):
                filtered_items.append(item)
            # Fallback: check if item content contains graph_id reference
            elif hasattr(item, "content") and self.graph_id in str(
                getattr(item, "content", "")
            ):
                filtered_items.append(item)

        return RetrieverResult(items=filtered_items)

    def __getattr__(self, name):
        """Delegate other attributes to wrapped retriever"""
        return getattr(self.base_retriever, name)


class MultiTenantVectorRetriever(MultiTenantRetriever):
    """
    Multi-tenant vector retriever factory.
    Compatible with Neo4j GraphRAG factory patterns.
    """

    @classmethod
    def create(
        cls,
        driver: Driver,
        index_name: str,
        embedder: Embedder,
        graph_id: str,
        return_properties: list[str] | None = None,
        **kwargs,
    ) -> "MultiTenantVectorRetriever":
        """
        Factory method following Neo4j GraphRAG patterns.

        Args:
            driver: Neo4j driver instance
            index_name: Base index name (will be made tenant-specific)
            embedder: Neo4j GraphRAG embedder
            graph_id: Tenant identifier
            return_properties: Properties to return from search
            **kwargs: Additional VectorRetriever arguments
        """
        # Create tenant-specific index name
        tenant_index_name = f"{index_name}_{graph_id}"

        # Create base Neo4j GraphRAG retriever
        base_retriever = VectorRetriever(
            driver=driver,
            index_name=tenant_index_name,
            embedder=embedder,
            return_properties=return_properties or ["text", "chunk_index"],
            **kwargs,
        )

        return cls(base_retriever, graph_id)


class MultiTenantVectorCypherRetriever(MultiTenantRetriever):
    """
    Multi-tenant vector+cypher retriever factory.
    Compatible with Neo4j GraphRAG factory patterns.
    """

    @classmethod
    def create(
        cls,
        driver: Driver,
        index_name: str,
        embedder: Embedder,
        retrieval_query: str,
        graph_id: str,
        **kwargs,
    ) -> "MultiTenantVectorCypherRetriever":
        """
        Factory method with automatic graph_id injection in Cypher queries.

        Args:
            driver: Neo4j driver instance
            index_name: Base index name (will be made tenant-specific)
            embedder: Neo4j GraphRAG embedder
            retrieval_query: Cypher query template
            graph_id: Tenant identifier
            **kwargs: Additional VectorCypherRetriever arguments
        """
        # Create tenant-specific index name
        tenant_index_name = f"{index_name}_{graph_id}"

        # Inject parameterized graph_id filter into Cypher query
        safe_query = cls._inject_graph_id_filter(retrieval_query)

        # Create base Neo4j GraphRAG retriever
        base_retriever = VectorCypherRetriever(
            driver=driver,
            index_name=tenant_index_name,
            embedder=embedder,
            retrieval_query=safe_query,
            **kwargs,
        )

        return cls(base_retriever, graph_id)

    @staticmethod
    def _inject_graph_id_filter(query: str) -> str:
        """
        Inject parameterized graph_id filter into Cypher queries.

        Uses $graph_id parameter — never interpolates values directly into Cypher.
        The caller must pass {"graph_id": graph_id_value} as query_params.
        """
        if "MATCH" not in query or "$graph_id" in query:
            return query

        lines = query.split("\n")
        modified_lines = []
        filter_added = False

        for line in lines:
            modified_lines.append(line)
            if line.strip().startswith("MATCH") and not filter_added:
                modified_lines.append("WHERE node.graph_id = $graph_id")
                filter_added = True

        return "\n".join(modified_lines)


class MultiTenantHybridRetriever(MultiTenantRetriever):
    """
    Multi-tenant hybrid retriever factory.
    Compatible with Neo4j GraphRAG factory patterns.
    """

    @classmethod
    def create(
        cls,
        driver: Driver,
        vector_index_name: str,
        fulltext_index_name: str,
        embedder: Embedder,
        graph_id: str,
        **kwargs,
    ) -> "MultiTenantHybridRetriever":
        """
        Factory method for hybrid (vector + fulltext) search.

        Args:
            driver: Neo4j driver instance
            vector_index_name: Base vector index name
            fulltext_index_name: Base fulltext index name
            embedder: Neo4j GraphRAG embedder
            graph_id: Tenant identifier
            **kwargs: Additional HybridRetriever arguments
        """
        # Create tenant-specific index names
        tenant_vector_index = f"{vector_index_name}_{graph_id}"
        tenant_fulltext_index = f"{fulltext_index_name}_{graph_id}"

        # Create base Neo4j GraphRAG hybrid retriever
        base_retriever = HybridRetriever(
            driver=driver,
            vector_index_name=tenant_vector_index,
            fulltext_index_name=tenant_fulltext_index,
            embedder=embedder,
            **kwargs,
        )

        return cls(base_retriever, graph_id)


class MultiTenantKGWriter:
    """
    Multi-tenant wrapper for Neo4j KG Writer.
    Automatically injects graph_id into all nodes and relationships.

    DESIGN PRINCIPLES:
    - Simple wrapper around Neo4jWriter
    - Automatic tenant metadata injection
    - Compatible with Neo4j GraphRAG pipelines
    """

    _MAX_INGESTION_SOURCE_LEN = 512

    @staticmethod
    def _sanitize_source(raw: str | None) -> str | None:
        """Strip null bytes, leading/trailing whitespace, and enforce max length.

        Returns None when the cleaned result is empty or the input was None.
        """
        if raw is None:
            return None
        cleaned = raw.replace("\x00", "").strip()[: MultiTenantKGWriter._MAX_INGESTION_SOURCE_LEN]
        return cleaned or None

    def __init__(
        self,
        base_writer: Neo4jWriter,
        graph_id: str,
        user_id: str | None = None,
        ingestion_source: str | None = None,
    ):
        """
        Args:
            base_writer: Neo4j GraphRAG writer instance
            graph_id: Tenant identifier
            user_id: Optional user identifier for additional isolation
            ingestion_source: Document filename, connector ID, or API name that
                provided the data being written — stored as the bitemporal
                ingestion_source property on every node and relationship.
        """
        self.base_writer = base_writer
        self.graph_id = graph_id
        self.user_id = user_id
        self.ingestion_source = ingestion_source

    # Labels that legitimately have no `name` property (chunks/documents carry
    # `text` / `path` instead). Empty-name entity filtering only applies to
    # other labels — i.e. things the LLM classified as entity types.
    _LEXICAL_LABELS = frozenset({"Chunk", "Document"})

    async def run(self, graph: Neo4jGraph) -> None:
        """Write graph with automatic tenant metadata injection.

        Also drops LLM-extracted entities whose `name` property is missing or
        empty (TASK-061 / STORY-025). Empty-name entities accumulate junk
        relationships and never deduplicate (the resolver groups by `name`),
        so the cleanest place to handle them is at write time.
        """
        now = datetime.now(UTC)

        # === TASK-061: drop empty-name entities before we touch anything else ===
        # Build the set of dropped node IDs so we can also filter the relationships
        # that would have pointed at them (avoids dangling MERGEs downstream).
        dropped_ids: set[str] = set()
        kept_nodes = []
        for node in graph.nodes:
            label = getattr(node, "label", None)
            if label in self._LEXICAL_LABELS:
                kept_nodes.append(node)
                continue
            raw_name = (node.properties or {}).get("name") if node.properties else None
            if not (isinstance(raw_name, str) and raw_name.strip()):
                dropped_ids.add(node.id)
                continue
            kept_nodes.append(node)

        if dropped_ids:
            logger.info(
                "MultiTenantKGWriter: dropped %d empty-name entities for tenant %s, source=%s",
                len(dropped_ids), self.graph_id, self.ingestion_source,
            )
            graph.nodes = kept_nodes
            # Drop relationships that would dangle on a removed node
            kept_rels = [
                r for r in graph.relationships
                if r.start_node_id not in dropped_ids
                and r.end_node_id not in dropped_ids
            ]
            dropped_rel_count = len(graph.relationships) - len(kept_rels)
            if dropped_rel_count:
                logger.info(
                    "MultiTenantKGWriter: dropped %d relationships pointing to filtered nodes",
                    dropped_rel_count,
                )
                graph.relationships = kept_rels

        # Inject graph_id, transaction_time, and bitemporal provenance into all nodes
        for node in graph.nodes:
            if not node.properties:
                node.properties = {}
            # graph_id: unconditional overwrite -- this IS the tenant isolation boundary.
            # Never change to setdefault(); a caller-supplied graph_id in node.properties
            # would bypass multi-tenancy. See security-findings.md Finding 4, TASK-005 review.
            node.properties.update(
                {
                    "graph_id": self.graph_id,
                    "created_by": "multi_tenant_pipeline",
                    "transaction_time": now,
                    "ingestion_time": now,
                }
            )
            # Always strip any caller/LLM-provided ingestion_source and apply sanitized
            # writer value to prevent prompt injection via long or null-byte-bearing strings.
            node.properties.pop("ingestion_source", None)
            safe = self._sanitize_source(self.ingestion_source)
            if safe is not None:
                node.properties["ingestion_source"] = safe

            # Add user_id if provided
            if self.user_id:
                node.properties["user_id"] = self.user_id

        # Inject graph_id, transaction_time, and bitemporal provenance into all relationships
        for rel in graph.relationships:
            if not rel.properties:
                rel.properties = {}
            # graph_id: unconditional overwrite -- this IS the tenant isolation boundary.
            # Never change to setdefault(); a caller-supplied graph_id in rel.properties
            # would bypass multi-tenancy. See security-findings.md Finding 4, TASK-005 review.
            rel.properties.update(
                {
                    "graph_id": self.graph_id,
                    "created_by": "multi_tenant_pipeline",
                    "transaction_time": now,
                    "ingestion_time": now,
                }
            )
            # Always strip any caller/LLM-provided ingestion_source and apply sanitized
            # writer value to prevent prompt injection via long or null-byte-bearing strings.
            rel.properties.pop("ingestion_source", None)
            safe = self._sanitize_source(self.ingestion_source)
            if safe is not None:
                rel.properties["ingestion_source"] = safe

            # Add user_id if provided
            if self.user_id:
                rel.properties["user_id"] = self.user_id

        logger.info(
            f"Writing graph with {len(graph.nodes)} nodes and {len(graph.relationships)} relationships for tenant {self.graph_id}"
        )

        # Delegate to base writer
        return await self.base_writer.run(graph)

    def __getattr__(self, name):
        """Delegate other attributes to wrapped writer"""
        return getattr(self.base_writer, name)


# ==================== FACTORY FUNCTIONS FOR FASTAPI COMPATIBILITY ====================


def create_multi_tenant_vector_retriever(
    driver: Driver,
    embedder: Embedder,
    graph_id: str,
    index_name: str = "entity_embeddings",
    return_properties: list[str] | None = None,
) -> MultiTenantVectorRetriever:
    """
    FastAPI-compatible factory function for vector retrievers.

    Usage in FastAPI services:
        # Ensure sync driver is connected for GraphRAG components
        neo4j_client.connect_sync()
        retriever = create_multi_tenant_vector_retriever(
            driver=neo4j_client.sync_driver,
            embedder=openai_embedder,
            graph_id=str(graph_id)
        )
    """
    return MultiTenantVectorRetriever.create(
        driver=driver,
        index_name=index_name,
        embedder=embedder,
        graph_id=graph_id,
        return_properties=return_properties,
    )


def create_multi_tenant_hybrid_retriever(
    driver: Driver,
    embedder: Embedder,
    graph_id: str,
    vector_index_name: str = "entity_embeddings",
    fulltext_index_name: str = "entity_text_fulltext",
) -> MultiTenantHybridRetriever:
    """
    FastAPI-compatible factory function for hybrid retrievers.

    Usage in FastAPI services:
        # Ensure sync driver is connected for GraphRAG components
        neo4j_client.connect_sync()
        retriever = create_multi_tenant_hybrid_retriever(
            driver=neo4j_client.sync_driver,
            embedder=openai_embedder,
            graph_id=str(graph_id)
        )
    """
    return MultiTenantHybridRetriever.create(
        driver=driver,
        vector_index_name=vector_index_name,
        fulltext_index_name=fulltext_index_name,
        embedder=embedder,
        graph_id=graph_id,
    )


def create_multi_tenant_kg_writer(
    driver: Driver,
    graph_id: str,
    user_id: str | None = None,
    neo4j_database: str = "neo4j",
) -> MultiTenantKGWriter:
    """
    FastAPI-compatible factory function for KG writers.

    Usage in FastAPI services:
        # Ensure sync driver is connected for GraphRAG components
        neo4j_client.connect_sync()
        writer = create_multi_tenant_kg_writer(
            driver=neo4j_client.sync_driver,
            graph_id=str(graph_id),
            user_id=current_user_id
        )
    """
    base_writer = Neo4jWriter(driver=driver, neo4j_database=neo4j_database)

    return MultiTenantKGWriter(base_writer, graph_id, user_id)
