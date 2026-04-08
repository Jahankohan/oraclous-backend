"""
Multi-Tenant Retrieval Service - Neo4j GraphRAG Foundation
Clean, maintainable retrieval service following Neo4j GraphRAG patterns with FastAPI compatibility.

DESIGN PRINCIPLES:
- Neo4j GraphRAG components as foundation  
- Multi-tenant wrappers for perfect isolation
- Simple factory patterns (no complex inheritance)
- FastAPI compatible with proper async support
- Direct driver access following existing patterns
"""

from typing import List, Dict, Any, Optional
from uuid import UUID

from neo4j_graphrag.embeddings import OpenAIEmbeddings
from neo4j_graphrag.types import RetrieverResult

from app.components.multi_tenant_components import (
    MultiTenantVectorRetriever,
    MultiTenantVectorCypherRetriever,
    MultiTenantHybridRetriever,
    create_multi_tenant_vector_retriever,
    create_multi_tenant_hybrid_retriever
)
from app.core.neo4j_client import neo4j_client
from app.core.config import settings
from app.core.logging import get_logger

logger = get_logger(__name__)


class RetrievalService:
    """
    Multi-tenant retrieval service using Neo4j GraphRAG foundation.
    
    FEATURES:
    - Semantic search using Neo4j GraphRAG VectorRetriever
    - Hybrid search combining vector + fulltext search  
    - Graph-aware search with relationship context
    - Perfect tenant isolation with graph_id filtering
    - FastAPI compatible with async support
    - Simple, maintainable code following factory patterns
    """
    
    def __init__(self, driver=None, embedder=None):
        """
        Initialize with Neo4j GraphRAG components.

        Args:
            driver: Neo4j driver (defaults to global client, connected lazily on first use)
            embedder: OpenAI embedder (defaults to configured instance)
        """
        # Store driver — if None, connection happens lazily on first retriever use
        self.driver = driver

        # Create OpenAI embedder (Neo4j GraphRAG pattern)
        self.embedder = embedder or OpenAIEmbeddings(
            model=settings.EMBEDDING_MODEL or "text-embedding-3-large",
            api_key=settings.OPENAI_API_KEY
        )

        # Simple retriever cache for performance
        self._retriever_cache = {}

        logger.info("RetrievalService initialized with Neo4j GraphRAG components")
    
    # ==================== ENTITY SEARCH (SEMANTIC) ====================
    
    async def similarity_search_entities(
        self,
        query: str,
        graph_id: UUID,
        k: int = 5,
        threshold: float = 0.7
    ) -> List[Dict[str, Any]]:
        """
        Semantic similarity search on entities using Neo4j GraphRAG VectorRetriever.
        
        Args:
            query: Search query text
            graph_id: Tenant graph identifier
            k: Number of results to return
            threshold: Similarity threshold (0.0 to 1.0)
            
        Returns:
            List of entity dictionaries with similarity scores
        """
        try:
            # Get multi-tenant entity retriever
            retriever = self._get_entity_retriever(graph_id)
            
            # Perform semantic search
            result = retriever.get_search_results(query_text=query, top_k=k)
            
            # Convert to standard format
            entities = self._convert_retriever_result_to_entities(result, threshold)
            
            logger.info(f"Entity similarity search found {len(entities)} results for graph {graph_id}")
            return entities
            
        except Exception as e:
            logger.error(f"Entity similarity search failed for graph {graph_id}: {e}")
            return []
    
    # ==================== CHUNK SEARCH (SEMANTIC) ====================
    
    async def similarity_search_chunks(
        self,
        query: str,
        graph_id: UUID,
        k: int = 10,
        threshold: float = 0.7
    ) -> List[Dict[str, Any]]:
        """
        Semantic similarity search on text chunks using Neo4j GraphRAG VectorRetriever.
        
        Args:
            query: Search query text
            graph_id: Tenant graph identifier
            k: Number of results to return
            threshold: Similarity threshold (0.0 to 1.0)
            
        Returns:
            List of chunk dictionaries with similarity scores
        """
        try:
            # Get multi-tenant chunk retriever
            retriever = self._get_chunk_retriever(graph_id)
            
            # Perform semantic search
            result = retriever.get_search_results(query_text=query, top_k=k)
            
            # Convert to standard format
            chunks = self._convert_retriever_result_to_chunks(result, threshold)
            
            logger.info(f"Chunk similarity search found {len(chunks)} results for graph {graph_id}")
            return chunks
            
        except Exception as e:
            logger.error(f"Chunk similarity search failed for graph {graph_id}: {e}")
            return []
    
    # ==================== HYBRID SEARCH ====================
    
    async def hybrid_search(
        self,
        query: str,
        graph_id: UUID,
        k: int = 10,
        alpha: float = 0.5
    ) -> List[Dict[str, Any]]:
        """
        Hybrid search combining vector similarity and fulltext search.
        
        Args:
            query: Search query text
            graph_id: Tenant graph identifier
            k: Number of results to return
            alpha: Balance between vector (1.0) and fulltext (0.0) search
            
        Returns:
            List of hybrid search results with combined scores
        """
        try:
            # Get multi-tenant hybrid retriever
            retriever = self._get_hybrid_retriever(graph_id)
            
            # Perform hybrid search
            result = retriever.get_search_results(
                query_text=query, 
                top_k=k,
                # Pass alpha as additional parameter if supported
                **({'alpha': alpha} if alpha != 0.5 else {})
            )
            
            # Convert to standard format
            hybrid_results = self._convert_retriever_result_to_entities(result)
            
            logger.info(f"Hybrid search found {len(hybrid_results)} results for graph {graph_id}")
            return hybrid_results
            
        except Exception as e:
            logger.error(f"Hybrid search failed for graph {graph_id}: {e}")
            # Fallback to semantic search
            logger.info("Falling back to semantic search")
            return await self.similarity_search_entities(query, graph_id, k=k)
    
    # ==================== GRAPH-AWARE SEARCH ====================
    
    async def graph_aware_search(
        self,
        query: str,
        graph_id: UUID,
        k: int = 5,
        include_relationships: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Graph-aware search that includes relationship context using VectorCypherRetriever.
        
        Args:
            query: Search query text
            graph_id: Tenant graph identifier
            k: Number of results to return
            include_relationships: Whether to include relationship context
            
        Returns:
            List of results with rich graph context
        """
        try:
            # Get multi-tenant vector+cypher retriever
            retriever = self._get_vector_cypher_retriever(graph_id)
            
            # Perform graph-aware search
            result = retriever.get_search_results(query_text=query, top_k=k)
            
            # Convert to format with graph context
            graph_results = self._convert_to_graph_aware_format(result)
            
            logger.info(f"Graph-aware search found {len(graph_results)} results for graph {graph_id}")
            return graph_results
            
        except Exception as e:
            logger.error(f"Graph-aware search failed for graph {graph_id}: {e}")
            # Fallback to regular semantic search
            return await self.similarity_search_entities(query, graph_id, k=k)
    
    # ==================== RETRIEVER FACTORIES (CACHED) ====================

    def _ensure_sync_connected(self):
        """Connect the sync Neo4j driver on first use (lazy initialization)."""
        if self.driver is None:
            neo4j_client.connect_sync()
            self.driver = neo4j_client.sync_driver

    def _get_entity_retriever(self, graph_id: UUID) -> MultiTenantVectorRetriever:
        """Get cached multi-tenant entity retriever using factory pattern."""
        self._ensure_sync_connected()
        cache_key = f"entity_{graph_id}"

        if cache_key not in self._retriever_cache:
            # Use factory function for clean creation
            self._retriever_cache[cache_key] = create_multi_tenant_vector_retriever(
                driver=self.driver,
                embedder=self.embedder,
                graph_id=str(graph_id),
                index_name="entity_embeddings",
                return_properties=["id", "name", "type", "description"]
            )
        
        return self._retriever_cache[cache_key]
    
    def _get_chunk_retriever(self, graph_id: UUID) -> MultiTenantVectorRetriever:
        """Get cached multi-tenant chunk retriever using factory pattern."""
        self._ensure_sync_connected()
        cache_key = f"chunk_{graph_id}"

        if cache_key not in self._retriever_cache:
            # Use factory function for clean creation
            self._retriever_cache[cache_key] = create_multi_tenant_vector_retriever(
                driver=self.driver,
                embedder=self.embedder,
                graph_id=str(graph_id),
                index_name="text_embeddings_primary",
                return_properties=["text", "chunk_index", "source"]
            )
        
        return self._retriever_cache[cache_key]
    
    def _get_hybrid_retriever(self, graph_id: UUID) -> MultiTenantHybridRetriever:
        """Get cached multi-tenant hybrid retriever using factory pattern."""
        self._ensure_sync_connected()
        cache_key = f"hybrid_{graph_id}"

        if cache_key not in self._retriever_cache:
            # Use factory function for clean creation
            self._retriever_cache[cache_key] = create_multi_tenant_hybrid_retriever(
                driver=self.driver,
                embedder=self.embedder,
                graph_id=str(graph_id),
                vector_index_name="entity_embeddings",
                fulltext_index_name="entity_text_fulltext"
            )
        
        return self._retriever_cache[cache_key]
    
    def _get_vector_cypher_retriever(self, graph_id: UUID) -> MultiTenantVectorCypherRetriever:
        """Get cached multi-tenant vector+cypher retriever using factory pattern."""
        self._ensure_sync_connected()
        cache_key = f"vector_cypher_{graph_id}"

        if cache_key not in self._retriever_cache:
            # Define graph-aware retrieval query
            retrieval_query = """
            WITH node AS chunk, score
            MATCH (chunk)<-[:FROM_CHUNK]-(entity:__Entity__)-[r]->(related_entity:__Entity__)
            WHERE r.confidence > 0.5
            RETURN 
                chunk.text AS context,
                chunk.chunk_index AS chunk_index,
                collect(DISTINCT {
                    entity: entity.name,
                    type: labels(entity)[0],
                    relationship: type(r),
                    related_entity: related_entity.name,
                    confidence: r.confidence
                }) AS knowledge_graph_context,
                score
            ORDER BY score DESC
            """
            
            # Create retriever using factory pattern
            self._retriever_cache[cache_key] = MultiTenantVectorCypherRetriever.create(
                driver=self.driver,
                index_name="text_embeddings_primary",
                embedder=self.embedder,
                retrieval_query=retrieval_query,
                graph_id=str(graph_id)
            )
        
        return self._retriever_cache[cache_key]
    
    # ==================== RESULT CONVERSION HELPERS ====================
    
    def _convert_retriever_result_to_entities(
        self, 
        result: RetrieverResult, 
        threshold: float = 0.0
    ) -> List[Dict[str, Any]]:
        """Convert Neo4j GraphRAG retriever result to entity format."""
        entities = []
        
        for item in result.items:
            # Get score (similarity)
            score = getattr(item, 'score', 0.0) or 0.0
            
            # Apply threshold filter
            if score < threshold:
                continue
            
            # Get metadata
            metadata = getattr(item, 'metadata', {}) or {}
            
            entity = {
                "id": metadata.get("id", ""),
                "name": metadata.get("name", ""),
                "type": metadata.get("type", ""),
                "description": metadata.get("description", ""),
                "score": score,
                "properties": metadata
            }
            entities.append(entity)
        
        return entities
    
    def _convert_retriever_result_to_chunks(
        self, 
        result: RetrieverResult, 
        threshold: float = 0.0
    ) -> List[Dict[str, Any]]:
        """Convert Neo4j GraphRAG retriever result to chunk format."""
        chunks = []
        
        for item in result.items:
            # Get score (similarity)
            score = getattr(item, 'score', 0.0) or 0.0
            
            # Apply threshold filter
            if score < threshold:
                continue
            
            # Get content and metadata
            content = getattr(item, 'content', '')
            metadata = getattr(item, 'metadata', {}) or {}
            
            chunk = {
                "text": content or metadata.get("text", ""),
                "chunk_index": metadata.get("chunk_index", 0),
                "source": metadata.get("source", ""),
                "score": score,
                "metadata": metadata
            }
            chunks.append(chunk)
        
        return chunks
    
    def _convert_to_graph_aware_format(self, result: RetrieverResult) -> List[Dict[str, Any]]:
        """Convert retriever result to format with graph context."""
        graph_results = []
        
        for item in result.items:
            score = getattr(item, 'score', 0.0) or 0.0
            content = getattr(item, 'content', '')
            metadata = getattr(item, 'metadata', {}) or {}
            
            # Extract graph context if available
            graph_context = metadata.get('knowledge_graph_context', [])
            
            result_item = {
                "text": content or metadata.get("context", ""),
                "chunk_index": metadata.get("chunk_index", 0),
                "score": score,
                "graph_context": {
                    "entities": [ctx.get("entity", "") for ctx in graph_context],
                    "relationships": [
                        f"{ctx.get('entity', '')} {ctx.get('relationship', '')} {ctx.get('related_entity', '')}"
                        for ctx in graph_context
                    ],
                    "confidence_scores": [ctx.get("confidence", 0.0) for ctx in graph_context]
                }
            }
            graph_results.append(result_item)
        
        return graph_results
    
    # ==================== CACHE MANAGEMENT ====================
    
    def clear_cache(self, graph_id: Optional[UUID] = None):
        """
        Clear retriever cache for performance management.
        
        Args:
            graph_id: If provided, clear cache for specific graph. If None, clear all.
        """
        if graph_id:
            # Clear cache for specific graph
            keys_to_remove = [
                key for key in self._retriever_cache.keys() 
                if str(graph_id) in key
            ]
            for key in keys_to_remove:
                del self._retriever_cache[key]
            
            logger.info(f"Cleared retriever cache for graph {graph_id}")
        else:
            # Clear all cache
            self._retriever_cache.clear()
            logger.info("Cleared all retriever cache")


# ==================== FASTAPI DEPENDENCY INJECTION ====================

def get_retrieval_service() -> RetrievalService:
    """
    FastAPI dependency factory for RetrievalService.
    
    Usage:
        @router.get("/search")
        async def search_endpoint(
            retrieval_service: RetrievalService = Depends(get_retrieval_service)
        ):
            return await retrieval_service.similarity_search_entities(...)
    """
    return RetrievalService()
