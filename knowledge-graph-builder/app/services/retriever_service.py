"""
Multi-Tenant Retrieval Service - Neo4j GraphRAG Foundation
Replaces search_service.py with Neo4j GraphRAG components and multi-tenant wrappers
"""

from typing import List, Dict, Any, Optional
from uuid import UUID

from neo4j_graphrag.retrievers import VectorRetriever, HybridRetriever
from neo4j_graphrag.embeddings import OpenAIEmbeddings
from neo4j_graphrag.types import RetrieverResult

from app.components.multi_tenant_components import (
    MultiTenantVectorRetriever, 
    MultiTenantVectorCypherRetriever,
    MultiTenantHybridRetriever
)
from app.core.neo4j_client import neo4j_client
from app.core.dependencies import get_neo4j_driver, get_openai_embeddings
from app.core.logging import get_logger

logger = get_logger(__name__)


class RetrievalService:
    """
    Multi-tenant retrieval service using Neo4j GraphRAG foundation.
    Provides semantic search, hybrid search, and graph-aware retrieval with perfect tenant isolation.
    
    DESIGN PRINCIPLES:
    - Neo4j GraphRAG components as foundation
    - Multi-tenant wrappers for perfect isolation
    - Clean delegation pattern (no complex inheritance)
    - FastAPI dependency injection ready
    """
    
    def __init__(self, driver=None, embedder=None):
        """
        Initialize with dependency injection support.
        
        Args:
            driver: Neo4j driver (injected via FastAPI)
            embedder: OpenAI embeddings (injected via FastAPI)
        """
        self.driver = driver or get_neo4j_driver()
        self.embedder = embedder or get_openai_embeddings()
        self._retriever_cache = {}  # Cache retrievers per graph_id
        
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
            threshold: Similarity threshold (0.0-1.0)
            
        Returns:
            List of entity dictionaries with similarity scores
        """
        try:
            # Get multi-tenant entity retriever
            retriever = self._get_entity_retriever(graph_id)
            
            # Search using Neo4j GraphRAG
            result = retriever.search(query_text=query, top_k=k)
            
            # Convert Neo4j GraphRAG result to expected format
            entities = self._convert_to_entity_format(result, threshold)
            
            logger.info(f"Entity similarity search found {len(entities)} results for graph {graph_id}")
            return entities
            
        except Exception as e:
            logger.error(f"Entity similarity search failed for graph {graph_id}: {e}")
            return []  # Return empty list instead of raising
    
    async def similarity_search_chunks(
        self,
        query: str,
        graph_id: UUID,
        k: int = 5,
        threshold: float = 0.7
    ) -> List[Dict[str, Any]]:
        """
        Semantic similarity search on text chunks using Neo4j GraphRAG VectorRetriever.
        
        Args:
            query: Search query text
            graph_id: Tenant graph identifier
            k: Number of results to return
            threshold: Similarity threshold (0.0-1.0)
            
        Returns:
            List of chunk dictionaries with similarity scores
        """
        try:
            # Get multi-tenant chunk retriever
            retriever = self._get_chunk_retriever(graph_id)
            
            # Search using Neo4j GraphRAG
            result = retriever.search(query_text=query, top_k=k)
            
            # Convert Neo4j GraphRAG result to expected format
            chunks = self._convert_to_chunk_format(result, threshold)
            
            logger.info(f"Chunk similarity search found {len(chunks)} results for graph {graph_id}")
            return chunks
            
        except Exception as e:
            logger.error(f"Chunk similarity search failed for graph {graph_id}: {e}")
            return []  # Return empty list instead of raising
    
    # ==================== HYBRID SEARCH ====================
    
    async def hybrid_search(
        self,
        query: str,
        graph_id: UUID,
        k: int = 10,
        semantic_weight: float = 0.7,
        keyword_weight: float = 0.3
    ) -> List[Dict[str, Any]]:
        """
        Hybrid search combining semantic and keyword search using Neo4j GraphRAG HybridRetriever.
        
        Args:
            query: Search query text
            graph_id: Tenant graph identifier
            k: Number of results to return
            semantic_weight: Weight for semantic search (0.0-1.0)
            keyword_weight: Weight for keyword search (0.0-1.0)
            
        Returns:
            List of result dictionaries with combined scores
        """
        try:
            # Get multi-tenant hybrid retriever
            retriever = self._get_hybrid_retriever(graph_id)
            
            # Configure weights (Neo4j GraphRAG HybridRetriever supports this)
            retriever_config = {
                "vector_weight": semantic_weight,
                "fulltext_weight": keyword_weight
            }
            
            # Search using Neo4j GraphRAG
            result = retriever.search(query_text=query, top_k=k, **retriever_config)
            
            # Convert to expected format
            hybrid_results = self._convert_to_hybrid_format(result)
            
            logger.info(f"Hybrid search returned {len(hybrid_results)} results for graph {graph_id}")
            return hybrid_results[:k]  # Ensure we don't exceed k results
            
        except Exception as e:
            logger.error(f"Hybrid search failed for graph {graph_id}: {e}")
            # Fallback to semantic search only
            logger.info("Falling back to semantic search")
            return await self.similarity_search_entities(query, graph_id, k=k, threshold=0.5)
    
    # ==================== FULLTEXT SEARCH (KEEP EXISTING) ====================
    
    async def fulltext_search_entities(
        self,
        query: str,
        graph_id: UUID,
        limit: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Fulltext search on entities using direct Cypher queries.
        Keep this method as-is since it's working well and uses direct queries.
        """
        try:
            cypher_query = """
            CALL db.index.fulltext.queryNodes('entity_fulltext', $query)
            YIELD node, score
            WHERE node.graph_id = $graph_id
            RETURN node.id as id,
                   node.name as name,
                   node.type as type,
                   node.description as description,
                   labels(node) as labels,
                   score,
                   node{.*} as properties
            ORDER BY score DESC
            LIMIT $limit
            """
            
            result = await neo4j_client.execute_query(cypher_query, {
                "query": query,
                "graph_id": str(graph_id),
                "limit": limit
            })
            
            logger.info(f"Fulltext search found {len(result)} entities for graph {graph_id}")
            return result
            
        except Exception as e:
            logger.error(f"Fulltext search failed for graph {graph_id}: {e}")
            return []  # Return empty results if fulltext index doesn't exist yet
    
    # ==================== GRAPH-AWARE SEARCH (NEW) ====================
    
    async def graph_aware_search(
        self,
        query: str,
        graph_id: UUID,
        k: int = 5,
        include_relationships: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Graph-aware search that includes relationship context.
        Uses VectorCypherRetriever for enhanced results.
        """
        try:
            # Get multi-tenant vector+cypher retriever
            retriever = self._get_vector_cypher_retriever(graph_id)
            
            # Search with graph context
            result = retriever.search(query_text=query, top_k=k)
            
            # Convert to format that includes graph context
            graph_results = self._convert_to_graph_aware_format(result)
            
            logger.info(f"Graph-aware search found {len(graph_results)} results for graph {graph_id}")
            return graph_results
            
        except Exception as e:
            logger.error(f"Graph-aware search failed for graph {graph_id}: {e}")
            # Fallback to regular semantic search
            return await self.similarity_search_entities(query, graph_id, k=k)
    
    # ==================== RETRIEVER FACTORIES (CACHED) ====================
    
    def _get_entity_retriever(self, graph_id: UUID) -> MultiTenantVectorRetriever:
        """Get cached multi-tenant entity retriever for graph_id."""
        cache_key = f"entity_{graph_id}"
        
        if cache_key not in self._retriever_cache:
            # Create base Neo4j GraphRAG retriever
            base_retriever = VectorRetriever(
                driver=self.driver,
                index_name="entity_embeddings",  # Entity vector index
                embedder=self.embedder,
                return_properties=["id", "name", "type", "description"]
            )
            
            # Wrap with multi-tenant filtering
            self._retriever_cache[cache_key] = MultiTenantVectorRetriever(
                base_retriever=base_retriever,
                graph_id=str(graph_id)
            )
        
        return self._retriever_cache[cache_key]
    
    def _get_chunk_retriever(self, graph_id: UUID) -> MultiTenantVectorRetriever:
        """Get cached multi-tenant chunk retriever for graph_id."""
        cache_key = f"chunk_{graph_id}"
        
        if cache_key not in self._retriever_cache:
            # Create base Neo4j GraphRAG retriever
            base_retriever = VectorRetriever(
                driver=self.driver,
                index_name="text_embeddings_primary",  # Chunk vector index
                embedder=self.embedder,
                return_properties=["text", "chunk_index", "source"]
            )
            
            # Wrap with multi-tenant filtering
            self._retriever_cache[cache_key] = MultiTenantVectorRetriever(
                base_retriever=base_retriever,
                graph_id=str(graph_id)
            )
        
        return self._retriever_cache[cache_key]
    
    def _get_hybrid_retriever(self, graph_id: UUID) -> MultiTenantHybridRetriever:
        """Get cached multi-tenant hybrid retriever for graph_id."""
        cache_key = f"hybrid_{graph_id}"
        
        if cache_key not in self._retriever_cache:
            # Create base Neo4j GraphRAG hybrid retriever
            base_retriever = HybridRetriever(
                driver=self.driver,
                vector_index_name="entity_embeddings",
                fulltext_index_name="entity_text_fulltext",
                embedder=self.embedder
            )
            
            # Wrap with multi-tenant filtering
            self._retriever_cache[cache_key] = MultiTenantHybridRetriever(
                base_retriever=base_retriever,
                graph_id=str(graph_id)
            )
        
        return self._retriever_cache[cache_key]
    
    def _get_vector_cypher_retriever(self, graph_id: UUID) -> MultiTenantVectorCypherRetriever:
        """Get cached multi-tenant vector+cypher retriever for graph_id."""
        cache_key = f"vector_cypher_{graph_id}"
        
        if cache_key not in self._retriever_cache:
            # Enhanced Cypher query that includes relationship context
            retrieval_query = """
            WITH node AS entity, score
            MATCH (entity)-[r]->(related)
            WHERE entity.graph_id = $graph_id 
            AND related.graph_id = $graph_id 
            AND r.graph_id = $graph_id
            
            RETURN 
                entity.id AS id,
                entity.name AS name,
                entity.type AS type,
                entity.description AS description,
                labels(entity) AS labels,
                score,
                collect(DISTINCT {
                    relationship: type(r),
                    related_entity: related.name,
                    related_type: labels(related)[0]
                })[..5] AS relationships,
                entity{.*} AS properties
            ORDER BY score DESC
            """
            
            self._retriever_cache[cache_key] = MultiTenantVectorCypherRetriever(
                driver=self.driver,
                index_name="entity_embeddings",
                embedder=self.embedder,
                retrieval_query=retrieval_query,
                graph_id=str(graph_id)
            )
        
        return self._retriever_cache[cache_key]
    
    # ==================== RESULT CONVERTERS ====================
    
    def _convert_to_entity_format(
        self, 
        result: RetrieverResult, 
        threshold: float
    ) -> List[Dict[str, Any]]:
        """Convert Neo4j GraphRAG result to expected entity format."""
        entities = []
        
        for item in result.items:
            # Extract score (Neo4j GraphRAG format)
            score = getattr(item, 'score', 0.0) or 0.0
            
            # Apply threshold filtering
            if score < threshold:
                continue
            
            # Extract metadata/content
            metadata = getattr(item, 'metadata', {}) or {}
            
            entity = {
                "id": metadata.get("id", ""),
                "name": metadata.get("name", ""),
                "type": metadata.get("type", ""),
                "description": metadata.get("description", ""),
                "labels": metadata.get("labels", []),
                "score": score,
                "properties": metadata.get("properties", {})
            }
            entities.append(entity)
        
        return entities
    
    def _convert_to_chunk_format(
        self, 
        result: RetrieverResult, 
        threshold: float
    ) -> List[Dict[str, Any]]:
        """Convert Neo4j GraphRAG result to expected chunk format."""
        chunks = []
        
        for item in result.items:
            # Extract score
            score = getattr(item, 'score', 0.0) or 0.0
            
            # Apply threshold filtering
            if score < threshold:
                continue
            
            # Extract content/metadata
            content = getattr(item, 'content', '') or ''
            metadata = getattr(item, 'metadata', {}) or {}
            
            chunk = {
                "text": content or metadata.get("text", ""),
                "chunk_index": metadata.get("chunk_index", 0),
                "source": metadata.get("source", ""),
                "score": score
            }
            chunks.append(chunk)
        
        return chunks
    
    def _convert_to_hybrid_format(self, result: RetrieverResult) -> List[Dict[str, Any]]:
        """Convert Neo4j GraphRAG hybrid result to expected format."""
        hybrid_results = []
        
        for item in result.items:
            # Extract scores (hybrid results may have multiple scores)
            total_score = getattr(item, 'score', 0.0) or 0.0
            metadata = getattr(item, 'metadata', {}) or {}
            
            entity = {
                "id": metadata.get("id", ""),
                "name": metadata.get("name", ""),
                "type": metadata.get("type", ""),
                "description": metadata.get("description", ""),
                "labels": metadata.get("labels", []),
                "combined_score": total_score,
                "semantic_score": metadata.get("vector_score", total_score * 0.7),  # Estimate
                "keyword_score": metadata.get("fulltext_score", total_score * 0.3),  # Estimate
                "properties": metadata.get("properties", {})
            }
            hybrid_results.append(entity)
        
        return hybrid_results
    
    def _convert_to_graph_aware_format(self, result: RetrieverResult) -> List[Dict[str, Any]]:
        """Convert Neo4j GraphRAG result with graph context to expected format."""
        graph_results = []
        
        for item in result.items:
            score = getattr(item, 'score', 0.0) or 0.0
            metadata = getattr(item, 'metadata', {}) or {}
            
            entity = {
                "id": metadata.get("id", ""),
                "name": metadata.get("name", ""),
                "type": metadata.get("type", ""),
                "description": metadata.get("description", ""),
                "score": score,
                "relationships": metadata.get("relationships", []),  # Graph context
                "graph_context": {
                    "direct_relationships": len(metadata.get("relationships", [])),
                    "connected_entities": [
                        rel.get("related_entity") 
                        for rel in metadata.get("relationships", [])
                    ][:5]  # Limit to 5 for readability
                }
            }
            graph_results.append(entity)
        
        return graph_results
    
    # ==================== CACHE MANAGEMENT ====================
    
    def clear_cache(self, graph_id: Optional[UUID] = None):
        """
        Clear retriever cache.
        
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
    
    # ==================== SEARCH SUGGESTIONS (TODO: FUTURE ENHANCEMENT) ====================
    
    async def get_search_suggestions(
        self,
        partial_query: str,
        graph_id: UUID,
        limit: int = 5
    ) -> List[str]:
        """
        TODO: Get search suggestions based on partial query.
        This could use entity names, popular searches, or query completion.
        
        For now, return empty list as placeholder.
        """
        # TODO: Implement search suggestions
        # Could use:
        # - Entity name prefix matching
        # - Popular search patterns
        # - Query completion based on graph content
        
        logger.debug(f"Search suggestions requested for '{partial_query}' (not implemented)")
        return []
    
    async def get_related_terms(
        self,
        query: str,
        graph_id: UUID,
        limit: int = 10
    ) -> List[str]:
        """
        TODO: Get terms related to the search query.
        Could use entity relationships, co-occurrence, or embedding similarity.
        
        For now, return empty list as placeholder.
        """
        # TODO: Implement related terms
        # Could use:
        # - Entity relationship analysis
        # - Term co-occurrence in chunks
        # - Embedding-based similarity
        
        logger.debug(f"Related terms requested for '{query}' (not implemented)")
        return []


# ==================== DEPENDENCY INJECTION SETUP ====================

def get_retrieval_service(
    driver=None,
    embedder=None
) -> RetrievalService:
    """
    Factory function for dependency injection.
    Can be used with FastAPI Depends().
    """
    return RetrievalService(driver=driver, embedder=embedder)


# ==================== GLOBAL INSTANCE (BACKWARD COMPATIBILITY) ====================

# Create global instance for backward compatibility
# Your existing code can continue to use retrieval_service
retrieval_service = RetrievalService()