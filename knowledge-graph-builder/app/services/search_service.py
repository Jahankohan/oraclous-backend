from typing import List, Dict, Any, Optional, Tuple
from uuid import UUID
from app.core.neo4j_client import neo4j_client
from app.services.embedding_service import embedding_service
from app.core.logging import get_logger

logger = get_logger(__name__)

class SearchService:
    """Service for semantic and hybrid search functionality"""
    
    def __init__(self):
        pass
    
    async def similarity_search_entities(
        self,
        query: str,
        graph_id: UUID,
        k: int = 5,
        threshold: float = 0.7
    ) -> List[Dict[str, Any]]:
        """Perform semantic similarity search on entities"""
        
        if not embedding_service.is_initialized():
            raise ValueError("Embedding service not initialized")
        
        try:
            # Generate query embedding
            query_embedding = await embedding_service.embed_text(query)
            
            # Vector similarity search
            cypher_query = """
            CALL db.index.vector.queryNodes('entity_embeddings', $k, $query_embedding)
            YIELD node, score
            WHERE node.graph_id = $graph_id AND score >= $threshold
            RETURN node.id as id,
                   node.name as name,
                   node.type as type,
                   node.description as description,
                   labels(node) as labels,
                   score,
                   node{.*} as properties
            ORDER BY score DESC
            """
            
            result = await neo4j_client.execute_query(cypher_query, {
                "k": k,
                "query_embedding": query_embedding,
                "graph_id": str(graph_id),
                "threshold": threshold
            })
            
            logger.info(f"Similarity search found {len(result)} entities")
            return result
            
        except Exception as e:
            logger.error(f"Entity similarity search failed: {e}")
            raise
    
    async def similarity_search_chunks(
        self,
        query: str, 
        graph_id: UUID,
        k: int = 5,
        threshold: float = 0.7
    ) -> List[Dict[str, Any]]:
        """Perform semantic similarity search on text chunks"""
        
        if not embedding_service.is_initialized():
            raise ValueError("Embedding service not initialized")
        
        try:
            # Generate query embedding
            query_embedding = await embedding_service.embed_text(query)
            
            # Vector similarity search
            cypher_query = """
            CALL db.index.vector.queryNodes('chunk_embeddings', $k, $query_embedding)
            YIELD node, score  
            WHERE node.graph_id = $graph_id AND score >= $threshold
            RETURN node.id as id,
                   node.text as text,
                   node.source as source,
                   node.chunk_index as chunk_index,
                   score,
                   node{.*} as properties
            ORDER BY score DESC
            """
            
            result = await neo4j_client.execute_query(cypher_query, {
                "k": k,
                "query_embedding": query_embedding,
                "graph_id": str(graph_id),
                "threshold": threshold
            })
            
            logger.info(f"Chunk similarity search found {len(result)} results")
            return result
            
        except Exception as e:
            logger.error(f"Chunk similarity search failed: {e}")
            raise
    
    async def fulltext_search_entities(
        self,
        query: str,
        graph_id: UUID,
        limit: int = 10
    ) -> List[Dict[str, Any]]:
        """Perform fulltext search on entities"""
        
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
            
            logger.info(f"Fulltext search found {len(result)} entities")
            return result
            
        except Exception as e:
            logger.error(f"Fulltext search failed: {e}")
            # Return empty results if fulltext index doesn't exist yet
            return []
    
    async def hybrid_search(
        self,
        query: str,
        graph_id: UUID,
        k: int = 10,
        semantic_weight: float = 0.7,
        keyword_weight: float = 0.3
    ) -> List[Dict[str, Any]]:
        """Perform hybrid search combining semantic and keyword search"""
        
        try:
            # Get semantic results
            semantic_results = await self.similarity_search_entities(
                query, graph_id, k=k, threshold=0.5
            )
            
            # Get keyword results
            keyword_results = await self.fulltext_search_entities(
                query, graph_id, limit=k
            )
            
            # Combine and rerank results
            combined_results = self._combine_search_results(
                semantic_results, keyword_results,
                semantic_weight, keyword_weight
            )
            
            # Sort by combined score and limit
            combined_results.sort(key=lambda x: x.get("combined_score", 0), reverse=True)
            
            logger.info(f"Hybrid search returned {len(combined_results[:k])} results")
            return combined_results[:k]
            
        except Exception as e:
            logger.error(f"Hybrid search failed: {e}")
            # Fallback to semantic search only
            return await self.similarity_search_entities(query, graph_id, k=k)
    
    def _combine_search_results(
        self,
        semantic_results: List[Dict[str, Any]],
        keyword_results: List[Dict[str, Any]], 
        semantic_weight: float,
        keyword_weight: float
    ) -> List[Dict[str, Any]]:
        """Combine and rerank search results"""
        
        # Create lookup for semantic scores
        semantic_scores = {r["id"]: r["score"] for r in semantic_results}
        keyword_scores = {r["id"]: r["score"] for r in keyword_results}
        
        # Get all unique entities
        all_entity_ids = set(semantic_scores.keys()) | set(keyword_scores.keys())
        
        combined_results = []
        for entity_id in all_entity_ids:
            # Get the full entity data (prefer semantic result)
            entity_data = None
            for result in semantic_results + keyword_results:
                if result["id"] == entity_id:
                    entity_data = result.copy()
                    break
            
            if entity_data:
                # Calculate combined score
                semantic_score = semantic_scores.get(entity_id, 0)
                keyword_score = keyword_scores.get(entity_id, 0)
                
                combined_score = (
                    semantic_weight * semantic_score + 
                    keyword_weight * keyword_score
                )
                
                entity_data["combined_score"] = combined_score
                entity_data["semantic_score"] = semantic_score
                entity_data["keyword_score"] = keyword_score
                
                combined_results.append(entity_data)
        
        return combined_results

search_service = SearchService()
