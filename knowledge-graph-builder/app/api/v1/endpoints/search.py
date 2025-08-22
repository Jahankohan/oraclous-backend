from fastapi import APIRouter, Depends, HTTPException, status, Query
from sqlalchemy.ext.asyncio import AsyncSession
from typing import List, Optional
from uuid import UUID
from pydantic import BaseModel

from app.api.dependencies import get_current_user_id, get_database
from app.services.search_service import search_service
from app.services.embedding_service import embedding_service
from app.core.logging import get_logger

router = APIRouter()
logger = get_logger(__name__)

class SearchRequest(BaseModel):
    query: str
    search_type: str = "hybrid"  # semantic, keyword, hybrid
    k: int = 10
    threshold: float = 0.7

class SearchResponse(BaseModel):
    results: List[dict]
    total: int
    search_type: str
    query: str

@router.post("/graphs/{graph_id}/search", response_model=SearchResponse)
async def search_graph(
    graph_id: UUID,
    request: SearchRequest,
    user_id: str = Depends(get_current_user_id)
):
    """Search entities in a knowledge graph using various methods"""
    
    try:
        # Initialize embeddings if needed for semantic search
        if request.search_type in ["semantic", "hybrid"]:
            if not embedding_service.is_initialized():
                success = await embedding_service.initialize_embeddings(
                    provider="openai", user_id=user_id
                )
                if not success:
                    # Fallback to keyword search
                    request.search_type = "keyword"
        
        # Perform search based on type
        if request.search_type == "semantic":
            results = await search_service.similarity_search_entities(
                query=request.query,
                graph_id=graph_id,
                k=request.k,
                threshold=request.threshold
            )
            
        elif request.search_type == "keyword":
            results = await search_service.fulltext_search_entities(
                query=request.query,
                graph_id=graph_id,
                limit=request.k
            )
            
        elif request.search_type == "hybrid":
            results = await search_service.hybrid_search(
                query=request.query,
                graph_id=graph_id,
                k=request.k
            )
        else:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid search_type. Use: semantic, keyword, or hybrid"
            )
        
        return SearchResponse(
            results=results,
            total=len(results),
            search_type=request.search_type,
            query=request.query
        )
        
    except Exception as e:
        logger.error(f"Search failed for graph {graph_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Search failed: {str(e)}"
        )


@router.post("/graphs/{graph_id}/search/similar")
async def find_similar_entities(
    graph_id: UUID,
    entity_id: str,
    k: int = 5,
    threshold: float = 0.7,
    user_id: str = Depends(get_current_user_id)
):
    """Find entities similar to a given entity using embeddings"""
    
    try:
        # Initialize embeddings if needed
        if not embedding_service.is_initialized():
            success = await embedding_service.initialize_embeddings(
                provider="openai", user_id=user_id
            )
            if not success:
                raise HTTPException(
                    status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                    detail="Embedding service unavailable"
                )
        
        # Get the source entity embedding
        source_query = """
        MATCH (source {id: $entity_id, graph_id: $graph_id})
        WHERE source.embedding IS NOT NULL
        RETURN source.embedding as embedding, source.name as name
        """
        
        source_result = await neo4j_client.execute_query(source_query, {
            "entity_id": entity_id,
            "graph_id": str(graph_id)
        })
        
        if not source_result:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Entity not found or has no embedding"
            )
        
        source_embedding = source_result[0]["embedding"]
        source_name = source_result[0]["name"]
        
        # Find similar entities
        similarity_query = """
        CALL db.index.vector.queryNodes('entity_embeddings', $k + 1, $source_embedding)
        YIELD node, score
        WHERE node.graph_id = $graph_id 
        AND node.id <> $entity_id
        AND score >= $threshold
        RETURN node.id as id,
               node.name as name,
               labels(node) as labels,
               score,
               node{.*} as properties
        ORDER BY score DESC
        LIMIT $k
        """
        
        similar_entities = await neo4j_client.execute_query(similarity_query, {
            "k": k,
            "source_embedding": source_embedding,
            "graph_id": str(graph_id),
            "entity_id": entity_id,
            "threshold": threshold
        })
        
        return {
            "source_entity": {
                "id": entity_id,
                "name": source_name
            },
            "similar_entities": similar_entities,
            "count": len(similar_entities),
            "threshold": threshold
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Similar entity search failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Similar entity search failed: {str(e)}"
        )

@router.get("/graphs/{graph_id}/search/recommendations/{entity_id}")
async def get_entity_recommendations(
    graph_id: UUID,
    entity_id: str,
    user_id: str = Depends(get_current_user_id)
):
    """Get entity recommendations based on graph structure and embeddings"""
    
    try:
        # Get direct neighbors
        neighbors_query = """
        MATCH (source {id: $entity_id, graph_id: $graph_id})-[r]-(neighbor)
        WHERE neighbor.graph_id = $graph_id
        RETURN neighbor.id as id,
               neighbor.name as name,
               type(r) as relationship,
               labels(neighbor) as labels
        LIMIT 10
        """
        
        neighbors = await neo4j_client.execute_query(neighbors_query, {
            "entity_id": entity_id,
            "graph_id": str(graph_id)
        })
        
        # Get similar entities if embeddings available
        similar_entities = []
        if embedding_service.is_initialized():
            try:
                similar_response = await find_similar_entities(
                    graph_id, entity_id, k=5, threshold=0.6, user_id=user_id
                )
                similar_entities = similar_response["similar_entities"]
            except:
                pass  # Ignore embedding errors for recommendations
        
        return {
            "entity_id": entity_id,
            "direct_neighbors": neighbors,
            "similar_entities": similar_entities,
            "recommendations": {
                "by_structure": neighbors[:5],
                "by_similarity": similar_entities[:5]
            }
        }
        
    except Exception as e:
        logger.error(f"Entity recommendations failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Entity recommendations failed: {str(e)}"
        )


@router.get("/graphs/{graph_id}/search/chunks")
async def search_chunks(
    graph_id: UUID,
    query: str = Query(..., description="Search query"),
    k: int = Query(default=5, description="Number of results"),
    threshold: float = Query(default=0.7, description="Similarity threshold"),
    user_id: str = Depends(get_current_user_id)
):
    """Search text chunks using semantic similarity"""
    
    try:
        # Initialize embeddings if needed
        if not embedding_service.is_initialized():
            success = await embedding_service.initialize_embeddings(
                provider="openai", user_id=user_id
            )
            if not success:
                raise HTTPException(
                    status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                    detail="Embedding service unavailable"
                )
        
        results = await search_service.similarity_search_chunks(
            query=query,
            graph_id=graph_id,
            k=k,
            threshold=threshold
        )
        
        return {
            "results": results,
            "total": len(results),
            "query": query
        }
        
    except Exception as e:
        logger.error(f"Chunk search failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Chunk search failed: {str(e)}"
        )
