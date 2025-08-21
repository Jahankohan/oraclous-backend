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

@router.post("/graphs/{graph_id}/embeddings/generate")
async def generate_embeddings(
    graph_id: UUID,
    user_id: str = Depends(get_current_user_id)
):
    """Generate embeddings for all entities in a graph"""
    
    try:
        # Initialize embeddings
        if not embedding_service.is_initialized():
            success = await embedding_service.initialize_embeddings(
                provider="openai", user_id=user_id
            )
            if not success:
                raise HTTPException(
                    status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                    detail="Embedding service unavailable"
                )
        
        # This would be implemented in the embedding service
        # For now, return a placeholder
        return {
            "status": "success",
            "message": "Embedding generation started",
            "graph_id": str(graph_id)
        }
        
    except Exception as e:
        logger.error(f"Embedding generation failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Embedding generation failed: {str(e)}"
        )
