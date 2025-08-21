from fastapi import APIRouter, Depends, HTTPException, status, BackgroundTasks
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from typing import Optional
from uuid import UUID
from pydantic import BaseModel

from app.api.dependencies import get_current_user_id, get_database
from app.models.graph import KnowledgeGraph
from app.services.enhanced_graph_service import enhanced_graph_service
from app.services.embedding_service import embedding_service
from app.services.vector_service import vector_service
from app.core.logging import get_logger

router = APIRouter()
logger = get_logger(__name__)

class EmbeddingRequest(BaseModel):
    provider: str = "openai"
    model: str = "text-embedding-3-small"
    generate_for_existing: bool = True

class EmbeddingResponse(BaseModel):
    status: str
    message: str
    nodes_processed: Optional[int] = None
    embedding_stats: Optional[dict] = None

class EmbeddingStatsResponse(BaseModel):
    total_nodes: int
    nodes_with_embeddings: int
    nodes_without_embeddings: int
    total_chunks: int
    chunks_with_embeddings: int
    embedding_coverage: float
    provider: Optional[str] = None
    model: Optional[str] = None
    dimension: Optional[int] = None

@router.post("/graphs/{graph_id}/embeddings/generate", response_model=EmbeddingResponse)
async def generate_embeddings(
    graph_id: UUID,
    request: EmbeddingRequest,
    background_tasks: BackgroundTasks,
    user_id: str = Depends(get_current_user_id),
    db: AsyncSession = Depends(get_database)
):
    """Generate embeddings for entities in a knowledge graph"""
    
    try:
        # Verify graph ownership
        result = await db.execute(
            select(KnowledgeGraph).where(
                KnowledgeGraph.id == graph_id,
                KnowledgeGraph.user_id == UUID(user_id)
            )
        )
        graph = result.scalar_one_or_none()
        
        if not graph:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Graph not found"
            )
        
        # Initialize embeddings
        success = await embedding_service.initialize_embeddings(
            provider=request.provider,
            model=request.model,
            user_id=user_id
        )
        
        if not success:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=f"Failed to initialize {request.provider} embeddings"
            )
        
        # Generate embeddings for existing nodes if requested
        if request.generate_for_existing:
            # Add to background tasks for async processing
            background_tasks.add_task(
                _process_embedding_generation,
                graph_id,
                user_id
            )
            
            return EmbeddingResponse(
                status="processing",
                message="Embedding generation started in background",
                nodes_processed=None
            )
        else:
            # Just initialize the service and create indexes
            await vector_service.create_vector_indexes(
                dimension=embedding_service.dimension
            )
            
            return EmbeddingResponse(
                status="ready",
                message="Embedding service initialized. New entities will get embeddings automatically.",
                nodes_processed=0
            )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Embedding generation failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Embedding generation failed: {str(e)}"
        )

@router.get("/graphs/{graph_id}/embeddings/stats", response_model=EmbeddingStatsResponse)
async def get_embedding_stats(
    graph_id: UUID,
    user_id: str = Depends(get_current_user_id),
    db: AsyncSession = Depends(get_database)
):
    """Get embedding statistics for a knowledge graph"""
    
    try:
        # Verify graph ownership
        result = await db.execute(
            select(KnowledgeGraph).where(
                KnowledgeGraph.id == graph_id,
                KnowledgeGraph.user_id == UUID(user_id)
            )
        )
        graph = result.scalar_one_or_none()
        
        if not graph:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Graph not found"
            )
        
        # Get embedding statistics
        stats = await enhanced_graph_service.get_embedding_stats(graph_id)
        
        # Add embedding service info if initialized
        provider = None
        model = None
        dimension = None
        
        if embedding_service.is_initialized():
            provider = embedding_service.provider
            model = embedding_service.model_name
            dimension = embedding_service.dimension
        
        return EmbeddingStatsResponse(
            **stats,
            provider=provider,
            model=model,
            dimension=dimension
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get embedding stats: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve embedding statistics"
        )

@router.post("/graphs/{graph_id}/embeddings/rebuild")
async def rebuild_vector_indexes(
    graph_id: UUID,
    user_id: str = Depends(get_current_user_id),
    db: AsyncSession = Depends(get_database)
):
    """Rebuild vector indexes for a graph"""
    
    try:
        # Verify graph ownership
        result = await db.execute(
            select(KnowledgeGraph).where(
                KnowledgeGraph.id == graph_id,
                KnowledgeGraph.user_id == UUID(user_id)
            )
        )
        graph = result.scalar_one_or_none()
        
        if not graph:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Graph not found"
            )
        
        if not embedding_service.is_initialized():
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Embedding service not initialized. Generate embeddings first."
            )
        
        # Rebuild vector indexes
        await vector_service.create_vector_indexes(
            dimension=embedding_service.dimension
        )
        
        return {
            "status": "success",
            "message": "Vector indexes rebuilt successfully",
            "dimension": embedding_service.dimension
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to rebuild indexes: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to rebuild vector indexes"
        )

async def _process_embedding_generation(graph_id: UUID, user_id: str):
    """Background task to process embedding generation"""
    
    try:
        nodes_processed = await enhanced_graph_service.generate_embeddings_for_existing_nodes(
            graph_id=graph_id,
            user_id=user_id,
            batch_size=50
        )
        
        logger.info(f"Background embedding generation completed: {nodes_processed} nodes processed")
        
    except Exception as e:
        logger.error(f"Background embedding generation failed: {e}")
