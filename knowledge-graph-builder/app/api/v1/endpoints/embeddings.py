from fastapi import APIRouter, Depends, HTTPException, status, BackgroundTasks
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from typing import Optional
from uuid import UUID
from pydantic import BaseModel

from app.services.background_jobs import process_embedding_generation_job
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
    logger.info(f"Generating embeddings for graph {graph_id} with provider {request.provider} and model {request.model}, {request.generate_for_existing}")
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
        
        logger.info(f"Step 3: Embeddings initialized successfully")
        
        # Generate embeddings for existing nodes if requested
        if request.generate_for_existing:
            logger.info("Step 4: Starting embedding generation for existing nodes")
            
            try:
                # Import the task here to make sure it's available
                from app.services.background_jobs import process_embedding_generation_job
                logger.info("Step 5: Successfully imported Celery task")
                
                # Start Celery task
                logger.info(f"Step 6: About to call process_embedding_generation_job.delay({str(graph_id)}, {user_id})")
                task = process_embedding_generation_job.delay(str(graph_id), user_id)
                logger.info(f"Step 7: Celery task started successfully with ID: {task.id}")
                
                return EmbeddingResponse(
                    status="processing",
                    message=f"Embedding generation started in background. Job ID: {task.id}",
                    nodes_processed=None
                )
                
            except Exception as e:
                logger.error(f"Step X: Failed to start background embedding job: {e}")
                logger.error(f"Exception type: {type(e)}")
                logger.error(f"Exception details: {str(e)}")
                import traceback
                logger.error(f"Traceback: {traceback.format_exc()}")
                
                # Fallback to FastAPI background task for development
                logger.info("Step 8: Falling back to FastAPI background task")
                background_tasks.add_task(
                    _process_embedding_generation,
                    graph_id,
                    user_id
                )
                
                return EmbeddingResponse(
                    status="processing",
                    message="Embedding generation started in background (fallback mode)",
                    nodes_processed=None
                )
        else:
            # Just initialize the service and create indexes
            logger.info("Step 4: Only initializing service, not generating for existing nodes")
            await vector_service.create_vector_indexes(
                dimension=embedding_service.dimension
            )
            
            return EmbeddingResponse(
                status="ready",
                message="Embedding service initialized. New entities will get embeddings automatically.",
                nodes_processed=None
            )
        
    except HTTPException:
        logger.error("HTTPException caught, re-raising")
        raise
    except Exception as e:
        logger.error(f"Unexpected error in embedding generation: {e}")
        logger.error(f"Exception type: {type(e)}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Embedding generation failed: {str(e)}"
        )

# Add this temporary diagnostic endpoint to your embeddings.py to debug

@router.get("/graphs/{graph_id}/debug/nodes")
async def debug_graph_nodes(
    graph_id: UUID,
    user_id: str = Depends(get_current_user_id),
    db: AsyncSession = Depends(get_database)
):
    """Debug endpoint to check node state in Neo4j"""
    
    try:
        from app.core.neo4j_client import neo4j_client
        
        # Query 1: Check all nodes for this graph
        query1 = """
        MATCH (n)
        WHERE n.graph_id = $graph_id
        RETURN n.id as id, n.name as name, 
               CASE WHEN n.embedding IS NOT NULL THEN 'YES' ELSE 'NO' END as has_embedding,
               size(n.embedding) as embedding_size,
               labels(n) as labels
        ORDER BY n.name
        """
        
        # Query 2: Count stats manually
        query2 = """
        MATCH (n)
        WHERE n.graph_id = $graph_id
        RETURN 
            count(n) as total_nodes,
            count(n.embedding) as nodes_with_embeddings,
            count(n) - count(n.embedding) as nodes_without_embeddings
        """
        
        # Query 3: Check embedding properties directly
        query3 = """
        MATCH (n)
        WHERE n.graph_id = $graph_id AND n.embedding IS NOT NULL
        RETURN n.id, n.name, size(n.embedding) as embedding_dimension
        LIMIT 5
        """
        
        nodes_result = await neo4j_client.execute_query(query1, {"graph_id": str(graph_id)})
        stats_result = await neo4j_client.execute_query(query2, {"graph_id": str(graph_id)})
        embedding_result = await neo4j_client.execute_query(query3, {"graph_id": str(graph_id)})
        
        return {
            "graph_id": str(graph_id),
            "all_nodes": [dict(record) for record in nodes_result],
            "manual_stats": dict(stats_result[0]) if stats_result else {},
            "nodes_with_embeddings_sample": [dict(record) for record in embedding_result],
            "queries_used": {
                "all_nodes": query1,
                "manual_stats": query2,
                "embeddings_sample": query3
            }
        }
        
    except Exception as e:
        logger.error(f"Debug query failed: {e}")
        return {"error": str(e), "graph_id": str(graph_id)}


# Also add this endpoint to manually trigger a single node embedding update for testing
@router.post("/graphs/{graph_id}/debug/single-embedding")
async def debug_single_embedding(
    graph_id: UUID,
    node_id: str,
    user_id: str = Depends(get_current_user_id),
    db: AsyncSession = Depends(get_database)
):
    """Debug endpoint to manually add embedding to a single node"""
    
    try:
        from app.core.neo4j_client import neo4j_client
        
        # Initialize embedding service
        if not embedding_service.is_initialized():
            await embedding_service.initialize_embeddings(
                provider="openai",
                model="text-embedding-3-small",
                user_id=user_id
            )
        
        # Get node info first
        get_query = """
        MATCH (n)
        WHERE n.graph_id = $graph_id AND n.id = $node_id
        RETURN n.id, n.name, coalesce(n.description, '') as description
        """
        
        result = await neo4j_client.execute_query(get_query, {
            "graph_id": str(graph_id),
            "node_id": node_id
        })
        
        if not result:
            return {"error": "Node not found"}
        
        node = result[0]
        
        # Generate embedding
        text = node["name"]
        if node["description"]:
            text += f" {node['description']}"
        
        embedding = await embedding_service.embed_text(text)
        
        # Update node with embedding
        update_query = """
        MATCH (n)
        WHERE n.graph_id = $graph_id AND n.id = $node_id
        SET n.embedding = $embedding
        RETURN n.id, n.name, size(n.embedding) as embedding_size
        """
        
        update_result = await neo4j_client.execute_query(update_query, {
            "graph_id": str(graph_id),
            "node_id": node_id,
            "embedding": embedding
        })
        
        return {
            "status": "success",
            "node_updated": dict(update_result[0]) if update_result else {},
            "embedding_dimension": len(embedding),
            "text_used": text
        }
        
    except Exception as e:
        logger.error(f"Debug single embedding failed: {e}")
        return {"error": str(e)}

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
