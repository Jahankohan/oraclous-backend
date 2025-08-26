from fastapi import APIRouter, Depends, HTTPException, status, BackgroundTasks
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, update, delete
from typing import List, Optional
from uuid import UUID
import uuid
from datetime import datetime

from app.api.dependencies import get_current_user_id, get_database
from app.models.graph import KnowledgeGraph, IngestionJob
from app.schemas.graph_schemas import (
    GraphCreate, GraphUpdate, GraphResponse, 
    IngestDataRequest, IngestionJobResponse, SchemaLearnRequest, SchemaUpdateRequest
)
from app.core.neo4j_client import neo4j_client
from app.services.background_jobs import (
    process_ingestion_job, 
    _create_similarity_relationships, 
    _detect_communities_and_cleanup,
    create_persistent_communities_task,
    update_community_embeddings_task,
    refresh_all_communities_task,
    optimize_all_graphs
)
from app.services.entity_extractor import entity_extractor, SchemaEvolutionConfig
from app.services.schema_service import schema_service
from app.services.enhanced_graph_service import enhanced_graph_service
from app.core.logging import get_logger

router = APIRouter()
logger = get_logger(__name__)

@router.post("/graphs", response_model=GraphResponse, status_code=status.HTTP_201_CREATED)
async def create_graph(
    graph_data: GraphCreate,
    user_id: str = Depends(get_current_user_id),
    db: AsyncSession = Depends(get_database)
):
    """Create a new knowledge graph"""
    
    # Create graph without neo4j_database field
    graph = KnowledgeGraph(
        name=graph_data.name,
        description=graph_data.description,
        user_id=UUID(user_id),
        schema_config=graph_data.schema_config or {}
    )
    
    db.add(graph)
    await db.commit()
    await db.refresh(graph)
    
    # Create schema constraints if provided
    if graph_data.schema_config:
        try:
            await schema_service.create_graph_constraints(
                graph_data.schema_config
            )
        except Exception as e:
            logger.warning(f"Failed to create schema constraints: {e}")
    
    logger.info(f"Created new graph: {graph.id} for user: {user_id}")
    return graph

@router.get("/graphs", response_model=List[GraphResponse])
async def list_graphs(
    user_id: str = Depends(get_current_user_id),
    db: AsyncSession = Depends(get_database)
):
    """List all graphs for the current user"""
    
    result = await db.execute(
        select(KnowledgeGraph).where(KnowledgeGraph.user_id == UUID(user_id))
    )
    graphs = result.scalars().all()
    
    return graphs

@router.get("/graphs/{graph_id}", response_model=GraphResponse)
async def get_graph(
    graph_id: UUID,
    user_id: str = Depends(get_current_user_id),
    db: AsyncSession = Depends(get_database)
):
    """Get a specific graph"""
    
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
    
    return graph

@router.put("/graphs/{graph_id}", response_model=GraphResponse)
async def update_graph(
    graph_id: UUID,
    graph_data: GraphUpdate,
    user_id: str = Depends(get_current_user_id),
    db: AsyncSession = Depends(get_database)
):
    """Update a graph"""
    
    # Check if graph exists and belongs to user
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
    
    # Update fields
    update_data = graph_data.model_dump(exclude_unset=True)
    if update_data:
        await db.execute(
            update(KnowledgeGraph)
            .where(KnowledgeGraph.id == graph_id)
            .values(**update_data)
        )
        await db.commit()
        await db.refresh(graph)
    
    logger.info(f"Updated graph: {graph_id}")
    return graph

@router.delete("/graphs/{graph_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_graph(
    graph_id: UUID,
    user_id: str = Depends(get_current_user_id),
    db: AsyncSession = Depends(get_database)
):
    """Delete a knowledge graph"""

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
    
    try:
        await enhanced_graph_service.delete_graph_data(graph_id)
    except Exception as e:
        logger.error(f"Failed to delete Neo4j data: {e}")
        # Continue with metadata deletion even if Neo4j cleanup fails
    
    # Delete metadata
    await db.execute(
        delete(KnowledgeGraph).where(KnowledgeGraph.id == graph_id)
    )
    await db.commit()
    
    logger.info(f"Deleted graph: {graph_id}")

@router.post("/graphs/{graph_id}/ingest", response_model=IngestionJobResponse)
async def ingest_data_corrected(
    graph_id: UUID,
    data: IngestDataRequest,
    background_tasks: BackgroundTasks,
    user_id: str = Depends(get_current_user_id),
    db: AsyncSession = Depends(get_database)
):
    """
    CORRECTED: Ingest data with unified schema evolution approach
    """
    
    # Check if graph exists and belongs to user
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
    
    # Create evolution config from request or use defaults
    evolution_config = SchemaEvolutionConfig(
        mode=data.evolution_mode or "guided",
        max_entities=data.max_entities or 20,
        max_relationships=data.max_relationships or 15
    )
    
    # Handle schema learning if needed
    saved_schema = data.graph_schema or graph.schema_config
    
    if not saved_schema and data.content:
        try:
            # Learn schema from content using unified approach
            learned_schema = await entity_extractor.learn_schema_from_text(
                data.content[:3000],
                user_id
            )
            
            # Update graph with learned schema
            await db.execute(
                update(KnowledgeGraph)
                .where(KnowledgeGraph.id == graph_id)
                .values(schema_config=learned_schema)
            )
            await db.commit()
            
            saved_schema = learned_schema
            logger.info(f"Auto-learned schema for graph {graph_id}: {learned_schema}")
            
        except Exception as e:
            logger.warning(f"Failed to auto-learn schema: {e}")
            # Continue with extraction without predefined schema
            saved_schema = None
    
    # Create ingestion job
    job = IngestionJob(
        graph_id=graph_id,
        source_type=data.source_type,
        source_content=data.content,
        status="pending"
        # evolution_mode=evolution_config.mode  # If field exists in your model
    )
    
    db.add(job)
    await db.commit()
    await db.refresh(job)
    
    # OPTION 1: Use existing background job system (RECOMMENDED)
    try:
        task = process_ingestion_job.delay(str(job.id), user_id)
        logger.info(f"Started background job {task.id} for ingestion {job.id}")
    except Exception as e:
        logger.error(f"Failed to start background job: {e}")
        # Fallback to background tasks
        background_tasks.add_task(
            _process_ingestion_fallback,
            job.id,
            user_id,
            graph_id,
            data.content,
            saved_schema,
            evolution_config
        )
    
    logger.info(f"Started ingestion job {job.id} with evolution mode: {evolution_config.mode}")
    
    return IngestionJobResponse(
        id=job.id,
        graph_id=job.graph_id,
        status=job.status,
        progress=job.progress,
        created_at=job.created_at,
        source_type=job.source_type,
        extracted_entities=0,
        extracted_relationships=0 
        # evolution_mode=evolution_config.mode  # If field exists
    )

async def _process_ingestion_fallback(
    job_id: UUID,
    user_id: str,
    graph_id: UUID,
    content: str,
    saved_schema: Optional[dict],
    evolution_config: SchemaEvolutionConfig
):
    """Fallback background task for ingestion processing"""
    
    try:
        # Import here to avoid circular imports
        from app.services.sync_ingestion_processor import run_extraction_with_schema_evolution
        
        # Run extraction with schema evolution
        result = await run_extraction_with_schema_evolution(
            content=content,
            user_id=user_id,
            graph_id=graph_id,
            schema=saved_schema,
            evolution_config=evolution_config
        )
        
        # Update job status (you'll need to implement this)
        # await update_job_status(str(job_id), "completed" if result["success"] else "failed")
        
    except Exception as e:
        logger.error(f"Background ingestion failed: {e}")
        # await update_job_status(str(job_id), "failed", error_message=str(e))

async def _process_sync_ingestion(job_id: str, user_id: str, db: AsyncSession):
    """Fallback sync processing for development with new architecture"""
    try:
        from app.services.background_jobs import _process_ingestion_job_async  # UPDATED
        await _process_ingestion_job_async(None, job_id, user_id)
    except Exception as e:
        logger.error(f"Sync ingestion processing failed: {e}")

@router.get("/graphs/{graph_id}/jobs", response_model=List[IngestionJobResponse])
async def list_ingestion_jobs(
    graph_id: UUID,
    user_id: str = Depends(get_current_user_id),
    db: AsyncSession = Depends(get_database)
):
    """List ingestion jobs for a graph"""
    
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
    
    # Get jobs
    result = await db.execute(
        select(IngestionJob)
        .where(IngestionJob.graph_id == graph_id)
        .order_by(IngestionJob.created_at.desc())
    )
    jobs = result.scalars().all()
    
    return jobs

@router.get("/graphs/{graph_id}/jobs/{job_id}", response_model=IngestionJobResponse)
async def get_ingestion_job(
    graph_id: UUID,
    job_id: UUID,
    user_id: str = Depends(get_current_user_id),
    db: AsyncSession = Depends(get_database)
):
    """Get details of a specific ingestion job"""
    
    # Verify graph ownership first
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
    
    # Get the specific job
    result = await db.execute(
        select(IngestionJob).where(
            IngestionJob.id == job_id,
            IngestionJob.graph_id == graph_id
        )
    )
    job = result.scalar_one_or_none()
    
    if not job:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Ingestion job not found"
        )
    
    return job

@router.post("/graphs/{graph_id}/schema/learn")
async def learn_graph_schema_corrected(
    graph_id: UUID,
    request: SchemaLearnRequest,
    user_id: str = Depends(get_current_user_id),
    db: AsyncSession = Depends(get_database)
):
    """
    CORRECTED: Learn schema from a text sample with evolution support
    """
    
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
    
    try:
        # Get current saved schema
        current_saved_schema = graph.schema_config
        
        # Create evolution config from request
        evolution_config = SchemaEvolutionConfig(
            mode=request.evolution_mode or "guided",
            max_entities=request.max_entities or 20,
            max_relationships=request.max_relationships or 15
        )
        
        # Learn/evolve schema using the unified approach
        if current_saved_schema and evolution_config.mode != "strict":
            # Evolve existing schema
            evolved_schema = await entity_extractor._evolve_schema_with_new_content(
                text=request.text_sample,
                user_id=user_id,
                base_schema=current_saved_schema,
                domain_context=request.domain_context
            )
            operation = "evolved"
        else:
            # Learn new schema from scratch
            evolved_schema = await entity_extractor.learn_schema_from_text(
                text=request.text_sample,
                user_id=user_id,
                provider="openai",
                use_diffbot=True
            )
            operation = "learned"
        
        # FIXED: Get existing schema without neo4j_database parameter
        existing_db_schema = await schema_service.get_existing_schema()
        
        # Consolidate with database schema if it exists
        if existing_db_schema.get("entities"):
            consolidated_schema = await schema_service.consolidate_schema(
                existing_db_schema.get("entities", []),
                evolved_schema
            )
        else:
            consolidated_schema = evolved_schema
        
        # Update graph schema with new evolution fields
        await db.execute(
            update(KnowledgeGraph)
            .where(KnowledgeGraph.id == graph_id)
            .values(
                schema_config=consolidated_schema,
                # Add new evolution tracking fields if they exist in your model
                # evolution_mode=evolution_config.mode,
                # last_schema_update=datetime.utcnow()
            )
        )
        await db.commit()
        
        logger.info(f"Schema {operation} for graph {graph_id}: {len(consolidated_schema.get('entities', []))} entities, {len(consolidated_schema.get('relationships', []))} relationships")
        
        return {
            "operation": operation,
            "schema": consolidated_schema,
            "entities_count": len(consolidated_schema.get("entities", [])),
            "relationships_count": len(consolidated_schema.get("relationships", [])),
            "evolution_mode": evolution_config.mode
        }
        
    except Exception as e:
        logger.error(f"Schema learning failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to learn schema: {str(e)}"
        )

@router.get("/graphs/{graph_id}/schema")
async def get_graph_schema(
    graph_id: UUID,
    include_database_schema: bool = False,
    user_id: str = Depends(get_current_user_id),
    db: AsyncSession = Depends(get_database)
):
    """Get current graph schema configuration"""
    
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
    
    response = {
        "graph_id": graph_id,
        "saved_schema": graph.schema_config or {"entities": [], "relationships": []},
        # "evolution_mode": graph.evolution_mode or "guided"  # If field exists
    }
    
    # Optionally include actual database schema
    if include_database_schema:
        try:
            # FIXED: Remove neo4j_database parameter
            db_schema = await schema_service.get_existing_schema()
            response["database_schema"] = db_schema
        except Exception as e:
            logger.warning(f"Failed to get database schema: {e}")
            response["database_schema"] = {"entities": [], "relationships": []}
    
    return response

@router.put("/graphs/{graph_id}/schema")
async def update_graph_schema(
    graph_id: UUID,
    request: SchemaUpdateRequest,
    user_id: str = Depends(get_current_user_id),
    db: AsyncSession = Depends(get_database)
):
    """Update graph schema configuration"""
    
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
    
    try:
        if not isinstance(request.graph_schema.get("entities"), list):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Schema entities must be a list"
            )
        
        if not isinstance(request.graph_schema.get("relationships"), list):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Schema relationships must be a list"
            )
        
        # Apply schema size limits
        evolution_config = SchemaEvolutionConfig()
        validated_schema = entity_extractor._validate_evolved_schema(request.graph_schema)
        
        # Update graph schema
        await db.execute(
            update(KnowledgeGraph)
            .where(KnowledgeGraph.id == graph_id)
            .values(
                schema_config=validated_schema,
            )
        )
        await db.commit()
        
        return {
            "success": True,
            "schema": validated_schema,
            "evolution_mode": request.evolution_mode or "guided"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Schema update failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to update schema: {str(e)}"
        )

@router.post("/graphs/{graph_id}/optimize/similarities")
async def create_similarity_relationships(
    graph_id: UUID,
    user_id: str = Depends(get_current_user_id),
    db: AsyncSession = Depends(get_database)
):
    """Create similarity relationships for existing graph"""
    
    # Verify graph ownership
    result = await db.execute(
        select(KnowledgeGraph).where(
            KnowledgeGraph.id == graph_id,
            KnowledgeGraph.user_id == UUID(user_id)
        )
    )
    graph = result.scalar_one_or_none()
    
    if not graph:
        raise HTTPException(status_code=404, detail="Graph not found")
    
    try:
        # Get chunks for this graph
        chunks_query = """
        MATCH (c:DocumentChunk {graph_id: $graph_id})
        RETURN c.id as id, c.embedding as embedding
        """
        
        result = await neo4j_client.execute_query(chunks_query, {
            "graph_id": str(graph_id)
        })
        
        chunks = [{"id": r["id"], "embedding": r["embedding"]} for r in result if r["embedding"]]
        
        # Create similarities
        similarity_count = await _create_similarity_relationships(
            graph_id=graph_id,
            chunks=chunks,
            entities_count=graph.node_count
        )
        
        return {
            "status": "success",
            "similarity_relationships_created": similarity_count,
            "message": f"Created {similarity_count} similarity relationships"
        }
        
    except Exception as e:
        logger.error(f"Similarity creation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/graphs/{graph_id}/optimize/communities")
async def detect_communities(
    graph_id: UUID,
    background_tasks: BackgroundTasks,
    user_id: str = Depends(get_current_user_id),
    db: AsyncSession = Depends(get_database)
):
    """Detect communities and cleanup graph (with background task option)"""
    
    # Verify graph ownership
    result = await db.execute(
        select(KnowledgeGraph).where(
            KnowledgeGraph.id == graph_id,
            KnowledgeGraph.user_id == UUID(user_id)
        )
    )
    graph = result.scalar_one_or_none()
    
    if not graph:
        raise HTTPException(status_code=404, detail="Graph not found")
    
    try:
        # Option 1: Synchronous execution (for small graphs)
        communities_found = await _detect_communities_and_cleanup(graph_id)
        
        # Option 2: For large graphs, you can use background task
        # task = create_persistent_communities_task.delay(str(graph_id), user_id)
        # return {"task_id": task.id, "status": "processing"}
        
        return {
            "status": "success",
            "communities_detected": communities_found,
            "message": f"Detected {communities_found} communities and cleaned up graph"
        }
        
    except Exception as e:
        logger.error(f"Community detection failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/graphs/{graph_id}/communities/create-async")
async def create_communities_async(
    graph_id: UUID,
    user_id: str = Depends(get_current_user_id),
    db: AsyncSession = Depends(get_database)
):
    """Create persistent communities in background task (for large graphs)"""
    
    # Verify graph ownership
    result = await db.execute(
        select(KnowledgeGraph).where(
            KnowledgeGraph.id == graph_id,
            KnowledgeGraph.user_id == UUID(user_id)
        )
    )
    graph = result.scalar_one_or_none()
    
    if not graph:
        raise HTTPException(status_code=404, detail="Graph not found")
    
    try:
        # Start background task
        task = create_persistent_communities_task.delay(str(graph_id), user_id)
        
        return {
            "task_id": task.id,
            "status": "processing",
            "message": "Community creation started in background",
            "check_status_url": f"/graphs/{graph_id}/tasks/{task.id}"
        }
        
    except Exception as e:
        logger.error(f"Background community creation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/graphs/{graph_id}/communities/embeddings")
async def update_community_embeddings(
    graph_id: UUID,
    user_id: str = Depends(get_current_user_id),
    db: AsyncSession = Depends(get_database)
):
    """Generate/update embeddings for community summaries"""
    
    # Verify graph ownership
    result = await db.execute(
        select(KnowledgeGraph).where(
            KnowledgeGraph.id == graph_id,
            KnowledgeGraph.user_id == UUID(user_id)
        )
    )
    graph = result.scalar_one_or_none()
    
    if not graph:
        raise HTTPException(status_code=404, detail="Graph not found")
    
    try:
        # Start background task for embedding generation
        task = update_community_embeddings_task.delay(str(graph_id), user_id)
        
        return {
            "task_id": task.id,
            "status": "processing",
            "message": "Community embedding generation started",
            "check_status_url": f"/graphs/{graph_id}/tasks/{task.id}"
        }
        
    except Exception as e:
        logger.error(f"Community embedding update failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/graphs/optimize/refresh-all")
async def refresh_all_communities(
    user_id: str = Depends(get_current_user_id),
):
    """Refresh communities for all user graphs"""
    
    try:
        # Start background task to refresh all communities
        task = refresh_all_communities_task.delay(user_id)
        
        return {
            "task_id": task.id,
            "status": "processing", 
            "message": "Community refresh started for all graphs",
            "check_status_url": f"/tasks/{task.id}/status"
        }
        
    except Exception as e:
        logger.error(f"Refresh all communities failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/admin/optimize/all-graphs")
async def optimize_all_graphs_endpoint(
    user_id: str = Depends(get_current_user_id),
    # TODO: Add admin role check here
    # admin_user: AdminUser = Depends(get_admin_user)
):
    """Admin endpoint: Optimize all graphs in the system"""
    
    # TODO: Add admin role verification
    # if not admin_user.is_admin:
    #     raise HTTPException(status_code=403, detail="Admin access required")
    
    try:
        # Start background task for system-wide optimization
        task = optimize_all_graphs.delay()
        
        return {
            "task_id": task.id,
            "status": "processing",
            "message": "System-wide graph optimization started",
            "check_status_url": f"/admin/tasks/{task.id}/status"
        }
        
    except Exception as e:
        logger.error(f"Admin optimization failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/graphs/{graph_id}/tasks/{task_id}")
async def get_task_status(
    graph_id: UUID,
    task_id: str,
    user_id: str = Depends(get_current_user_id),
    db: AsyncSession = Depends(get_database)
):
    """Get status of a background task for a specific graph"""
    
    # Verify graph ownership
    result = await db.execute(
        select(KnowledgeGraph).where(
            KnowledgeGraph.id == graph_id,
            KnowledgeGraph.user_id == UUID(user_id)
        )
    )
    graph = result.scalar_one_or_none()
    
    if not graph:
        raise HTTPException(status_code=404, detail="Graph not found")
    
    try:
        from app.services.background_jobs import celery_app
        
        # Get task result
        task_result = celery_app.AsyncResult(task_id)
        
        if task_result.state == 'PENDING':
            response = {
                "task_id": task_id,
                "status": "pending",
                "message": "Task is waiting to be processed"
            }
        elif task_result.state == 'PROGRESS':
            response = {
                "task_id": task_id,
                "status": "processing",
                "progress": task_result.info.get('progress', 0),
                "current": task_result.info.get('status', 'Processing...'),
                "result": task_result.info
            }
        elif task_result.state == 'SUCCESS':
            response = {
                "task_id": task_id,
                "status": "completed",
                "result": task_result.result
            }
        else:  # FAILURE
            response = {
                "task_id": task_id,
                "status": "failed",
                "error": str(task_result.info)
            }
            
        return response
        
    except Exception as e:
        logger.error(f"Task status check failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

