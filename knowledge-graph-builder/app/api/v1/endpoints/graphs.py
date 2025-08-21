from fastapi import APIRouter, Depends, HTTPException, status, BackgroundTasks
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, update, delete
from typing import List
from uuid import UUID
import uuid
from datetime import datetime

from app.api.dependencies import get_current_user_id, get_database
from app.models.graph import KnowledgeGraph, IngestionJob
from app.schemas.graph_schemas import (
    GraphCreate, GraphUpdate, GraphResponse, 
    IngestDataRequest, IngestionJobResponse
)
from app.services.background_jobs import process_ingestion_job
from app.services.entity_extractor import entity_extractor
from app.services.schema_service import schema_service
from app.services.graph_service import graph_service
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
    """Delete a graph and all its data"""
    
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
    
    # Delete Neo4j data
    try:
        await graph_service.delete_graph_data(graph_id)
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
async def ingest_data(
    graph_id: UUID,
    data: IngestDataRequest,
    background_tasks: BackgroundTasks,
    user_id: str = Depends(get_current_user_id),
    db: AsyncSession = Depends(get_database)
):
    """Ingest data into a knowledge graph (async processing)"""
    
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
    
    # Learn schema from text if not provided
    if not data.schema and not graph.schema_config:
        try:
            learned_schema = await entity_extractor.learn_schema_from_text(
                data.content, user_id
            )
            
            # Update graph with learned schema
            await db.execute(
                update(KnowledgeGraph)
                .where(KnowledgeGraph.id == graph_id)
                .values(schema_config=learned_schema)
            )
            await db.commit()
            
            logger.info(f"Learned schema for graph {graph_id}: {learned_schema}")
            
        except Exception as e:
            logger.warning(f"Failed to learn schema: {e}")
    
    # Create ingestion job
    job = IngestionJob(
        graph_id=graph_id,
        source_type=data.source_type,
        source_content=data.content,
        status="pending"
    )
    
    db.add(job)
    await db.commit()
    await db.refresh(job)
    
    # Trigger background processing
    try:
        # Use Celery for background processing
        task = process_ingestion_job.delay(str(job.id), user_id)
        logger.info(f"Started background job {task.id} for ingestion {job.id}")
        
    except Exception as e:
        logger.error(f"Failed to start background job: {e}")
        # Fall back to immediate processing for development
        background_tasks.add_task(
            _process_sync_ingestion, 
            str(job.id), 
            user_id, 
            db
        )
    
    logger.info(f"Created ingestion job: {job.id} for graph: {graph_id}")
    
    return job

async def _process_sync_ingestion(job_id: str, user_id: str, db: AsyncSession):
    """Fallback sync processing for development"""
    try:
        from app.services.background_jobs import _process_ingestion_job_async
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
async def learn_graph_schema(
    graph_id: UUID,
    text_sample: str,
    user_id: str = Depends(get_current_user_id),
    db: AsyncSession = Depends(get_database)
):
    """Learn schema from a text sample"""
    
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
        # Learn schema from text
        learned_schema = await entity_extractor.learn_schema_from_text(
            text_sample, user_id
        )
        
        # Get existing schema
        existing_schema = await schema_service.get_existing_schema(
            graph.neo4j_database
        )
        
        # Consolidate schemas
        consolidated_schema = await schema_service.consolidate_schema(
            existing_schema.get("entities", []),
            learned_schema
        )
        
        # Update graph schema
        await db.execute(
            update(KnowledgeGraph)
            .where(KnowledgeGraph.id == graph_id)
            .values(schema_config=consolidated_schema)
        )
        await db.commit()
        
        return consolidated_schema
        
    except Exception as e:
        logger.error(f"Schema learning failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to learn schema"
        )
