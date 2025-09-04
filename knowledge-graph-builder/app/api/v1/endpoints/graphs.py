# app/api/v1/endpoints/graphs.py - COMPLETE CORRECTED VERSION

from fastapi import APIRouter, Depends, HTTPException, status, BackgroundTasks
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, update
from typing import List
from uuid import UUID

# Core dependencies
from app.api.dependencies import get_current_user_id, get_database
from app.models.graph import KnowledgeGraph, IngestionJob
from app.schemas.graph_schemas import (
    GraphCreate, 
    GraphUpdate, 
    GraphResponse, 
    IngestDataRequest,
    IngestionJobResponse
)

# Services that exist after refactoring
from app.services.background_job_service import background_job_service
from app.core.logging import get_logger

router = APIRouter()
logger = get_logger(__name__)

# ==================== GRAPH CRUD ENDPOINTS ====================

@router.post("/graphs", response_model=GraphResponse, status_code=status.HTTP_201_CREATED)
async def create_graph(
    graph_data: GraphCreate,
    user_id: str = Depends(get_current_user_id),
    db: AsyncSession = Depends(get_database)
):
    """Create a new knowledge graph"""
    
    graph = KnowledgeGraph(
        name=graph_data.name,
        description=graph_data.description,
        user_id=UUID(user_id),
        schema_config=graph_data.schema_config or {}
    )
    
    db.add(graph)
    await db.commit()
    await db.refresh(graph)
    
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

# ==================== SIMPLIFIED INGESTION ENDPOINT ====================

@router.post("/graphs/{graph_id}/ingest", response_model=IngestionJobResponse)
async def ingest_data_corrected(
    graph_id: UUID,
    data: IngestDataRequest,
    background_tasks: BackgroundTasks,
    user_id: str = Depends(get_current_user_id),
    db: AsyncSession = Depends(get_database)
):
    """
    SIMPLIFIED: Ingest data using Neo4j GraphRAG pipeline only
    No schema evolution complexity - focus on core ingestion flow
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
    
    # Create ingestion job record
    job = IngestionJob(
        graph_id=graph_id,
        source_type=data.source_type or "text",
        source_content=data.content,
        status="pending"
    )
    
    db.add(job)
    await db.commit()
    await db.refresh(job)
    
    # Start background ingestion job using pipeline service
    try:
        job_result = background_job_service.start_ingestion_job(str(job.id), user_id)
        
        if job_result["status"] == "failed":
            logger.error(f"Failed to start background job: {job_result['message']}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to start background ingestion job"
            )
        
        logger.info(f"Started ingestion job {job.id} for graph {graph_id}")
        
        return IngestionJobResponse(
            id=job.id, # type: ignore
            graph_id=job.graph_id, # type: ignore
            status=job.status, # type: ignore
            progress=job.progress, # type: ignore
            created_at=job.created_at, # type: ignore
            source_type=job.source_type, # type: ignore
            extracted_entities=0,
            extracted_relationships=0
        )
        
    except Exception as e:
        logger.error(f"Ingestion job creation failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to create ingestion job: {str(e)}"
        )

# ==================== JOB MANAGEMENT ENDPOINTS ====================

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
