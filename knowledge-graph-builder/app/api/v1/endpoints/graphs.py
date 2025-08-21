from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, update, delete
from typing import List
from uuid import UUID
import uuid

from app.api.dependencies import get_current_user_id, get_database
from app.models.graph import KnowledgeGraph, IngestionJob
from app.schemas.graph_schemas import (
    GraphCreate, GraphUpdate, GraphResponse, 
    IngestDataRequest, IngestionJobResponse
)
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
    
    # Generate unique database name for Neo4j
    neo4j_db_name = f"graph_{uuid.uuid4().hex[:8]}"
    
    graph = KnowledgeGraph(
        name=graph_data.name,
        description=graph_data.description,
        user_id=UUID(user_id),
        neo4j_database=neo4j_db_name,
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

@router.delete("/graphs/{graph_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_graph(
    graph_id: UUID,
    user_id: str = Depends(get_current_user_id),
    db: AsyncSession = Depends(get_database)
):
    """Delete a graph"""
    
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
    
    # TODO: Clean up Neo4j database and related data
    
    await db.execute(
        delete(KnowledgeGraph).where(KnowledgeGraph.id == graph_id)
    )
    await db.commit()
    
    logger.info(f"Deleted graph: {graph_id}")

@router.post("/graphs/{graph_id}/ingest", response_model=IngestionJobResponse)
async def ingest_data(
    graph_id: UUID,
    data: IngestDataRequest,
    user_id: str = Depends(get_current_user_id),
    db: AsyncSession = Depends(get_database)
):
    """Ingest data into a knowledge graph"""
    
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
    
    # TODO: Trigger background processing
    logger.info(f"Created ingestion job: {job.id} for graph: {graph_id}")
    
    return job

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
        select(IngestionJob).where(IngestionJob.graph_id == graph_id)
    )
    jobs = result.scalars().all()
    
    return jobs
