# app/api/v1/endpoints/graphs.py - NEO4J-ONLY VERSION

from fastapi import APIRouter, Depends, HTTPException, status, BackgroundTasks
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, update
from typing import List, Any
from uuid import UUID
from datetime import datetime

# Core dependencies
from app.api.dependencies import get_current_user_id, get_database
from app.models.graph import IngestionJob  # Keep for ingestion jobs only
from app.schemas.graph_schemas import (
    GraphCreate, 
    GraphUpdate, 
    GraphResponse, 
    IngestDataRequest,
    IngestionJobResponse
)

# Neo4j Services
from app.services.graph_node_service import GraphNodeService
from app.services.background_job_service import background_job_service
from app.core.neo4j_client import neo4j_client
from app.core.logging import get_logger

router = APIRouter()
logger = get_logger(__name__)

# ==================== HELPER FUNCTIONS ====================

def convert_neo4j_datetime_to_python(neo4j_datetime: Any) -> datetime:
    """Convert Neo4j DateTime to Python datetime for Pydantic validation"""
    if isinstance(neo4j_datetime, str):
        # Handle ISO format strings from GraphNodeService
        return datetime.fromisoformat(neo4j_datetime.replace('Z', '+00:00'))
    elif hasattr(neo4j_datetime, 'to_native'):
        # Handle Neo4j DateTime objects
        return neo4j_datetime.to_native()
    elif isinstance(neo4j_datetime, datetime):
        # Already a Python datetime
        return neo4j_datetime
    else:
        # Fallback - try to parse as string
        return datetime.fromisoformat(str(neo4j_datetime).replace('Z', '+00:00'))

# ==================== NEO4J GRAPH CRUD ENDPOINTS ====================

@router.post("/graphs", response_model=GraphResponse, status_code=status.HTTP_201_CREATED)
async def create_graph(
    graph_data: GraphCreate,
    user_id: str = Depends(get_current_user_id)
):
    """Create a new knowledge graph in Neo4j"""
    
    try:
        # Use GraphNodeService with Neo4j sync driver for Neo4j operations
        if not neo4j_client.sync_driver:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Neo4j connection not available"
            )
            
        graph_service = GraphNodeService(neo4j_client.sync_driver)
        
        # Generate unique graph_id
        from uuid import uuid4
        graph_id = str(uuid4())
        
        # Create graph in Neo4j
        graph_result = graph_service.create_graph(
            graph_id=graph_id,
            user_id=user_id,
            name=graph_data.name,
            description=graph_data.description
        )
        
        logger.info(f"Created new Neo4j graph: {graph_result['graph_id']} for user: {user_id}")
        
        # Return response in expected format
        return GraphResponse(
            id=UUID(graph_result["graph_id"]),
            name=graph_result["name"],
            description=graph_result.get("description"),
            user_id=UUID(user_id),
            created_at=convert_neo4j_datetime_to_python(graph_result["created_at"]),
            updated_at=convert_neo4j_datetime_to_python(graph_result["updated_at"]),
            node_count=graph_result.get("node_count", 0),
            relationship_count=graph_result.get("relationship_count", 0),
            status=graph_result.get("status", "active"),
            schema_config=graph_data.schema_config or {}
        )
        
    except Exception as e:
        logger.error(f"Failed to create graph: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to create graph: {str(e)}"
        )

@router.get("/graphs", response_model=List[GraphResponse])
async def list_graphs(
    user_id: str = Depends(get_current_user_id)
):
    """List all graphs for the current user from Neo4j"""
    
    try:
        if not neo4j_client.sync_driver:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Neo4j connection not available"
            )
            
        graph_service = GraphNodeService(neo4j_client.sync_driver)
        graphs = graph_service.list_user_graphs(user_id)
        
        # Convert to GraphResponse format
        graph_responses = []
        for graph in graphs:
            graph_responses.append(GraphResponse(
                id=UUID(graph["graph_id"]),
                name=graph["name"],
                description=graph.get("description", ""),
                user_id=UUID(user_id),
                created_at=convert_neo4j_datetime_to_python(graph["created_at"]),
                updated_at=convert_neo4j_datetime_to_python(graph["updated_at"]),
                node_count=graph.get("node_count", 0),
                relationship_count=graph.get("relationship_count", 0),
                status=graph.get("status", "active"),
                schema_config={}  # Will be populated from graph metadata later
            ))
        
        return graph_responses
        
    except Exception as e:
        logger.error(f"Failed to list graphs: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to list graphs: {str(e)}"
        )

@router.get("/graphs/{graph_id}", response_model=GraphResponse)
async def get_graph(
    graph_id: UUID,
    user_id: str = Depends(get_current_user_id)
):
    """Get a specific graph by ID from Neo4j"""
    
    try:
        if not neo4j_client.sync_driver:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Neo4j connection not available"
            )
            
        graph_service = GraphNodeService(neo4j_client.sync_driver)
        graph = graph_service.get_graph(str(graph_id))
        
        if not graph:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Graph not found"
            )
            
        # Verify ownership
        if graph["user_id"] != user_id:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Access denied"
            )
        
        return GraphResponse(
            id=UUID(graph["graph_id"]),
            name=graph["name"],
            description=graph.get("description", ""),
            user_id=UUID(user_id),
            created_at=convert_neo4j_datetime_to_python(graph["created_at"]),
            updated_at=convert_neo4j_datetime_to_python(graph["updated_at"]),
            node_count=graph.get("node_count", 0),
            relationship_count=graph.get("relationship_count", 0),
            status=graph.get("status", "active"),
            schema_config={}  # Will be populated from graph metadata later
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get graph {graph_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get graph: {str(e)}"
        )

@router.put("/graphs/{graph_id}", response_model=GraphResponse)
async def update_graph(
    graph_id: UUID,
    graph_update: GraphUpdate,
    user_id: str = Depends(get_current_user_id)
):
    """Update a specific graph in Neo4j"""
    
    try:
        if not neo4j_client.sync_driver:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Neo4j connection not available"
            )
            
        graph_service = GraphNodeService(neo4j_client.sync_driver)
        
        # Verify graph exists and user has access
        existing_graph = graph_service.get_graph(str(graph_id))
        if not existing_graph:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Graph not found"
            )
            
        if existing_graph["user_id"] != user_id:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Access denied"
            )
        
        # Update graph
        updated_graph = graph_service.update_graph(
            graph_id=str(graph_id),
            user_id=user_id,
            name=graph_update.name,
            description=graph_update.description
        )
        
        if not updated_graph:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to update graph"
            )
        
        return GraphResponse(
            id=UUID(updated_graph["graph_id"]),
            name=updated_graph["name"],
            description=updated_graph.get("description", ""),
            user_id=UUID(user_id),
            created_at=convert_neo4j_datetime_to_python(updated_graph["created_at"]),
            updated_at=convert_neo4j_datetime_to_python(updated_graph["updated_at"]),
            node_count=updated_graph.get("node_count", 0),
            relationship_count=updated_graph.get("relationship_count", 0),
            status=updated_graph.get("status", "active"),
            schema_config=graph_update.schema_config or {}
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to update graph {graph_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to update graph: {str(e)}"
        )

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
    Ingest data using Neo4j GraphRAG pipeline.
    Verifies graph exists in Neo4j, then creates PostgreSQL ingestion job.
    """
    
    # Check if graph exists in Neo4j and belongs to user
    try:
        if not neo4j_client.sync_driver:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Neo4j connection not available"
            )
            
        graph_service = GraphNodeService(neo4j_client.sync_driver)
        neo4j_graph = graph_service.get_graph(str(graph_id))
        
        if not neo4j_graph:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Graph not found"
            )
            
        # Verify ownership
        if neo4j_graph["user_id"] != user_id:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Access denied"
            )
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to verify graph {graph_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to verify graph: {str(e)}"
        )
    
    # Create ingestion job record (still in PostgreSQL for job tracking)
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
