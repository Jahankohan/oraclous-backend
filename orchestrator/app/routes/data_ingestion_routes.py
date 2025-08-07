from fastapi import APIRouter, HTTPException, Request, Depends, BackgroundTasks
from typing import List, Optional
from uuid import UUID

from app.services.data_ingestion_service import DataIngestionService
from app.repositories.data_source_repository import DataSourceRepository
from app.repositories.ingestion_job_repository import IngestionJobRepository
from app.repositories.document_repository import DocumentRepository
from app.schemas.data_source_schema import (
    DataSourceCreate, DataSourceUpdate, DataSourceRead, DataSourceTest
)
from app.schemas.ingestion_job_schema import (
    IngestionJobCreate, IngestionJobRead, IngestionJobStart
)
from app.schemas.document_schema import DocumentSearch
from app.models.ingestion_job_model import IngestionStatus

router = APIRouter(prefix="/ingestion", tags=["Data Ingestion"])

async def get_ingestion_service(request: Request) -> DataIngestionService:
    """Dependency to get the data ingestion service"""
    data_source_repo = request.app.state.data_source_repository
    ingestion_job_repo = request.app.state.ingestion_job_repository
    document_repo = request.app.state.document_repository
    ingestion_registry = request.app.state.ingestion_registry
    
    return DataIngestionService(
        data_source_repo=data_source_repo,
        ingestion_job_repo=ingestion_job_repo,
        document_repo=document_repo,
        ingestion_registry=ingestion_registry
    )

# Data Source Endpoints
@router.post("/sources", response_model=DataSourceRead)
async def create_data_source(
    data: DataSourceCreate,
    service: DataIngestionService = Depends(get_ingestion_service)
):
    """Create a new data source"""
    try:
        return await service.create_data_source(data)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.get("/sources", response_model=List[DataSourceRead])
async def list_data_sources(
    owner_id: Optional[str] = None,
    service: DataIngestionService = Depends(get_ingestion_service)
):
    """List all data sources"""
    return await service.list_data_sources(owner_id=owner_id)

@router.get("/sources/{source_id}", response_model=DataSourceRead)
async def get_data_source(
    source_id: UUID,
    service: DataIngestionService = Depends(get_ingestion_service)
):
    """Get a specific data source"""
    source = await service.get_data_source(source_id)
    if not source:
        raise HTTPException(status_code=404, detail="Data source not found")
    return source

@router.put("/sources/{source_id}", response_model=DataSourceRead)
async def update_data_source(
    source_id: UUID,
    data: DataSourceUpdate,
    service: DataIngestionService = Depends(get_ingestion_service)
):
    """Update a data source"""
    source = await service.update_data_source(source_id, data)
    if not source:
        raise HTTPException(status_code=404, detail="Data source not found")
    return source

@router.delete("/sources/{source_id}")
async def delete_data_source(
    source_id: UUID,
    service: DataIngestionService = Depends(get_ingestion_service)
):
    """Delete a data source"""
    success = await service.delete_data_source(source_id)
    if not success:
        raise HTTPException(status_code=404, detail="Data source not found")
    return {"status": "deleted"}

@router.post("/sources/{source_id}/test")
async def test_data_source(
    source_id: UUID,
    service: DataIngestionService = Depends(get_ingestion_service)
):
    """Test connection to a data source"""
    try:
        success = await service.test_data_source(source_id)
        return {"success": success, "message": "Connection successful" if success else "Connection failed"}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.get("/sources/{source_id}/resources")
async def get_available_resources(
    source_id: UUID,
    service: DataIngestionService = Depends(get_ingestion_service)
):
    """Get available resources from a data source"""
    try:
        resources = await service.get_available_resources(source_id)
        return {
            "source_id": source_id,
            "resources": resources,
            "total_count": len(resources)
        }
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

# Ingestion Job Endpoints
@router.post("/jobs", response_model=IngestionJobRead)
async def create_ingestion_job(
    data: IngestionJobCreate,
    service: DataIngestionService = Depends(get_ingestion_service)
):
    """Create a new ingestion job"""
    try:
        return await service.create_ingestion_job(data)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.post("/jobs/{job_id}/start")
async def start_ingestion_job(
    job_id: UUID,
    start_data: Optional[IngestionJobStart] = None,
    service: DataIngestionService = Depends(get_ingestion_service)
):
    """Start an ingestion job"""
    try:
        config_override = start_data.config_override if start_data else None
        success = await service.start_ingestion_job(job_id, config_override)
        return {"job_id": job_id, "status": "started" if success else "failed"}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.post("/jobs/{job_id}/cancel")
async def cancel_ingestion_job(
    job_id: UUID,
    service: DataIngestionService = Depends(get_ingestion_service)
):
    """Cancel a running ingestion job"""
    success = await service.cancel_job(job_id)
    return {"job_id": job_id, "status": "cancelled" if success else "not_running"}

@router.get("/jobs", response_model=List[IngestionJobRead])
async def list_ingestion_jobs(
    source_id: Optional[UUID] = None,
    status: Optional[IngestionStatus] = None,
    service: DataIngestionService = Depends(get_ingestion_service)
):
    """List ingestion jobs with optional filters"""
    return await service.list_ingestion_jobs(source_id=source_id, status=status)

@router.get("/jobs/{job_id}", response_model=IngestionJobRead)
async def get_ingestion_job(
    job_id: UUID,
    service: DataIngestionService = Depends(get_ingestion_service)
):
    """Get a specific ingestion job"""
    job = await service.get_ingestion_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Ingestion job not found")
    return job

@router.get("/jobs/{job_id}/documents")
async def get_job_documents(
    job_id: UUID,
    service: DataIngestionService = Depends(get_ingestion_service)
):
    """Get all documents from a specific ingestion job"""
    documents = await service.get_documents_by_job(job_id)
    return {
        "job_id": job_id,
        "documents": documents,
        "total_count": len(documents)
    }

@router.delete("/jobs/{job_id}/documents")
async def delete_job_documents(
    job_id: UUID,
    service: DataIngestionService = Depends(get_ingestion_service)
):
    """Delete all documents from a specific job"""
    count = await service.delete_documents_by_job(job_id)
    return {"job_id": job_id, "deleted_count": count}

# Document Search Endpoints
@router.post("/documents/search")
async def search_documents(
    search_data: DocumentSearch,
    service: DataIngestionService = Depends(get_ingestion_service)
):
    """Search documents by content"""
    documents = await service.search_documents(
        query=search_data.query,
        limit=search_data.limit,
        source_ids=search_data.source_ids
    )
    return {
        "query": search_data.query,
        "documents": documents,
        "total_count": len(documents)
    }

# Quick ingestion endpoint (create source + job + start in one call)
@router.post("/quick-ingest")
async def quick_ingestion(
    source_data: DataSourceCreate,
    job_config: Optional[dict] = None,
    service: DataIngestionService = Depends(get_ingestion_service)
):
    """Create a data source, ingestion job, and start ingestion in one call"""
    try:
        # Create data source
        source = await service.create_data_source(source_data)
        
        # Create ingestion job
        job_data = IngestionJobCreate(
            source_id=source.id,
            config=job_config or {},
            owner_id=source_data.owner_id
        )
        job = await service.create_ingestion_job(job_data)
        
        # Start the job
        await service.start_ingestion_job(job.id)
        
        return {
            "source": source,
            "job": job,
            "status": "started"
        }
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

# Health check for ingestors
@router.get("/ingestors")
async def list_available_ingestors(request: Request):
    """List all available ingestors and their supported types"""
    registry = request.app.state.ingestion_registry
    ingestors = registry.list_ingestors()
    
    return {
        "ingestors": [
            {
                "name": ingestor.name,
                "supported_types": ingestor.supported_types
            }
            for ingestor in ingestors.values()
        ],
        "supported_types": registry.list_supported_types()
    }