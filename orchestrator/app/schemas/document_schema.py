from pydantic import BaseModel, Field
from typing import Optional, Dict, Any
from uuid import UUID
from datetime import datetime
from app.models.ingestion_job_model import IngestionStatus

class DocumentCreate(BaseModel):
    job_id: UUID
    source_id: UUID
    external_id: Optional[str] = None
    title: Optional[str] = None
    content: str
    content_type: Optional[str] = "text"
    metadata: Optional[Dict[str, Any]] = None
    embeddings: Optional[Dict[str, Any]] = None
    content_hash: Optional[str] = None
    size_bytes: Optional[int] = None
    owner_id: Optional[str] = None

class DocumentUpdate(BaseModel):
    title: Optional[str] = None
    content: Optional[str] = None
    content_type: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None
    embeddings: Optional[Dict[str, Any]] = None

class DocumentRead(BaseModel):
    id: UUID
    job_id: UUID
    source_id: UUID
    external_id: Optional[str]
    title: Optional[str]
    content: str
    content_type: Optional[str]
    metadata: Optional[Dict[str, Any]]
    embeddings: Optional[Dict[str, Any]]
    content_hash: Optional[str]
    size_bytes: Optional[int]
    owner_id: Optional[str]
    created_at: datetime
    updated_at: datetime
    
    # Related objects
    source: Optional['DataSourceRead'] = None
    job: Optional['IngestionJobRead'] = None

    class Config:
        from_attributes = True

class DocumentSearch(BaseModel):
    """Schema for document search requests"""
    query: str = Field(..., min_length=1)
    limit: int = Field(50, ge=1, le=1000)
    source_ids: Optional[list[UUID]] = None
    content_types: Optional[list[str]] = None

class DocumentBatch(BaseModel):
    """Schema for batch document operations"""
    document_ids: list[UUID]
    action: str  # "delete", "update_metadata", etc.
    parameters: Optional[Dict[str, Any]] = None

# Ingestion API request/response schemas
class IngestionRequest(BaseModel):
    """Request to start data ingestion"""
    source_id: UUID
    config: Optional[Dict[str, Any]] = {}

class IngestionResponse(BaseModel):
    """Response from ingestion request"""
    job_id: UUID
    status: IngestionStatus
    message: str

class IngestionSummary(BaseModel):
    """Summary of ingestion results"""
    job_id: UUID
    source_name: str
    source_type: str
    status: IngestionStatus
    documents_ingested: int
    errors_count: int
    duration_seconds: Optional[float] = None
    started_at: Optional[datetime]
    completed_at: Optional[datetime]

class AvailableResource(BaseModel):
    """Schema for available resources from a data source"""
    id: str
    name: str
    type: str
    metadata: Optional[Dict[str, Any]] = None

class ResourcesResponse(BaseModel):
    """Response containing available resources"""
    source_id: UUID
    resources: list[AvailableResource]
    total_count: int
    last_updated: datetime
