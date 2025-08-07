from pydantic import BaseModel, Field
from typing import Optional, Dict, Any
from uuid import UUID
from datetime import datetime
from app.models.ingestion_job_model import IngestionStatus

class IngestionJobCreate(BaseModel):
    source_id: UUID
    config: Optional[Dict[str, Any]] = {}
    owner_id: Optional[str] = None

class IngestionJobUpdate(BaseModel):
    status: Optional[IngestionStatus] = None
    config: Optional[Dict[str, Any]] = None
    documents_count: Optional[int] = None
    error_message: Optional[str] = None
    job_metadata: Optional[Dict[str, Any]] = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None

class IngestionJobRead(BaseModel):
    id: UUID
    source_id: UUID
    status: IngestionStatus
    config: Optional[Dict[str, Any]]
    documents_count: int
    error_message: Optional[str]
    job_metadata: Optional[Dict[str, Any]]
    owner_id: Optional[str]
    created_at: datetime
    started_at: Optional[datetime]
    completed_at: Optional[datetime]
    
    # Related objects
    source: Optional['DataSourceRead'] = None

    class Config:
        from_attributes = True

class IngestionJobStart(BaseModel):
    """Schema for starting an ingestion job"""
    job_id: UUID
    config_override: Optional[Dict[str, Any]] = None

class IngestionJobCancel(BaseModel):
    """Schema for cancelling an ingestion job"""
    job_id: UUID
    reason: Optional[str] = None
