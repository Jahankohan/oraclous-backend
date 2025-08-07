from pydantic import BaseModel, Field
from typing import Optional, Dict, Any
from uuid import UUID
from datetime import datetime
from app.models.data_source_model import DataSourceType

class DataSourceCreate(BaseModel):
    name: str = Field(..., min_length=1, max_length=255)
    type: DataSourceType
    config: Dict[str, Any]
    credentials_ref: Optional[str] = None  # Usually user_id for OAuth
    data_metadata: Optional[Dict[str, Any]] = None
    owner_id: Optional[str] = None

class DataSourceUpdate(BaseModel):
    name: Optional[str] = Field(None, min_length=1, max_length=255)
    config: Optional[Dict[str, Any]] = None
    credentials_ref: Optional[str] = None
    data_metadata: Optional[Dict[str, Any]] = None

class DataSourceRead(BaseModel):
    id: UUID
    name: str
    type: DataSourceType
    config: Dict[str, Any]
    credentials_ref: Optional[str]
    data_metadata: Optional[Dict[str, Any]]
    owner_id: Optional[str]
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True

class DataSourceTest(BaseModel):
    """Schema for testing data source connection"""
    source_id: UUID

class DataSourceResources(BaseModel):
    """Schema for listing available resources from a data source"""
    source_id: UUID
    refresh: bool = False  # Whether to refresh the resource list
