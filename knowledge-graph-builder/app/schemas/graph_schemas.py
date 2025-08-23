from pydantic import BaseModel, Field, ConfigDict
from typing import Optional, Dict, Any, List
from datetime import datetime
from uuid import UUID

class GraphCreate(BaseModel):
    """Schema for creating a new knowledge graph"""
    name: str = Field(..., min_length=1, max_length=255)
    description: Optional[str] = Field(None, max_length=1000)
    schema_config: Optional[Dict[str, Any]] = Field(None)

class GraphUpdate(BaseModel):
    """Schema for updating a knowledge graph"""
    name: Optional[str] = Field(None, min_length=1, max_length=255)
    description: Optional[str] = Field(None, max_length=1000)
    schema_config: Optional[Dict[str, Any]] = Field(None)

class GraphResponse(BaseModel):
    """Schema for knowledge graph response"""
    model_config = ConfigDict(from_attributes=True)
    
    id: UUID
    name: str
    description: Optional[str]
    user_id: UUID
    schema_config: Optional[Dict[str, Any]]
    created_at: datetime
    updated_at: datetime
    node_count: int
    relationship_count: int
    status: str

class IngestDataRequest(BaseModel):
    """Schema for data ingestion request"""
    content: str = Field(..., min_length=1)
    source_type: str = Field(default="text", pattern="^(text|pdf|url|api)$")
    schema: Optional[Dict[str, Any]] = Field(None)
    instructions: Optional[str] = Field(None)
    use_diffbot: bool = Field(default=True, description="Enable Diffbot NLP extraction")
    llm_provider: str = Field(default="openai", description="LLM provider for extraction")

class IngestionJobResponse(BaseModel):
    """Schema for ingestion job response"""
    id: UUID
    graph_id: UUID
    source_type: Optional[str]
    status: str
    progress: int
    error_message: Optional[str]
    extracted_entities: int
    extracted_relationships: int
    processed_chunks: int = 0
    similarity_relationships: int = 0
    communities_detected: int = 0
    error_message: Optional[str] = None
    created_at: datetime
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    
    class Config:
        from_attributes = True

class HealthResponse(BaseModel):
    """Health check response schema"""
    status: str
    service: str
    version: str
    timestamp: datetime
    dependencies: Dict[str, Any]
