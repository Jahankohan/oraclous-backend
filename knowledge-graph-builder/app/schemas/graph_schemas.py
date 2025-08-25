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

class SchemaLearnRequest(BaseModel):
    """Request for learning schema from text"""
    text_sample: str = Field(..., min_length=50, description="Sample text to learn schema from")
    domain_context: Optional[str] = Field(None, description="Domain context (e.g., 'medical', 'legal')")
    evolution_mode: Optional[str] = Field("guided", description="Schema evolution mode: strict, guided, permissive")
    max_entities: Optional[int] = Field(20, description="Maximum number of entity types")
    max_relationships: Optional[int] = Field(15, description="Maximum number of relationship types")

class SchemaUpdateRequest(BaseModel):
    """Request for updating graph schema"""
    graph_schema: dict = Field(..., description="Schema with entities and relationships")
    evolution_mode: Optional[str] = Field("guided", description="Schema evolution mode")

class IngestDataRequest(BaseModel):
    """Enhanced request for data ingestion with schema evolution"""
    content: str = Field(..., min_length=10)
    source_type: str = Field(default="text")
    graph_schema: Optional[dict] = None
    instructions: Optional[str] = None
    
    # Schema evolution parameters
    evolution_mode: Optional[str] = Field("guided", description="Schema evolution mode: strict, guided, permissive")
    max_entities: Optional[int] = Field(20, description="Maximum entity types allowed")
    max_relationships: Optional[int] = Field(15, description="Maximum relationship types allowed")
    allow_schema_evolution: Optional[bool] = Field(True, description="Allow schema to evolve during ingestion")

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

class SchemaValidationRequest(BaseModel):
    """Request for validating a schema"""
    entities: List[str] = Field(..., description="List of entity types")
    relationships: List[str] = Field(..., description="List of relationship types")

class SchemaValidationResponse(BaseModel):
    """Response for schema validation"""
    valid: bool
    entities_count: int
    relationships_count: int
    warnings: List[str] = []
    errors: List[str] = []

class SchemaEvolutionSettings(BaseModel):
    """Settings for schema evolution"""
    mode: str = Field("guided", description="Evolution mode")
    max_entities: int = Field(20, description="Maximum entities")
    max_relationships: int = Field(15, description="Maximum relationships")
    evolution_threshold: float = Field(0.3, description="Threshold for triggering evolution")
    auto_consolidate: bool = Field(True, description="Automatically consolidate similar entities")

class GraphConfiguration(BaseModel):
    """Complete graph configuration"""
    graph_schema: Dict[str, List[str]]
    evolution_settings: SchemaEvolutionSettings
    domain_context: Optional[str] = None
    custom_instructions: Optional[str] = None