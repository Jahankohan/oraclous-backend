from pydantic import BaseModel, Field
from typing import Optional, Dict, Any, List
from datetime import datetime
import uuid

from app.schemas.common import CredentialType, ToolCategory, ToolType


# Schema definitions
class ToolSchema(BaseModel):
    """Defines the input/output schema for a tool"""
    type: str = Field(..., description="Schema type (object, array, string, etc.)")
    properties: Optional[Dict[str, Any]] = Field(None, description="Object properties")
    required: Optional[List[str]] = Field(None, description="Required properties")
    items: Optional[Dict[str, Any]] = Field(None, description="Array item schema")
    description: Optional[str] = Field(None, description="Schema description")

class ToolCapability(BaseModel):
    """Represents a capability that a tool provides"""
    name: str = Field(..., description="Capability name")
    description: str = Field(..., description="Capability description")
    parameters: Optional[Dict[str, Any]] = Field(None, description="Capability parameters")

class CredentialRequirement(BaseModel):
    """Defines what credentials a tool needs"""
    type: CredentialType
    required: bool = True
    scopes: Optional[List[str]] = None
    description: Optional[str] = None

# Tool Definition (Registry Entry)
class ToolDefinition(BaseModel):
    """
    Tool Definition - Metadata stored in Tool Registry
    This is the 'blueprint' of a tool, not an executable instance
    """
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    name: str = Field(..., description="Human-readable tool name")
    description: str = Field(..., description="Tool description")
    version: str = Field(default="1.0.0", description="Tool version")
    icon: Optional[str] = Field(None, description="Tool icon URL")
    
    # Categorization
    category: ToolCategory = Field(..., description="Tool category")
    type: ToolType = Field(..., description="Tool implementation type")
    capabilities: List[ToolCapability] = Field(default_factory=list)
    tags: List[str] = Field(default_factory=list)
    
    # Schema definitions
    input_schema: ToolSchema = Field(..., description="Input data schema")
    output_schema: ToolSchema = Field(..., description="Output data schema")
    configuration_schema: Optional[ToolSchema] = Field(None, description="Configuration schema")
    
    # Requirements
    credential_requirements: List[CredentialRequirement] = Field(default_factory=list)
    dependencies: List[str] = Field(default_factory=list, description="Tool dependencies")
    
    # Metadata
    author: Optional[str] = None
    documentation_url: Optional[str] = None
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)


# Registry query helpers
class ToolQuery(BaseModel):
    """Helper class for complex tool queries"""
    text: Optional[str] = None
    category: Optional[ToolCategory] = None
    capabilities: Optional[List[str]] = None
    tags: Optional[List[str]] = None
    type: Optional[ToolType] = None
    limit: int = 10
    offset: int = 0