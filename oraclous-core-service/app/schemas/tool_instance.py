from pydantic import BaseModel, Field, validator
import uuid
from typing import Optional, Dict, Any, List
from datetime import datetime
from decimal import Decimal

from app.schemas.common import InstanceStatus

# Tool Instance (Workflow Execution)
class ToolInstance(BaseModel):
    """
    Tool Instance - Configured instance of a tool for workflow execution
    This represents a specific use of a tool with specific configuration
    """
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    workflow_id: uuid.UUID = Field(..., description="Parent workflow ID")
    tool_definition_id: str = Field(..., description="Reference to tool definition")
    user_id: uuid.UUID = Field(..., description="Owner user ID")
    
    # Instance configuration
    name: str = Field(..., description="Instance name (user-defined)")
    description: Optional[str] = Field(None, description="Instance description")
    configuration: Dict[str, Any] = Field(default_factory=dict)
    settings: Dict[str, Any] = Field(default_factory=dict)
    
    # Credential management
    credential_mappings: Dict[str, str] = Field(default_factory=dict, description="Maps credential_type -> credential_id")
    required_credentials: List[str] = Field(default_factory=list, description="List of required credential types")
    
    # Runtime state
    status: InstanceStatus = Field(default=InstanceStatus.PENDING)
    
    # Execution metadata
    last_execution_id: Optional[str] = None
    execution_count: int = 0
    total_credits_consumed: Decimal = Field(default=Decimal('0'))
    
    # Metadata
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)

    oauth_redirects: Optional[Dict[str, Optional[str]]] = None
    missing_credentials: Optional[List[str]] = None


class CreateInstanceRequest(BaseModel):
    """Request to create a new tool instance"""
    workflow_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    tool_definition_id: str = Field(..., description="Tool definition to instantiate")
    name: str = Field(..., description="Instance name")
    description: Optional[str] = Field(None, description="Instance description")
    configuration: Dict[str, Any] = Field(default_factory=dict, description="Initial configuration")
    settings: Dict[str, Any] = Field(default_factory=dict, description="Runtime settings")

    @validator('name')
    def validate_name(cls, v):
        if not v or len(v.strip()) == 0:
            raise ValueError('Name cannot be empty')
        if len(v) > 255:
            raise ValueError('Name too long (max 255 characters)')
        return v.strip()


class UpdateInstanceRequest(BaseModel):
    """Request to update an existing instance"""
    name: Optional[str] = Field(None, description="Updated name")
    description: Optional[str] = Field(None, description="Updated description")
    configuration: Optional[Dict[str, Any]] = Field(None, description="Updated configuration")
    settings: Optional[Dict[str, Any]] = Field(None, description="Updated settings")

    @validator('name')
    def validate_name(cls, v):
        if v is not None:
            if len(v.strip()) == 0:
                raise ValueError('Name cannot be empty')
            if len(v) > 255:
                raise ValueError('Name too long (max 255 characters)')
            return v.strip()
        return v


class ConfigureCredentialsRequest(BaseModel):
    """Request to configure credentials for an instance"""
    credential_mappings: Dict[str, str] = Field(..., description="Maps credential_type -> credential_id")


class InstanceCredentialStatus(BaseModel):
    """Status of instance credentials"""
    credential_type: str
    required: bool
    configured: bool
    valid: bool
    error_message: Optional[str] = None


class InstanceStatusResponse(BaseModel):
    """Complete status information for an instance"""
    instance: ToolInstance
    credentials_status: List[InstanceCredentialStatus]
    is_ready_for_execution: bool
    validation_errors: List[str] = Field(default_factory=list)


class ExecutionContext(BaseModel):
    """Context provided during tool execution"""
    instance_id: str
    workflow_id: str
    user_id: str
    job_id: str
    credentials: Optional[Dict[str, Any]] = None
    configuration: Dict[str, Any] = Field(default_factory=dict)
    settings: Dict[str, Any] = Field(default_factory=dict)


class ExecutionResult(BaseModel):
    """Result of tool execution"""
    success: bool
    data: Optional[Any] = None
    error_message: Optional[str] = None
    error_type: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)
    credits_consumed: Decimal = Field(default=Decimal('0'))
    processing_time_ms: Optional[int] = None


# Execution tracking schemas
class Execution(BaseModel):
    """Execution record"""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    workflow_id: str
    instance_id: str
    user_id: str
    
    status: str = Field(default='QUEUED')
    input_data: Optional[Dict[str, Any]] = None
    output_data: Optional[Dict[str, Any]] = None
    
    error_message: Optional[str] = None
    error_type: Optional[str] = None
    retry_count: int = 0
    max_retries: int = 3
    
    credits_consumed: Decimal = Field(default=Decimal('0'))
    processing_time_ms: Optional[int] = None
    execution_metadata: Dict[str, Any] = Field(default_factory=dict)
    
    created_at: datetime = Field(default_factory=datetime.utcnow)
    queued_at: Optional[datetime] = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None


class CreateExecutionRequest(BaseModel):
    """Request to create a new execution"""
    instance_id: str
    input_data: Dict[str, Any]
    max_retries: int = Field(default=3, ge=0, le=10)


class Job(BaseModel):
    """Job for queue processing"""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    job_type: str
    execution_id: str
    
    queue_name: str = 'default'
    priority: int = 0
    status: str = 'QUEUED'
    worker_id: Optional[str] = None
    
    job_data: Dict[str, Any]
    result_data: Optional[Dict[str, Any]] = None
    error_details: Optional[Dict[str, Any]] = None
    retry_count: int = 0
    
    created_at: datetime = Field(default_factory=datetime.utcnow)
    scheduled_at: Optional[datetime] = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None


# Response schemas
class InstanceListResponse(BaseModel):
    """Response for listing instances"""
    instances: List[ToolInstance]
    total: int
    page: int
    size: int


class ExecutionListResponse(BaseModel):
    """Response for listing executions"""
    executions: List[Execution]
    total: int
    page: int
    size: int