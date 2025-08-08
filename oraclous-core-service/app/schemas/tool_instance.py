from pydantic import BaseModel, Field
import uuid
from typing import Optional, Dict, Any
from datetime import datetime

from app.schemas.common import InstanceStatus

# Tool Instance (Workflow Execution)
class ToolInstance(BaseModel):
    """
    Tool Instance - Configured instance of a tool for workflow execution
    This represents a specific use of a tool with specific configuration
    """
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    workflow_id: str = Field(..., description="Parent workflow ID")
    tool_definition_id: str = Field(..., description="Reference to tool definition")
    user_id: str = Field(..., description="Owner user ID")
    
    # Instance configuration
    name: str = Field(..., description="Instance name (user-defined)")
    description: Optional[str] = Field(None, description="Instance description")
    configuration: Dict[str, Any] = Field(default_factory=dict)
    settings: Dict[str, Any] = Field(default_factory=dict)
    
    # Runtime state
    status: InstanceStatus = Field(default=InstanceStatus.PENDING)
    credential_id: Optional[str] = Field(None, description="Associated credential ID")
    
    # Metadata
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)


# Execution context for tool runs
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
    metadata: Dict[str, Any] = Field(default_factory=dict)
    credits_consumed: float = 0.0

