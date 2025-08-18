from pydantic import BaseModel, Field, validator
import uuid
from typing import Optional, Dict, Any, List
from datetime import datetime
from decimal import Decimal
from enum import Enum


# Enums for workflow management
class WorkflowStatus(str, Enum):
    DRAFT = "DRAFT"
    ACTIVE = "ACTIVE"
    PAUSED = "PAUSED"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    ARCHIVED = "ARCHIVED"


class WorkflowExecutionStatus(str, Enum):
    QUEUED = "QUEUED"
    RUNNING = "RUNNING"
    SUCCESS = "SUCCESS"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"
    PAUSED = "PAUSED"


class TriggerType(str, Enum):
    MANUAL = "MANUAL"
    SCHEDULED = "SCHEDULED"
    API = "API"
    WEBHOOK = "WEBHOOK"


class PermissionType(str, Enum):
    VIEW = "VIEW"
    EXECUTE = "EXECUTE"
    EDIT = "EDIT"
    ADMIN = "ADMIN"


# Core workflow schemas
class WorkflowNode(BaseModel):
    """Represents a single node in the workflow"""
    id: str = Field(..., description="Unique node identifier")
    type: str = Field(..., description="Node type (tool, condition, etc.)")
    tool_definition_id: Optional[str] = Field(None, description="Tool definition ID if this is a tool node")
    instance_id: Optional[str] = Field(None, description="Tool instance ID if configured")
    
    # Node configuration
    name: str = Field(..., description="Human-readable node name")
    description: Optional[str] = Field(None, description="Node description")
    configuration: Dict[str, Any] = Field(default_factory=dict, description="Node-specific configuration")
    
    # Position for UI
    position: Dict[str, float] = Field(default_factory=dict, description="Node position in workflow editor")
    
    # Execution settings
    is_required: bool = Field(default=True, description="Whether this node is required for workflow success")
    timeout_seconds: Optional[int] = Field(None, description="Execution timeout")
    retry_count: int = Field(default=0, description="Number of retries on failure")


class WorkflowEdge(BaseModel):
    """Represents a connection between workflow nodes"""
    id: str = Field(..., description="Unique edge identifier")
    source_node_id: str = Field(..., description="Source node ID")
    target_node_id: str = Field(..., description="Target node ID")
    
    # Edge properties
    condition: Optional[Dict[str, Any]] = Field(None, description="Conditional logic for this edge")
    data_mapping: Optional[Dict[str, str]] = Field(None, description="Maps output fields to input fields")
    
    # Edge metadata
    label: Optional[str] = Field(None, description="Edge label for UI")
    is_conditional: bool = Field(default=False, description="Whether this edge has conditions")


class Workflow(BaseModel):
    """Main workflow schema"""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    name: str = Field(..., description="Workflow name")
    description: Optional[str] = Field(None, description="Workflow description")
    owner_id: str = Field(..., description="Workflow owner user ID")
    
    # Workflow structure
    nodes: List[WorkflowNode] = Field(default_factory=list, description="Workflow nodes")
    edges: List[WorkflowEdge] = Field(default_factory=list, description="Workflow connections")
    
    # LangGraph integration
    chat_history: List[Dict[str, Any]] = Field(default_factory=list, description="Creation conversation")
    generation_prompt: Optional[str] = Field(None, description="Original user prompt")
    generation_metadata: Dict[str, Any] = Field(default_factory=dict, description="Generation metadata")
    
    # Configuration
    settings: Dict[str, Any] = Field(default_factory=dict, description="Workflow settings")
    variables: Dict[str, Any] = Field(default_factory=dict, description="Workflow variables")
    
    # State
    status: WorkflowStatus = Field(default=WorkflowStatus.DRAFT)
    version: str = Field(default="1.0.0")
    is_template: bool = Field(default=False)
    
    # Execution tracking
    last_execution_id: Optional[str] = None
    total_executions: int = 0
    successful_executions: int = 0
    
    # Resource estimates
    estimated_credits_per_run: Decimal = Field(default=Decimal('0'))
    total_credits_consumed: Decimal = Field(default=Decimal('0'))
    
    # Metadata
    tags: List[str] = Field(default_factory=list)
    category: Optional[str] = None
    is_public: bool = Field(default=False)
    
    # Timestamps
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)

    @validator('name')
    def validate_name(cls, v):
        if not v or len(v.strip()) == 0:
            raise ValueError('Workflow name cannot be empty')
        if len(v) > 255:
            raise ValueError('Workflow name too long (max 255 characters)')
        return v.strip()


class CreateWorkflowRequest(BaseModel):
    """Request to create a new workflow"""
    name: str = Field(..., description="Workflow name")
    description: Optional[str] = Field(None, description="Workflow description")
    
    # Creation method
    generation_prompt: Optional[str] = Field(None, description="Natural language prompt for LangGraph")
    template_id: Optional[str] = Field(None, description="Template ID if creating from template")
    
    # Initial structure (if not using prompt or template)
    nodes: Optional[List[WorkflowNode]] = Field(None, description="Initial nodes")
    edges: Optional[List[WorkflowEdge]] = Field(None, description="Initial edges")
    
    # Configuration
    settings: Dict[str, Any] = Field(default_factory=dict)
    variables: Dict[str, Any] = Field(default_factory=dict)
    tags: List[str] = Field(default_factory=list)
    category: Optional[str] = None

    @validator('name')
    def validate_name(cls, v):
        if not v or len(v.strip()) == 0:
            raise ValueError('Workflow name cannot be empty')
        return v.strip()


class UpdateWorkflowRequest(BaseModel):
    """Request to update workflow"""
    name: Optional[str] = None
    description: Optional[str] = None
    nodes: Optional[List[WorkflowNode]] = None
    edges: Optional[List[WorkflowEdge]] = None
    settings: Optional[Dict[str, Any]] = None
    variables: Optional[Dict[str, Any]] = None
    tags: Optional[List[str]] = None
    category: Optional[str] = None


class WorkflowExecution(BaseModel):
    """Workflow execution record"""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    workflow_id: str = Field(..., description="Workflow ID")
    user_id: str = Field(..., description="User who triggered execution")
    
    # Execution details
    status: WorkflowExecutionStatus = Field(default=WorkflowExecutionStatus.QUEUED)
    trigger_type: TriggerType = Field(default=TriggerType.MANUAL)
    
    # Input/Output
    input_parameters: Dict[str, Any] = Field(default_factory=dict)
    output_data: Optional[Dict[str, Any]] = None
    
    # Progress tracking
    total_steps: int = 0
    completed_steps: int = 0
    failed_steps: int = 0
    
    # Error handling
    error_message: Optional[str] = None
    error_type: Optional[str] = None
    failed_node_id: Optional[str] = None
    
    # Timing
    queued_at: datetime = Field(default_factory=datetime.utcnow)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    
    # Resource usage
    credits_consumed: Decimal = Field(default=Decimal('0'))
    processing_time_ms: Optional[int] = None
    
    # Control features
    can_pause: bool = True
    can_resume: bool = True
    paused_at: Optional[datetime] = None
    resumed_at: Optional[datetime] = None
    
    # Metadata
    execution_metadata: Dict[str, Any] = Field(default_factory=dict)
    
    # Timestamps
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)


class CreateWorkflowExecutionRequest(BaseModel):
    """Request to execute a workflow"""
    workflow_id: str = Field(..., description="Workflow to execute")
    input_parameters: Dict[str, Any] = Field(default_factory=dict, description="Input parameters")
    trigger_type: TriggerType = Field(default=TriggerType.MANUAL)


class WorkflowTemplate(BaseModel):
    """Workflow template schema"""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    name: str = Field(..., description="Template name")
    description: Optional[str] = Field(None, description="Template description")
    category: Optional[str] = None
    
    # Template structure
    template_nodes: List[WorkflowNode] = Field(..., description="Template nodes")
    template_edges: List[WorkflowEdge] = Field(..., description="Template connections")
    
    # Configuration
    parameters: Dict[str, Any] = Field(default_factory=dict, description="Template parameters")
    required_tools: List[str] = Field(default_factory=list, description="Required tool definition IDs")
    
    # Metadata
    author_id: Optional[str] = None
    tags: List[str] = Field(default_factory=list)
    difficulty_level: str = Field(default='BEGINNER')
    estimated_time_minutes: Optional[int] = None
    estimated_credits: Optional[Decimal] = None
    
    # Usage tracking
    usage_count: int = 0
    average_rating: Decimal = Field(default=Decimal('0'))
    
    # Publication
    is_published: bool = False
    is_featured: bool = False
    
    # Timestamps
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)


class WorkflowShare(BaseModel):
    """Workflow sharing schema"""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    workflow_id: str = Field(..., description="Shared workflow ID")
    owner_id: str = Field(..., description="Workflow owner ID")
    shared_with_id: Optional[str] = Field(None, description="User ID (null for public)")
    
    # Permissions
    permission_type: PermissionType = Field(default=PermissionType.VIEW)
    can_reshare: bool = False
    
    # Sharing metadata
    share_token: Optional[str] = None
    expires_at: Optional[datetime] = None
    
    # Usage tracking
    access_count: int = 0
    last_accessed_at: Optional[datetime] = None
    
    # Timestamps
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)


# Response schemas
class WorkflowListResponse(BaseModel):
    """Response for listing workflows"""
    workflows: List[Workflow]
    total: int
    page: int
    size: int


class WorkflowExecutionListResponse(BaseModel):
    """Response for listing executions"""
    executions: List[WorkflowExecution]
    total: int
    page: int
    size: int


class WorkflowValidationResult(BaseModel):
    """Result of workflow validation"""
    is_valid: bool
    errors: List[str] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)
    missing_credentials: List[str] = Field(default_factory=list)
    estimated_credits: Decimal = Field(default=Decimal('0'))


class GenerateWorkflowRequest(BaseModel):
    """Request to generate workflow from prompt"""
    prompt: str = Field(..., description="Natural language description of desired workflow")
    context: Dict[str, Any] = Field(default_factory=dict, description="Additional context")
    preferred_tools: List[str] = Field(default_factory=list, description="Preferred tool definition IDs")
    
    @validator('prompt')
    def validate_prompt(cls, v):
        if not v or len(v.strip()) == 0:
            raise ValueError('Prompt cannot be empty')
        if len(v) > 5000:
            raise ValueError('Prompt too long (max 5000 characters)')
        return v.strip()


class GenerateWorkflowResponse(BaseModel):
    """Response from workflow generation"""
    workflow: Workflow
    confidence_score: float = Field(ge=0.0, le=1.0, description="Generation confidence")
    suggestions: List[str] = Field(default_factory=list, description="Improvement suggestions")
    alternative_approaches: List[Dict[str, Any]] = Field(default_factory=list, description="Alternative workflow approaches")