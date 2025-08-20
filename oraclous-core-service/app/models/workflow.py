# app/models/workflow.py - FIXED VERSION
import uuid
from sqlalchemy import Column, String, Text, JSON, Numeric, Boolean, ForeignKey, DateTime
from sqlalchemy.dialects.postgresql import UUID, ARRAY
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from app.models.base import Base, TimestampMixin, UUIDMixin


class WorkflowDB(Base, UUIDMixin, TimestampMixin):
    __tablename__ = "workflows"
    
    # REMOVE the custom __init__ method - let UUIDMixin handle UUID generation
    
    # Basic workflow information
    name = Column(String(255), nullable=False)
    description = Column(Text)
    owner_id = Column(UUID(as_uuid=True), nullable=False, index=True)  # FIXED: Use UUID type
    
    # Workflow structure
    nodes = Column(JSON, nullable=False, default=list)  # List of workflow nodes
    edges = Column(JSON, nullable=False, default=list)  # Connections between nodes
    
    # LangGraph integration
    chat_history = Column(JSON, default=list)  # Conversation that created this workflow
    generation_prompt = Column(Text)           # Original user prompt
    generation_metadata = Column(JSON, default=dict)  # LangGraph generation details
    
    # Workflow configuration
    settings = Column(JSON, default=dict)      # Workflow-level settings
    variables = Column(JSON, default=dict)     # Workflow variables/parameters
    
    # State and status
    status = Column(String(50), nullable=False, default='DRAFT', index=True)
    version = Column(String(50), default="1.0.0")
    is_template = Column(Boolean, default=False)
    
    # Execution tracking
    last_execution_id = Column(UUID(as_uuid=True), nullable=True)  # FIXED: Use UUID type
    total_executions = Column(Numeric(10, 0), default=0)
    successful_executions = Column(Numeric(10, 0), default=0)
    
    # Resource estimates
    estimated_credits_per_run = Column(Numeric(10, 4), default=0)
    total_credits_consumed = Column(Numeric(12, 4), default=0)
    
    # Metadata and organization
    tags = Column(ARRAY(String), default=list)
    category = Column(String(100))
    is_public = Column(Boolean, default=False)
    
    # Relationships
    instances = relationship("ToolInstanceDB", back_populates="workflow", cascade="all, delete-orphan")
    executions = relationship("WorkflowExecutionDB", back_populates="workflow", cascade="all, delete-orphan")
    
    def __repr__(self):
        return f"<Workflow {self.name} ({self.id})>"


class WorkflowExecutionDB(Base, UUIDMixin, TimestampMixin):
    __tablename__ = "workflow_executions"
    
    # REMOVE the custom __init__ method - let UUIDMixin handle UUID generation
    
    # Execution identification - FIXED: Use UUID type for foreign key
    workflow_id = Column(UUID(as_uuid=True), ForeignKey("workflows.id"), nullable=False, index=True)
    user_id = Column(UUID(as_uuid=True), nullable=False, index=True)
    
    # Execution details
    status = Column(String(50), nullable=False, default='QUEUED', index=True)
    trigger_type = Column(String(50), default='MANUAL')  # MANUAL, SCHEDULED, API, etc.
    
    # Input/Output
    input_parameters = Column(JSON, default=dict)
    output_data = Column(JSON)
    
    # Progress tracking
    total_steps = Column(Numeric(5, 0), default=0)
    completed_steps = Column(Numeric(5, 0), default=0)
    failed_steps = Column(Numeric(5, 0), default=0)
    
    # Error handling
    error_message = Column(Text)
    error_type = Column(String(100))
    failed_node_id = Column(String(255))  # ID of node where execution failed
    
    # Timing
    queued_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    started_at = Column(DateTime(timezone=True), nullable=True)
    completed_at = Column(DateTime(timezone=True), nullable=True)
    
    # Resource usage
    credits_consumed = Column(Numeric(10, 4), default=0)
    processing_time_ms = Column(Numeric(12, 0))
    
    # Control features
    can_pause = Column(Boolean, default=True)
    can_resume = Column(Boolean, default=True)
    paused_at = Column(DateTime(timezone=True), nullable=True)
    resumed_at = Column(DateTime(timezone=True), nullable=True)
    
    # Metadata
    execution_metadata = Column(JSON, default=dict)
    
    # Relationships
    workflow = relationship("WorkflowDB", back_populates="executions")
    
    def __repr__(self):
        return f"<WorkflowExecution {self.id} - {self.status}>"


class WorkflowTemplateDB(Base, UUIDMixin, TimestampMixin):
    __tablename__ = "workflow_templates"
    
    # REMOVE the custom __init__ method
    
    # Template information
    name = Column(String(255), nullable=False)
    description = Column(Text)
    category = Column(String(100))
    
    # Template structure
    template_nodes = Column(JSON, nullable=False)
    template_edges = Column(JSON, nullable=False)
    
    # Configuration
    parameters = Column(JSON, default=dict)  # Template parameters that can be customized
    required_tools = Column(ARRAY(String), default=list)  # Required tool definition IDs
    
    # Metadata
    author_id = Column(UUID(as_uuid=True))  # FIXED: Use UUID type
    tags = Column(ARRAY(String), default=list)
    difficulty_level = Column(String(20), default='BEGINNER')  # BEGINNER, INTERMEDIATE, ADVANCED
    estimated_time_minutes = Column(Numeric(5, 0))
    estimated_credits = Column(Numeric(10, 4))
    
    # Usage tracking
    usage_count = Column(Numeric(10, 0), default=0)
    average_rating = Column(Numeric(3, 2), default=0)
    
    # Publication
    is_published = Column(Boolean, default=False)
    is_featured = Column(Boolean, default=False)
    
    def __repr__(self):
        return f"<WorkflowTemplate {self.name} ({self.id})>"


class WorkflowShareDB(Base, UUIDMixin, TimestampMixin):
    __tablename__ = "workflow_shares"
    
    # REMOVE the custom __init__ method
    
    # Sharing details - FIXED: Use UUID types
    workflow_id = Column(UUID(as_uuid=True), nullable=False, index=True)
    owner_id = Column(UUID(as_uuid=True), nullable=False)      # Workflow owner
    shared_with_id = Column(UUID(as_uuid=True), nullable=True) # Specific user (null for public)
    
    # Permissions
    permission_type = Column(String(20), nullable=False, default='VIEW')  # VIEW, EXECUTE, EDIT, ADMIN
    can_reshare = Column(Boolean, default=False)
    
    # Sharing metadata
    share_token = Column(String(255), unique=True)  # For public sharing
    expires_at = Column(DateTime(timezone=True), nullable=True)
    
    # Usage tracking
    access_count = Column(Numeric(10, 0), default=0)
    last_accessed_at = Column(DateTime(timezone=True), nullable=True)
    
    def __repr__(self):
        return f"<WorkflowShare {self.workflow_id} -> {self.shared_with_id or 'PUBLIC'}>"
