import uuid
from sqlalchemy import Column, String, Text, JSON, ForeignKey, Numeric
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import relationship
from app.models.base import Base, TimestampMixin, UUIDMixin
from app.models.workflow import WorkflowDB


class ToolInstanceDB(Base, UUIDMixin, TimestampMixin):
    __tablename__ = "tool_instances"
    
    # Override id to use UUID generation
    def __init__(self, **kwargs):
        if 'id' not in kwargs:
            kwargs['id'] = str(uuid.uuid4())
        super().__init__(**kwargs)
    
    # Basic instance information
    name = Column(String(255), nullable=False)
    description = Column(Text)
    
    # Relationships
    workflow_id = Column(UUID, ForeignKey("workflows.id"), nullable=False, index=True)
    workflow = relationship("WorkflowDB", back_populates="instances")
    user_id = Column(UUID, nullable=False, index=True)  # User who owns this instance
    
    # Configuration
    configuration = Column(JSON, default=dict)  # Tool-specific configuration
    settings = Column(JSON, default=dict)      # Runtime settings
    
    # Credential Management
    credential_mappings = Column(JSON, default=dict)  # Maps credential_type -> credential_id
    required_credentials = Column(JSON, default=list) # List of required credential types
    
    # Status and State
    status = Column(String(50), nullable=False, default='PENDING', index=True)
    
    # Execution History
    last_execution_id = Column(UUID, nullable=True)
    execution_count = Column(Numeric(10, 0), default=0)
    total_credits_consumed = Column(Numeric(10, 4), default=0)
    
    # Relationships
    tool_definition = relationship("ToolDefinitionDB", back_populates="instances")
    tool_definition_id = Column(UUID, ForeignKey("tool_definitions.id"), nullable=False, index=True)
    
    def __repr__(self):
        return f"<ToolInstance {self.name} ({self.id})>"
