import uuid
from sqlalchemy import Column, String, Text, JSON, ForeignKey, Numeric, DateTime
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from app.models.base import Base, TimestampMixin, UUIDMixin

class ExecutionDB(Base, UUIDMixin, TimestampMixin):
    __tablename__ = "executions"
    
    def __init__(self, **kwargs):
        if 'id' not in kwargs:
            kwargs['id'] = str(uuid.uuid4())
        super().__init__(**kwargs)
    
    # Execution identification
    workflow_id = Column(UUID, nullable=False, index=True)
    instance_id = Column(String, ForeignKey('tool_instances.id'), nullable=False)
    user_id = Column(UUID, nullable=False, index=True)
    
    # Execution details
    status = Column(String(50), nullable=False, default='QUEUED', index=True)
    input_data = Column(JSON)
    output_data = Column(JSON)
    
    # Error handling
    error_message = Column(Text)
    error_type = Column(String(100))
    retry_count = Column(Numeric(3, 0), default=0)
    max_retries = Column(Numeric(3, 0), default=3)
    
    # Timing
    queued_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    started_at = Column(DateTime(timezone=True), nullable=True)
    completed_at = Column(DateTime(timezone=True), nullable=True)
    
    # Resource usage
    credits_consumed = Column(Numeric(10, 4), default=0)
    processing_time_ms = Column(Numeric(10, 0))
    
    # Metadata
    execution_metadata = Column(JSON, default=dict)
    
    # Relationships
    instance = relationship("ToolInstanceDB")
    
    def __repr__(self):
        return f"<Execution {self.id} - {self.status}>"
