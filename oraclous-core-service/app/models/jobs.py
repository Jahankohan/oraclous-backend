from sqlalchemy import Column, String, JSON, ForeignKey, Numeric, DateTime
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import relationship
from app.models.base import Base, TimestampMixin, UUIDMixin

class JobDB(Base, UUIDMixin, TimestampMixin):
    __tablename__ = "jobs"
    
    # REMOVE the custom __init__ method
    
    # Job identification - FIXED: Use UUID type for foreign key
    job_type = Column(String(100), nullable=False)  # 'tool_execution', 'workflow_execution', etc.
    execution_id = Column(UUID(as_uuid=True), ForeignKey('executions.id'), nullable=False)
    
    # Job queue information
    queue_name = Column(String(100), default='default')
    priority = Column(Numeric(3, 0), default=0)  # Higher number = higher priority
    
    # Job status
    status = Column(String(50), nullable=False, default='QUEUED', index=True)
    worker_id = Column(String(255))  # ID of worker processing this job
    
    # Job data
    job_data = Column(JSON, nullable=False)  # Input data for job processing
    result_data = Column(JSON)               # Job results
    
    # Error handling
    error_details = Column(JSON)
    retry_count = Column(Numeric(3, 0), default=0)
    
    # Timing
    scheduled_at = Column(DateTime(timezone=True), nullable=True)
    started_at = Column(DateTime(timezone=True), nullable=True)
    completed_at = Column(DateTime(timezone=True), nullable=True)
    
    # Relationships
    execution = relationship("ExecutionDB")
    
    def __repr__(self):
        return f"<Job {self.job_type} - {self.status}>"
