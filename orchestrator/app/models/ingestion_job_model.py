from sqlalchemy import Column, String, DateTime, Text, Enum, Integer, ForeignKey
from sqlalchemy.dialects.postgresql import UUID, JSONB
from sqlalchemy.orm import relationship
from app.models.base import Base
import uuid
import datetime
from enum import Enum as PyEnum

class IngestionStatus(str, PyEnum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

class IngestionJob(Base):
    __tablename__ = "ingestion_jobs"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    source_id = Column(UUID(as_uuid=True), ForeignKey("data_sources.id"), nullable=False)
    status = Column(Enum(IngestionStatus), default=IngestionStatus.PENDING)
    config = Column(JSONB)  # Job-specific configuration
    documents_count = Column(Integer, default=0)
    error_message = Column(Text)
    job_metadata = Column(JSONB)  # Job metadata and results
    owner_id = Column(String, nullable=True)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)
    started_at = Column(DateTime)
    completed_at = Column(DateTime)

    # Relationship
    source = relationship("DataSource")
