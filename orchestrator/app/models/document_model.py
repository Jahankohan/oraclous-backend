from sqlalchemy import Column, String, DateTime, Text, ForeignKey, Integer
from sqlalchemy.dialects.postgresql import UUID, JSONB
from sqlalchemy.orm import relationship
from app.models.base import Base
import uuid
import datetime

class Document(Base):
    __tablename__ = "documents"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    job_id = Column(UUID(as_uuid=True), ForeignKey("ingestion_jobs.id"), nullable=False)
    source_id = Column(UUID(as_uuid=True), ForeignKey("data_sources.id"), nullable=False)
    external_id = Column(String)  # ID from the external source
    title = Column(String)
    content = Column(Text, nullable=False)
    content_type = Column(String)  # text, html, markdown, etc.
    doc_metadata = Column(JSONB)  # Document metadata
    embeddings = Column(JSONB)  # Store embeddings if needed
    content_hash = Column(String)  # For deduplication
    size_bytes = Column(Integer)
    owner_id = Column(String, nullable=True)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.datetime.utcnow, onupdate=datetime.datetime.utcnow)

    # Relationships
    job = relationship("IngestionJob")
    source = relationship("DataSource")
