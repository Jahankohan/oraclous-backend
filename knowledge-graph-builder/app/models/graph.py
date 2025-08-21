from sqlalchemy import Column, String, Text, DateTime, Integer, Boolean, JSON
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.sql import func
import uuid
from app.core.database import Base

class KnowledgeGraph(Base):
    """Knowledge graph metadata model"""
    __tablename__ = "knowledge_graphs"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    name = Column(String(255), nullable=False)
    description = Column(Text)
    user_id = Column(UUID(as_uuid=True), nullable=False)
    schema_config = Column(JSON)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())
    node_count = Column(Integer, default=0)
    relationship_count = Column(Integer, default=0)
    status = Column(String(50), default="active")

class IngestionJob(Base):
    """Data ingestion job tracking"""
    __tablename__ = "ingestion_jobs"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    graph_id = Column(UUID(as_uuid=True), nullable=False)
    source_type = Column(String(50))  # 'text', 'pdf', 'url', 'api'
    source_content = Column(Text)
    status = Column(String(50), default="pending")
    progress = Column(Integer, default=0)
    error_message = Column(Text)
    extracted_entities = Column(Integer, default=0)
    extracted_relationships = Column(Integer, default=0)
    credits_consumed = Column(String(20), default="0")
    started_at = Column(DateTime(timezone=True))
    completed_at = Column(DateTime(timezone=True))
    created_at = Column(DateTime(timezone=True), server_default=func.now())
