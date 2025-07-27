from sqlalchemy import Column, String, DateTime, JSON, Enum
from sqlalchemy.dialects.postgresql import UUID
from app.models.base import Base
import enum
import uuid
import datetime


class WorkflowStatus(str, enum.Enum):
    draft = "draft"
    ready = "ready"
    error = "error"

class Workflow(Base):
    __tablename__ = "workflows"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    name = Column(String, nullable=False)
    description = Column(String)
    owner_id = Column(String, nullable=True)
    nodes = Column(JSON, nullable=False)
    edges = Column(JSON, nullable=False)
    status = Column(Enum(WorkflowStatus), default=WorkflowStatus.draft)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.datetime.utcnow, onupdate=datetime.datetime.utcnow)
