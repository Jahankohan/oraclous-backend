# app/models/task_model.py
import uuid
from sqlalchemy import Column, String, ForeignKey, DateTime, Enum, JSON
from sqlalchemy.dialects.postgresql import UUID, JSONB
from sqlalchemy.orm import relationship
from datetime import datetime
from enum import Enum as PyEnum
from app.models.base import Base


class TaskStatus(PyEnum):
    CREATED = "created"
    RUNNING = "running"
    PAUSED = "paused"
    FAILED = "failed"
    COMPLETED = "completed"


class Task(Base):
    __tablename__ = "tasks"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    workflow_id = Column(UUID(as_uuid=True), ForeignKey("workflows.id"), nullable=False)
    status = Column(Enum(TaskStatus), default=TaskStatus.CREATED)
    inputs = Column(JSONB, nullable=True)
    outputs = Column(JSONB, nullable=True)
    logs = Column(JSONB, nullable=True)
    token_usage = Column(String, nullable=True)
    metadata = Column(JSONB, nullable=True)
    started_at = Column(DateTime, default=datetime.utcnow)
    finished_at = Column(DateTime, nullable=True)
    owner_id = Column(String, nullable=True)

    workflow = relationship("Workflow")
