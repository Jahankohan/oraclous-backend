import uuid

from sqlalchemy import (
    JSON,
    Column,
    DateTime,
    String,
    Text,
)
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.sql import func

from app.core.database import Base


class Organization(Base):
    """First-class organization entity (TASK-201).

    Backed by both PostgreSQL (this table, source of truth for metadata)
    and a synced Neo4j ``:Organization`` node used for ReBAC ownership
    edges. ``org_id`` everywhere == ``str(Organization.id)``.
    """

    __tablename__ = "organizations"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    name = Column(String(255), nullable=False)
    description = Column(Text, nullable=True)
    owner_user_id = Column(UUID(as_uuid=True), nullable=False)
    settings = Column(JSON, nullable=False, default=dict)
    status = Column(String(50), default="active")
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )
