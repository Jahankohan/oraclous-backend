from sqlalchemy import Column, String, DateTime, Text, ARRAY
from sqlalchemy.dialects.postgresql import UUID, JSONB
from app.models.base import Base
from sqlalchemy.orm import relationship
import uuid
import datetime

class MCPServer(Base):
    __tablename__ = "mcp_servers"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    name = Column(String, unique=True, nullable=False)
    description = Column(Text)
    url = Column(String, nullable=False)
    token = Column(Text)
    category = Column(String)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.datetime.utcnow, onupdate=datetime.datetime.utcnow)

    tools = relationship("Tool", back_populates="mcp_server", cascade="all, delete-orphan")

