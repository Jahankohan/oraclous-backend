from sqlalchemy import Column, String, DateTime, Text, ARRAY, ForeignKey
from sqlalchemy.dialects.postgresql import UUID, JSONB
from app.models.base import Base
from sqlalchemy.orm import relationship
import uuid
import datetime

class Prompt(Base):
    __tablename__ = "prompts"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    mcp_server_id = Column(UUID(as_uuid=True), ForeignKey("mcp_servers.id"))
    name = Column(String, unique=True, nullable=False)
    description = Column(Text)
    input_schema = Column(JSONB)
    output_example = Column(JSONB)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.datetime.utcnow, onupdate=datetime.datetime.utcnow)
