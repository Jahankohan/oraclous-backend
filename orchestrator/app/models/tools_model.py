from sqlalchemy import Column, String, DateTime, Text, ARRAY
from sqlalchemy.dialects.postgresql import UUID, JSONB
from app.models.base import Base
import uuid
import datetime

class Tool(Base):
    __tablename__ = "tools"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    name = Column(String, unique=True, nullable=False)
    description = Column(Text)
    url = Column(String, nullable=False)
    input_schema = Column(JSONB)
    output_example = Column(JSONB)
    category = Column(String)
    tags = Column(ARRAY(String))
    created_at = Column(DateTime, default=datetime.datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.datetime.utcnow, onupdate=datetime.datetime.utcnow)
