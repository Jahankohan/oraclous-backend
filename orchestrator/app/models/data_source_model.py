from sqlalchemy import Column, String, DateTime, Text, Enum
from sqlalchemy.dialects.postgresql import UUID, JSONB
from app.models.base import Base
import uuid
import datetime
from enum import Enum as PyEnum

class DataSourceType(str, PyEnum):
    GOOGLE_DRIVE = "google_drive"
    NOTION = "notion"
    DATABASE = "database"
    FILE_SYSTEM = "file_system"
    WEB_SCRAPER = "web_scraper"

class DataSource(Base):
    __tablename__ = "data_sources"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    name = Column(String, nullable=False)
    type = Column(Enum(DataSourceType), nullable=False)
    config = Column(JSONB, nullable=False)  # Source-specific configuration
    credentials_ref = Column(String)  # Reference to user credentials (user_id)
    data_metadata = Column(JSONB)  # Additional metadata
    owner_id = Column(String, nullable=True)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.datetime.utcnow, onupdate=datetime.datetime.utcnow)
