from sqlalchemy import Column, String, DateTime, JSON
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.sql import func
import uuid
from app.models.base_model import Base

class Credential(Base):
    __tablename__ = "credentials"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    provider = Column(String, nullable=False)
    type = Column(String, nullable=False)       # oauth, api_key, service_account
    encrypted_data = Column(String, nullable=False)
    cred_metadata = Column(JSON, nullable=True)  # {"connection_name": "Postgres Prod"}
    created_by = Column(UUID(as_uuid=True), nullable=False, index=True) # User reference
    created_at = Column(DateTime, server_default=func.now())
    updated_at = Column(DateTime, onupdate=func.now())
