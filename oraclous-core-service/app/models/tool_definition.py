from sqlalchemy import Column, String, Text
from sqlalchemy.orm import relationship
from sqlalchemy.dialects.postgresql import ARRAY, JSONB, UUID
from app.models.base import Base, TimestampMixin, UUIDMixin


class ToolDefinitionDB(Base, UUIDMixin, TimestampMixin):
    __tablename__ = "tool_definitions"
    
    # REMOVE the custom __init__ method
    
    name = Column(String(255), nullable=False, index=True)
    description = Column(Text)
    version = Column(String(50), default="1.0.0")
    icon = Column(String(255))
    
    # Categorization
    category = Column(String(50), nullable=False, index=True)
    type = Column(String(50), nullable=False, index=True)
    capabilities = Column(JSONB, default=list)  # List of ToolCapability dicts
    tags = Column(ARRAY(String), default=list)
    
    # Schemas (stored as JSON)
    input_schema = Column(JSONB, nullable=False)
    output_schema = Column(JSONB, nullable=False)
    configuration_schema = Column(JSONB)
    
    # Requirements
    credential_requirements = Column(JSONB, default=list)  # List of CredentialRequirement dicts
    dependencies = Column(ARRAY(String), default=list)

    instances = relationship("ToolInstanceDB", back_populates="tool_definition")
    
    # Metadata
    author = Column(String(255))
    documentation_url = Column(String(500))
