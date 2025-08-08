import uuid
from sqlalchemy import Column, String, Text, JSON
from sqlalchemy.dialects.postgresql import ARRAY
from app.models.base import Base, TimestampMixin, UUIDMixin


class ToolDefinitionDB(Base, UUIDMixin, TimestampMixin):
    __tablename__ = "tool_definitions"
    
    # Override id to use UUID generation
    def __init__(self, **kwargs):
        if 'id' not in kwargs:
            kwargs['id'] = str(uuid.uuid4())
        super().__init__(**kwargs)
    
    name = Column(String(255), nullable=False, index=True)
    description = Column(Text)
    version = Column(String(50), default="1.0.0")
    icon = Column(String(255))
    
    # Categorization
    category = Column(String(50), nullable=False, index=True)
    type = Column(String(50), nullable=False, index=True)
    capabilities = Column(JSON, default=list)  # List of ToolCapability dicts
    tags = Column(ARRAY(String), default=list)
    
    # Schemas (stored as JSON)
    input_schema = Column(JSON, nullable=False)
    output_schema = Column(JSON, nullable=False)
    configuration_schema = Column(JSON)
    
    # Requirements
    credential_requirements = Column(JSON, default=list)  # List of CredentialRequirement dicts
    dependencies = Column(ARRAY(String), default=list)
    
    # Metadata
    author = Column(String(255))
    documentation_url = Column(String(500))
