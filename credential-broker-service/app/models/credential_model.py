from sqlalchemy import Column, String, JSON, DateTime, Enum
from sqlalchemy.dialects.postgresql import UUID as PG_UUID
import uuid
from app.models.enums import CredentialType
from app.models.base_model import BaseModel


class UserCredential(BaseModel):
    __tablename__ = "user_credentials"
    id = Column(PG_UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    name = Column(String, nullable=True) # Custom name for later uses
    provider = Column(String, nullable=False) # provider; if oauth: google, notion, github, etc
    user_id = Column(PG_UUID(as_uuid=True), nullable=False) # The owner of the data
    tool_id = Column(PG_UUID(as_uuid=True), nullable=False) # Tool specific
    encrypted_cred = Column(String, nullable=False) # Encrypted credentials
    cred_type = Column(Enum(CredentialType), nullable=True) # Type of credentials
