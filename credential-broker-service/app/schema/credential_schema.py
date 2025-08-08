from pydantic import BaseModel
from typing import Optional, Dict, Literal
from uuid import UUID

class RequestCredentials(BaseModel):
    user_id: UUID
    tool_id: Optional[UUID] = None

class RequestCredentialsResponse(BaseModel):
    id: UUID
    name: Optional[str] = None
    provider: str
    user_id: UUID
    tool_id: UUID
    cred_type: Literal["oauth", "api_key", "raw"]
    credential: Dict

class CreateCredential(BaseModel):
    tool_id: UUID
    user_id: UUID
    name: Optional[str] = None
    provider: str
    cred_type: Literal["oauth", "api_key", "raw"]
    credential: Dict

class CredentialOut(BaseModel):
    id: UUID
    name: Optional[str]
    provider: str
    user_id: UUID
    tool_id: UUID
    cred_type: Literal["oauth", "api_key", "raw"]

class CredentialsUpdate(BaseModel):
    id: UUID
    name: Optional[str] = None
    provider: str
    user_id: UUID
    tool_id: UUID
    cred_type: Literal["oauth", "api_key", "raw"]
    credential: Dict
