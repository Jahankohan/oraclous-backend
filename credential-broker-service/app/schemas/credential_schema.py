from pydantic import BaseModel
from uuid import UUID
from datetime import datetime
from typing import Optional, Dict

class CredentialCreate(BaseModel):
    provider: str
    type: str
    data: str
    metadata: Optional[Dict] = None
    created_by: UUID


class OAuthCredentialCreate(BaseModel):
    provider: str
    data: Dict  # {access_token, refresh_token, expires_at, scopes}
    metadata: Dict = {}
    created_by: str

class CredentialResponse(BaseModel):
    id: UUID
    provider: str
    type: str
    metadata: Optional[Dict[str, str]] = {}
    created_at: datetime
    updated_at: Optional[datetime]

    class Config:
        orm_mode = True
