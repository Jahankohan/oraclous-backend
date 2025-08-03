from pydantic import BaseModel
from uuid import UUID
from datetime import datetime
from typing import Optional, Dict

class CredentialCreate(BaseModel):
    provider: str
    type: str
    data: Dict  # Changed from str to Dict to match OAuth data
    cred_metadata: Optional[Dict] = None


class OAuthCredentialCreate(BaseModel):
    provider: str
    data: Dict  # {access_token, refresh_token, expires_at, scopes}
    cred_metadata: Dict = {}
    created_by: UUID  # Changed from str to UUID for consistency

class CredentialResponse(BaseModel):
    id: UUID
    provider: str
    type: str
    cred_metadata: Optional[Dict] = {}
    created_at: datetime
    updated_at: Optional[datetime]

    class Config:
        orm_mode = True
        # Map the model field name to schema field name
