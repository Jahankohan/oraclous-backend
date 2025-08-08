from pydantic import BaseModel
from typing import Optional, Dict, Literal, List
from uuid import UUID
from datetime import datetime

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

class CredentialTokenOut(BaseModel):
    access_token: str
    expires_at: Optional[datetime]
    scopes: List[str]
    provider: str
    user_id: UUID

class CredentialBrokerSuccessResponse(BaseModel):
    success: bool = True
    access_token: str
    expires_at: Optional[datetime]
    scopes: List[str]
    provider: str
    user_id: UUID

class CredentialBrokerErrorResponse(BaseModel):
    success: bool = False
    error_code: Optional[str]
    error_message: Optional[str]
    login_url: Optional[str]
    missing_scopes: Optional[List[str]]

class ProvidersResponse(BaseModel):
    user_id: UUID
    providers: dict

class AvailableDataSourcesResponse(BaseModel):
    user_id: UUID
    available_data_sources: dict

class RuntimeTokenRequest(BaseModel):
    user_id: UUID
    provider: str
    required_scopes: Optional[List[str]] = None