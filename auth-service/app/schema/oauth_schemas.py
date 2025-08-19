from pydantic import BaseModel
from typing import Optional, List
from datetime import datetime

class TokenRefreshRequest(BaseModel):
    user_id: str
    provider: str
    state: Optional[str] = "/"

class TokenRefreshResponse(BaseModel):
    success: bool
    access_token: Optional[str] = None
    login_url: Optional[str] = None
    expires_at: Optional[datetime] = None
    error: Optional[str] = None

class ScopeValidationRequest(BaseModel):
    user_id: str
    provider: str
    required_scopes: List[str]


class ScopeValidationResponse(BaseModel):
    valid: bool
    missing_scopes: List[str]
    current_scopes: List[str]
    token_expired: bool
    needs_reauth: bool
    login_url: Optional[str] = None
    error: Optional[str] = None

class UserTokensResponse(BaseModel):
    user_id: str
    providers: List[dict]  # List of {provider: str, scopes: List[str], expires_at: datetime, has_refresh_token: bool}

class RuntimeTokenResponse(BaseModel):
    user_id: str
    provider: str
    access_token: str
    expires_at: Optional[datetime]
    scopes: List[str]
    refresh_token: Optional[str] = None


class EnsureAccessRequest(BaseModel):
    user_id: str
    required_scopes: Optional[List[str]] = []
    state: Optional[str] = "/"


class EnsureAccessResponse(BaseModel):
    action: str  # "ok" or "reauthenticate"
    login_url: Optional[str] = None
    current_scopes: List[str] = []
    missing_scopes: List[str] = []
    token: Optional[dict] = None
    error: Optional[str] = None