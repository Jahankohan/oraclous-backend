"""Pydantic schemas for Agent Service Account API."""
from typing import Optional

from pydantic import BaseModel, Field


# ── Request models ─────────────────────────────────────────────────────────

class CreateServiceAccountRequest(BaseModel):
    name: str = Field(..., min_length=1, max_length=128)
    description: str = Field(default="", max_length=512)
    level: str = Field(default="reader", pattern="^(reader|writer|admin)$")
    expires_at: Optional[str] = Field(
        default=None, description="ISO-8601 datetime for grant expiry (null = no expiry)"
    )


class UpdateServiceAccountRequest(BaseModel):
    name: Optional[str] = Field(default=None, min_length=1, max_length=128)
    description: Optional[str] = Field(default=None, max_length=512)


class AddGraphGrantRequest(BaseModel):
    graph_id: str = Field(..., min_length=1)
    level: str = Field(..., pattern="^(reader|writer|admin)$")
    expires_at: Optional[str] = Field(
        default=None, description="ISO-8601 datetime — TTL ≤ 90 days recommended"
    )


# ── Response models ────────────────────────────────────────────────────────

class ServiceAccountResponse(BaseModel):
    service_account_id: str
    name: str
    description: str
    home_graph_id: str
    tenant_id: str
    status: str
    key_prefix: str
    created_at: str
    last_used_at: Optional[str] = None


class ServiceAccountCreatedResponse(ServiceAccountResponse):
    """Extended response returned ONLY at creation — includes raw api_key."""
    api_key: str = Field(..., description="Raw API key — shown once, never stored in plaintext")


class ServiceAccountRotatedResponse(BaseModel):
    service_account_id: str
    key_prefix: str
    api_key: str = Field(..., description="New raw API key — shown once")
    rotated_at: str


class GraphGrantResponse(BaseModel):
    graph_id: str
    graph_name: Optional[str] = None
    level: str
    source: Optional[str] = None
    granted_by: Optional[str] = None
    granted_at: Optional[str] = None
    expires_at: Optional[str] = None
