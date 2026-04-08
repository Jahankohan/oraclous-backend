"""Pydantic schemas for Agent Service Account API."""

from pydantic import BaseModel, Field

# ── Request models ─────────────────────────────────────────────────────────


class CreateServiceAccountRequest(BaseModel):
    name: str = Field(..., min_length=1, max_length=128)
    description: str = Field(default="", max_length=512)
    level: str = Field(default="reader", pattern="^(reader|writer|admin)$")
    expires_at: str | None = Field(
        default=None,
        description="ISO-8601 datetime for grant expiry (null = no expiry)",
    )


class UpdateServiceAccountRequest(BaseModel):
    name: str | None = Field(default=None, min_length=1, max_length=128)
    description: str | None = Field(default=None, max_length=512)


class AddGraphGrantRequest(BaseModel):
    graph_id: str = Field(..., min_length=1)
    level: str = Field(..., pattern="^(reader|writer|admin)$")
    expires_at: str | None = Field(
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
    last_used_at: str | None = None


class ServiceAccountCreatedResponse(ServiceAccountResponse):
    """Extended response returned ONLY at creation — includes raw api_key."""

    api_key: str = Field(
        ..., description="Raw API key — shown once, never stored in plaintext"
    )


class ServiceAccountRotatedResponse(BaseModel):
    service_account_id: str
    key_prefix: str
    api_key: str = Field(..., description="New raw API key — shown once")
    rotated_at: str


class GraphGrantResponse(BaseModel):
    graph_id: str
    graph_name: str | None = None
    level: str
    source: str | None = None
    granted_by: str | None = None
    granted_at: str | None = None
    expires_at: str | None = None
