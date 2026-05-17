"""Pydantic schemas for the Organization API (TASK-201)."""

from pydantic import BaseModel, Field

# ── Request models ─────────────────────────────────────────────────────────


class OrganizationCreate(BaseModel):
    name: str = Field(..., min_length=1, max_length=255)
    description: str = Field(default="", max_length=2000)
    settings: dict = Field(default_factory=dict)


class OrganizationUpdate(BaseModel):
    name: str | None = Field(default=None, min_length=1, max_length=255)
    description: str | None = Field(default=None, max_length=2000)
    settings: dict | None = Field(default=None)


# ── Response models ────────────────────────────────────────────────────────


class OrganizationResponse(BaseModel):
    id: str
    name: str
    description: str
    owner_user_id: str
    settings: dict
    status: str
    created_at: str
    updated_at: str
    org_role: str | None = Field(
        default=None,
        description="The calling user's role on this org: owner|admin|member",
    )
