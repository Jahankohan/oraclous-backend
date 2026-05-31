"""Pydantic schemas for the Organization API (TASK-201)."""

from pydantic import BaseModel, Field

# A DNS-label-safe slug: lowercase letters, digits, hyphens.
_SLUG_PATTERN = r"^[a-z0-9-]+$"

# ── Request models ─────────────────────────────────────────────────────────


class OrganizationCreate(BaseModel):
    name: str = Field(..., min_length=1, max_length=255)
    description: str = Field(default="", max_length=2000)
    settings: dict = Field(default_factory=dict)
    slug: str | None = Field(
        default=None,
        min_length=1,
        max_length=63,
        pattern=_SLUG_PATTERN,
        description="Optional subdomain slug; auto-generated from name if omitted",
    )
    logo_url: str | None = Field(default=None, max_length=512)


class OrganizationUpdate(BaseModel):
    name: str | None = Field(default=None, min_length=1, max_length=255)
    description: str | None = Field(default=None, max_length=2000)
    settings: dict | None = Field(default=None)
    logo_url: str | None = Field(default=None, max_length=512)


# ── Response models ────────────────────────────────────────────────────────


class OrganizationResponse(BaseModel):
    id: str
    name: str
    slug: str
    description: str
    logo_url: str | None = None
    owner_user_id: str
    settings: dict
    status: str
    created_at: str
    updated_at: str
    org_role: str | None = Field(
        default=None,
        description="The calling user's role on this org: owner|admin|member",
    )


class PublicOrganizationResponse(BaseModel):
    """Minimal, non-sensitive org info served by the unauthenticated
    by-slug lookup. Deliberately excludes owner, settings, members, counts."""

    id: str
    name: str
    slug: str
    status: str
    logo_url: str | None = None


class PublicInvitationResponse(BaseModel):
    """Minimal invitation info for the pre-accept invitation screen
    (unauthenticated peek by token)."""

    org_name: str
    org_logo_url: str | None = None
    invited_email: str
    status: str
