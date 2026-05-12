"""Schemas for agent integration layer (STORY-022)."""

from pydantic import BaseModel, Field


class PublishAgentRequest(BaseModel):
    slug: str = Field(
        pattern=r"^[a-z0-9-]{3,64}$", description="URL-safe unique identifier"
    )
    cors_origins: list[str] = Field(default_factory=list)
    rate_limit_rpm: int = Field(default=60, ge=1, le=1000)
    egress_url: str | None = None


class PublishAgentResponse(BaseModel):
    slug: str
    endpoint_url: str
    integration_key: str  # shown ONCE at publish time
    key_last4: str


class PublishedAgentResponse(BaseModel):
    agent_id: str
    slug: str
    cors_origins: list[str]
    rate_limit_rpm: int
    egress_url: str | None
    key_last4: str
    published_at: int
    unpublished_at: int | None


class RotateKeyResponse(BaseModel):
    integration_key: str  # new key, shown once
    key_last4: str
